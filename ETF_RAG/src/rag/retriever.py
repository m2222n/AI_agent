"""
하이브리드 검색 모듈 (FAISS Dense + Kiwi BM25 Sparse + MMR)

검색 흐름:
0. 쿼리에서 ETF 이름/티커 추출 → 직접 매칭 (exact match)
1. 쿼리 → Kiwi 형태소 분석 → BM25 키워드 검색 (sparse)
2. 쿼리 → OpenAI 임베딩 → FAISS 벡터 검색 (dense)
3. RRF (Reciprocal Rank Fusion)로 두 결과를 결합
4. MMR (Maximal Marginal Relevance)로 다양성 확보
5. 최종 top-k 반환
"""

import logging
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import SIMILARITY_THRESHOLD, TOP_K_RESULTS, HYBRID_SEARCH

logger = logging.getLogger(__name__)

# Kiwi 형태소 분석기 (싱글턴)
_kiwi = None


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def tokenize_korean(text: str) -> List[str]:
    """Kiwi 형태소 분석기로 한국어 텍스트를 토큰화 (명사/동사/형용사 추출)"""
    kiwi = _get_kiwi()
    tokens = []
    for token in kiwi.tokenize(text):
        # NNG(일반명사), NNP(고유명사), VV(동사), VA(형용사), SL(외국어)
        if token.tag in ("NNG", "NNP", "VV", "VA", "SL"):
            tokens.append(token.form)
    return tokens




class HybridRetriever:
    """FAISS(dense) + BM25(sparse) 하이브리드 검색기"""

    def __init__(self, vectorstore: FAISS, documents: List[Document]):
        self.vectorstore = vectorstore
        self.documents = documents

        # ETF 이름 → Document 인덱스 매핑 (정확 매칭용)
        self._name_index: Dict[str, int] = {}
        self._ticker_index: Dict[str, int] = {}
        for i, doc in enumerate(documents):
            name = doc.metadata.get("name", "")
            ticker = doc.metadata.get("ticker", "")
            if name:
                self._name_index[name.lower()] = i
            if ticker:
                self._ticker_index[ticker] = i

        # BM25 인덱스 구축
        tokenized_corpus = [tokenize_korean(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(
            f"HybridRetriever 초기화: {len(documents)}개 문서, "
            f"dense_weight={HYBRID_SEARCH['dense_weight']}, "
            f"sparse_weight={HYBRID_SEARCH['sparse_weight']}"
        )

    def _match_etf_by_name(self, query: str) -> List[Tuple[Document, float]]:
        """질문에서 ETF 이름/티커를 찾아 직접 매칭.

        전략: 실제 ETF 이름 목록을 질문 텍스트에 대해 매칭.
        긴 이름부터 매칭하여 "KODEX 200선물인버스2X"가 "KODEX 200"보다 먼저 매칭.

        반환: 매칭된 [(Document, 1.0), ...] — 매칭 시 최고 점수 부여
        """
        q_lower = query.lower()
        matched = []
        seen_tickers = set()

        # 1) 6자리 티커 직접 매칭
        for ticker_match in re.finditer(r"\b(\d{6})\b", query):
            ticker = ticker_match.group(1)
            if ticker in self._ticker_index and ticker not in seen_tickers:
                idx = self._ticker_index[ticker]
                matched.append((self.documents[idx], 1.0))
                seen_tickers.add(ticker)

        # 2) ETF 이름 매칭 — 긴 이름부터 시도 (greedy matching)
        sorted_names = sorted(self._name_index.keys(), key=len, reverse=True)
        for doc_name in sorted_names:
            if doc_name in q_lower:
                idx = self._name_index[doc_name]
                ticker = self.documents[idx].metadata.get("ticker", "")
                if ticker not in seen_tickers:
                    matched.append((self.documents[idx], 1.0))
                    seen_tickers.add(ticker)

        if matched:
            logger.info(f"ETF 이름 매칭: {[d.metadata['name'] for d, _ in matched]}")

        return matched

    def search(
        self, query: str, final_k: Optional[int] = None, use_mmr: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        하이브리드 검색 실행

        Args:
            query: 검색 쿼리
            final_k: 최종 반환 문서 수
            use_mmr: MMR 적용 여부 (True면 다양성 확보)

        Returns:
            [(Document, score), ...] — score가 높을수록 관련도 높음
        """
        if final_k is None:
            final_k = HYBRID_SEARCH["final_k"]

        # 0. ETF 이름/티커 직접 매칭 (정확도 최우선)
        name_matched = self._match_etf_by_name(query)
        matched_tickers = {d.metadata.get("ticker") for d, _ in name_matched}

        # 이름 매칭만으로 충분하면 (질문이 특정 ETF만 묻는 경우) 바로 반환
        if len(name_matched) >= final_k:
            return name_matched[:final_k]

        dense_k = HYBRID_SEARCH["dense_k"]
        bm25_k = HYBRID_SEARCH["bm25_k"]

        # 1. FAISS dense 검색
        dense_results = self.vectorstore.similarity_search_with_score(query, k=dense_k)

        # 2. BM25 sparse 검색
        query_tokens = tokenize_korean(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # BM25 상위 k개 인덱스
        sorted_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:bm25_k]

        # 3. RRF (Reciprocal Rank Fusion) 결합
        rrf_scores = {}
        rrf_k = 60  # RRF 상수 (표준값)

        dense_weight = HYBRID_SEARCH["dense_weight"]
        sparse_weight = HYBRID_SEARCH["sparse_weight"]

        # Dense 결과 RRF
        for rank, (doc, _faiss_dist) in enumerate(dense_results):
            doc_key = self._doc_key(doc)
            rrf_score = dense_weight * (1.0 / (rrf_k + rank + 1))
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + rrf_score

        # Sparse 결과 RRF
        for rank, idx in enumerate(sorted_bm25_indices):
            if bm25_scores[idx] <= 0:
                continue
            doc = self.documents[idx]
            doc_key = self._doc_key(doc)
            rrf_score = sparse_weight * (1.0 / (rrf_k + rank + 1))
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + rrf_score

        # doc_key → Document 매핑
        doc_map = {}
        for doc, _ in dense_results:
            doc_map[self._doc_key(doc)] = doc
        for idx in sorted_bm25_indices:
            doc = self.documents[idx]
            doc_map[self._doc_key(doc)] = doc

        # RRF 정렬 — MMR 적용 시 후보를 넓게 가져감
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fetch_k = final_k * 3 if use_mmr else final_k
        candidates = []
        for doc_key, score in sorted_results[:fetch_k]:
            if doc_key in doc_map:
                candidates.append((doc_map[doc_key], score))

        # 4. MMR — 관련성과 다양성의 균형
        if use_mmr and len(candidates) > final_k:
            candidates = self._apply_mmr(
                candidates, final_k, lambda_param=HYBRID_SEARCH.get("mmr_lambda", 0.7)
            )

        # 5. 이름 매칭 결과를 최상위에 병합 (중복 제거)
        if name_matched:
            remaining_k = final_k - len(name_matched)
            hybrid_filtered = [
                (doc, score) for doc, score in candidates
                if doc.metadata.get("ticker") not in matched_tickers
            ][:remaining_k]
            return name_matched + hybrid_filtered

        return candidates[:final_k]

    def _apply_mmr(
        self,
        candidates: List[Tuple[Document, float]],
        k: int,
        lambda_param: float = 0.7,
    ) -> List[Tuple[Document, float]]:
        """MMR (Maximal Marginal Relevance) 적용

        BM25 토큰 기반 유사도로 문서 간 다양성 확보.
        lambda_param: 1.0이면 관련성만, 0.0이면 다양성만.
        """
        if len(candidates) <= k:
            return candidates

        # 각 문서의 BM25 토큰 벡터 (간소화: 토큰 집합 기반 Jaccard 유사도)
        doc_tokens = [set(tokenize_korean(doc.page_content)) for doc, _ in candidates]

        selected = [0]  # 첫 번째 (최고 RRF 점수)는 무조건 선택
        remaining = list(range(1, len(candidates)))

        while len(selected) < k and remaining:
            best_idx = None
            best_mmr = -float("inf")

            for idx in remaining:
                # 관련성: RRF 점수 (이미 정규화됨)
                relevance = candidates[idx][1]

                # 다양성: 이미 선택된 문서들과의 최대 유사도
                max_sim = 0.0
                for sel_idx in selected:
                    sim = self._jaccard_similarity(doc_tokens[idx], doc_tokens[sel_idx])
                    if sim > max_sim:
                        max_sim = sim

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [candidates[i] for i in selected]

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard 유사도 (토큰 집합 기반)"""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """Document의 고유 키 생성 (ticker 우선, 없으면 id)"""
        return doc.metadata.get("ticker", "") or doc.metadata.get("id", "")


def retrieve_relevant_docs(
    retriever, query: str, k: int = TOP_K_RESULTS
) -> Tuple[Optional[str], List[dict]]:
    """
    관련 문서 검색 (하이브리드 또는 FAISS 단독)

    retriever: HybridRetriever 또는 FAISS (하위 호환)
    """
    # 하이브리드 검색
    if isinstance(retriever, HybridRetriever):
        results = retriever.search(query, final_k=k)
        if not results:
            return None, []

        # RRF 최소 점수 필터링 — 무관한 결과 제거
        min_score = HYBRID_SEARCH.get("min_rrf_score", 0.0)
        results = [(doc, score) for doc, score in results if score >= min_score]
        if not results:
            logger.info(f"모든 검색 결과가 min_rrf_score({min_score}) 미달 — 관련 문서 없음")
            return None, []

        context_parts = []
        sources = []
        for doc, score in results:
            label = doc.metadata.get("id") or doc.metadata.get("ticker", "")
            context_parts.append(f"[{label}] {doc.page_content}")
            sources.append({
                "id": doc.metadata.get("id", doc.metadata.get("ticker", "")),
                "name": doc.metadata["name"],
                "ticker": doc.metadata["ticker"],
                "relevance_score": round(score * 100, 1),  # RRF score → 백분율 표시
            })

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    # Fallback: FAISS 단독 검색 (하위 호환)
    results = retriever.similarity_search_with_score(query, k=k)
    filtered_results = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]

    if not filtered_results:
        return None, []

    context_parts = []
    sources = []
    for doc, score in filtered_results:
        label = doc.metadata.get("id") or doc.metadata.get("ticker", "")
        context_parts.append(f"[{label}] {doc.page_content}")
        sources.append({
            "id": doc.metadata.get("id", doc.metadata.get("ticker", "")),
            "name": doc.metadata["name"],
            "ticker": doc.metadata["ticker"],
            "relevance_score": round(1 - score / 2, 2),
        })

    context = "\n\n---\n\n".join(context_parts)
    return context, sources
