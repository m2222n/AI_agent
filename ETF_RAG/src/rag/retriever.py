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
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import SIMILARITY_THRESHOLD, TOP_K_RESULTS, HYBRID_SEARCH, RERANK, COHERE_API_KEY, PERSIST_DIR
from src.rag.utils import compute_docs_hash as _compute_docs_hash

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




# BM25 캐시 디렉토리
BM25_CACHE_DIR = PERSIST_DIR / "bm25_cache"


def _load_bm25_cache(docs_hash: str) -> Optional[Tuple[BM25Okapi, List[List[str]]]]:
    """캐시된 BM25 인덱스 로드. 해시 불일치 또는 파일 없으면 None."""
    cache_path = BM25_CACHE_DIR / "bm25_index.pkl"
    hash_path = BM25_CACHE_DIR / "docs_hash.txt"
    try:
        if not cache_path.exists() or not hash_path.exists():
            return None
        saved_hash = hash_path.read_text().strip()
        if saved_hash != docs_hash:
            logger.info(f"BM25 캐시 해시 불일치: {saved_hash} != {docs_hash}")
            return None
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        bm25 = data["bm25"]
        tokenized_corpus = data["tokenized_corpus"]
        logger.info(f"BM25 캐시 로드 성공 ({len(tokenized_corpus)}개 문서)")
        return bm25, tokenized_corpus
    except Exception as e:
        logger.warning(f"BM25 캐시 로드 실패: {e}")
        return None


def _save_bm25_cache(
    bm25: BM25Okapi, tokenized_corpus: List[List[str]], docs_hash: str
) -> None:
    """BM25 인덱스를 디스크에 캐싱."""
    try:
        BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = BM25_CACHE_DIR / "bm25_index.pkl"
        hash_path = BM25_CACHE_DIR / "docs_hash.txt"
        with open(cache_path, "wb") as f:
            pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)
        hash_path.write_text(docs_hash)
        logger.info(f"BM25 캐시 저장 완료 ({len(tokenized_corpus)}개 문서, hash={docs_hash})")
    except Exception as e:
        logger.warning(f"BM25 캐시 저장 실패: {e}")


class HybridRetriever:
    """FAISS(dense) + BM25(sparse) 하이브리드 검색기"""

    # 영문→한글 별칭 (종목명에 영문이 포함된 경우의 한글 표기)
    _NAME_ALIASES = {
        "posco": "포스코",
        "lg": "엘지",
        "sk": "에스케이",
        "sdi": "에스디아이",
        "dx": "디엑스",
        "hd": "에이치디",
        "ks": "케이에스",
    }

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
                name_lower = name.lower()
                self._name_index[name_lower] = i
                # 영문→한글 별칭도 등록 (POSCO홀딩스 → 포스코홀딩스)
                alias = self._make_korean_alias(name_lower)
                if alias and alias != name_lower:
                    self._name_index[alias] = i
            if ticker:
                self._ticker_index[ticker] = i

        # BM25 인덱스 구축 (pickle 캐시 활용)
        docs_hash = _compute_docs_hash(documents)
        cached = _load_bm25_cache(docs_hash)
        if cached is not None:
            self.bm25, _tokenized = cached
        else:
            tokenized_corpus = [tokenize_korean(doc.page_content) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            _save_bm25_cache(self.bm25, tokenized_corpus, docs_hash)

        logger.info(
            f"HybridRetriever 초기화: {len(documents)}개 문서, "
            f"dense_weight={HYBRID_SEARCH['dense_weight']}, "
            f"sparse_weight={HYBRID_SEARCH['sparse_weight']}"
        )

    @classmethod
    def _make_korean_alias(cls, name: str) -> Optional[str]:
        """영문 포함 종목명의 한글 별칭 생성.
        예: 'posco홀딩스' → '포스코홀딩스'"""
        result = name
        changed = False
        for eng, kor in cls._NAME_ALIASES.items():
            if eng in result:
                result = result.replace(eng, kor)
                changed = True
        return result if changed else None

    def _match_etf_by_name(self, query: str) -> List[Tuple[Document, float]]:
        """질문에서 ETF 이름/티커를 찾아 직접 매칭.

        전략:
        1. 6자리 티커 직접 매칭 (score=1.0)
        2. ETF 전체 이름이 쿼리에 포함 (score=1.0, 긴 이름 우선)
        3. 쿼리 키워드가 ETF 이름에 포함 — 부분 매칭 (score=0.8)
           예: "반도체" → "KODEX 반도체", "나스닥" → "TIGER 미국나스닥100"

        반환: 매칭된 [(Document, score), ...]
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

        # 2) ETF 전체 이름 매칭 — 긴 이름부터 시도 (greedy matching)
        sorted_names = sorted(self._name_index.keys(), key=len, reverse=True)
        for doc_name in sorted_names:
            if doc_name in q_lower:
                idx = self._name_index[doc_name]
                ticker = self.documents[idx].metadata.get("ticker", "")
                if ticker not in seen_tickers:
                    matched.append((self.documents[idx], 1.0))
                    seen_tickers.add(ticker)

        # 2b) 접두사 매칭 — 쿼리 내 연속 단어가 ETF 이름의 시작 부분과 일치
        #     "TIGER 차이나전기차 ETF" → "TIGER 차이나전기차SOLACTIVE" 매칭
        if not matched:
            q_words = q_lower.split()
            for w_count in range(min(3, len(q_words)), 0, -1):
                if matched:
                    break
                prefix = " ".join(q_words[:w_count])
                if len(prefix) < 5:
                    continue
                for doc_name in sorted_names:
                    # doc_name이 prefix로 시작하거나, 공백 없는 버전으로 비교
                    if (doc_name.startswith(prefix)
                            or doc_name.replace(" ", "").startswith(prefix.replace(" ", ""))):
                        idx = self._name_index[doc_name]
                        ticker = self.documents[idx].metadata.get("ticker", "")
                        if ticker not in seen_tickers:
                            matched.append((self.documents[idx], 0.95))
                            seen_tickers.add(ticker)
                            break  # prefix 길이별로 1개만

        # 3) 부분 매칭 — 쿼리의 주요 키워드가 ETF 이름에 포함되는지
        #    정확 매칭이 없을 때만 시도 (너무 많은 결과 방지)
        if not matched:
            # 쿼리에서 의미있는 키워드 추출 (2글자 이상, 일반어 제외)
            _STOP_WORDS = {"etf", "알리", "최근", "성과", "보유", "종목",
                           "투자", "좋", "위험", "어때", "알려"}
            query_keywords = []
            for token in tokenize_korean(query):
                t = token.lower()
                if len(t) >= 2 and t not in _STOP_WORDS:
                    query_keywords.append(t)

            # 이름이 짧은 순으로 정렬 → 대표 ETF 우선 (KODEX 반도체 > KODEX AI반도체핵심장비)
            names_by_short = sorted(self._name_index.keys(), key=len)

            # 키워드별로 대표 1개씩 매칭 (각 키워드의 최단 이름 ETF)
            kw_matched = {}  # keyword → (doc, score)
            for doc_name in names_by_short:
                for kw in query_keywords:
                    if kw in doc_name and kw not in kw_matched:
                        idx = self._name_index[doc_name]
                        ticker = self.documents[idx].metadata.get("ticker", "")
                        if ticker not in seen_tickers:
                            kw_matched[kw] = (self.documents[idx], 0.8)
                            seen_tickers.add(ticker)
                            break

            matched.extend(kw_matched.values())

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

        if not query or not query.strip():
            logger.warning("빈 검색 쿼리 — 빈 결과 반환")
            return []

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

        # 4. Cohere Rerank — cross-encoder 재정렬 (API 키 있을 때만)
        if RERANK.get("enabled") and len(candidates) > 1:
            candidates = self._rerank(query, candidates)

        # 5. MMR — 관련성과 다양성의 균형
        if use_mmr and len(candidates) > final_k:
            candidates = self._apply_mmr(
                candidates, final_k, lambda_param=HYBRID_SEARCH.get("mmr_lambda", 0.7)
            )

        # 6. 이름 매칭 결과를 최상위에 병합 (중복 제거)
        if name_matched:
            remaining_k = final_k - len(name_matched)
            hybrid_filtered = [
                (doc, score) for doc, score in candidates
                if doc.metadata.get("ticker") not in matched_tickers
            ][:remaining_k]
            return name_matched + hybrid_filtered

        return candidates[:final_k]

    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """Cohere Rerank로 후보 문서 재정렬.

        RRF 결합 후 호출되며, cross-encoder가 query-document 관련성을 직접 평가.
        실패 시 원래 순서 그대로 반환 (graceful fallback).
        """
        try:
            import cohere

            co = cohere.ClientV2(api_key=COHERE_API_KEY)
            docs_text = [doc.page_content for doc, _ in candidates]
            top_n = RERANK.get("top_n", HYBRID_SEARCH.get("final_k", 5))

            response = co.rerank(
                model=RERANK.get("model", "rerank-v3.5"),
                query=query,
                documents=docs_text,
                top_n=min(top_n, len(candidates)),
            )

            reranked = []
            for result in response.results:
                doc, _ = candidates[result.index]
                reranked.append((doc, result.relevance_score))

            logger.info(
                f"Cohere Rerank 완료: {len(candidates)}개 → {len(reranked)}개, "
                f"top score={reranked[0][1]:.4f}" if reranked else "empty"
            )
            return reranked

        except Exception as e:
            logger.warning(f"Cohere Rerank 실패, RRF 결과 사용: {e}")
            return candidates

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
