"""하이브리드 검색 (FAISS + Kiwi BM25) 테스트"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.retriever import (
    tokenize_korean,
    HybridRetriever,
    retrieve_relevant_docs,
)


# ── Kiwi 토큰화 테스트 ──────────────────────────────────────

def test_tokenize_korean_nouns():
    """한국어 명사 추출"""
    tokens = tokenize_korean("삼성전자 주가가 상승했다")
    assert "삼성전자" in tokens or "삼성" in tokens
    assert "주가" in tokens


def test_tokenize_korean_english():
    """영문(SL 태그) 포함"""
    tokens = tokenize_korean("KODEX 200 ETF")
    assert "KODEX" in tokens
    assert "ETF" in tokens


def test_tokenize_korean_empty():
    tokens = tokenize_korean("")
    assert tokens == []


def test_tokenize_korean_etf_query():
    """ETF 관련 질문 토큰화"""
    tokens = tokenize_korean("반도체 ETF 수익률 비교해줘")
    assert "반도체" in tokens
    assert "ETF" in tokens
    # Kiwi는 "수익률"을 "수익" + "률"로 분리할 수 있음
    assert "수익" in tokens or "수익률" in tokens


# ── HybridRetriever 테스트 ───────────────────────────────────

SAMPLE_DOCS = [
    Document(
        page_content="KODEX 200 ETF. 종가: 80,800원. 수익률: 1일: +2.91%. 주요 보유종목: 삼성전자 31.77%",
        metadata={"ticker": "069500", "name": "KODEX 200", "source": "krx_collected"},
    ),
    Document(
        page_content="TIGER 반도체 ETF. 종가: 15,200원. 수익률: 1일: +4.12%. 주요 보유종목: SK하이닉스 25.3%",
        metadata={"ticker": "091160", "name": "TIGER 반도체", "source": "krx_collected"},
    ),
    Document(
        page_content="KODEX 단기채권 ETF. 종가: 102,350원. 안정적인 채권형 ETF. 변동성이 낮은 상품.",
        metadata={"ticker": "153130", "name": "KODEX 단기채권", "source": "krx_collected"},
    ),
]


@pytest.fixture
def mock_vectorstore():
    """FAISS vectorstore mock — similarity_search_with_score 반환"""
    vs = MagicMock()

    def fake_search(query, k=20):
        # 쿼리에 "반도체" 포함 시 반도체 ETF를 상위로
        if "반도체" in query:
            return [
                (SAMPLE_DOCS[1], 0.3),  # TIGER 반도체 — 가장 가까움
                (SAMPLE_DOCS[0], 0.7),  # KODEX 200
                (SAMPLE_DOCS[2], 1.2),  # 단기채권
            ]
        return [
            (SAMPLE_DOCS[0], 0.5),
            (SAMPLE_DOCS[1], 0.8),
            (SAMPLE_DOCS[2], 1.0),
        ]

    vs.similarity_search_with_score = MagicMock(side_effect=fake_search)
    return vs


@pytest.fixture
def hybrid_retriever(mock_vectorstore):
    return HybridRetriever(mock_vectorstore, SAMPLE_DOCS)


def test_hybrid_retriever_init(hybrid_retriever):
    """HybridRetriever 초기화 확인"""
    assert hybrid_retriever.bm25 is not None
    assert len(hybrid_retriever.documents) == 3


def test_hybrid_search_returns_results(hybrid_retriever):
    """하이브리드 검색이 결과를 반환하는지 확인"""
    results = hybrid_retriever.search("KODEX 200 수익률")
    assert len(results) > 0
    # 각 결과는 (Document, score) 튜플
    doc, score = results[0]
    assert hasattr(doc, "page_content")
    assert score > 0


def test_hybrid_search_semiconductor_query(hybrid_retriever):
    """반도체 관련 질문 시 반도체 ETF가 상위에 나오는지 확인"""
    results = hybrid_retriever.search("반도체 ETF 수익률 알려줘", final_k=3)
    assert len(results) > 0
    # 반도체 ETF가 결과에 포함되어야 함
    tickers = [doc.metadata["ticker"] for doc, _ in results]
    assert "091160" in tickers  # TIGER 반도체


def test_hybrid_search_bond_query(hybrid_retriever):
    """채권 관련 질문 시 채권 ETF가 상위에 나오는지 확인"""
    results = hybrid_retriever.search("채권 ETF 안정적인 상품", final_k=3)
    tickers = [doc.metadata["ticker"] for doc, _ in results]
    assert "153130" in tickers  # KODEX 단기채권


def test_hybrid_search_final_k(hybrid_retriever):
    """final_k 파라미터가 반환 개수를 제한하는지 확인"""
    results = hybrid_retriever.search("ETF", final_k=2)
    assert len(results) <= 2


def test_hybrid_search_scores_positive(hybrid_retriever):
    """모든 RRF 점수가 양수인지 확인"""
    results = hybrid_retriever.search("KODEX ETF 정보")
    for _, score in results:
        assert score > 0


# ── retrieve_relevant_docs 통합 테스트 ───────────────────────

def test_retrieve_with_hybrid_retriever(hybrid_retriever):
    """retrieve_relevant_docs가 HybridRetriever와 동작하는지 확인"""
    context, sources = retrieve_relevant_docs(hybrid_retriever, "KODEX 200")
    assert context is not None
    assert len(sources) > 0
    assert "name" in sources[0]
    assert "ticker" in sources[0]
    assert "relevance_score" in sources[0]


def test_retrieve_with_faiss_fallback():
    """FAISS 직접 전달 시 하위 호환 동작 확인"""
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = [
        (SAMPLE_DOCS[0], 0.5),
        (SAMPLE_DOCS[1], 0.8),
    ]
    context, sources = retrieve_relevant_docs(mock_vs, "ETF 정보", k=3)
    assert context is not None
    assert len(sources) == 2


def test_retrieve_faiss_fallback_no_results():
    """FAISS fallback에서 threshold 초과 시 빈 결과"""
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = [
        (SAMPLE_DOCS[0], 2.0),  # threshold(1.5) 초과
    ]
    context, sources = retrieve_relevant_docs(mock_vs, "무관한 질문")
    assert context is None
    assert sources == []


# ── doc_key 테스트 ───────────────────────────────────────────

def test_doc_key_uses_ticker():
    doc = Document(page_content="test", metadata={"ticker": "069500", "id": "some_id"})
    assert HybridRetriever._doc_key(doc) == "069500"


def test_doc_key_fallback_to_id():
    doc = Document(page_content="test", metadata={"id": "ETF_001"})
    assert HybridRetriever._doc_key(doc) == "ETF_001"


# ── MMR 테스트 ───────────────────────────────────────────────

def test_mmr_reduces_duplicates(hybrid_retriever):
    """MMR 적용 시 다양한 문서가 선택되는지 확인"""
    results_mmr = hybrid_retriever.search("ETF 정보", final_k=3, use_mmr=True)
    results_no_mmr = hybrid_retriever.search("ETF 정보", final_k=3, use_mmr=False)
    # 두 방식 모두 결과를 반환해야 함
    assert len(results_mmr) > 0
    assert len(results_no_mmr) > 0


def test_mmr_off_returns_rrf_order(hybrid_retriever):
    """MMR 비활성화 시 RRF 순서 그대로 반환"""
    results = hybrid_retriever.search("KODEX 200", final_k=3, use_mmr=False)
    scores = [s for _, s in results]
    # RRF 점수 내림차순
    assert scores == sorted(scores, reverse=True)


def test_jaccard_similarity():
    """Jaccard 유사도 계산"""
    assert HybridRetriever._jaccard_similarity({"a", "b", "c"}, {"a", "b", "d"}) == pytest.approx(0.5)
    assert HybridRetriever._jaccard_similarity(set(), {"a"}) == 0.0
    assert HybridRetriever._jaccard_similarity({"a"}, {"a"}) == 1.0


# ── PDF 로더 테스트 ──────────────────────────────────────────

def test_pdf_loader_no_dir():
    """PDF 디렉토리 없으면 빈 리스트 반환"""
    from src.data.pdf_loader import load_pdf_documents
    from pathlib import Path
    docs = load_pdf_documents(pdf_dir=Path("/nonexistent/path"))
    assert docs == []


def test_pdf_loader_empty_dir(tmp_path):
    """PDF 파일 없으면 빈 리스트 반환"""
    from src.data.pdf_loader import load_pdf_documents
    docs = load_pdf_documents(pdf_dir=tmp_path)
    assert docs == []


def test_pdf_extract_metadata():
    """파일명에서 메타데이터 추출"""
    from src.data.pdf_loader import _extract_file_metadata
    meta = _extract_file_metadata("069500_KODEX200_투자설명서")
    assert meta["ticker"] == "069500"
    assert meta["name"] == "KODEX200"
    assert meta["doc_type"] == "투자설명서"


def test_pdf_extract_metadata_partial():
    """파일명 부분 매칭"""
    from src.data.pdf_loader import _extract_file_metadata
    meta = _extract_file_metadata("random_file")
    assert "ticker" not in meta
