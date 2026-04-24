"""tests/test_rerank.py — Cohere Rerank 통합 테스트."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.retriever import HybridRetriever


SAMPLE_DOCS = [
    Document(
        page_content="KODEX 200 ETF. 종가: 80,800원. 수익률: +2.91%. 삼성전자 31.77%",
        metadata={"ticker": "069500", "name": "KODEX 200", "source": "krx"},
    ),
    Document(
        page_content="TIGER 반도체 ETF. 종가: 15,200원. 수익률: +4.12%. SK하이닉스 25.3%",
        metadata={"ticker": "091160", "name": "TIGER 반도체", "source": "krx"},
    ),
    Document(
        page_content="KODEX 단기채권 ETF. 종가: 102,350원. 안정적인 채권형 ETF.",
        metadata={"ticker": "153130", "name": "KODEX 단기채권", "source": "krx"},
    ),
    Document(
        page_content="TIGER 미국S&P500 ETF. 종가: 18,500원. 미국 대형주 추종.",
        metadata={"ticker": "360750", "name": "TIGER 미국S&P500", "source": "krx"},
    ),
]


@pytest.fixture
def mock_vectorstore():
    vs = MagicMock()
    vs.similarity_search_with_score = MagicMock(
        return_value=[(doc, 0.5 + i * 0.2) for i, doc in enumerate(SAMPLE_DOCS)]
    )
    return vs


@pytest.fixture
def retriever(mock_vectorstore):
    return HybridRetriever(mock_vectorstore, SAMPLE_DOCS)


# ── _rerank 메서드 직접 테스트 ──────────────────────────────


class TestRerankMethod:
    """_rerank() 메서드 단위 테스트."""

    def test_rerank_reorders_candidates(self, retriever):
        """Cohere Rerank가 후보를 재정렬."""
        candidates = [(SAMPLE_DOCS[0], 0.5), (SAMPLE_DOCS[1], 0.4), (SAMPLE_DOCS[2], 0.3)]

        # Mock Cohere 응답: 2번째(반도체)를 1위로 재정렬
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(index=1, relevance_score=0.95),
            MagicMock(index=0, relevance_score=0.72),
            MagicMock(index=2, relevance_score=0.15),
        ]

        with patch("src.rag.retriever.RERANK", {"enabled": True, "model": "rerank-v3.5", "top_n": 3}), \
             patch("src.rag.retriever.COHERE_API_KEY", "test-key"), \
             patch("cohere.ClientV2") as mock_cohere:
            mock_cohere.return_value.rerank.return_value = mock_result
            result = retriever._rerank("반도체 ETF", candidates)

        assert len(result) == 3
        # 반도체 ETF가 1위로 올라옴
        assert result[0][0].metadata["ticker"] == "091160"
        assert result[0][1] == 0.95

    def test_rerank_respects_top_n(self, retriever):
        """top_n 설정에 따라 결과 수 제한."""
        candidates = [(doc, 0.5 - i * 0.1) for i, doc in enumerate(SAMPLE_DOCS)]

        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(index=1, relevance_score=0.9),
            MagicMock(index=0, relevance_score=0.7),
        ]

        with patch("src.rag.retriever.RERANK", {"enabled": True, "model": "rerank-v3.5", "top_n": 2}), \
             patch("src.rag.retriever.COHERE_API_KEY", "test-key"), \
             patch("cohere.ClientV2") as mock_cohere:
            mock_cohere.return_value.rerank.return_value = mock_result
            result = retriever._rerank("ETF 비교", candidates)

        assert len(result) == 2

    def test_rerank_fallback_on_error(self, retriever):
        """Cohere API 실패 시 원래 순서 유지."""
        candidates = [(SAMPLE_DOCS[0], 0.5), (SAMPLE_DOCS[1], 0.4)]

        with patch("src.rag.retriever.RERANK", {"enabled": True, "model": "rerank-v3.5", "top_n": 5}), \
             patch("src.rag.retriever.COHERE_API_KEY", "test-key"), \
             patch("cohere.ClientV2") as mock_cohere:
            mock_cohere.return_value.rerank.side_effect = Exception("API error")
            result = retriever._rerank("test query", candidates)

        # 원래 candidates 그대로 반환
        assert len(result) == 2
        assert result[0][0].metadata["ticker"] == "069500"
        assert result[0][1] == 0.5

    def test_rerank_fallback_on_import_error(self, retriever):
        """cohere 패키지 없을 때 graceful fallback."""
        candidates = [(SAMPLE_DOCS[0], 0.5)]

        with patch("src.rag.retriever.RERANK", {"enabled": True, "model": "rerank-v3.5", "top_n": 5}), \
             patch("src.rag.retriever.COHERE_API_KEY", "test-key"), \
             patch.dict("sys.modules", {"cohere": None}):
            result = retriever._rerank("test", candidates)

        assert result == candidates

    def test_rerank_uses_page_content(self, retriever):
        """rerank에 page_content 텍스트를 전달하는지 확인."""
        candidates = [(SAMPLE_DOCS[0], 0.5), (SAMPLE_DOCS[1], 0.4)]

        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(index=0, relevance_score=0.8),
            MagicMock(index=1, relevance_score=0.6),
        ]

        with patch("src.rag.retriever.RERANK", {"enabled": True, "model": "rerank-v3.5", "top_n": 5}), \
             patch("src.rag.retriever.COHERE_API_KEY", "test-key"), \
             patch("cohere.ClientV2") as mock_cohere:
            mock_cohere.return_value.rerank.return_value = mock_result
            retriever._rerank("KODEX 200", candidates)

        # rerank 호출 시 documents에 page_content가 전달됨
        call_kwargs = mock_cohere.return_value.rerank.call_args
        docs_arg = call_kwargs.kwargs.get("documents") or call_kwargs[1].get("documents")
        assert docs_arg[0] == SAMPLE_DOCS[0].page_content
        assert docs_arg[1] == SAMPLE_DOCS[1].page_content


# ── search() 통합 테스트 (Rerank 활성/비활성) ──────────────


class TestSearchWithRerank:
    """search()에서 Rerank 동작 통합 테스트."""

    def test_search_with_rerank_enabled(self, retriever):
        """RERANK enabled일 때 _rerank가 호출됨."""
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(index=0, relevance_score=0.9),
            MagicMock(index=1, relevance_score=0.7),
        ]

        with patch("src.rag.retriever.RERANK", {"enabled": True, "model": "rerank-v3.5", "top_n": 5}), \
             patch("src.rag.retriever.COHERE_API_KEY", "test-key"), \
             patch("cohere.ClientV2") as mock_cohere:
            mock_cohere.return_value.rerank.return_value = mock_result
            results = retriever.search("반도체 ETF 추천", final_k=2)

        assert len(results) > 0
        # Cohere rerank가 실제로 호출됨
        mock_cohere.return_value.rerank.assert_called_once()

    def test_search_with_rerank_disabled(self, retriever):
        """RERANK disabled일 때 _rerank 미호출."""
        with patch("src.rag.retriever.RERANK", {"enabled": False}), \
             patch("cohere.ClientV2") as mock_cohere:
            results = retriever.search("반도체 ETF", final_k=2)

        assert len(results) > 0
        mock_cohere.return_value.rerank.assert_not_called()

    def test_search_name_match_bypasses_rerank(self, retriever):
        """이름 매칭으로 충분하면 rerank 미호출."""
        with patch("src.rag.retriever.RERANK", {"enabled": True, "model": "rerank-v3.5", "top_n": 5}), \
             patch("src.rag.retriever.COHERE_API_KEY", "test-key"), \
             patch("cohere.ClientV2") as mock_cohere:
            # KODEX 200 + TIGER 반도체 + KODEX 단기채권 = 3개 이름 매칭, final_k=3 이면 바로 반환
            results = retriever.search(
                "KODEX 200 TIGER 반도체 KODEX 단기채권", final_k=3
            )

        # 이름 매칭 3개로 충분 → rerank 미호출
        mock_cohere.return_value.rerank.assert_not_called()
        assert len(results) == 3

    def test_search_rerank_then_mmr(self, retriever):
        """Rerank → MMR 순서로 적용됨."""
        # Rerank가 4개 반환, MMR이 final_k=2로 줄임
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(index=i, relevance_score=0.9 - i * 0.1) for i in range(4)
        ]

        with patch("src.rag.retriever.RERANK", {"enabled": True, "model": "rerank-v3.5", "top_n": 10}), \
             patch("src.rag.retriever.COHERE_API_KEY", "test-key"), \
             patch("cohere.ClientV2") as mock_cohere:
            mock_cohere.return_value.rerank.return_value = mock_result
            results = retriever.search("ETF 추천", final_k=2, use_mmr=True)

        assert len(results) <= 2


class TestRerankConfig:
    """config.py RERANK 설정 검증."""

    def test_rerank_config_keys(self):
        """RERANK dict에 필수 키 존재."""
        from config import RERANK
        assert "enabled" in RERANK
        assert "model" in RERANK
        assert "top_n" in RERANK

    def test_rerank_model_name(self):
        """Rerank 모델명 확인."""
        from config import RERANK
        assert "rerank" in RERANK["model"]

    def test_rerank_top_n_reasonable(self):
        """top_n이 합리적 범위."""
        from config import RERANK
        assert 1 <= RERANK["top_n"] <= 20
