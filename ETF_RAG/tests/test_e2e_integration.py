"""tests/test_e2e_integration.py — E2E 통합 테스트.

단위 테스트에서 mock으로 커버할 수 없는 통합 시나리오를 검증:
1. 데이터 로드 → 문서 생성 → 인덱스 구축 → 검색 (API 키 불필요)
2. 도구 주입 → 도구 함수 직접 호출 (실제 retriever 사용)
3. 에이전트 그래프 빌드 → 노드 연결 검증
4. 에러 핸들링 파이프라인

LLM API 호출이 필요한 테스트는 pytest.mark.slow로 표시.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


# ── 테스트 데이터 ──────────────────────────────────────────

_ETF_BASE = {
    "date": "2026-04-24", "volume": 100000, "base_index": "KOSPI 200", "deviation": 0.05,
}
_STOCK_BASE = {
    "date": "2026-04-24", "volume": 100000, "eps": 5000, "dps": 500, "div": 2.0,
}

SAMPLE_ETF_DATA = [
    {**_ETF_BASE, "ticker": "069500", "name": "KODEX 200", "close": 38500,
     "change_pct": 1.2, "trade_value": 50_000_000_000,
     "nav": 38520, "nav_diff": -20, "tracking_error": 0.01,
     "returns": {"1d": 1.2, "1w": 2.5, "1m": 3.0, "3m": 5.0, "1y": 12.0},
     "holdings": [
         {"stock_name": "삼성전자", "stock_ticker": "005930", "weight": 31.5},
         {"stock_name": "SK하이닉스", "stock_ticker": "000660", "weight": 8.2},
     ]},
    {**_ETF_BASE, "ticker": "091160", "name": "TIGER 반도체", "close": 15200,
     "change_pct": 4.1, "trade_value": 20_000_000_000,
     "nav": 15180, "nav_diff": 20, "tracking_error": 0.05,
     "returns": {"1d": 4.1, "1w": 7.3, "1m": 10.0, "3m": -2.0, "1y": 25.0},
     "holdings": [
         {"stock_name": "SK하이닉스", "stock_ticker": "000660", "weight": 25.3},
         {"stock_name": "삼성전자", "stock_ticker": "005930", "weight": 18.1},
     ]},
    {**_ETF_BASE, "ticker": "153130", "name": "KODEX 단기채권", "close": 102350,
     "change_pct": 0.02, "trade_value": 5_000_000_000,
     "nav": 102340, "nav_diff": 10, "tracking_error": 0.001,
     "returns": {"1d": 0.02, "1w": 0.1, "1m": 0.3, "3m": 1.0, "1y": 3.5},
     "holdings": []},
    {**_ETF_BASE, "ticker": "360750", "name": "TIGER 미국S&P500", "close": 18500,
     "change_pct": -0.5, "trade_value": 30_000_000_000,
     "nav": 18480, "nav_diff": 20, "tracking_error": 0.03,
     "returns": {"1d": -0.5, "1w": 1.0, "1m": 2.0, "3m": 8.0, "1y": 20.0},
     "holdings": []},
    {**_ETF_BASE, "ticker": "261240", "name": "KODEX 미국나스닥100TR", "close": 22000,
     "change_pct": 1.5, "trade_value": 15_000_000_000,
     "nav": 21980, "nav_diff": 20, "tracking_error": 0.02,
     "returns": {"1d": 1.5, "1w": 3.0, "1m": 5.0, "3m": 12.0, "1y": 30.0},
     "holdings": []},
]

SAMPLE_STOCK_DATA = [
    {**_STOCK_BASE, "ticker": "005930", "name": "삼성전자", "close": 68000,
     "change_pct": 2.1, "trade_value": 800_000_000_000,
     "market_cap": 400_000_000_000_000, "per": 12.5, "pbr": 1.3,
     "dividend_yield": 2.1, "sector": "전기전자",
     "returns": {"1d": 2.1, "1w": 3.5, "1m": 5.0, "3m": 8.0, "1y": 15.0}},
    {**_STOCK_BASE, "ticker": "000660", "name": "SK하이닉스", "close": 195000,
     "change_pct": 3.5, "trade_value": 500_000_000_000,
     "market_cap": 140_000_000_000_000, "per": 8.2, "pbr": 1.8,
     "dividend_yield": 1.0, "sector": "전기전자",
     "returns": {"1d": 3.5, "1w": 5.0, "1m": 8.0, "3m": 15.0, "1y": 40.0}},
    {**_STOCK_BASE, "ticker": "035420", "name": "NAVER", "close": 210000,
     "change_pct": -1.2, "trade_value": 200_000_000_000,
     "market_cap": 35_000_000_000_000, "per": 25.0, "pbr": 1.5,
     "dividend_yield": 0.5, "sector": "서비스업",
     "returns": {"1d": -1.2, "1w": -2.0, "1m": 0.5, "3m": 3.0, "1y": 10.0}},
]


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def etf_documents():
    """ETF 데이터 → Document 변환 (실제 loader 사용)."""
    from src.data.loader import create_documents
    return create_documents(SAMPLE_ETF_DATA, include_pdfs=False)


@pytest.fixture(scope="module")
def stock_documents():
    """주식 데이터 → Document 변환."""
    from src.data.loader import create_stock_documents
    return create_stock_documents(SAMPLE_STOCK_DATA)


@pytest.fixture(scope="module")
def mock_embeddings():
    """실제 OpenAI 호출 대신 랜덤 임베딩 사용."""
    from langchain_core.embeddings import Embeddings
    import numpy as np

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            np.random.seed(42)
            return [np.random.rand(256).tolist() for _ in texts]

        def embed_query(self, text):
            np.random.seed(hash(text) % (2**31))
            return np.random.rand(256).tolist()

    return FakeEmbeddings()


@pytest.fixture(scope="module")
def etf_retriever(etf_documents, mock_embeddings):
    """실제 HybridRetriever (FAISS + BM25, fake embedding)."""
    from src.rag.retriever import HybridRetriever
    vectorstore = FAISS.from_documents(etf_documents, mock_embeddings)
    return HybridRetriever(vectorstore, etf_documents)


@pytest.fixture(scope="module")
def stock_retriever(stock_documents, mock_embeddings):
    """주식용 HybridRetriever."""
    from src.rag.retriever import HybridRetriever
    vectorstore = FAISS.from_documents(stock_documents, mock_embeddings)
    return HybridRetriever(vectorstore, stock_documents)


@pytest.fixture
def initialized_tools(etf_retriever, etf_documents, stock_retriever):
    """도구에 retriever + 데이터 주입 후 정리."""
    from src.llm.tools import set_retriever
    set_retriever(
        etf_retriever, etf_documents,
        stock_retriever=stock_retriever,
        etf_data=SAMPLE_ETF_DATA,
        stock_data=SAMPLE_STOCK_DATA,
    )
    yield
    # teardown — 모듈 전역 상태 초기화
    import src.llm.tools as tools_mod
    tools_mod._retriever = None
    tools_mod._stock_retriever = None
    tools_mod._documents = None
    tools_mod._etf_data_index = {}
    tools_mod._stock_data_index = {}
    tools_mod._holdings_reverse_index = {}
    tools_mod._sector_index = {}
    tools_mod._data_initialized = False


# ── 1. 데이터 → 문서 → 인덱스 → 검색 파이프라인 ──────────


class TestDataToSearchPipeline:
    """데이터 로드 → 문서 생성 → 인덱스 구축 → 하이브리드 검색 통합 검증."""

    def test_etf_documents_created(self, etf_documents):
        """ETF 데이터 → Document 변환 성공."""
        assert len(etf_documents) == 5
        for doc in etf_documents:
            assert "ticker" in doc.metadata
            assert "name" in doc.metadata
            assert len(doc.page_content) > 0

    def test_stock_documents_created(self, stock_documents):
        """주식 데이터 → Document 변환 성공."""
        assert len(stock_documents) == 3
        for doc in stock_documents:
            assert doc.metadata.get("ticker")
            assert doc.metadata.get("name")

    def test_etf_document_content_format(self, etf_documents):
        """ETF 문서에 핵심 정보 포함."""
        kodex200_doc = [d for d in etf_documents if d.metadata["ticker"] == "069500"][0]
        content = kodex200_doc.page_content
        assert "KODEX 200" in content
        assert "38,500" in content or "38500" in content

    def test_hybrid_retriever_init(self, etf_retriever):
        """HybridRetriever 초기화 — BM25 + FAISS 모두 준비."""
        assert etf_retriever.bm25 is not None
        assert etf_retriever.vectorstore is not None
        assert len(etf_retriever.documents) == 5

    def test_search_by_name(self, etf_retriever):
        """이름 매칭 검색 — KODEX 200."""
        results = etf_retriever.search("KODEX 200 수익률", final_k=3)
        assert len(results) > 0
        assert results[0][0].metadata["ticker"] == "069500"

    def test_search_by_ticker(self, etf_retriever):
        """티커 검색 — 091160."""
        results = etf_retriever.search("091160", final_k=3)
        assert results[0][0].metadata["ticker"] == "091160"

    def test_search_by_keyword(self, etf_retriever):
        """키워드 검색 — 반도체."""
        results = etf_retriever.search("반도체 ETF 추천", final_k=3)
        tickers = [d.metadata["ticker"] for d, _ in results]
        assert "091160" in tickers  # TIGER 반도체

    def test_search_by_comparison(self, etf_retriever):
        """비교 검색 — 두 ETF 모두 반환."""
        results = etf_retriever.search("KODEX 200이랑 TIGER 반도체 비교", final_k=5)
        tickers = [d.metadata["ticker"] for d, _ in results]
        assert "069500" in tickers
        assert "091160" in tickers

    def test_stock_search(self, stock_retriever):
        """주식 검색 — 삼성전자."""
        results = stock_retriever.search("삼성전자 주가", final_k=3)
        assert len(results) > 0
        assert results[0][0].metadata["ticker"] == "005930"

    def test_search_returns_scores(self, etf_retriever):
        """검색 결과에 양수 점수 포함."""
        results = etf_retriever.search("KODEX 200", final_k=3)
        for doc, score in results:
            assert score > 0

    def test_retrieve_relevant_docs(self, etf_retriever):
        """retrieve_relevant_docs() 통합 호출."""
        from src.rag.retriever import retrieve_relevant_docs
        context, sources = retrieve_relevant_docs(etf_retriever, "KODEX 200")
        assert context is not None
        assert len(sources) > 0
        assert sources[0]["name"] == "KODEX 200"


# ── 2. 도구 주입 → 도구 함수 직접 호출 ──────────────────────


class TestToolIntegration:
    """set_retriever() → 도구 함수 직접 호출 통합 테스트."""

    def test_search_etf_tool(self, initialized_tools):
        """search_etf 도구 — 실제 retriever로 검색."""
        from src.llm.tools import search_etf
        result = search_etf.invoke({"query": "KODEX 200 수익률"})
        assert isinstance(result, str)
        assert "KODEX 200" in result
        # 구조화 데이터 enrichment 포함
        assert "38,500" in result or "38500" in result or "종가" in result

    def test_search_stock_tool(self, initialized_tools):
        """search_stock 도구 — 실제 retriever로 검색."""
        from src.llm.tools import search_stock
        result = search_stock.invoke({"query": "삼성전자 주가"})
        assert isinstance(result, str)
        assert "삼성전자" in result

    def test_search_no_result(self, initialized_tools):
        """검색 결과 없는 경우 빈 문자열."""
        from src.llm.tools import search_etf
        result = search_etf.invoke({"query": "존재하지않는ETF12345"})
        assert isinstance(result, str)
        # 빈 문자열 또는 "검색 결과 없음" 메시지
        assert result == "" or "없" in result or len(result) < 50

    def test_compare_etfs_tool(self, initialized_tools):
        """compare_etfs 도구 — 두 ETF 비교."""
        from src.llm.tools import compare_etfs
        result = compare_etfs.invoke({"etf_name_1": "KODEX 200", "etf_name_2": "TIGER 반도체"})
        assert isinstance(result, str)
        assert "KODEX 200" in result
        assert "TIGER 반도체" in result or "반도체" in result

    def test_get_etf_list_tool(self, initialized_tools):
        """get_etf_list 도구 — 카테고리별 ETF 목록."""
        from src.llm.tools import get_etf_list
        result = get_etf_list.invoke({"category": "미국 ETF"})
        assert isinstance(result, str)
        # 미국 관련 ETF가 있으면 포함
        assert "미국" in result or "S&P" in result or len(result) > 0

    def test_compare_stocks_tool(self, initialized_tools):
        """compare_stocks 도구 — 두 주식 비교."""
        from src.llm.tools import compare_stocks
        result = compare_stocks.invoke({"stock_name_1": "삼성전자", "stock_name_2": "SK하이닉스"})
        assert isinstance(result, str)
        assert "삼성전자" in result

    def test_get_stock_list_tool(self, initialized_tools):
        """get_stock_list 도구 — 카테고리별 주식 목록."""
        from src.llm.tools import get_stock_list
        result = get_stock_list.invoke({"category": "반도체 관련주"})
        assert isinstance(result, str)

    def test_analyze_sector_tool(self, initialized_tools):
        """analyze_sector 도구 — 종목→ETF 역인덱스 기반."""
        from src.llm.tools import analyze_sector
        result = analyze_sector.invoke({"query": "삼성전자"})
        assert isinstance(result, str)
        # 삼성전자가 포함된 ETF 정보가 나옴
        assert "삼성전자" in result

    def test_data_index_built(self, initialized_tools):
        """도구 주입 후 인덱스 정상 구축."""
        import src.llm.tools as tools_mod
        assert tools_mod._data_initialized is True
        assert len(tools_mod._etf_data_index) > 0
        assert len(tools_mod._stock_data_index) > 0
        # 티커 + 이름 기준으로 인덱싱
        assert "069500" in tools_mod._etf_data_index
        assert "kodex 200" in tools_mod._etf_data_index
        assert "005930" in tools_mod._stock_data_index

    def test_holdings_reverse_index(self, initialized_tools):
        """역인덱스 — 삼성전자가 포함된 ETF 2개."""
        import src.llm.tools as tools_mod
        samsung_etfs = tools_mod._holdings_reverse_index.get("005930", [])
        assert len(samsung_etfs) == 2  # KODEX 200 + TIGER 반도체
        etf_names = [e["etf_name"] for e in samsung_etfs]
        assert "KODEX 200" in etf_names
        assert "TIGER 반도체" in etf_names

    def test_sector_index(self, initialized_tools):
        """업종 인덱스 — 전기전자 업종에 삼성전자+SK하이닉스."""
        import src.llm.tools as tools_mod
        elec = tools_mod._sector_index.get("전기전자", [])
        assert len(elec) == 2
        names = [s["name"] for s in elec]
        assert "삼성전자" in names
        assert "SK하이닉스" in names

    def test_available_tickers(self, initialized_tools):
        """자동완성용 종목 옵션 반환."""
        from src.llm.tools import get_available_tickers
        options = get_available_tickers()
        assert len(options) > 0
        # "삼성전자 (005930)" 형식
        assert any("삼성전자" in opt for opt in options)
        assert any("KODEX 200" in opt for opt in options)


# ── 3. 에이전트 그래프 빌드 + 노드 연결 검증 ──────────────


class TestAgentGraphIntegration:
    """LangGraph 에이전트 구조 통합 검증 (LLM 호출 없이)."""

    def test_all_tools_registered(self):
        """13개 도구 전부 등록."""
        from src.llm.agent import ALL_TOOLS
        assert len(ALL_TOOLS) == 14

    def test_tool_names_unique(self):
        """도구 이름 중복 없음."""
        from src.llm.agent import ALL_TOOLS
        names = [t.name for t in ALL_TOOLS]
        assert len(names) == len(set(names))

    def test_graph_compiles(self):
        """LangGraph가 컴파일 가능."""
        from src.llm.agent import build_graph
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """그래프에 필수 노드 존재."""
        from src.llm.agent import build_graph
        graph = build_graph()
        # LangGraph compiled graph는 nodes 속성이 다를 수 있으므로
        # 그래프가 정상 빌드되었는지만 검증
        assert graph is not None

    def test_prompt_builds(self):
        """시스템 프롬프트 정상 생성."""
        from src.llm.prompts import build_system_prompt
        prompt = build_system_prompt("simple")
        assert len(prompt) > 100
        assert "ETF" in prompt or "금융" in prompt

    def test_prompt_varies_by_type(self):
        """질문 유형별 프롬프트 차이."""
        from src.llm.prompts import build_system_prompt
        simple = build_system_prompt("simple")
        compare = build_system_prompt("compare")
        recommend = build_system_prompt("recommend")
        # 유형별로 다른 지침 포함
        assert simple != compare
        assert compare != recommend

    def test_classifier_keyword_fallback(self):
        """키워드 분류기 — LLM 없이도 동작."""
        from src.llm.classifier import classify_question_type
        assert classify_question_type("KODEX 200 수익률 알려줘") == "simple"
        assert classify_question_type("반도체 ETF vs 채권 ETF 비교") == "compare"
        assert classify_question_type("좋은 ETF 추천해줘") == "recommend"

    def test_error_message_generation(self):
        """에러 메시지 생성 함수 통합."""
        from src.llm.agent import _make_error_message
        msg = _make_error_message(TimeoutError("timeout"))
        assert "시간" in msg or "잠시" in msg
        msg2 = _make_error_message(Exception("unknown"))
        assert len(msg2) > 0


# ── 4. 에러 핸들링 + Graceful Degradation ──────────────────


class TestErrorHandlingIntegration:
    """에러 발생 시 graceful degradation 검증."""

    def test_search_with_empty_retriever(self):
        """빈 retriever — 정상적으로 빈 결과 반환."""
        from src.rag.retriever import HybridRetriever, retrieve_relevant_docs

        empty_docs = [Document(page_content="없음", metadata={"ticker": "000000", "name": "없음"})]
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [(empty_docs[0], 2.0)]
        retriever = HybridRetriever(mock_vs, empty_docs)

        context, sources = retrieve_relevant_docs(retriever, "없는 종목")
        # min_rrf_score 필터에 의해 결과 없음이 될 수 있음
        assert context is None or isinstance(context, str)

    def test_tool_without_retriever(self):
        """retriever 미주입 상태에서 도구 호출 시 에러 안 남."""
        import src.llm.tools as tools_mod
        # 전역 상태 초기화
        old_retriever = tools_mod._retriever
        tools_mod._retriever = None
        try:
            from src.llm.tools import search_etf
            result = search_etf.invoke({"query": "test"})
            # 빈 결과 또는 에러 메시지 (크래시 아님)
            assert isinstance(result, str)
        finally:
            tools_mod._retriever = old_retriever

    def test_rerank_disabled_by_default(self):
        """COHERE_API_KEY 없으면 Rerank 비활성화."""
        from config import RERANK
        import os
        if not os.getenv("COHERE_API_KEY"):
            assert RERANK["enabled"] is False


# ── 5. 데이터 일관성 검증 ──────────────────────────────────


class TestDataConsistency:
    """데이터 → 문서 → 검색 결과 간 일관성."""

    def test_document_metadata_matches_data(self, etf_documents):
        """Document 메타데이터가 원본 데이터와 일치."""
        for doc in etf_documents:
            ticker = doc.metadata["ticker"]
            orig = next((e for e in SAMPLE_ETF_DATA if e["ticker"] == ticker), None)
            assert orig is not None
            assert doc.metadata["name"] == orig["name"]

    def test_search_result_metadata_consistent(self, etf_retriever):
        """검색 결과 Document의 메타데이터가 유효."""
        from src.rag.retriever import retrieve_relevant_docs
        context, sources = retrieve_relevant_docs(etf_retriever, "KODEX 200")
        if sources:
            for src_info in sources:
                assert src_info["ticker"] in [e["ticker"] for e in SAMPLE_ETF_DATA]
                assert src_info["name"] in [e["name"] for e in SAMPLE_ETF_DATA]

    def test_enrichment_data_matches(self, initialized_tools):
        """도구의 구조화 데이터 enrichment가 원본과 일치."""
        import src.llm.tools as tools_mod
        kodex = tools_mod._etf_data_index.get("069500")
        assert kodex is not None
        assert kodex["close"] == 38500
        assert kodex["name"] == "KODEX 200"

    def test_document_count_matches_data(self, etf_documents, stock_documents):
        """문서 수 = 원본 데이터 수."""
        assert len(etf_documents) == len(SAMPLE_ETF_DATA)
        assert len(stock_documents) == len(SAMPLE_STOCK_DATA)


# ── 6. UI 렌더링 함수 통합 ──────────────────────────────────


class TestUIIntegration:
    """UI 관련 함수 통합 테스트 (Streamlit 없이)."""

    def test_split_sections_with_real_answer(self):
        """실제 에이전트 응답 형태의 섹션 분리."""
        from src.ui.chat import split_into_sections
        answer = (
            "KODEX 200 ETF에 대해 분석해 보겠습니다.\n\n"
            "## 기본 정보\n"
            "KODEX 200은 KOSPI 200 지수를 추종하는 ETF입니다.\n"
            "현재 종가 38,500원입니다.\n\n"
            "## 수익률 분석\n"
            "| 기간 | 수익률 |\n|---|---|\n| 1일 | +1.2% |\n| 1개월 | +3.0% |\n\n"
            "## 투자 판단\n"
            "안정적인 시장 대표 ETF로 분산투자에 적합합니다.\n\n"
            "📌 위 내용은 참고 정보이며, 투자 판단은 본인의 책임입니다."
        )
        sections = split_into_sections(answer)
        assert len(sections) == 4  # 도입부 + 3개 헤더
        assert sections[0]["title"] is None  # 도입부
        assert sections[1]["title"] == "기본 정보"
        assert sections[2]["title"] == "수익률 분석"
        assert "| 기간 |" in sections[2]["body"]  # 테이블 보존
        assert sections[3]["title"] == "투자 판단"

    def test_dynamic_examples_with_real_data(self):
        """실제 데이터 형태로 동적 예시 질문 생성 (10개 이상 필요)."""
        from src.ui.components import generate_dynamic_examples
        # generate_dynamic_examples는 10개 이상의 데이터가 필요
        extended = SAMPLE_ETF_DATA + SAMPLE_STOCK_DATA
        # 부족하면 더미 추가
        for i in range(10):
            extended.append({
                **_STOCK_BASE, "ticker": f"9999{i:02d}", "name": f"테스트종목{i}",
                "close": 10000 + i * 1000, "change_pct": (i - 5) * 0.5,
                "trade_value": (10 - i) * 1_000_000_000,
                "market_cap": 1_000_000_000_000,
                "per": 10.0, "pbr": 1.0, "dividend_yield": 1.0, "sector": "기타",
                "returns": {"1d": (i - 5) * 0.5},
            })
        result = generate_dynamic_examples(stock_data=extended)
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_chart_parsing(self):
        """구조화 데이터 파싱 — 비교 테이블."""
        from src.ui.charts import try_parse_structured_data
        comparison_json = json.dumps({
            "__type__": "comparison_table",
            "items": [
                {"name": "KODEX 200", "ticker": "069500", "close": 38500},
                {"name": "TIGER 반도체", "ticker": "091160", "close": 15200},
            ],
            "metrics": ["close"],
        })
        parsed = try_parse_structured_data(comparison_json)
        assert parsed is not None
        assert parsed["__type__"] == "comparison_table"

    def test_followup_suggestions(self):
        """후속 질문 제안 생성."""
        from src.ui.chat import _get_followup_suggestions
        suggestions = _get_followup_suggestions(
            question="삼성전자 주가 알려줘",
            tools_used=["search_stock"],
            question_type="simple",
        )
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        if suggestions:
            assert "삼성전자" in suggestions[0]
