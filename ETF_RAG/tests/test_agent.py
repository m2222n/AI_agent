"""LangGraph 에이전트 + 도구 테스트"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.llm.tools import search_etf, compare_etfs, get_etf_list, set_retriever, ALL_TOOLS
from src.llm.agent import (
    AgentState,
    should_call_tools,
    COMPLEX_TYPES,
    build_graph,
)


# ── Fixtures ──────────────────────────────────────────────

SAMPLE_DOCS = [
    Document(
        page_content="KODEX 200 ETF. 종가: 80,800원. 수익률: 1일: +2.91%.",
        metadata={"ticker": "069500", "name": "KODEX 200", "source": "krx_collected"},
    ),
    Document(
        page_content="TIGER 반도체 ETF. 종가: 15,200원. 수익률: 1일: +4.12%.",
        metadata={"ticker": "091160", "name": "TIGER 반도체", "source": "krx_collected"},
    ),
]


@pytest.fixture
def mock_retriever():
    """도구에 mock retriever 주입"""
    retriever = MagicMock()
    retriever.documents = SAMPLE_DOCS
    set_retriever(retriever, SAMPLE_DOCS)
    return retriever


# ── 도구 테스트 ───────────────────────────────────────────

def test_all_tools_list():
    """도구 목록에 3개 도구가 있는지 확인"""
    assert len(ALL_TOOLS) == 3
    names = {t.name for t in ALL_TOOLS}
    assert names == {"search_etf", "compare_etfs", "get_etf_list"}


def test_search_etf_with_results(mock_retriever):
    """search_etf가 검색 결과를 반환하는지 확인"""
    with patch("src.rag.retriever.retrieve_relevant_docs") as mock_search:
        mock_search.return_value = (
            "KODEX 200 ETF 정보...",
            [{"ticker": "069500", "name": "KODEX 200", "relevance_score": 100.0}],
        )
        result = search_etf.invoke({"query": "KODEX 200 수익률"})
        assert "KODEX 200" in result
        assert "069500" in result


def test_search_etf_no_results(mock_retriever):
    """검색 결과 없을 때 빈 문자열 반환"""
    with patch("src.rag.retriever.retrieve_relevant_docs") as mock_search:
        mock_search.return_value = (None, [])
        result = search_etf.invoke({"query": "비트코인 ETF"})
        assert result == ""


def test_compare_etfs(mock_retriever):
    """compare_etfs 도구가 두 ETF 정보를 반환하는지 확인"""
    with patch("src.rag.retriever.retrieve_relevant_docs") as mock_search:
        mock_search.side_effect = [
            ("KODEX 200 정보...", [{"ticker": "069500", "name": "KODEX 200", "relevance_score": 100.0}]),
            ("TIGER 반도체 정보...", [{"ticker": "091160", "name": "TIGER 반도체", "relevance_score": 95.0}]),
        ]
        result = compare_etfs.invoke({"etf_name_1": "KODEX 200", "etf_name_2": "TIGER 반도체"})
        assert "ETF 1" in result
        assert "ETF 2" in result


def test_get_etf_list(mock_retriever):
    """get_etf_list 도구가 목록을 반환하는지 확인"""
    with patch("src.rag.retriever.retrieve_relevant_docs") as mock_search:
        mock_search.return_value = (
            "반도체 ETF 목록...",
            [{"ticker": "091160", "name": "TIGER 반도체", "relevance_score": 90.0}],
        )
        result = get_etf_list.invoke({"category": "반도체"})
        assert "반도체" in result


def test_get_etf_list_no_results(mock_retriever):
    """카테고리 검색 결과 없을 때"""
    with patch("src.rag.retriever.retrieve_relevant_docs") as mock_search:
        mock_search.return_value = (None, [])
        result = get_etf_list.invoke({"category": "우주"})
        assert "찾지 못했습니다" in result


# ── 라우팅 로직 테스트 ────────────────────────────────────

def test_should_call_tools_with_tool_calls():
    """tool_calls가 있으면 'tools' 반환"""
    state: AgentState = {
        "messages": [AIMessage(content="", tool_calls=[{"id": "1", "name": "search_etf", "args": {"query": "test"}}])],
        "question_type": "simple",
        "tool_call_count": 0,
    }
    assert should_call_tools(state) == "tools"


def test_should_call_tools_no_tool_calls():
    """tool_calls가 없으면 'end' 반환"""
    state: AgentState = {
        "messages": [AIMessage(content="답변입니다.")],
        "question_type": "simple",
        "tool_call_count": 0,
    }
    assert should_call_tools(state) == "end"


def test_should_call_tools_limit_exceeded():
    """도구 호출 횟수 초과 시 'end' 반환"""
    state: AgentState = {
        "messages": [AIMessage(content="", tool_calls=[{"id": "1", "name": "search_etf", "args": {"query": "test"}}])],
        "question_type": "simple",
        "tool_call_count": 2,
    }
    assert should_call_tools(state) == "end"


# ── 모델 라우팅 테스트 ────────────────────────────────────

def test_complex_types():
    """복잡한 질문 유형 정의 확인"""
    assert "compare" in COMPLEX_TYPES
    assert "recommend" in COMPLEX_TYPES
    assert "risk" in COMPLEX_TYPES
    assert "simple" not in COMPLEX_TYPES
    assert "general" not in COMPLEX_TYPES


# ── 그래프 빌드 테스트 ────────────────────────────────────

def test_build_graph():
    """그래프가 정상적으로 컴파일되는지 확인"""
    graph = build_graph()
    assert graph is not None
