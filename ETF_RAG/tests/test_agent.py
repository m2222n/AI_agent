"""LangGraph 에이전트 + 도구 테스트"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from src.llm.tools import (
    search_etf, compare_etfs, get_etf_list, set_retriever, ALL_TOOLS,
    _find_structured_data, _extract_comparison_fields,
)
from src.llm.agent import (
    AgentState,
    should_call_tools,
    COMPLEX_TYPES,
    build_graph,
    stream_agent,
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
    """도구 목록에 4개 도구가 있는지 확인"""
    assert len(ALL_TOOLS) == 4
    names = {t.name for t in ALL_TOOLS}
    assert names == {"search_etf", "compare_etfs", "get_etf_list", "search_stock"}


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


# ── 스트리밍 테스트 ──────────────────────────────────────

def test_stream_agent_yields_question_type():
    """stream_agent가 첫 이벤트로 question_type을 반환하는지 확인"""
    with patch("src.llm.agent.classify_with_llm", return_value="simple"), \
         patch("src.llm.agent.get_agent") as mock_agent:
        # agent.stream()이 빈 이터레이터 반환
        mock_agent.return_value.stream.return_value = iter([])

        events = list(stream_agent("KODEX 200 수익률"))
        assert events[0] == {"event": "question_type", "data": "simple"}
        assert events[-1]["event"] == "done"


def test_stream_agent_token_accumulation():
    """stream_mode='messages' 토큰이 누적되어 전달되는지 확인"""
    chunk1 = AIMessageChunk(content="안녕")
    chunk2 = AIMessageChunk(content="하세요")
    meta = {"langgraph_node": "agent"}

    mock_events = [
        ("messages", (chunk1, meta)),
        ("messages", (chunk2, meta)),
    ]

    with patch("src.llm.agent.classify_with_llm", return_value="simple"), \
         patch("src.llm.agent.get_agent") as mock_agent:
        mock_agent.return_value.stream.return_value = iter(mock_events)

        events = list(stream_agent("테스트 질문"))
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["data"] == "안녕"
        assert token_events[1]["data"] == "안녕하세요"  # 누적

        done_event = events[-1]
        assert done_event["event"] == "done"
        assert done_event["data"]["answer"] == "안녕하세요"


def test_stream_agent_tool_call_events():
    """updates 모드에서 도구 호출 이벤트가 전달되는지 확인"""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "search_etf", "args": {"query": "KODEX 200"}}],
    )
    tool_result_msg = ToolMessage(content="KODEX 200 ETF 정보...", tool_call_id="tc1")

    mock_events = [
        ("updates", {"agent": {"messages": [tool_call_msg]}}),
        ("updates", {"tools": {"messages": [tool_result_msg]}}),
    ]

    with patch("src.llm.agent.classify_with_llm", return_value="simple"), \
         patch("src.llm.agent.get_agent") as mock_agent:
        mock_agent.return_value.stream.return_value = iter(mock_events)

        events = list(stream_agent("KODEX 200 수익률"))
        tool_calls = [e for e in events if e["event"] == "tool_call"]
        tool_results = [e for e in events if e["event"] == "tool_result"]

        assert len(tool_calls) == 1
        assert tool_calls[0]["data"]["name"] == "search_etf"
        assert len(tool_results) == 1


def test_stream_agent_model_routing():
    """복잡한 질문 유형에서 GPT-4o 모델이 선택되는지 확인"""
    with patch("src.llm.agent.classify_with_llm", return_value="compare"), \
         patch("src.llm.agent.get_agent") as mock_agent:
        mock_agent.return_value.stream.return_value = iter([])

        events = list(stream_agent("KODEX 200 vs TIGER 반도체"))
        done_event = events[-1]
        assert done_event["data"]["model"] == "gpt-4o"
        assert done_event["data"]["question_type"] == "compare"


def test_stream_agent_skips_tool_call_chunks():
    """도구 호출 청크(tool_call_chunks)는 token 이벤트로 전달되지 않아야 함"""
    tool_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "search_etf", "args": '{"query": "test"}', "id": "tc1", "index": 0}],
    )
    text_chunk = AIMessageChunk(content="답변입니다")
    meta = {"langgraph_node": "agent"}

    mock_events = [
        ("messages", (tool_chunk, meta)),
        ("messages", (text_chunk, meta)),
    ]

    with patch("src.llm.agent.classify_with_llm", return_value="simple"), \
         patch("src.llm.agent.get_agent") as mock_agent:
        mock_agent.return_value.stream.return_value = iter(mock_events)

        events = list(stream_agent("테스트"))
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) == 1
        assert token_events[0]["data"] == "답변입니다"


# ── 구조화 비교 데이터 테스트 ────────────────────────────

SAMPLE_ETF_DATA = [
    {
        "ticker": "069500", "name": "KODEX 200", "date": "20260408",
        "close": 80800, "nav": 80647, "change_pct": 2.91,
        "volume": 14703488, "trade_value": 1184866376189,
        "deviation": -0.17, "tracking_error": 0.05,
        "returns": {"1d": 2.91, "1w": 5.12, "1m": -1.23, "3m": 3.45, "1y": 12.34},
        "holdings": [
            {"stock_ticker": "005930", "stock_name": "삼성전자", "weight": 31.77},
            {"stock_ticker": "000660", "stock_name": "SK하이닉스", "weight": 8.12},
        ],
    },
    {
        "ticker": "091160", "name": "TIGER 반도체", "date": "20260408",
        "close": 15200, "nav": 15180, "change_pct": 4.12,
        "volume": 5000000, "trade_value": 76000000000,
        "deviation": -0.13, "tracking_error": 0.08,
        "returns": {"1d": 4.12, "1w": 7.50, "1m": 2.10, "3m": -0.50, "1y": 20.00},
        "holdings": [
            {"stock_ticker": "005930", "stock_name": "삼성전자", "weight": 25.0},
        ],
    },
]


@pytest.fixture
def mock_retriever_with_data():
    """구조화 데이터 인덱스를 포함한 mock retriever"""
    retriever = MagicMock()
    retriever.documents = SAMPLE_DOCS
    set_retriever(retriever, SAMPLE_DOCS, etf_data=SAMPLE_ETF_DATA)
    return retriever


def test_find_structured_data_by_name(mock_retriever_with_data):
    """이름으로 구조화 데이터 조회"""
    result = _find_structured_data("KODEX 200")
    assert result is not None
    assert result["ticker"] == "069500"


def test_find_structured_data_by_ticker(mock_retriever_with_data):
    """티커로 구조화 데이터 조회"""
    result = _find_structured_data("091160")
    assert result is not None
    assert result["name"] == "TIGER 반도체"


def test_find_structured_data_not_found(mock_retriever_with_data):
    """존재하지 않는 종목 조회"""
    result = _find_structured_data("존재하지않는ETF")
    assert result is None


def test_extract_comparison_fields_etf(mock_retriever_with_data):
    """ETF 비교 필드 추출"""
    fields = _extract_comparison_fields(SAMPLE_ETF_DATA[0])
    assert fields["name"] == "KODEX 200"
    assert fields["close"] == 80800
    assert fields["nav"] == 80647
    assert fields["return_1d"] == 2.91
    assert fields["return_1y"] == 12.34
    assert fields["asset_type"] == "etf"
    assert len(fields["top_holdings"]) == 2
    assert fields["top_holdings"][0]["name"] == "삼성전자"


def test_compare_etfs_structured(mock_retriever_with_data):
    """구조화 데이터로 비교 시 JSON 반환"""
    result = compare_etfs.invoke({"etf_name_1": "KODEX 200", "etf_name_2": "TIGER 반도체"})
    assert '"__type__": "comparison_table"' in result
    assert "KODEX 200" in result
    assert "TIGER 반도체" in result

    import json
    json_part = result.split("\n\n---\n\n")[0]
    data = json.loads(json_part)
    assert data["__type__"] == "comparison_table"
    assert len(data["items"]) == 2


def test_compare_etfs_fallback_when_no_index():
    """인덱스 없으면 기존 텍스트 검색 fallback"""
    retriever = MagicMock()
    retriever.documents = SAMPLE_DOCS
    set_retriever(retriever, SAMPLE_DOCS, etf_data=[], stock_data=[])  # 빈 인덱스

    with patch("src.rag.retriever.retrieve_relevant_docs") as mock_search:
        mock_search.side_effect = [
            ("KODEX 200 정보...", [{"ticker": "069500", "name": "KODEX 200", "relevance_score": 100.0}]),
            ("TIGER 반도체 정보...", [{"ticker": "091160", "name": "TIGER 반도체", "relevance_score": 95.0}]),
        ]
        result = compare_etfs.invoke({"etf_name_1": "KODEX 200", "etf_name_2": "TIGER 반도체"})
        assert "ETF 1" in result
        assert "ETF 2" in result


def test_stream_agent_structured_data_event():
    """structured_data 이벤트가 전달되는지 확인"""
    import json
    comparison_json = json.dumps({
        "__type__": "comparison_table",
        "items": [{"name": "A"}, {"name": "B"}],
    })
    tool_result_msg = ToolMessage(
        content=f'{comparison_json}\n\n---\n\nA vs B',
        tool_call_id="tc1",
    )

    mock_events = [
        ("updates", {"tools": {"messages": [tool_result_msg]}}),
    ]

    with patch("src.llm.agent.classify_with_llm", return_value="compare"), \
         patch("src.llm.agent.get_agent") as mock_agent:
        mock_agent.return_value.stream.return_value = iter(mock_events)

        events = list(stream_agent("A vs B"))
        structured_events = [e for e in events if e["event"] == "structured_data"]
        assert len(structured_events) == 1
        assert '"__type__"' in structured_events[0]["data"]
