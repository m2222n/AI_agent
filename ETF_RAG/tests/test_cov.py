"""CoV (Chain of Verification) + Structured Output 테스트"""
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

from src.llm.agent import (
    _extract_tool_evidence,
    verify_answer,
    should_verify,
    should_call_tools,
    force_answer,
    _route_after_agent,
    build_graph,
    COV_TYPES,
    COMPLEX_TYPES,
    AgentState,
    QuestionClassification,
)


# ── _extract_tool_evidence 테스트 ─────────────────────────

def test_extract_tool_evidence_basic():
    """ToolMessage에서 증거 추출"""
    messages = [
        HumanMessage(content="질문"),
        ToolMessage(content="삼성전자 종가 70,000원", tool_call_id="1"),
    ]
    evidence = _extract_tool_evidence(messages)
    assert "70,000원" in evidence


def test_extract_tool_evidence_multiple():
    """여러 ToolMessage 결합"""
    messages = [
        ToolMessage(content="데이터1", tool_call_id="1"),
        ToolMessage(content="데이터2", tool_call_id="2"),
    ]
    evidence = _extract_tool_evidence(messages)
    assert "데이터1" in evidence
    assert "데이터2" in evidence


def test_extract_tool_evidence_no_tools():
    """ToolMessage 없으면 빈 문자열"""
    messages = [HumanMessage(content="질문"), AIMessage(content="답변")]
    evidence = _extract_tool_evidence(messages)
    assert evidence == ""


def test_extract_tool_evidence_structured_data():
    """구조화 데이터(__type__) 포함 시 텍스트 부분만 추출"""
    content = '{"__type__": "comparison_table", "items": []}---\n삼성전자 정보'
    messages = [ToolMessage(content=content, tool_call_id="1")]
    evidence = _extract_tool_evidence(messages)
    assert "삼성전자 정보" in evidence


def test_extract_tool_evidence_truncation():
    """긴 도구 결과 2000자로 절단"""
    long_content = "x" * 5000
    messages = [ToolMessage(content=long_content, tool_call_id="1")]
    evidence = _extract_tool_evidence(messages)
    assert len(evidence) <= 2000


# ── should_verify 테스트 ──────────────────────────────────

def test_should_verify_complex_with_tools():
    """compare/recommend/risk + 도구 결과 있으면 verify"""
    for qtype in COV_TYPES:
        state: AgentState = {
            "messages": [
                ToolMessage(content="데이터", tool_call_id="1"),
                AIMessage(content="최종 답변입니다"),
            ],
            "question_type": qtype,
            "tool_call_count": 1,
        }
        assert should_verify(state) == "verify"


def test_should_verify_general_skipped():
    """general 유형은 검증 스킵 (도구 없이 답변하는 유형)"""
    state: AgentState = {
        "messages": [
            ToolMessage(content="데이터", tool_call_id="1"),
            AIMessage(content="최종 답변"),
        ],
        "question_type": "general",
        "tool_call_count": 1,
    }
    assert should_verify(state) == "end"


def test_should_verify_no_tools():
    """도구 결과 없으면 검증 스킵"""
    state: AgentState = {
        "messages": [AIMessage(content="일반 답변")],
        "question_type": "compare",
        "tool_call_count": 0,
    }
    assert should_verify(state) == "end"


def test_should_verify_tool_calls_present():
    """tool_calls가 있는 AIMessage(아직 답변 아님)는 스킵"""
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": "search_etf", "args": {}, "id": "1"}]
    state: AgentState = {
        "messages": [msg],
        "question_type": "compare",
        "tool_call_count": 0,
    }
    assert should_verify(state) == "end"


# ── _route_after_agent 테스트 ─────────────────────────────

def test_route_tools_priority():
    """도구 호출이 있으면 tools로 라우팅 (verify보다 우선)"""
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": "search_etf", "args": {}, "id": "1"}]
    state: AgentState = {
        "messages": [msg],
        "question_type": "compare",
        "tool_call_count": 0,
    }
    assert _route_after_agent(state) == "tools"


def test_route_verify_after_answer():
    """최종 답변 + compare → verify"""
    state: AgentState = {
        "messages": [
            ToolMessage(content="근거", tool_call_id="1"),
            AIMessage(content="비교 결과입니다"),
        ],
        "question_type": "compare",
        "tool_call_count": 1,
    }
    assert _route_after_agent(state) == "verify"


def test_route_end_general():
    """general 유형 최종 답변 → end (CoV 대상 아님)"""
    state: AgentState = {
        "messages": [AIMessage(content="답변")],
        "question_type": "general",
        "tool_call_count": 0,
    }
    assert _route_after_agent(state) == "end"


# ── verify_answer 테스트 ──────────────────────────────────

@patch("src.llm.agent._get_classifier")
def test_verify_answer_pass(mock_cls):
    """검증 통과 시 빈 메시지 반환"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="판정: 통과\n모든 수치가 일치합니다.")
    mock_cls.return_value = mock_llm

    state: AgentState = {
        "messages": [
            ToolMessage(content="삼성전자 PER 8.5배", tool_call_id="1"),
            AIMessage(content="삼성전자의 PER은 8.5배입니다."),
        ],
        "question_type": "compare",
        "tool_call_count": 1,
    }
    result = verify_answer(state)
    assert result["messages"] == []  # 통과 → 수정 없음


@patch("src.llm.agent._get_classifier")
def test_verify_answer_revision(mock_cls):
    """수정 필요 시 수정된 답변 반환"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="판정: 수정필요\n문제점: PER 8.5가 아니라 9.2\n수정 답변: 삼성전자의 PER은 9.2배입니다. 이는 업종 평균 대비 적절한 수준입니다."
    )
    mock_cls.return_value = mock_llm

    state: AgentState = {
        "messages": [
            ToolMessage(content="삼성전자 PER 9.2배", tool_call_id="1"),
            AIMessage(content="삼성전자의 PER은 8.5배입니다."),
        ],
        "question_type": "compare",
        "tool_call_count": 1,
    }
    result = verify_answer(state)
    assert len(result["messages"]) == 1
    assert "9.2" in result["messages"][0].content


@patch("src.llm.agent._get_classifier")
def test_verify_answer_no_evidence(mock_cls):
    """도구 결과 없으면 검증 스킵"""
    state: AgentState = {
        "messages": [AIMessage(content="일반 답변")],
        "question_type": "compare",
        "tool_call_count": 0,
    }
    result = verify_answer(state)
    assert result["messages"] == []


@patch("src.llm.agent._get_classifier")
def test_verify_answer_error_graceful(mock_cls):
    """검증 중 오류 → 빈 메시지 (기존 답변 유지)"""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("API 오류")
    mock_cls.return_value = mock_llm

    state: AgentState = {
        "messages": [
            ToolMessage(content="데이터", tool_call_id="1"),
            AIMessage(content="답변"),
        ],
        "question_type": "compare",
        "tool_call_count": 1,
    }
    result = verify_answer(state)
    assert result["messages"] == []  # 오류 시 기존 답변 유지


# ── QuestionClassification 스키마 테스트 ──────────────────

def test_question_classification_schema():
    """Pydantic 스키마 기본 동작"""
    qc = QuestionClassification(question_type="simple")
    assert qc.question_type == "simple"


def test_question_classification_all_types():
    """모든 유형이 스키마로 생성 가능"""
    for qtype in ["simple", "compare", "recommend", "risk", "general"]:
        qc = QuestionClassification(question_type=qtype)
        assert qc.question_type == qtype


# ── build_graph 테스트 ────────────────────────────────────

def test_build_graph_has_verify_node():
    """그래프에 verify 노드 존재"""
    graph = build_graph()
    graph_repr = graph.get_graph()
    # nodes는 dict 또는 list — 키/값에서 verify 확인
    if hasattr(graph_repr, 'nodes'):
        nodes = graph_repr.nodes
        if isinstance(nodes, dict):
            assert "verify" in nodes
        else:
            node_names = [n.id if hasattr(n, 'id') else str(n) for n in nodes]
            assert "verify" in node_names


def test_cov_types_includes_complex():
    """COV_TYPES는 COMPLEX_TYPES를 포함 (비교/추천/위험 + 단순/기술적 등)"""
    assert COMPLEX_TYPES.issubset(COV_TYPES)
    assert "simple" in COV_TYPES
    assert "technical" in COV_TYPES
    assert "general" not in COV_TYPES  # general은 도구 없이 답변하므로 제외


# ── CoV 확대 적용 테스트 ──────────────────────────────────

def test_should_verify_simple_with_tools():
    """simple 유형 + 도구 결과 있으면 verify (확대 적용)"""
    state: AgentState = {
        "messages": [
            ToolMessage(content="삼성전자 종가 70,000원", tool_call_id="1"),
            AIMessage(content="삼성전자의 현재 종가는 70,000원입니다."),
        ],
        "question_type": "simple",
        "tool_call_count": 1,
    }
    assert should_verify(state) == "verify"


def test_should_verify_technical_with_tools():
    """technical 유형 + 도구 결과 있으면 verify"""
    state: AgentState = {
        "messages": [
            ToolMessage(content="RSI: 72.5 (과매수)", tool_call_id="1"),
            AIMessage(content="RSI가 72.5로 과매수 구간입니다."),
        ],
        "question_type": "technical",
        "tool_call_count": 1,
    }
    assert should_verify(state) == "verify"


# ── force_answer 개선 테스트 ──────────────────────────────

def test_force_answer_includes_prior_evidence():
    """force_answer 시 이전 도구 결과 요약이 포함되는지 확인"""
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": "search_etf", "args": {}, "id": "tc1"}]
    state: AgentState = {
        "messages": [
            ToolMessage(content="KODEX 200 종가 45,000원", tool_call_id="prev1"),
            msg,
        ],
        "question_type": "simple",
        "tool_call_count": 2,
    }
    result = force_answer(state)
    assert len(result["messages"]) == 1
    # 이전 도구 결과 요약이 force_content에 포함
    content = result["messages"][0].content
    assert "KODEX 200" in content or "45,000" in content
    assert "이전 도구 결과 요약" in content


def test_force_answer_no_prior_evidence():
    """이전 도구 결과 없으면 기본 메시지만"""
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": "search_etf", "args": {}, "id": "tc1"}]
    state: AgentState = {
        "messages": [msg],
        "question_type": "simple",
        "tool_call_count": 2,
    }
    result = force_answer(state)
    content = result["messages"][0].content
    assert "더 이상 도구를 호출할 수 없습니다" in content
    assert "이전 도구 결과 요약" not in content
