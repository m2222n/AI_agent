"""
LangGraph 기반 ETF 에이전트

흐름:
1. 사용자 질문 → LLM이 도구 선택 (Function Calling)
2. 도구 실행 → 검색 결과 반환
3. 검색 결과 부족 시 재검색 (최대 1회)
4. 최종 답변 생성
5. (CoV) 복잡 질문 시 답변 검증 → 수정

모델 라우팅:
- 단순 질문 (simple, general) → GPT-4o-mini (비용 절감)
- 복잡 질문 (compare, recommend, risk) → GPT-4o
"""

import json
import logging
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.llm.tools import ALL_TOOLS
from src.llm.prompts import build_system_prompt

logger = logging.getLogger(__name__)


# ── State 정의 ────────────────────────────────────────────

class AgentState(TypedDict):
    """에이전트 상태"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question_type: str
    tool_call_count: int


# ── 모델 라우팅 ───────────────────────────────────────────

# 복잡한 질문 유형 → GPT-4o
COMPLEX_TYPES = {"compare", "recommend", "risk"}

# CoV 적용 대상 질문 유형
COV_TYPES = {"compare", "recommend", "risk"}

_models = {}


def _get_model(question_type: str) -> ChatOpenAI:
    """질문 유형에 따라 모델 선택 (캐싱)"""
    if question_type in COMPLEX_TYPES:
        model_name = "gpt-4o"
    else:
        model_name = "gpt-4o-mini"

    if model_name not in _models:
        _models[model_name] = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            timeout=60,
            streaming=True,
        ).bind_tools(ALL_TOOLS)
        logger.info(f"모델 초기화: {model_name}")

    return _models[model_name]


# ── LLM 분류기 ────────────────────────────────────────────

_classifier_model = None


def _get_classifier():
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=15)
    return _classifier_model


# ── Structured Output 스키마 ──────────────────────────────

class QuestionClassification(BaseModel):
    """질문 유형 분류 결과 (Structured Output)"""
    question_type: str = Field(
        description="질문 유형: simple, compare, recommend, risk, general 중 하나"
    )


_structured_classifier = None


def _get_structured_classifier():
    """Structured Output 분류기 반환 (캐싱)"""
    global _structured_classifier
    if _structured_classifier is None:
        base = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=15)
        _structured_classifier = base.with_structured_output(QuestionClassification)
    return _structured_classifier


def classify_with_llm(question: str) -> str:
    """LLM으로 질문 유형 분류 (Structured Output + 키워드 fallback)"""
    prompt = f"""다음 질문을 분류하세요. 반드시 아래 5가지 중 하나를 선택합니다.

- simple: 특정 ETF/주식의 가격, 수익률, NAV, 거래량 등 단순 정보 질문
- compare: 두 개 이상의 ETF/주식을 비교하는 질문
- recommend: ETF/주식 추천, 카테고리 탐색, 목록 요청
- risk: 투자 위험, 변동성, 손실 가능성 질문
- general: ETF/주식 일반 개념, 용어 설명

질문: {question}"""

    valid_types = {"simple", "compare", "recommend", "risk", "general"}

    try:
        # Structured Output으로 분류 (JSON 스키마 강제)
        result = _get_structured_classifier().invoke([HumanMessage(content=prompt)])
        qtype = result.question_type.strip().lower()
        if qtype in valid_types:
            return qtype
        # 유효하지 않은 유형이면 general fallback
        logger.warning(f"Structured Output 유효하지 않은 유형: {qtype}")
        return "general"
    except Exception as e:
        logger.warning(f"Structured Output 분류 실패, 기존 방식 시도: {e}")

    # fallback: 기존 텍스트 파싱 방식
    try:
        result = _get_classifier().invoke([HumanMessage(content=prompt + "\n분류:")])
        answer = result.content.strip().lower()
        for vt in valid_types:
            if vt in answer:
                return vt
        return "general"
    except Exception as e2:
        logger.warning(f"LLM 분류 실패, 키워드 fallback: {e2}")
        from src.llm.classifier import classify_question_type
        return classify_question_type(question)


# ── 에러 메시지 ───────────────────────────────────────────

def _make_error_message(error: Exception) -> str:
    """예외 유형에 따라 사용자 친화적 에러 메시지 생성"""
    error_type = type(error).__name__
    error_str = str(error).lower()

    if "rate" in error_str or "429" in error_str or "RateLimit" in error_type:
        return "⚠️ API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
    if "timeout" in error_str or "timed out" in error_str:
        return "⚠️ 응답 시간이 초과되었습니다. 다시 시도해주세요."
    if "connection" in error_str or "network" in error_str:
        return "⚠️ 네트워크 연결에 문제가 있습니다. 인터넷 연결을 확인해주세요."
    if "auth" in error_str or "api key" in error_str or "401" in error_str:
        return "⚠️ API 인증에 실패했습니다. API 키를 확인해주세요."

    return f"⚠️ 일시적인 오류가 발생했습니다. 다시 시도해주세요. ({error_type})"


# ── 그래프 노드 ───────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """LLM 호출 (도구 선택 또는 최종 답변)"""
    question_type = state["question_type"]
    model = _get_model(question_type)

    messages = list(state["messages"])
    try:
        response = model.invoke(messages)
    except Exception as e:
        logger.error(f"LLM 호출 실패: {e}")
        error_msg = _make_error_message(e)
        return {"messages": [AIMessage(content=error_msg)]}

    return {"messages": [response]}


def _strip_chart_json(content: str) -> tuple:
    """도구 결과에서 차트 JSON을 분리.

    Returns:
        (LLM에 전달할 텍스트, 차트 JSON 원본 또는 None)
    """
    if '"__type__": "technical_chart"' not in content:
        return content, None

    # chart_json + "\n\n---\n\n" + text_result 구조
    parts = content.split("\n\n---\n\n", 1)
    if len(parts) == 2:
        chart_json = parts[0]
        text_only = parts[1]
        return f"[기술적 분석 차트 이미지 생성됨]\n\n{text_only}", chart_json

    return content, None


def call_tools(state: AgentState) -> dict:
    """도구 실행"""
    last_message = state["messages"][-1]

    tool_map = {t.name: t for t in ALL_TOOLS}
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        logger.info(f"도구 호출: {tool_name}({tool_args})")

        try:
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
            else:
                result = f"알 수 없는 도구: {tool_name}"
        except Exception as e:
            logger.error(f"도구 실행 실패 ({tool_name}): {e}")
            result = f"검색 중 오류가 발생했습니다: {type(e).__name__}"

        result_str = str(result)

        # 차트 JSON은 LLM context에서 제거 (base64 이미지가 수십 KB → 토큰 낭비 + 혼란)
        # 원본은 _raw_tool_results에 보관하여 stream에서 structured_data 이벤트 발행
        llm_content, chart_json = _strip_chart_json(result_str)

        msg = ToolMessage(content=llm_content, tool_call_id=tool_call["id"])
        if chart_json:
            msg.additional_kwargs["_chart_json"] = chart_json
        tool_messages.append(msg)

    return {
        "messages": tool_messages,
        "tool_call_count": state["tool_call_count"] + 1,
    }


# ── CoV (Chain of Verification) ──────────────────────────

def _extract_tool_evidence(messages: Sequence[BaseMessage]) -> str:
    """메시지에서 도구 결과(ToolMessage)를 수집하여 검증 근거로 반환"""
    evidence_parts = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # 구조화 데이터 JSON은 제외 (비교 테이블 등)
            content = msg.content
            if '"__type__"' in content:
                # JSON 앞의 텍스트 부분만 추출
                parts = content.split("---")
                content = parts[-1] if len(parts) > 1 else content[:500]
            evidence_parts.append(content[:1000])  # 각 도구 결과 최대 1000자
    return "\n---\n".join(evidence_parts)


def verify_answer(state: AgentState) -> dict:
    """
    CoV: 최종 답변의 수치/주장이 도구 결과와 일치하는지 검증.
    불일치 발견 시 수정된 답변을 반환.
    """
    messages = list(state["messages"])

    # 최종 답변 찾기
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            answer = msg.content
            break

    if not answer:
        return {"messages": []}

    # 도구 결과 수집
    evidence = _extract_tool_evidence(messages)
    if not evidence:
        return {"messages": []}

    # 검증 프롬프트
    verify_prompt = f"""당신은 금융 데이터 검증 전문가입니다.
아래 [답변]에 포함된 수치(가격, 수익률, PER, PBR, 시가총액, 거래량 등)가
[도구 결과]의 데이터와 일치하는지 검증하세요.

[도구 결과]
{evidence[:3000]}

[답변]
{answer}

검증 규칙:
1. 답변의 수치가 도구 결과에 있는 수치와 다르면 "불일치"
2. 도구 결과에 없는 수치를 답변이 만들어냈으면 "허위"
3. 모든 수치가 일치하면 "통과"

결과를 아래 형식으로 답하세요:
- 판정: 통과 / 수정필요
- 문제점: (수정필요인 경우만) 불일치/허위 수치를 구체적으로 나열
- 수정 답변: (수정필요인 경우만) 도구 결과의 정확한 수치로 수정한 전체 답변"""

    try:
        classifier = _get_classifier()
        result = classifier.invoke([HumanMessage(content=verify_prompt)])
        verification = result.content

        if "통과" in verification and "수정필요" not in verification:
            logger.info("CoV 검증 통과")
            return {"messages": []}

        # 수정 필요 — 수정된 답변 추출
        logger.info("CoV 검증: 수정 필요 감지")

        # "수정 답변:" 이후의 텍스트를 추출
        if "수정 답변:" in verification:
            revised = verification.split("수정 답변:", 1)[1].strip()
            if len(revised) > 50:  # 의미 있는 길이의 수정 답변이 있을 때만
                logger.info(f"CoV 수정 적용 ({len(revised)}자)")
                return {"messages": [AIMessage(content=revised)]}

        # 수정 답변 추출 실패 시 — 별도 LLM 호출로 수정
        revise_prompt = f"""아래 검증 결과를 바탕으로 답변을 수정하세요.
원래 답변의 구조와 톤은 유지하되, 잘못된 수치만 도구 결과의 정확한 값으로 교체하세요.

[검증 결과]
{verification}

[도구 결과]
{evidence[:2000]}

[원래 답변]
{answer}

수정된 답변:"""

        revised_result = classifier.invoke([HumanMessage(content=revise_prompt)])
        revised = revised_result.content.strip()
        if len(revised) > 50:
            logger.info(f"CoV 2차 수정 적용 ({len(revised)}자)")
            return {"messages": [AIMessage(content=revised)]}

    except Exception as e:
        logger.warning(f"CoV 검증 실패 (무시): {e}")

    return {"messages": []}


# ── 라우팅 함수 ───────────────────────────────────────────

def should_call_tools(state: AgentState) -> str:
    """도구 호출이 필요한지 판단"""
    last_message = state["messages"][-1]

    # AIMessage이고 tool_calls가 있으면 도구 실행
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # 재검색 제한 (최대 2회)
        if state["tool_call_count"] >= 2:
            logger.info("도구 호출 횟수 초과 — 최종 답변으로 전환")
            return "force_answer"
        return "tools"

    return "end"


def force_answer(state: AgentState) -> dict:
    """도구 호출 횟수 초과 시, 현재까지의 정보로 최종 답변을 생성하도록 강제."""
    logger.info("force_answer: 도구 호출 제한 도달, 현재 정보로 답변 강제 생성")

    last_message = state["messages"][-1]
    force_content = "[시스템] 더 이상 도구를 호출할 수 없습니다. 지금까지 수집한 정보를 바탕으로 사용자에게 최선의 답변을 작성하세요. 정보가 부족하면 부족하다고 안내하세요."

    # 각 tool_call에 대해 ToolMessage를 생성 (LangGraph는 모든 tool_call에 대응하는 ToolMessage를 기대)
    tool_messages = []
    for tc in last_message.tool_calls:
        tool_messages.append(ToolMessage(content=force_content, tool_call_id=tc["id"]))

    return {"messages": tool_messages}


def should_verify(state: AgentState) -> str:
    """최종 답변 후 CoV 검증이 필요한지 판단"""
    last_message = state["messages"][-1]
    question_type = state["question_type"]

    # 최종 답변(tool_calls 없는 AIMessage)이고, CoV 대상 유형이면 검증
    if (isinstance(last_message, AIMessage)
            and not last_message.tool_calls
            and last_message.content
            and question_type in COV_TYPES):
        # 도구 호출이 있었는지 확인 (도구 없이 답변한 경우는 스킵)
        has_tool_results = any(
            isinstance(msg, ToolMessage) for msg in state["messages"]
        )
        if has_tool_results:
            return "verify"

    return "end"


# ── 그래프 빌드 ───────────────────────────────────────────

def build_graph() -> StateGraph:
    """LangGraph 에이전트 그래프 구성 (CoV 포함)"""
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_node("force_answer", force_answer)
    graph.add_node("verify", verify_answer)

    # 진입점
    graph.set_entry_point("agent")

    # agent → tools / force_answer / verify(최종답변) / END
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "force_answer": "force_answer", "verify": "verify", "end": END},
    )

    # tools → agent (도구 결과를 LLM에 전달)
    graph.add_edge("tools", "agent")

    # force_answer → agent (강제 답변 생성)
    graph.add_edge("force_answer", "agent")

    # verify → END
    graph.add_edge("verify", END)

    return graph.compile()


def _route_after_agent(state: AgentState) -> str:
    """agent 노드 후 라우팅: tools / force_answer / verify / end"""
    # 먼저 도구 호출 여부 확인
    tool_route = should_call_tools(state)
    if tool_route in ("tools", "force_answer"):
        return tool_route

    # 도구 호출 아니면 검증 여부 확인
    return should_verify(state)


# ── 에이전트 실행 ─────────────────────────────────────────

_compiled_graph = None


def get_agent():
    """컴파일된 에이전트 그래프 반환 (싱글턴)"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        logger.info("LangGraph 에이전트 빌드 완료 (CoV 포함)")
    return _compiled_graph


def run_agent(question: str, chat_history: list = None) -> dict:
    """
    에이전트 실행 (비스트리밍)

    Args:
        question: 사용자 질문
        chat_history: 이전 대화 히스토리 [{"role": ..., "content": ...}, ...]

    Returns:
        {"answer": str, "question_type": str, "model": str}
    """
    agent = get_agent()

    # 질문 유형 분류
    question_type = classify_with_llm(question)
    logger.info(f"질문 유형: {question_type}")

    # 시스템 프롬프트
    system_prompt = build_system_prompt(question_type)

    # 메시지 구성
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    # 대화 히스토리 추가
    if chat_history:
        for msg in chat_history[-10:]:  # 최근 10개
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    # 에이전트 실행
    initial_state: AgentState = {
        "messages": messages,
        "question_type": question_type,
        "tool_call_count": 0,
    }

    final_state = agent.invoke(initial_state)

    # 최종 답변 추출
    answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.content
            break

    model_used = "gpt-4o" if question_type in COMPLEX_TYPES else "gpt-4o-mini"

    return {
        "answer": answer,
        "question_type": question_type,
        "model": model_used,
    }


def stream_agent(question: str, chat_history: list = None):
    """
    에이전트 토큰 스트리밍 실행 — stream_mode=["messages", "updates"] 사용

    Yields: {"event": str, "data": ...}
        - {"event": "question_type", "data": "simple"}
        - {"event": "tool_call", "data": {"name": ..., "args": ...}}
        - {"event": "tool_result", "data": "검색 결과 요약"}
        - {"event": "token", "data": "누적된 답변 텍스트"}
        - {"event": "done", "data": {"answer": ..., "model": ...}}
    """
    agent = get_agent()

    question_type = classify_with_llm(question)
    yield {"event": "question_type", "data": question_type}

    system_prompt = build_system_prompt(question_type)
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    if chat_history:
        for msg in chat_history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    initial_state: AgentState = {
        "messages": messages,
        "question_type": question_type,
        "tool_call_count": 0,
    }

    final_answer = ""
    model_used = "gpt-4o" if question_type in COMPLEX_TYPES else "gpt-4o-mini"
    cov_applied = False

    try:
        # 두 모드 동시 사용: messages(토큰 스트리밍) + updates(도구 호출 감지)
        for event in agent.stream(initial_state, stream_mode=["messages", "updates"]):
            # stream_mode가 리스트이면 (mode, data) 튜플로 반환됨
            if not isinstance(event, tuple) or len(event) != 2:
                continue

            mode, data = event

            if mode == "messages":
                # (AIMessageChunk, metadata) 튜플
                msg_chunk, metadata = data
                node = metadata.get("langgraph_node", "")

                if isinstance(msg_chunk, AIMessageChunk):
                    # 도구 호출 청크는 건너뜀 (updates 모드에서 처리)
                    if msg_chunk.tool_call_chunks:
                        continue
                    # 텍스트 토큰 누적
                    if msg_chunk.content:
                        if node == "verify":
                            # CoV 수정 답변이 오면 기존 답변 대체
                            if not cov_applied:
                                cov_applied = True
                                final_answer = ""
                                yield {"event": "cov_revision", "data": "검증 수정 적용 중..."}
                            final_answer += msg_chunk.content
                            yield {"event": "token", "data": final_answer}
                        else:
                            final_answer += msg_chunk.content
                            yield {"event": "token", "data": final_answer}

            elif mode == "updates":
                # 노드별 상태 업데이트 (도구 호출/결과 감지용)
                if not isinstance(data, dict):
                    continue
                for node_name, node_output in data.items():
                    if node_name == "tools":
                        for msg in node_output.get("messages", []):
                            if isinstance(msg, ToolMessage):
                                yield {"event": "tool_result", "data": msg.content[:100]}
                                # 차트 JSON (call_tools에서 분리하여 additional_kwargs에 보관)
                                chart_json = msg.additional_kwargs.get("_chart_json")
                                if chart_json:
                                    yield {"event": "structured_data", "data": chart_json}
                                # 비교 테이블 등 다른 구조화 데이터 (content에 남아있는 경우)
                                elif '"__type__"' in msg.content:
                                    yield {"event": "structured_data", "data": msg.content}
                    elif node_name == "agent":
                        for msg in node_output.get("messages", []):
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    yield {"event": "tool_call", "data": {"name": tc["name"], "args": tc["args"]}}
                            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                                # 최종 답변 (토큰 스트리밍이 안 된 경우 fallback)
                                if not final_answer:
                                    final_answer = msg.content
                                    logger.info(f"[stream] updates fallback: {len(final_answer)}자")
                                    yield {"event": "token", "data": final_answer}
                    elif node_name == "verify":
                        for msg in node_output.get("messages", []):
                            if isinstance(msg, AIMessage) and msg.content:
                                # CoV 수정 답변으로 대체
                                if not cov_applied:
                                    cov_applied = True
                                    final_answer = msg.content
                                    yield {"event": "cov_revision", "data": "검증 수정 완료"}
                                    yield {"event": "token", "data": final_answer}

    except Exception as e:
        import traceback
        logger.error(f"에이전트 스트리밍 오류: {e}\n{traceback.format_exc()}")
        error_msg = _make_error_message(e)
        if not final_answer:
            final_answer = error_msg
        yield {"event": "error", "data": error_msg}

    yield {"event": "done", "data": {
        "answer": final_answer,
        "model": model_used,
        "question_type": question_type,
        "cov_applied": cov_applied,
    }}
