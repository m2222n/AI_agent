"""
LangGraph 기반 ETF 에이전트

흐름:
1. 사용자 질문 → LLM이 도구 선택 (Function Calling)
2. 도구 실행 → 검색 결과 반환
3. 검색 결과 부족 시 재검색 (최대 1회)
4. 최종 답변 생성

모델 라우팅:
- 단순 질문 (simple, general) → GPT-4o-mini (비용 절감)
- 복잡 질문 (compare, recommend, risk) → GPT-4o
"""

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


def classify_with_llm(question: str) -> str:
    """LLM으로 질문 유형 분류 (키워드 classifier 대체)"""
    prompt = f"""다음 질문을 분류하세요. 반드시 아래 5가지 중 하나만 답하세요.

- simple: 특정 ETF의 가격, 수익률, NAV, 거래량 등 단순 정보 질문
- compare: 두 개 이상의 ETF를 비교하는 질문
- recommend: ETF 추천, 카테고리 탐색, 목록 요청
- risk: 투자 위험, 변동성, 손실 가능성 질문
- general: ETF 일반 개념, 용어 설명

질문: {question}
분류:"""

    try:
        result = _get_classifier().invoke([HumanMessage(content=prompt)])
        answer = result.content.strip().lower()
        valid_types = {"simple", "compare", "recommend", "risk", "general"}
        # 응답에서 유효한 유형 추출
        for vt in valid_types:
            if vt in answer:
                return vt
        return "general"
    except Exception as e:
        logger.warning(f"LLM 분류 실패, 키워드 fallback: {e}")
        from src.llm.classifier import classify_question_type
        return classify_question_type(question)


# ── 그래프 노드 ───────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """LLM 호출 (도구 선택 또는 최종 답변)"""
    question_type = state["question_type"]
    model = _get_model(question_type)

    messages = list(state["messages"])
    response = model.invoke(messages)

    return {"messages": [response]}


def call_tools(state: AgentState) -> dict:
    """도구 실행"""
    last_message = state["messages"][-1]

    tool_map = {t.name: t for t in ALL_TOOLS}
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        logger.info(f"도구 호출: {tool_name}({tool_args})")

        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(tool_args)
        else:
            result = f"알 수 없는 도구: {tool_name}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {
        "messages": tool_messages,
        "tool_call_count": state["tool_call_count"] + 1,
    }


# ── 라우팅 함수 ───────────────────────────────────────────

def should_call_tools(state: AgentState) -> str:
    """도구 호출이 필요한지 판단"""
    last_message = state["messages"][-1]

    # AIMessage이고 tool_calls가 있으면 도구 실행
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # 재검색 제한 (최대 2회)
        if state["tool_call_count"] >= 2:
            logger.info("도구 호출 횟수 초과 — 최종 답변으로 전환")
            return "end"
        return "tools"

    return "end"


# ── 그래프 빌드 ───────────────────────────────────────────

def build_graph() -> StateGraph:
    """LangGraph 에이전트 그래프 구성"""
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)

    # 진입점
    graph.set_entry_point("agent")

    # 조건부 엣지: agent → tools 또는 END
    graph.add_conditional_edges(
        "agent",
        should_call_tools,
        {"tools": "tools", "end": END},
    )

    # tools → agent (도구 결과를 LLM에 전달)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── 에이전트 실행 ─────────────────────────────────────────

_compiled_graph = None


def get_agent():
    """컴파일된 에이전트 그래프 반환 (싱글턴)"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        logger.info("LangGraph 에이전트 빌드 완료")
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
                elif node_name == "agent":
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {"event": "tool_call", "data": {"name": tc["name"], "args": tc["args"]}}

    yield {"event": "done", "data": {"answer": final_answer, "model": model_used, "question_type": question_type}}
