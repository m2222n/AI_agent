import time

import streamlit as st

from src.llm.agent import stream_agent
from src.ui.charts import try_parse_comparison, render_comparison, try_parse_structured_data, render_structured_data
from src.utils.logging import log_interaction

QUESTION_TYPE_LABELS = {
    "simple": "📝 단순 정보",
    "compare": "⚖️ 비교 분석",
    "recommend": "💡 추천",
    "risk": "⚠️ 위험 분석",
    "general": "📚 일반 질문"
}

MODEL_LABELS = {
    "gpt-4o-mini": "⚡ GPT-4o-mini",
    "gpt-4o": "🧠 GPT-4o",
}


def _get_user_error_message(error: Exception) -> str:
    """예외 유형에 따라 사용자 친화적 메시지 반환"""
    error_str = str(error).lower()
    if "rate" in error_str or "429" in error_str:
        return "⚠️ API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
    if "timeout" in error_str or "timed out" in error_str:
        return "⚠️ 응답 시간이 초과되었습니다. 질문을 다시 시도해주세요."
    if "connection" in error_str or "network" in error_str:
        return "⚠️ 네트워크 연결에 문제가 있습니다. 인터넷 연결을 확인해주세요."
    if "api key" in error_str or "auth" in error_str:
        return "⚠️ API 인증에 실패했습니다. 설정을 확인해주세요."
    return "⚠️ 일시적인 오류가 발생했습니다. 다시 시도해주세요."


def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        "messages": [],
        "last_sources": [],
        "last_answer": "",
        "last_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_chat_history():
    """대화 히스토리 표시"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 저장된 구조화 데이터가 있으면 차트도 렌더링
            comparison = message.get("comparison_data")
            if comparison:
                render_structured_data(comparison)


def process_question(question: str, client=None, retriever=None):
    """LangGraph 에이전트를 통한 질문 처리

    Args:
        question: 사용자 질문
        client: OpenAI 클라이언트 (하위 호환, 에이전트에서는 미사용)
        retriever: HybridRetriever (하위 호환, tools.py에서 주입)
    """
    total_start_time = time.time()

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.last_question = question
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        status_placeholder = st.empty()
        chart_placeholder = st.container()

        question_type = None
        model_used = None
        full_response = ""
        comparison_data = None

        try:
            for event in stream_agent(question, st.session_state.messages[:-1]):
                if event["event"] == "question_type":
                    question_type = event["data"]
                    st.session_state.last_question_type = question_type
                    type_label = QUESTION_TYPE_LABELS.get(question_type, question_type)
                    status_placeholder.caption(f"질문 유형: {type_label}")

                elif event["event"] == "tool_call":
                    tool_name = event["data"]["name"]
                    status_placeholder.caption(f"🔍 {tool_name} 검색 중...")

                elif event["event"] == "structured_data":
                    parsed = try_parse_structured_data(event["data"])
                    if parsed:
                        comparison_data = parsed

                elif event["event"] == "token":
                    full_response = event["data"]
                    answer_placeholder.markdown(full_response)

                elif event["event"] == "error":
                    st.warning(event["data"])

                elif event["event"] == "done":
                    full_response = event["data"]["answer"]
                    model_used = event["data"]["model"]
                    question_type = event["data"].get("question_type", question_type)

        except Exception as e:
            error_msg = _get_user_error_message(e)
            st.error(error_msg)
            full_response = error_msg

        total_time = time.time() - total_start_time

        if full_response:
            answer_placeholder.markdown(full_response)
        st.session_state.last_answer = full_response

        # 구조화 데이터 렌더링 (비교 테이블, 기술적 차트 등)
        if comparison_data:
            with chart_placeholder:
                render_structured_data(comparison_data)

        # 메시지 저장 (비교 데이터 포함)
        msg = {"role": "assistant", "content": full_response}
        if comparison_data:
            msg["comparison_data"] = comparison_data
        st.session_state.messages.append(msg)

        # 성능 지표 표시
        model_label = MODEL_LABELS.get(model_used, model_used or "")
        type_label = QUESTION_TYPE_LABELS.get(question_type, question_type or "")
        status_placeholder.caption(
            f"질문 유형: {type_label} | 모델: {model_label} | "
            f"⏱️ {total_time:.1f}s"
        )

        # 로그 저장
        log_interaction(
            question=question,
            answer=full_response,
            sources=[],
            question_type=question_type or "general",
            search_time=0,
            llm_time=total_time,
            total_time=total_time
        )
