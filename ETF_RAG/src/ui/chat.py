import logging
import time
import traceback

import streamlit as st

from src.llm.agent import stream_agent
from src.ui.charts import try_parse_comparison, render_comparison, try_parse_structured_data, render_structured_data
from src.utils.logging import log_interaction

logger = logging.getLogger(__name__)

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

TOOL_DISPLAY_NAMES = {
    "search_etf": "📋 ETF 정보 검색",
    "compare_etfs": "⚖️ ETF 비교 분석",
    "get_etf_list": "📋 ETF 목록 조회",
    "search_stock": "📋 주식 정보 검색",
    "compare_stocks": "⚖️ 주식 비교 분석",
    "get_stock_list": "📋 주식 목록 조회",
    "get_realtime_price": "💰 실시간 시세 조회",
    "analyze_sector": "🏭 섹터 분석",
    "get_technical_indicators": "📊 기술적 지표 분석",
    "get_stock_correlation": "🔗 상관관계 분석",
    "simulate_portfolio": "💼 포트폴리오 시뮬레이션",
    "get_financial_statements": "📑 재무제표 조회",
    "predict_price_outlook": "🔮 가격 전망 분석",
}


def _get_followup_suggestions(question: str, tools_used: list, question_type: str) -> list[str]:
    """사용된 도구와 질문 유형 기반으로 후속 질문 2~3개 제안"""
    suggestions = []

    # 질문에서 종목명 추출 (간단한 휴리스틱)
    # 주요 종목명이 있으면 후속 질문에 활용
    stock_names = []
    common_stocks = ["삼성전자", "SK하이닉스", "현대차", "LG에너지솔루션", "카카오",
                     "네이버", "셀트리온", "기아", "포스코홀딩스", "삼성SDI"]
    for name in common_stocks:
        if name in question:
            stock_names.append(name)

    etf_names = []
    common_etfs = ["KODEX 200", "TIGER 200", "KODEX 레버리지", "TIGER 미국S&P500"]
    for name in common_etfs:
        if name in question:
            etf_names.append(name)

    target = stock_names[0] if stock_names else (etf_names[0] if etf_names else "")

    if "search_stock" in tools_used or "search_etf" in tools_used:
        if target:
            suggestions.append(f"{target} 기술적 분석해줘")
            suggestions.append(f"{target} 앞으로 전망은?")
    elif "get_technical_indicators" in tools_used:
        if target:
            suggestions.append(f"{target} 재무제표 보여줘")
            suggestions.append(f"{target} 최근 실적은 어때?")
    elif "predict_price_outlook" in tools_used:
        if target:
            suggestions.append(f"{target} 기술적 분석해줘")
    elif "compare_etfs" in tools_used or "compare_stocks" in tools_used:
        if len(stock_names) >= 1:
            suggestions.append(f"{stock_names[0]} 기술적 분석해줘")
    elif "get_financial_statements" in tools_used:
        if target:
            suggestions.append(f"{target} 기술적 분석해줘")
            suggestions.append(f"{target} 앞으로 전망은?")

    # 일반적인 후속 질문 추가
    if question_type == "simple" and target:
        if f"{target} 기술적 분석해줘" not in suggestions:
            suggestions.append(f"{target} 기술적 분석해줘")
    if not suggestions and target:
        suggestions.append(f"{target} 기술적 분석해줘")

    return suggestions[:3]


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
            event_count = 0
            token_count = 0
            tool_results_summary = []
            tools_used = []
            status_placeholder.caption("🔄 에이전트 시작...")
            for event in stream_agent(question, st.session_state.messages[:-1]):
                event_count += 1
                event_type = event.get("event", "unknown")
                logger.info(f"[chat] event #{event_count}: {event_type}")

                if event_type == "question_type":
                    question_type = event["data"]
                    st.session_state.last_question_type = question_type
                    type_label = QUESTION_TYPE_LABELS.get(question_type, question_type)
                    status_placeholder.caption(f"질문 유형: {type_label}")

                elif event_type == "tool_call":
                    tool_name = event["data"]["name"]
                    tools_used.append(tool_name)
                    display_name = TOOL_DISPLAY_NAMES.get(tool_name, f"🔍 {tool_name}")
                    status_placeholder.caption(f"{display_name} 중...")

                elif event_type == "tool_result":
                    tool_results_summary.append(str(event["data"])[:200])

                elif event_type == "structured_data":
                    parsed = try_parse_structured_data(event["data"])
                    if parsed:
                        comparison_data = parsed

                elif event_type == "token":
                    token_count += 1
                    full_response = event["data"]
                    answer_placeholder.markdown(full_response)

                elif event_type == "error":
                    st.warning(event["data"])

                elif event_type == "done":
                    done_answer = event["data"]["answer"]
                    model_used = event["data"]["model"]
                    question_type = event["data"].get("question_type", question_type)
                    logger.info(f"[chat] done: answer_len={len(done_answer)}, tokens={token_count}, model={model_used}")
                    # done의 answer가 더 길면 사용 (토큰 스트리밍이 안 됐을 수 있음)
                    if len(done_answer) > len(full_response):
                        full_response = done_answer
                        answer_placeholder.markdown(full_response)

            logger.info(f"[chat] stream 종료: {event_count}개 이벤트, {token_count}개 토큰, 답변 {len(full_response)}자")
            if not full_response:
                logger.warning(f"[chat] 빈 응답! events={event_count}, tokens={token_count}, tool_results={len(tool_results_summary)}")
                # 빈 응답 시 사용자에게 안내 메시지 표시
                full_response = "죄송합니다. 답변 생성에 실패했습니다. 다시 시도해주세요."
                answer_placeholder.markdown(full_response)

        except Exception as e:
            logger.error(f"[chat] stream 오류: {e}\n{traceback.format_exc()}")
            user_msg = _get_user_error_message(e)
            st.error(user_msg)
            full_response = user_msg
            # 재시도 버튼
            if st.button("🔄 다시 시도", key=f"retry_{hash(question)}"):
                st.session_state.messages.pop()  # 실패한 user 메시지 제거
                st.session_state["_retry_question"] = question
                st.rerun()

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

        # 성능 지표 — 간결한 캡션 (일반 사용자에게 부담 없게)
        status_placeholder.caption(f"⏱️ {total_time:.1f}초")

        # 후속 질문 제안
        followups = _get_followup_suggestions(question, tools_used, question_type or "")
        if followups:
            cols = st.columns(len(followups))
            for col, fq in zip(cols, followups):
                with col:
                    if st.button(f"💬 {fq}", key=f"followup_{hash(fq)}",
                                 use_container_width=True):
                        st.session_state["_retry_question"] = fq
                        st.rerun()

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
