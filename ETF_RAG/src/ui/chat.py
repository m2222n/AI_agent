import time

import streamlit as st

from src.rag.retriever import retrieve_relevant_docs
from src.llm.classifier import classify_question_type
from src.llm.client import (
    call_llm_streaming, LLMError, RateLimitExceededError, ConnectionFailedError
)
from src.utils.logging import log_interaction

QUESTION_TYPE_LABELS = {
    "simple": "📝 단순 정보",
    "compare": "⚖️ 비교 분석",
    "recommend": "💡 추천",
    "risk": "⚠️ 위험 분석",
    "general": "📚 일반 질문"
}


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


def process_question(question: str, client, vectorstore):
    """질문 처리: 분류 → 검색 → LLM 스트리밍 → 로깅"""
    total_start_time = time.time()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 질문 유형 분류
        question_type = classify_question_type(question)
        st.session_state.last_question_type = question_type

        # 관련 문서 검색
        search_start_time = time.time()
        context, sources = retrieve_relevant_docs(vectorstore, question)
        search_time = time.time() - search_start_time

        st.session_state.last_sources = sources
        st.session_state.last_question = question

        st.caption(f"질문 유형: {QUESTION_TYPE_LABELS.get(question_type, question_type)}")

        # LLM 스트리밍 호출
        llm_start_time = time.time()
        try:
            response_stream = call_llm_streaming(
                client, context, question, st.session_state.messages, question_type
            )
        except RateLimitExceededError:
            st.error("⚠️ API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
            return
        except ConnectionFailedError:
            st.error("⚠️ 네트워크 연결 오류가 발생했습니다. 인터넷 연결을 확인해주세요.")
            return
        except LLMError as e:
            st.error(f"⚠️ 오류 발생: {e}")
            return

        # 스트리밍 응답 표시
        answer_placeholder = st.empty()
        full_response = ""

        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                answer_placeholder.markdown(full_response + "▌")

        llm_time = time.time() - llm_start_time
        total_time = time.time() - total_start_time

        answer_placeholder.markdown(full_response)
        st.session_state.last_answer = full_response

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

        # 참고 ETF 표시
        if sources:
            st.divider()
            st.markdown("**🔍 검색된 ETF 정보:**")
            for src in sources:
                st.write(
                    f"- **{src['id']}** {src['name']} ({src['ticker']}) "
                    f"- 관련도: {src['relevance_score']:.0%}"
                )

        # 성능 지표 표시
        st.caption(
            f"⏱️ 응답시간: {total_time*1000:.0f}ms "
            f"(검색: {search_time*1000:.0f}ms, LLM: {llm_time*1000:.0f}ms)"
        )

        # 로그 저장
        log_interaction(
            question=question,
            answer=full_response,
            sources=sources,
            question_type=question_type,
            search_time=search_time,
            llm_time=llm_time,
            total_time=total_time
        )
