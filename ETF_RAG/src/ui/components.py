import streamlit as st

from src.utils.logging import log_feedback

EXAMPLE_QUESTIONS = [
    "KODEX 200 ETF에 대해 알려줘",
    "삼성전자 주가 정보 알려줘",
    "반도체 관련 ETF 추천해줘",
    "삼성전자랑 SK하이닉스 비교해줘",
]


def render_example_questions():
    """대화 시작 전 예시 질문 버튼 표시"""
    if not st.session_state.messages:
        st.markdown("### 💡 이런 질문을 해보세요:")
        col1, col2 = st.columns(2)

        with col1:
            if st.button(EXAMPLE_QUESTIONS[0], use_container_width=True):
                st.session_state.example_q = EXAMPLE_QUESTIONS[0]
                st.rerun()
            if st.button(EXAMPLE_QUESTIONS[2], use_container_width=True):
                st.session_state.example_q = EXAMPLE_QUESTIONS[2]
                st.rerun()

        with col2:
            if st.button(EXAMPLE_QUESTIONS[1], use_container_width=True):
                st.session_state.example_q = EXAMPLE_QUESTIONS[1]
                st.rerun()
            if st.button(EXAMPLE_QUESTIONS[3], use_container_width=True):
                st.session_state.example_q = EXAMPLE_QUESTIONS[3]
                st.rerun()


def render_feedback_buttons():
    """좋아요/싫어요 피드백 버튼"""
    if st.session_state.last_answer:
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 4])

        with col1:
            if st.button("👍 도움됨", key="feedback_positive"):
                log_feedback(
                    st.session_state.last_question,
                    st.session_state.last_answer,
                    "positive"
                )
                st.success("피드백 감사합니다!")

        with col2:
            if st.button("👎 별로", key="feedback_negative"):
                log_feedback(
                    st.session_state.last_question,
                    st.session_state.last_answer,
                    "negative"
                )
                st.info("개선에 참고하겠습니다!")


def render_reset_button():
    """대화 초기화 버튼"""
    if st.session_state.messages:
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.last_answer = ""
            st.session_state.last_question = ""
            st.rerun()
