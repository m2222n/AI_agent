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
    """좋아요/싫어요 피드백 버튼 + 부정 피드백 시 사유 입력"""
    if not st.session_state.get("last_answer"):
        return

    # 이미 피드백을 남긴 경우 표시하지 않음
    if st.session_state.get("feedback_submitted"):
        return

    st.divider()
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("👍 도움됨", key="feedback_positive"):
            log_feedback(
                st.session_state.last_question,
                st.session_state.last_answer,
                "positive"
            )
            st.session_state.feedback_submitted = True
            st.toast("피드백 감사합니다!", icon="👍")
            st.rerun()

    with col2:
        if st.button("👎 아쉬워요", key="feedback_negative"):
            st.session_state.show_feedback_detail = True
            st.rerun()

    # 부정 피드백 상세 사유 입력
    if st.session_state.get("show_feedback_detail"):
        reason = st.radio(
            "어떤 점이 아쉬웠나요?",
            ["정보가 부정확해요", "원하는 답변이 아니에요", "더 자세한 정보가 필요해요", "기타"],
            key="feedback_reason",
            horizontal=True,
        )
        detail = ""
        if reason == "기타":
            detail = st.text_input("구체적으로 알려주세요:", key="feedback_detail_input")

        if st.button("피드백 제출", key="submit_feedback_detail"):
            feedback_text = f"negative:{reason}"
            if detail:
                feedback_text += f" - {detail}"
            log_feedback(
                st.session_state.last_question,
                st.session_state.last_answer,
                feedback_text
            )
            st.session_state.feedback_submitted = True
            st.session_state.show_feedback_detail = False
            st.toast("개선에 참고하겠습니다!", icon="🙏")
            st.rerun()


def render_reset_button():
    """대화 초기화 버튼"""
    if st.session_state.messages:
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.last_answer = ""
            st.session_state.last_question = ""
            st.session_state.feedback_submitted = False
            st.session_state.show_feedback_detail = False
            st.rerun()
