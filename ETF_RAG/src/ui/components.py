import streamlit as st

from src.utils.logging import log_feedback

# 카테고리별 예시 질문
EXAMPLE_CATEGORIES = {
    "기본 조회": {
        "icon": "🔍",
        "questions": [
            "삼성전자 주가 정보 알려줘",
            "KODEX 200 ETF에 대해 알려줘",
        ],
    },
    "비교 분석": {
        "icon": "⚖️",
        "questions": [
            "삼성전자랑 SK하이닉스 비교해줘",
            "KODEX 200이랑 TIGER 200 비교해줘",
        ],
    },
    "기술적 분석": {
        "icon": "📊",
        "questions": [
            "삼성전자 기술적 분석해줘",
            "현대차 앞으로 어떨까?",
        ],
    },
    "심화 분석": {
        "icon": "🧪",
        "questions": [
            "반도체 관련 ETF 추천해줘",
            "삼성전자 60% SK하이닉스 40% 포트폴리오 시뮬레이션",
        ],
    },
}

# 기능 소개 카드 (tab_index: st.tabs 0-indexed)
# 탭 순서: 0=종합채팅, 1=기술적분석, 2=재무제표, 3=가격전망, 4=비교분석
# action="sidebar_stock" → 사이드바 주식 탭 전환 (tab_index 무시)
FEATURE_CARDS = [
    {
        "icon": "📈",
        "title": "실시간 시세",
        "desc": "ETF/주식 4,300+ 종목의 최신 가격과 수익률",
        "action": "sidebar_stock",
    },
    {
        "icon": "📊",
        "title": "기술적 분석",
        "desc": "11개 지표 + 차트 자동 생성 (MA, RSI, MACD 등)",
        "tab_index": 1,  # 기술적 분석
    },
    {
        "icon": "📑",
        "title": "재무제표",
        "desc": "분기별 매출·영업이익·순이익 추이와 성장률",
        "tab_index": 2,  # 재무제표
    },
    {
        "icon": "🔮",
        "title": "가격 전망",
        "desc": "기술적+펀더멘털+회귀모델 3축 분석",
        "tab_index": 3,  # 가격 전망
    },
    {
        "icon": "⚖️",
        "title": "비교/시뮬레이션",
        "desc": "종목 비교, 포트폴리오 백테스트",
        "tab_index": 4,  # 비교 분석
    },
]


def render_welcome():
    """첫 방문자용 웰컴 화면 (대화 없을 때만)"""
    if st.session_state.messages:
        return

    # 기능 소개 카드
    st.markdown(
        '<p style="text-align:center; color:#888; margin-bottom:0.5rem; font-size:0.9rem;">'
        "무엇을 도와드릴까요?</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(FEATURE_CARDS))
    for col, card in zip(cols, FEATURE_CARDS):
        with col:
            if st.button(
                f'{card["icon"]} {card["title"]}',
                key=f"card_{card['title']}",
                use_container_width=True,
                help=card["desc"],
            ):
                if card.get("action") == "sidebar_stock":
                    st.session_state["_goto_sidebar_stock"] = True
                else:
                    st.session_state["_goto_tab"] = card["tab_index"]
                st.rerun()
            st.caption(card["desc"])

    st.markdown("")  # spacer

    # 예시 질문 라벨
    st.caption("💡 예시 질문")

    # 카테고리별 예시 질문
    for cat_name, cat_info in EXAMPLE_CATEGORIES.items():
        st.markdown(
            f'<p style="font-size:0.8rem; color:#666; margin:0.6rem 0 0.3rem; font-weight:500;">'
            f'{cat_info["icon"]} {cat_name}</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, q in enumerate(cat_info["questions"]):
            with cols[i % 2]:
                if st.button(q, use_container_width=True, key=f"ex_{cat_name}_{i}"):
                    st.session_state.example_q = q
                    st.rerun()


def render_example_questions():
    """대화 시작 전 예시 질문 — render_welcome()으로 대체"""
    render_welcome()


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
