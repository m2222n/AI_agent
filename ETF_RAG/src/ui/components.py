import logging
from typing import Optional

import streamlit as st

from src.utils.logging import log_feedback

logger = logging.getLogger(__name__)

# 하드코딩 예시 (데이터 없을 때 fallback)
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


def generate_dynamic_examples(
    etf_data: Optional[list] = None,
    stock_data: Optional[list] = None,
) -> Optional[dict]:
    """당일 수집 데이터 기반 동적 예시 질문 생성.

    Returns:
        카테고리별 예시 dict (EXAMPLE_CATEGORIES와 동일 형식) 또는 None (데이터 부족)
    """
    all_data = []
    if etf_data:
        all_data.extend(etf_data)
    if stock_data:
        all_data.extend(stock_data)

    if len(all_data) < 10:
        return None

    # change_pct가 있는 종목만 필터 (종가 0 제외)
    with_change = [
        d for d in all_data
        if d.get("change_pct") is not None
        and d.get("close", 0) > 0
        and d.get("name")
    ]
    if len(with_change) < 5:
        return None

    # 급등 상위 (change_pct > 0)
    gainers = sorted(with_change, key=lambda d: d.get("change_pct", 0), reverse=True)
    # 급락 하위 (change_pct < 0)
    losers = sorted(with_change, key=lambda d: d.get("change_pct", 0))
    # 거래대금 상위
    by_volume = sorted(all_data, key=lambda d: d.get("trade_value", 0), reverse=True)

    # 이름 추출 헬퍼 (ETF는 길어서 앞 부분만)
    def _name(d: dict) -> str:
        name = d["name"]
        return name if len(name) <= 15 else name[:15]

    categories = {}

    # 1) 오늘의 급등주 — 상위 2개
    top_gainers = [g for g in gainers[:5] if g.get("change_pct", 0) > 0][:2]
    if top_gainers:
        questions = []
        for g in top_gainers:
            pct = g["change_pct"]
            questions.append(f"{_name(g)} 오늘 {pct:+.1f}% 왜 올랐어?")
        categories["오늘의 급등주"] = {"icon": "🔥", "questions": questions}

    # 2) 오늘의 급락주 — 하위 2개
    top_losers = [l for l in losers[:5] if l.get("change_pct", 0) < 0][:2]
    if top_losers:
        questions = []
        for l in top_losers:
            pct = l["change_pct"]
            questions.append(f"{_name(l)} {pct:+.1f}% 하락, 기술적 분석해줘")
        categories["오늘의 급락주"] = {"icon": "📉", "questions": questions}

    # 3) 거래대금 상위 — 상위 2개
    top_volume = [v for v in by_volume[:5] if v.get("trade_value", 0) > 0 and v.get("name")][:2]
    if top_volume:
        questions = []
        for v in top_volume:
            questions.append(f"{_name(v)} 앞으로 어떨까?")
        categories["거래대금 TOP"] = {"icon": "💰", "questions": questions}

    # 4) 비교 질문 — 급등 1위 vs 거래대금 1위 (겹치지 않을 때)
    if top_gainers and top_volume:
        g_name = _name(top_gainers[0])
        v_name = _name(top_volume[0])
        if g_name != v_name:
            categories["비교 분석"] = {
                "icon": "⚖️",
                "questions": [f"{g_name}이랑 {v_name} 비교해줘"],
            }

    if not categories:
        return None

    return categories

# 기능 소개 카드 (tab_index: st.tabs 0-indexed)
# 탭 순서: 0=종합채팅, 1=기술적분석, 2=재무제표, 3=가격전망, 4=비교분석, 5=섹터
# action="sidebar_stock" → 사이드바 주식 탭 전환 (tab_index 무시)
FEATURE_CARDS = [
    {
        "icon": "💬",
        "title": "AI 채팅",
        "desc": "자연어로 자유롭게 질문하세요",
        "detail": "\"삼성전자 알려줘\", \"반도체 ETF 추천\" 등 자유로운 대화",
        "tab_index": 0,
    },
    {
        "icon": "📊",
        "title": "기술적 분석",
        "desc": "11개 지표 + 차트 자동 생성",
        "detail": "MA, RSI, MACD, 볼린저, 일목균형표, 스토캐스틱 등",
        "tab_index": 1,
    },
    {
        "icon": "📑",
        "title": "재무제표",
        "desc": "분기별 실적 추이와 성장률",
        "detail": "매출, 영업이익, 순이익, 마진율 — 2015년부터",
        "tab_index": 2,
    },
    {
        "icon": "🔮",
        "title": "가격 전망",
        "desc": "AI 기반 3축 종합 분석",
        "detail": "기술적 + 펀더멘털 + Ridge 회귀 모델",
        "tab_index": 3,
    },
    {
        "icon": "⚖️",
        "title": "비교 분석",
        "desc": "종목 비교, 포트폴리오 백테스트",
        "detail": "수익률/MDD/샤프 비교, KODEX 200 벤치마크",
        "tab_index": 4,
    },
    {
        "icon": "📰",
        "title": "뉴스 & 섹터",
        "desc": "실시간 뉴스 감성 분석 + 업종 분석",
        "detail": "Google News 기반 긍정/부정 판정, 업종별 등락",
        "tab_index": 5,
    },
]


def _render_example_categories(categories: dict, key_prefix: str = "ex") -> None:
    """카테고리별 예시 질문 버튼 렌더링 (공통 로직)."""
    for cat_name, cat_info in categories.items():
        st.markdown(
            f'<p style="font-size:0.8rem; color:#666; margin:0.6rem 0 0.3rem; font-weight:500;">'
            f'{cat_info["icon"]} {cat_name}</p>',
            unsafe_allow_html=True,
        )
        questions = cat_info["questions"]
        cols = st.columns(min(len(questions), 2))
        for i, q in enumerate(questions):
            with cols[i % len(cols)]:
                if st.button(q, use_container_width=True, key=f"{key_prefix}_{cat_name}_{i}"):
                    st.session_state.example_q = q
                    st.rerun()


def _render_feature_cards():
    """기능 소개 카드 — HTML 기반 시각적 카드."""
    # 3열 × 2행 배치
    for row_start in range(0, len(FEATURE_CARDS), 3):
        row_cards = FEATURE_CARDS[row_start:row_start + 3]
        cols = st.columns(len(row_cards))
        for col, card in zip(cols, row_cards):
            with col:
                st.markdown(
                    f'<div class="welcome-card">'
                    f'<div class="welcome-card-icon">{card["icon"]}</div>'
                    f'<div class="welcome-card-title">{card["title"]}</div>'
                    f'<div class="welcome-card-desc">{card["desc"]}</div>'
                    f'<div class="welcome-card-detail">{card.get("detail", "")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    f'{card["title"]} 바로가기',
                    key=f"card_{card['title']}",
                    use_container_width=True,
                ):
                    if card.get("action") == "sidebar_stock":
                        st.session_state["_goto_sidebar_stock"] = True
                    else:
                        st.session_state["_goto_tab"] = card.get("tab_index", 0)
                    st.rerun()


def render_welcome(
    etf_data: Optional[list] = None,
    stock_data: Optional[list] = None,
):
    """첫 방문자용 웰컴 화면 (대화 없을 때만)"""
    if st.session_state.messages:
        return

    # 서비스 소개
    st.markdown(
        '<div class="welcome-hero">'
        '<p class="welcome-subtitle">'
        f'ETF &middot; 주식 <strong>{len(etf_data or []) + len(stock_data or []):,}</strong> 종목의 실시간 데이터를 AI가 분석합니다.<br>'
        '아래에서 원하는 기능을 선택하거나, 채팅창에 자유롭게 질문하세요.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # 기능 카드
    _render_feature_cards()

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

    # 동적 예시 질문 (당일 급등/급락/거래대금 기반)
    dynamic = generate_dynamic_examples(etf_data, stock_data)
    if dynamic:
        st.markdown(
            '<p class="welcome-section-title">🔥 오늘의 추천 질문</p>',
            unsafe_allow_html=True,
        )
        _render_example_categories(dynamic, key_prefix="dyn")
        st.markdown("")

    # 기본 예시 질문 (항상 표시)
    st.markdown(
        '<p class="welcome-section-title">💡 이렇게 물어보세요</p>',
        unsafe_allow_html=True,
    )
    _render_example_categories(EXAMPLE_CATEGORIES, key_prefix="ex")


def render_example_questions(
    etf_data: Optional[list] = None,
    stock_data: Optional[list] = None,
):
    """대화 시작 전 예시 질문 — render_welcome()으로 대체"""
    render_welcome(etf_data, stock_data)


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
