"""
커스텀 CSS 스타일 — Streamlit 기본 UI 개선

적용 항목:
- 채팅 메시지 스타일 (배경색, 둥근 모서리)
- 사이드바 카드 스타일
- 예시 질문 버튼 스타일
- 비교 차트 테이블 스타일
- 웰컴 카드 스타일
- 반응형 모바일 대응
"""

CUSTOM_CSS = """
<style>
/* ── 색상 변수 ─────────────────────────────────── */
:root {
    --primary: #2563EB;
    --primary-light: rgba(37, 99, 235, 0.08);
    --primary-border: rgba(37, 99, 235, 0.15);
    --text-primary: #1F2937;
    --text-secondary: #6B7280;
    --bg-subtle: rgba(107, 114, 128, 0.04);
    --border-light: rgba(128, 128, 128, 0.15);
    --radius: 10px;
}

/* ── 전역 ───────────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

/* 본문 텍스트 가독성 (line-height) */
.stMarkdown, .stChatMessage {
    line-height: 1.7 !important;
}

/* ── 헤더 영역 ──────────────────────────────────── */
h1 {
    font-size: 1.8rem !important;
    margin-bottom: 0 !important;
    color: var(--text-primary) !important;
}

/* ── 채팅 메시지 ────────────────────────────────── */
.stChatMessage {
    border-radius: var(--radius) !important;
    margin-bottom: 0.5rem !important;
    padding: 0.8rem 1rem !important;
}

/* 사용자 메시지 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: var(--primary-light) !important;
}

/* 어시스턴트 메시지 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: var(--bg-subtle) !important;
}

/* ── 버튼 ─────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 0.9rem !important;
    transition: all 0.15s ease !important;
    border-color: var(--border-light) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    border-color: var(--primary-border) !important;
}

/* ── 사이드바 ───────────────────────────────────── */
section[data-testid="stSidebar"] {
    width: 320px !important;
}

section[data-testid="stSidebar"] .stExpander {
    border-radius: 8px !important;
    border: 1px solid var(--border-light) !important;
    margin-bottom: 0.3rem !important;
}

/* ── 비교 테이블 (가로 스크롤 지원) ───────────── */
.stMarkdown table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* 테이블 감싸는 div에 가로 스크롤 */
.stMarkdown div:has(> table) {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
}

.stMarkdown table th {
    background-color: var(--primary-light) !important;
    font-weight: 600 !important;
    padding: 0.6rem !important;
    white-space: nowrap !important;
}

.stMarkdown table td {
    padding: 0.5rem 0.6rem !important;
}

/* ── 상태 캡션 ──────────────────────────────────── */
.stCaption {
    opacity: 0.6 !important;
    font-size: 0.82rem !important;
}

/* ── 메트릭 카드 ────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: var(--bg-subtle) !important;
    border-radius: var(--radius) !important;
    padding: 0.8rem !important;
}

/* ── 토스트 알림 ────────────────────────────────── */
.stToast {
    border-radius: var(--radius) !important;
}

/* ── 차트 영역 ──────────────────────────────────── */
[data-testid="stBarChart"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── 피드백 라디오 버튼 ─────────────────────────── */
.stRadio > div {
    gap: 0.5rem !important;
}

/* ── divider ────────────────────────────────────── */
hr {
    margin-top: 0.8rem !important;
    margin-bottom: 0.8rem !important;
    opacity: 0.2 !important;
}

/* ── 채팅 입력 ──────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    border-radius: var(--radius) !important;
}

/* ── 웰컴 화면 ────────────────────────────────────── */
.welcome-hero {
    text-align: center;
    margin-bottom: 1rem;
}

.welcome-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
}

.welcome-section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 0.8rem 0 0.4rem;
}

.welcome-card {
    background: var(--bg-subtle);
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    padding: 1rem;
    text-align: center;
    transition: all 0.2s ease;
    margin-bottom: 0.3rem;
}

.welcome-card:hover {
    border-color: var(--primary-border);
    box-shadow: 0 2px 12px rgba(37, 99, 235, 0.08);
    transform: translateY(-2px);
}

.welcome-card-icon {
    font-size: 1.8rem;
    margin-bottom: 0.4rem;
}

.welcome-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.welcome-card-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}

.welcome-card-detail {
    font-size: 0.75rem;
    color: #9CA3AF;
    line-height: 1.4;
}

/* ── 모바일 반응형: 태블릿 ───────────────────────── */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 100% !important;
    }

    section[data-testid="stSidebar"] {
        width: 280px !important;
    }

    h1 {
        font-size: 1.4rem !important;
    }

    .stButton > button {
        font-size: 0.85rem !important;
        padding: 0.5rem 0.7rem !important;
    }

    .stMarkdown table {
        font-size: 0.85rem !important;
    }

    /* 탭 라벨 축소 */
    [role="tab"] {
        font-size: 0.82rem !important;
        padding: 0.4rem 0.5rem !important;
    }

    /* 멀티컬럼 줄바꿈 허용 (3~4컬럼 → 2열로 축소) */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.3rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 45% !important;
        min-width: 45% !important;
    }

    /* 웰컴 카드 축소 */
    .welcome-card {
        padding: 0.7rem;
    }
    .welcome-card-icon {
        font-size: 1.4rem;
    }
    .welcome-card-detail {
        display: none;
    }
}

/* ── 모바일 반응형: 소형 폰 (<480px) ─────────────── */
@media (max-width: 480px) {
    .main .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }

    h1 {
        font-size: 1.25rem !important;
    }

    /* 채팅 메시지 패딩 축소 */
    .stChatMessage {
        padding: 0.6rem 0.7rem !important;
    }

    /* 테이블 모바일 최적화 */
    .stMarkdown table {
        font-size: 0.8rem !important;
    }
    .stMarkdown table th,
    .stMarkdown table td {
        padding: 0.35rem 0.4rem !important;
    }

    /* 메트릭 카드 컴팩트 */
    [data-testid="stMetric"] {
        padding: 0.5rem !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }

    /* 캡션 최소 크기 보장 */
    .stCaption {
        font-size: 0.8rem !important;
    }

    /* 탭 라벨 더 축소 */
    [role="tab"] {
        font-size: 0.75rem !important;
        padding: 0.35rem 0.3rem !important;
    }

    /* 사이드바 너비 축소 */
    section[data-testid="stSidebar"] {
        width: 260px !important;
    }

    /* 멀티컬럼 → 세로 스택 (st.columns 모바일 대응) */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.4rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
}
</style>
"""


def inject_custom_css():
    """Streamlit 페이지에 커스텀 CSS 주입"""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
