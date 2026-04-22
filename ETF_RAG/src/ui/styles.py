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

/* ── 비교 테이블 ────────────────────────────────── */
.stMarkdown table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

.stMarkdown table th {
    background-color: var(--primary-light) !important;
    font-weight: 600 !important;
    padding: 0.6rem !important;
}

.stMarkdown table td {
    padding: 0.5rem 0.6rem !important;
}

/* ── 상태 캡션 ──────────────────────────────────── */
.stCaption {
    opacity: 0.6 !important;
    font-size: 0.78rem !important;
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

/* ── 모바일 반응형 ──────────────────────────────── */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    section[data-testid="stSidebar"] {
        width: 280px !important;
    }

    h1 {
        font-size: 1.4rem !important;
    }

    .stButton > button {
        font-size: 0.8rem !important;
        padding: 0.5rem !important;
    }

    .stMarkdown table {
        font-size: 0.85rem !important;
    }
}
</style>
"""


def inject_custom_css():
    """Streamlit 페이지에 커스텀 CSS 주입"""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
