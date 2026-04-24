"""
ETF RAG 챗봇 - Streamlit Entry Point

LLM 기반 ETF 질의응답 시스템
- RAG 파이프라인: FAISS + Kiwi BM25 하이브리드 검색
- 에이전트: LangGraph + Function Calling + 모델 라우팅
- UI: Streamlit
"""

import logging
from pathlib import Path

import streamlit as st
import streamlit.components.v1

from config import is_langsmith_enabled
from src.data.db_downloader import ensure_db
from src.data.loader import (
    load_etf_data as _load_etf_data,
    load_stock_data as _load_stock_data,
    create_documents,
    create_stock_documents,
)
from src.rag.vectorstore import create_vectorstore as _create_vectorstore
from src.rag.retriever import HybridRetriever
from src.llm.client import get_api_key, APIKeyMissingError
from src.llm.tools import set_retriever
from src.ui.sidebar import render_sidebar
from src.ui.chat import init_session_state, render_chat_history, process_question
from src.ui.components import render_example_questions, render_feedback_buttons, render_reset_button
from src.ui.styles import inject_custom_css
from src.ui.tabs import render_technical_tab, render_financial_tab, render_comparison_tab, render_outlook_tab, render_sector_tab

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="데이터베이스 준비 중... (최초 실행 시 1~2분)")
def download_db():
    """Streamlit Cloud 시작 시 GitHub Release에서 DB 다운로드 (1회)."""
    db_path = Path(__file__).parent / "src" / "data" / "etf_rag.db"
    result = ensure_db(db_path)
    logger.info(f"DB 다운로드 결과: {'성공' if result else '실패/스킵'}")
    return result


@st.cache_resource(show_spinner="ETF/주식 데이터 로딩 중... (4,200+ 종목)")
def load_all_data():
    """ETF + 주식 데이터 로드 + 문서 생성."""
    etf_data = _load_etf_data()
    logger.info(f"ETF 데이터 로드: {len(etf_data)}종목")

    try:
        stock_data = _load_stock_data()
        logger.info(f"주식 데이터 로드 성공: {len(stock_data)}종목")
    except Exception as e:
        logger.error(f"주식 데이터 로드 실패: {e}", exc_info=True)
        stock_data = []

    documents = create_documents(etf_data)
    stock_documents = create_stock_documents(stock_data) if stock_data else []
    return etf_data, stock_data, documents, stock_documents


@st.cache_resource(show_spinner="검색 인덱스 구축 중... (임베딩 + BM25)")
def build_retrievers(_documents, _stock_documents):
    """FAISS 벡터스토어 + BM25 하이브리드 검색기 생성."""
    vectorstore = _create_vectorstore(_documents, prefix="etf")
    etf_retriever = HybridRetriever(vectorstore, _documents)

    stock_retriever = None
    if _stock_documents:
        try:
            stock_vectorstore = _create_vectorstore(_stock_documents, prefix="stock")
            stock_retriever = HybridRetriever(stock_vectorstore, _stock_documents)
            logger.info(f"주식 retriever 초기화 성공: {len(_stock_documents)}개 문서")
        except Exception as e:
            logger.error(f"주식 retriever 초기화 실패: {e}", exc_info=True)

    return etf_retriever, stock_retriever


def init_all():
    """전체 초기화 파이프라인 (단계별 spinner 표시)."""
    # Step 1: DB 다운로드
    download_db()

    # Step 2: 데이터 로드 + 문서 생성
    etf_data, stock_data, documents, stock_documents = load_all_data()

    # Step 3: 검색 인덱스 구축
    etf_retriever, stock_retriever = build_retrievers(documents, stock_documents)

    # Step 4: 에이전트 도구 주입
    set_retriever(etf_retriever, documents, stock_retriever=stock_retriever,
                  etf_data=etf_data, stock_data=stock_data)

    return etf_data, stock_data


def main():
    st.set_page_config(page_title="투자 AI 어시스턴트", page_icon="📈", layout="wide")
    inject_custom_css()
    # 헤더
    st.markdown(
        '<h1 style="margin-bottom:0;">📈 투자 AI 어시스턴트</h1>'
        '<p style="color:#888; font-size:0.9rem; margin-top:0.2rem;">'
        'ETF &middot; 주식 &middot; 기술적 분석 &middot; 재무제표 &middot; 가격 전망</p>',
        unsafe_allow_html=True,
    )

    # OpenAI API 키 확인 (Streamlit Cloud: st.secrets, 로컬: .env)
    try:
        secrets = None
        try:
            if len(st.secrets) > 0:
                secrets = st.secrets
        except Exception:
            pass
        get_api_key(secrets)
    except APIKeyMissingError:
        st.error("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        st.info("Streamlit Cloud: Settings → Secrets에서 설정하세요.")
        st.stop()

    # 전체 초기화 (단계별 spinner: DB 다운로드 → 데이터 로드 → 인덱스 구축)
    try:
        etf_data, stock_data = init_all()
    except Exception as e:
        logger.error(f"초기화 실패: {e}")
        st.error("데이터베이스 초기화에 실패했습니다. 페이지를 새로고침해주세요.")
        st.caption(f"오류 상세: {type(e).__name__}: {e}")
        st.stop()

    # 사이드바
    render_sidebar(etf_data, stock_data)

    # 세션 상태
    init_session_state()

    # 예시/후속 질문 처리 (탭 외부에서 pop — rerun 시 탭 상태와 무관하게 동작)
    example_question = st.session_state.pop("example_q", None)
    retry_question = st.session_state.pop("_retry_question", None)
    pending_question = retry_question or example_question
    if pending_question:
        logger.info(f"[app] pending_question: {pending_question} (retry={retry_question}, example={example_question})")

    # chat_input은 탭 밖에 배치 (Streamlit은 chat_input을 항상 하단 고정)
    user_input = st.chat_input("ETF/주식에 대해 궁금한 점을 물어보세요...")
    question = pending_question or user_input

    # 질문 입력 시 종합 채팅 탭으로 자동 전환
    if user_input and "_goto_tab" not in st.session_state:
        st.session_state["_goto_tab"] = 0

    # 탭 UI
    tab_chat, tab_tech, tab_fin, tab_outlook, tab_cmp, tab_sector = st.tabs([
        "💬 종합 채팅", "📊 기술적 분석", "📑 재무제표", "🔮 가격 전망", "⚖️ 비교 분석", "🏭 섹터"
    ])

    with tab_chat:
        # 대화 히스토리
        render_chat_history()

        # 예시 질문 (동적 + 기본)
        render_example_questions(etf_data, stock_data)

        if question:
            process_question(question)

        # 피드백 + 초기화
        render_feedback_buttons()
        render_reset_button()

    with tab_tech:
        render_technical_tab()

    with tab_fin:
        render_financial_tab()

    with tab_cmp:
        render_comparison_tab()

    with tab_outlook:
        render_outlook_tab()

    with tab_sector:
        render_sector_tab()

    # JS 기반 탭 전환 (카드 클릭 / 홈 버튼 / 사이드바 전환)
    js_commands = []

    # 메인 탭 전환
    goto_tab = st.session_state.pop("_goto_tab", None)
    if goto_tab is not None:
        js_commands.append(f"""
            const main = window.parent.document.querySelector(
                '.stMainBlockContainer, [data-testid="stAppViewBlockContainer"]'
            );
            if (main) {{
                const tabs = main.querySelectorAll('[role="tab"]');
                if (tabs.length > {goto_tab}) tabs[{goto_tab}].click();
            }}
        """)

    # 사이드바 주식 탭 전환 + 검색 포커스
    goto_sidebar = st.session_state.pop("_goto_sidebar_stock", None)
    if goto_sidebar:
        js_commands.append("""
            const sidebar = window.parent.document.querySelector(
                '[data-testid="stSidebar"]'
            );
            if (sidebar) {
                const sTabs = sidebar.querySelectorAll('[role="tab"]');
                // 주식 탭은 두 번째 (index 1)
                if (sTabs.length > 1) sTabs[1].click();
                // 검색 input에 포커스
                setTimeout(function() {
                    const inputs = sidebar.querySelectorAll('input[type="text"]');
                    if (inputs.length > 0) inputs[inputs.length - 1].focus();
                }, 300);
            }
        """)

    if js_commands:
        combined_js = "\n".join(js_commands)
        st.components.v1.html(
            f"<script>{combined_js}</script>",
            height=0,
        )


if __name__ == "__main__":
    main()
