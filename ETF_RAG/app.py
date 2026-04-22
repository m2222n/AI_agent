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
from src.ui.tabs import render_technical_tab, render_financial_tab, render_comparison_tab, render_outlook_tab

logger = logging.getLogger(__name__)


@st.cache_resource
def load_etf_data():
    return _load_etf_data()


@st.cache_resource
def load_stock_data():
    try:
        data = _load_stock_data()
        logger.info(f"주식 데이터 로드 성공: {len(data)}종목")
        return data
    except Exception as e:
        logger.error(f"주식 데이터 로드 실패: {e}", exc_info=True)
        return []


@st.cache_resource
def download_db():
    """Streamlit Cloud 시작 시 GitHub Release에서 DB 다운로드 (1회)."""
    db_path = Path(__file__).parent / "src" / "data" / "etf_rag.db"
    result = ensure_db(db_path)
    logger.info(f"DB 다운로드 결과: {'성공' if result else '실패/스킵'}")
    return result


@st.cache_resource
def init_retriever():
    """하이브리드 검색기 초기화 + 에이전트 도구 주입 (ETF + 주식)"""
    etf_data = load_etf_data()
    logger.info(f"ETF 데이터 로드: {len(etf_data)}종목")
    documents = create_documents(etf_data)
    vectorstore = _create_vectorstore(documents, prefix="etf")
    etf_retriever = HybridRetriever(vectorstore, documents)

    # 주식 retriever (데이터 있을 때만, 실패해도 진행)
    stock_data = load_stock_data()
    logger.info(f"주식 데이터 로드: {len(stock_data)}종목")
    stock_retriever = None
    if stock_data:
        try:
            stock_documents = create_stock_documents(stock_data)
            stock_vectorstore = _create_vectorstore(stock_documents, prefix="stock")
            stock_retriever = HybridRetriever(stock_vectorstore, stock_documents)
            logger.info(f"주식 retriever 초기화 성공: {len(stock_documents)}개 문서")
        except Exception as e:
            logger.error(f"주식 retriever 초기화 실패 (인덱스는 생성): {e}", exc_info=True)

    # LangGraph 도구에 retriever + 원본 데이터 주입 (구조화 비교용)
    # stock_retriever 실패해도 stock_data 인덱스는 반드시 생성
    set_retriever(etf_retriever, documents, stock_retriever=stock_retriever,
                  etf_data=etf_data, stock_data=stock_data)

    return etf_retriever


def main():
    st.set_page_config(page_title="투자 AI 어시스턴트", page_icon="📈", layout="wide")
    inject_custom_css()
    st.markdown(
        '<h1 style="margin-bottom:0;">📈 투자 AI 어시스턴트</h1>'
        '<p style="color:#888; font-size:0.9rem; margin-top:0.2rem;">'
        'ETF &middot; 주식 &middot; 기술적 분석 &middot; 재무제표 &middot; 가격 전망</p>',
        unsafe_allow_html=True,
    )

    # 사이드바
    etf_data = load_etf_data()
    stock_data = load_stock_data()
    render_sidebar(etf_data, stock_data)

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

    # DB 다운로드 (Streamlit Cloud: 최초 실행 시 GitHub Release에서 다운로드)
    with st.spinner("데이터베이스 준비 중... (최초 실행 시 1~2분 소요)"):
        download_db()

    # 하이브리드 검색기 + 에이전트 초기화
    try:
        with st.spinner("데이터베이스 로딩 중..."):
            init_retriever()
    except Exception as e:
        logger.error(f"검색기 초기화 실패: {e}")
        st.error("데이터베이스 초기화에 실패했습니다. 페이지를 새로고침해주세요.")
        st.caption(f"오류 상세: {type(e).__name__}: {e}")
        st.stop()

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

    # 탭 UI
    tab_chat, tab_tech, tab_fin, tab_cmp, tab_outlook = st.tabs([
        "💬 종합 채팅", "📊 기술적 분석", "📑 재무제표", "⚖️ 비교 분석", "🔮 가격 전망"
    ])

    with tab_chat:
        # 대화 히스토리
        render_chat_history()

        # 예시 질문
        render_example_questions()

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


if __name__ == "__main__":
    main()
