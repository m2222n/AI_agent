"""
ETF RAG 챗봇 - Streamlit Entry Point

LLM 기반 ETF 질의응답 시스템
- RAG 파이프라인: FAISS + Kiwi BM25 하이브리드 검색
- 에이전트: LangGraph + Function Calling + 모델 라우팅
- UI: Streamlit
"""

import streamlit as st

from config import is_langsmith_enabled
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


@st.cache_resource
def load_etf_data():
    return _load_etf_data()


@st.cache_resource
def load_stock_data():
    return _load_stock_data()


@st.cache_resource
def init_retriever():
    """하이브리드 검색기 초기화 + 에이전트 도구 주입 (ETF + 주식)"""
    etf_data = load_etf_data()
    documents = create_documents(etf_data)
    vectorstore = _create_vectorstore(documents)
    etf_retriever = HybridRetriever(vectorstore, documents)

    # 주식 retriever (데이터 있을 때만)
    stock_data = load_stock_data()
    stock_retriever = None
    if stock_data:
        stock_documents = create_stock_documents(stock_data)
        stock_vectorstore = _create_vectorstore(stock_documents)
        stock_retriever = HybridRetriever(stock_vectorstore, stock_documents)

    # LangGraph 도구에 retriever 주입
    set_retriever(etf_retriever, documents, stock_retriever=stock_retriever)

    return etf_retriever


def main():
    st.set_page_config(page_title="투자 질의응답 챗봇", page_icon="📈", layout="wide")
    st.title("📈 투자 질의응답 챗봇")
    st.caption("LangGraph 에이전트 기반 ETF/주식 투자 정보 검색 시스템")

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

    # 하이브리드 검색기 + 에이전트 초기화
    with st.spinner("데이터베이스 로딩 중..."):
        init_retriever()

    # 세션 상태
    init_session_state()

    # 대화 히스토리
    render_chat_history()

    # 예시 질문
    render_example_questions()
    example_question = st.session_state.pop("example_q", None)

    # 채팅 입력
    user_input = st.chat_input("ETF/주식에 대해 궁금한 점을 물어보세요...")
    question = example_question or user_input

    if question:
        process_question(question)

    # 피드백 + 초기화
    render_feedback_buttons()
    render_reset_button()


if __name__ == "__main__":
    main()
