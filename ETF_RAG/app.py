"""
ETF RAG 챗봇 - Streamlit Entry Point

LLM 기반 ETF 질의응답 시스템
- RAG 파이프라인: FAISS + Kiwi BM25 하이브리드 검색
- 에이전트: LangGraph + Function Calling + 모델 라우팅
- UI: Streamlit
"""

import streamlit as st

from src.data.loader import load_etf_data as _load_etf_data, create_documents
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
def init_retriever():
    """하이브리드 검색기 초기화 + 에이전트 도구 주입"""
    etf_data = load_etf_data()
    documents = create_documents(etf_data)
    vectorstore = _create_vectorstore(documents)
    retriever = HybridRetriever(vectorstore, documents)

    # LangGraph 도구에 retriever 주입
    set_retriever(retriever, documents)

    return retriever


def main():
    st.set_page_config(page_title="ETF 질의응답 챗봇", page_icon="📈", layout="wide")
    st.title("📈 ETF 질의응답 챗봇")
    st.caption("LangGraph 에이전트 기반 ETF 투자 정보 검색 시스템")

    # 사이드바
    etf_data = load_etf_data()
    render_sidebar(etf_data)

    # OpenAI API 키 확인 (에이전트에서 OPENAI_API_KEY 환경변수 사용)
    try:
        get_api_key(st.secrets)
    except APIKeyMissingError:
        st.error("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        st.info("Streamlit Cloud: Settings → Secrets에서 설정하세요.")
        st.stop()

    # 하이브리드 검색기 + 에이전트 초기화
    with st.spinner("ETF 데이터베이스 로딩 중..."):
        init_retriever()

    # 세션 상태
    init_session_state()

    # 대화 히스토리
    render_chat_history()

    # 예시 질문
    render_example_questions()
    example_question = st.session_state.pop("example_q", None)

    # 채팅 입력
    user_input = st.chat_input("ETF에 대해 궁금한 점을 물어보세요...")
    question = example_question or user_input

    if question:
        process_question(question)

    # 피드백 + 초기화
    render_feedback_buttons()
    render_reset_button()


if __name__ == "__main__":
    main()
