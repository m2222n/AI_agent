"""
FastAPI 백엔드 초기화 (Streamlit-free).

app.py의 init_all()(@st.cache_resource로 감싸진 4단계)을 데코레이터 없이 복제한다.
underlying 함수들은 전부 Streamlit 비의존이므로 그대로 호출 가능.
초기화 결과(retriever 등)는 src.llm.tools._state 프로세스 전역에 들어가므로
run_agent/stream_agent가 투명하게 읽는다. AppState에는 init 상태만 보관한다.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, Request

from src.llm.client import get_api_key
from src.data.db_downloader import ensure_db
from src.data.loader import (
    load_etf_data,
    load_stock_data,
    create_documents,
    create_stock_documents,
)
from src.rag.vectorstore import create_vectorstore
from src.rag.retriever import HybridRetriever
from src.llm.tools import set_retriever

logger = logging.getLogger(__name__)

# DB 경로는 config.DB_PATH 단일 출처 사용(ETF_DATA_DIR 볼륨 반영). 과거 하드코딩은
# config·_schema와 어긋날 수 있어 통일.
from config import DB_PATH as _DB_PATH  # noqa: E402


@dataclass
class AppState:
    """앱 초기화 상태. /health가 읽고, /chat·/stream이 ready를 가드로 사용."""
    ready: bool = False
    error: Optional[str] = None


def require_ready(request: Request) -> None:
    """라우터용 공유 가드 의존성. ready 아니면 503.

    라우터는 app 참조가 없으므로 Request에서 app.state를 꺼낸다.
    (main.py의 _require_ready와 동일 동작 — /chat·/stream은 그대로 둠.)
    """
    state: AppState = request.app.state.app_state
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail=f"초기화 중/실패: {state.error or 'initializing'}",
        )


def run_init() -> None:
    """동기·블로킹 초기화 — uvicorn lifespan에서 run_in_threadpool로 호출한다.

    실패 시 예외를 그대로 raise (lifespan이 잡아 AppState.error에 기록).
    """
    # 0) OpenAI 키 선검증 (Streamlit secrets 없이 .env/환경변수만)
    get_api_key(streamlit_secrets=None)

    # 1) GitHub Release에서 SQLite DB 다운로드 (이미 있으면 ensure_db가 스킵)
    ensure_db(_DB_PATH)

    # 2) 데이터 로드 + 문서 생성 (app.py load_all_data와 동일, 주식은 graceful fallback)
    etf_data = load_etf_data()
    logger.info(f"ETF 데이터 로드: {len(etf_data)}종목")
    try:
        stock_data = load_stock_data()
        logger.info(f"주식 데이터 로드 성공: {len(stock_data)}종목")
    except Exception as e:
        logger.error(f"주식 데이터 로드 실패: {e}", exc_info=True)
        stock_data = []

    documents = create_documents(etf_data)
    stock_documents = create_stock_documents(stock_data) if stock_data else []

    # 3) 검색 인덱스 구축 (app.py build_retrievers와 동일)
    vectorstore = create_vectorstore(documents, prefix="etf")
    etf_retriever = HybridRetriever(vectorstore, documents)

    stock_retriever = None
    if stock_documents:
        try:
            stock_vectorstore = create_vectorstore(stock_documents, prefix="stock")
            stock_retriever = HybridRetriever(stock_vectorstore, stock_documents)
            logger.info(f"주식 retriever 초기화 성공: {len(stock_documents)}개 문서")
        except Exception as e:
            logger.error(f"주식 retriever 초기화 실패: {e}", exc_info=True)

    # 4) 에이전트 도구에 주입 (프로세스 전역 상태)
    set_retriever(etf_retriever, documents, stock_retriever=stock_retriever,
                  etf_data=etf_data, stock_data=stock_data)
    logger.info("FastAPI 백엔드 초기화 완료")
