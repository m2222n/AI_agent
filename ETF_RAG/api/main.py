"""
ETF RAG FastAPI 백엔드 (Phase F-1 골격).

기존 Streamlit 앱(app.py)과 병행. LangGraph 에이전트(run_agent/stream_agent)를
REST + SSE로 노출해 Streamlit 없이도 구동 가능하게 한다.

실행 (repo root에서):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

주의:
    - 단일 워커 전용. set_retriever가 프로세스 전역 상태를 쓰므로 --workers N이면
      워커별 재init이 일어난다 (ensure_db는 파일 존재 시 스킵, FAISS는 캐시되어 비용은 적음).
    - /stream의 token 이벤트는 델타가 아니라 "누적 전체 답변 텍스트"다 (기존 chat.py 계약).
      클라이언트는 append가 아니라 replace로 처리해야 한다.
"""

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from src.llm.agent import run_agent, stream_agent

from api.deps import AppState, run_init
from api.db import init_models
from api.models import ChatRequest, ChatResponse, HealthResponse
from api.tabs import router as tabs_router
from api.auth import router as auth_router
from api.user_data import router as user_data_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 1회 초기화. 실패해도 서버는 떠서 /health가 에러를 보고한다."""
    state: AppState = app.state.app_state
    # 사용자 DB 테이블 보장 (멱등, 싸다) — RAG init/API_SKIP_INIT과 무관하게 항상.
    init_models()
    if os.getenv("API_SKIP_INIT") == "1":
        # 테스트: 실제 DB 다운로드/임베딩 우회 (retriever는 테스트에서 mock 주입)
        state.ready, state.error = True, None
        logger.info("API_SKIP_INIT=1 → 초기화 우회")
    else:
        try:
            await run_in_threadpool(run_init)
            state.ready, state.error = True, None
        except Exception as e:  # noqa: BLE001 — 모든 init 실패를 상태로 보고
            state.ready, state.error = False, f"{type(e).__name__}: {e}"
            logger.error(f"초기화 실패: {state.error}", exc_info=True)
    yield


app = FastAPI(title="ETF RAG API", version="0.1.0", lifespan=lifespan)
app.state.app_state = AppState()

# CORS: CORS_ORIGINS 환경변수(쉼표 구분)로 제어, 미설정 시 "*"(로컬/dev).
# 프로덕션은 프론트 배포 origin을 지정. "*"일 때만 credentials 비활성(브라우저 제약).
_cors_env = os.getenv("CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] or ["*"]
_allow_credentials = _cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5개 데이터 탭 + 자동완성 엔드포인트 (/tabs/*)
app.include_router(tabs_router)
# JWT 이메일 인증 (/auth/*)
app.include_router(auth_router)
# 유저별 저장 — 관심종목/대화이력 (/me/*)
app.include_router(user_data_router)


def _require_ready() -> None:
    state: AppState = app.state.app_state
    if not state.ready:
        raise HTTPException(status_code=503, detail=f"초기화 중/실패: {state.error or 'initializing'}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    state: AppState = app.state.app_state
    return HealthResponse(ready=state.ready, error=state.error)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """비스트리밍 채팅. 동기 run_agent를 threadpool에서 실행해 이벤트 루프 보호."""
    _require_ready()
    history = [m.model_dump() for m in req.chat_history] if req.chat_history else None
    result = await run_in_threadpool(run_agent, req.question, history)
    return ChatResponse(**result)


@app.post("/stream")
async def stream(req: ChatRequest) -> EventSourceResponse:
    """SSE 스트리밍. 동기 제너레이터를 iterate_in_threadpool로 async 변환.

    이벤트는 이름을 열거하지 않고 stream_agent가 내보내는 대로 전부 통과시킨다
    (question_type/tool_call/tool_result/structured_data/token/cov_revision/error/done).
    data가 dict면 JSON 직렬화, str이면 그대로 보낸다.
    """
    _require_ready()
    history = [m.model_dump() for m in req.chat_history] if req.chat_history else None

    async def _source():
        async for ev in iterate_in_threadpool(stream_agent(req.question, history)):
            data = ev["data"]
            if not isinstance(data, str):
                data = json.dumps(data, ensure_ascii=False)
            yield {"event": ev["event"], "data": data}

    return EventSourceResponse(_source())
