"""
운영용 cron 엔드포인트 — 재부팅 없이 DB를 최신 Release DB로 교체.

배경: 프로덕션 백엔드는 부팅 시(run_init)에만 ensure_db로 DB를 받는다.
keep-alive ping이 컨테이너를 계속 깨워두면 재부팅이 안 일어나 볼륨의 DB가
며칠씩 stale해진다(신선도 검사가 있어도 트리거될 기회=재부팅이 없음).

이 엔드포인트는 daily-collect가 Release DB를 갱신한 직후 X-Cron-Token으로
호출되어, 재부팅 없이 최신 DB를 볼륨에 내려받고 메모리 상태를 재초기화한다.
push/paper의 cron 엔드포인트와 동일한 인라인 토큰 검증 패턴을 따른다.
"""

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.deps import run_init, verify_cron_token, _DB_PATH
from src.data.db_downloader import ensure_db
from src.data import technical

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# 동시에 두 refresh가 진입해 DB 파일을 지우는 중 재초기화가 겹치는 것을 막는다.
# 단일 워커 전제라도 daily-collect 재시도 등으로 요청이 겹칠 수 있다.
_refresh_lock = threading.Lock()


class RefreshDbResponse(BaseModel):
    ok: bool
    refreshed: bool  # 실제로 DB를 내려받아 교체했는지
    detail: str


def _purge_db_files() -> None:
    """DB 본체 + WAL/SHM 사이드카를 삭제. ensure_db가 이후 새 파일을 받게 한다.

    ensure_db 내부 unlink는 .db 본체만 지운다. WAL 모드라 잔여 -wal/-shm이
    새 DB와 섞이면 손상으로 이어질 수 있어 사이드카까지 명시적으로 정리한다.
    """
    for suffix in ("", "-wal", "-shm"):
        p = _DB_PATH.parent / (_DB_PATH.name + suffix)
        p.unlink(missing_ok=True)


@router.post("/refresh-db", response_model=RefreshDbResponse)
def refresh_db(_: None = Depends(verify_cron_token)) -> RefreshDbResponse:
    """볼륨 DB를 최신 Release DB로 강제 교체 후 메모리 상태 재초기화. X-Cron-Token 보호.

    순서: (1) DB 파일+사이드카 삭제 → (2) ensure_db로 최신 Release 재다운로드
    → (3) technical 싱글톤 커넥션/캐시 리셋(삭제된 옛 inode 핸들 방지)
    → (4) run_init 재실행(retriever/인덱스/data_index 재구축, FAISS/BM25는 해시로 자동 재빌드).
    """
    if not _refresh_lock.acquire(blocking=False):
        # 이미 다른 refresh가 진행 중 — 중복 실행 방지
        return RefreshDbResponse(ok=True, refreshed=False, detail="이미 진행 중")

    try:
        logger.info("DB 새로고침 시작 — 볼륨 DB 삭제 후 최신 Release 재다운로드")
        _purge_db_files()

        downloaded = ensure_db(_DB_PATH)
        if not downloaded:
            # 다운로드 실패 — DB가 사라진 상태이므로 다음 부팅/재시도에서 복구.
            # 여기서 예외를 던져 cron이 실패로 인지하게 한다.
            raise HTTPException(503, "DB 다운로드 실패 — 다음 재시도 필요")

        # 삭제된 옛 파일 핸들을 잡고 있는 technical 싱글톤을 리셋(가장 위험한 stale 지점).
        technical.reset_db_connection()

        # retriever/인덱스/data_index 재구축. ensure_db 재호출은 방금 받은 신선한
        # DB에 대해 통과하므로 재다운로드 없음(idempotent).
        run_init()

        logger.info("DB 새로고침 완료 — 최신 데이터 반영")
        return RefreshDbResponse(ok=True, refreshed=True, detail="최신 DB로 교체 완료")
    finally:
        _refresh_lock.release()
