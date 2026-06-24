"""사용자 DB (인증/관심종목/대화이력) — 동기 SQLAlchemy 2.0.

주가용 stock DB(src/data/etf_rag.db, raw sqlite3, read-only)와 완전히 분리된 엔진.
postgresql://(prod, psycopg2)와 sqlite:///(dev/test) 모두 동일 Engine/Session API.

FastAPI는 non-async(def) 핸들러를 threadpool에서 실행하므로 blocking Session이
이벤트 루프를 막지 않는다 → 코드베이스의 sync 패턴과 일관.
"""

import logging
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL

logger = logging.getLogger(__name__)

# sqlite는 커넥션 생성 스레드에서만 사용 가능 — FastAPI threadpool(다중 스레드) 위해 해제.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_models() -> None:
    """테이블이 없으면 생성. 멱등 — API_SKIP_INIT과 무관하게 매 부팅 호출(테이블만, 싸다)."""
    from api import models_db  # noqa: F401 — 모델 등록(임포트 부수효과)
    Base.metadata.create_all(engine)
    _migrate_add_columns()
    logger.info("사용자 DB 테이블 준비 완료 (%s)", engine.url.drivername)


def _migrate_add_columns() -> None:
    """create_all은 기존 테이블에 신규 컬럼을 추가하지 못함(Alembic 미도입).
    이미 users 테이블이 있는 환경(기존 가입자 보유 DB)을 위한 경량 멱등 마이그레이션.
    sqlite/postgres 모두 ADD COLUMN IF NOT EXISTS를 지원하지 않는 경우가 있어
    inspector로 존재 여부를 먼저 확인한다."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if "users" not in insp.get_table_names():
        return  # create_all이 방금 최신 스키마로 생성 → 보정 불필요
    cols = {c["name"] for c in insp.get_columns("users")}
    if "nickname" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN nickname VARCHAR(40)"))
        logger.info("마이그레이션: users.nickname 컬럼 추가")

    # paper_accounts.started_at (라운드 결산 기간 산정용). 기존 행은 created_at으로 채움.
    if "paper_accounts" in insp.get_table_names():
        pa_cols = {c["name"] for c in insp.get_columns("paper_accounts")}
        if "started_at" not in pa_cols:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE paper_accounts ADD COLUMN started_at TIMESTAMP"
                ))
                conn.execute(text(
                    "UPDATE paper_accounts SET started_at = created_at "
                    "WHERE started_at IS NULL"
                ))
            logger.info("마이그레이션: paper_accounts.started_at 컬럼 추가")


def get_db() -> Iterator[Session]:
    """요청 스코프 세션 의존성."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
