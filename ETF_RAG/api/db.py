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
    logger.info("사용자 DB 테이블 준비 완료 (%s)", engine.url.drivername)


def get_db() -> Iterator[Session]:
    """요청 스코프 세션 의존성."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
