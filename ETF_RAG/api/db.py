"""사용자 DB (인증/관심종목/대화이력) — 동기 SQLAlchemy 2.0.

주가용 stock DB(src/data/etf_rag.db, raw sqlite3, read-only)와 완전히 분리된 엔진.
postgresql://(prod, psycopg2)와 sqlite:///(dev/test) 모두 동일 Engine/Session API.

FastAPI는 non-async(def) 핸들러를 threadpool에서 실행하므로 blocking Session이
이벤트 루프를 막지 않는다 → 코드베이스의 sync 패턴과 일관.
"""

import logging
import os
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL

logger = logging.getLogger(__name__)

# alembic.ini는 프로젝트 루트(api/의 부모). 부팅 시 프로그램적으로 마이그레이션 실행.
_ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"

# sqlite는 커넥션 생성 스레드에서만 사용 가능 — FastAPI threadpool(다중 스레드) 위해 해제.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_models() -> None:
    """사용자 DB 스키마 준비. 멱등 — API_SKIP_INIT과 무관하게 매 부팅 호출.

    테스트(API_SKIP_INIT=1)는 Alembic을 타지 않고 create_all만 한다(픽스처가
    인메모리 엔진에 이미 create_all 하므로 무해·빠름, 894 테스트 경로 불변).
    프로덕션은 Alembic으로 마이그레이션(run_migrations). 어느 경로든 실패 시
    create_all fallback으로 최소한 테이블은 보장(가용성 우선, 현행 철학).
    """
    from api import models_db  # noqa: F401 — 모델 등록(임포트 부수효과)

    if os.getenv("API_SKIP_INIT") == "1":
        Base.metadata.create_all(engine)
        logger.info("사용자 DB 테이블 준비 완료 (create_all, 테스트 경로)")
        return

    try:
        run_migrations()
        logger.info("사용자 DB 마이그레이션 완료 (%s)", engine.url.drivername)
    except Exception:  # noqa: BLE001 — 마이그레이션 실패해도 테이블은 보장
        logger.error("Alembic 마이그레이션 실패 — create_all fallback", exc_info=True)
        Base.metadata.create_all(engine)


def run_migrations() -> None:
    """Alembic으로 스키마를 head까지 올린다. 재바인딩된 db.engine을 그대로 사용.

    부팅 시점의 DB 상태에 따라 분기:
    - alembic_version 존재 → upgrade head (정상 경로, 이후 매 배포)
    - users만 존재(기존 라이브 DB) → stamp head (DDL 0건, 버전만 각인 — 최초 1회)
    - 빈 DB → upgrade head (처음부터 전체 스키마 생성, 신규 환경)

    Alembic이 자체 엔진을 만들지 않도록 config.attributes["connection"]에 현재
    엔진 커넥션을 주입한다(테스트의 StaticPool 인메모리 엔진과 연결 유지 목적).
    """
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", str(engine.url))

    insp = inspect(engine)
    tables = set(insp.get_table_names())

    with engine.connect() as conn:
        cfg.attributes["connection"] = conn
        if "alembic_version" in tables:
            command.upgrade(cfg, "head")
        elif "users" in tables:
            # 기존 라이브 DB: 스키마는 이미 최신, 버전만 각인(마이그레이션 SQL 미실행).
            command.stamp(cfg, "head")
            logger.info("기존 DB 감지 — alembic stamp head (DDL 없음)")
        else:
            command.upgrade(cfg, "head")


def get_db() -> Iterator[Session]:
    """요청 스코프 세션 의존성."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
