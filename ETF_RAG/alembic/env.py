"""Alembic 마이그레이션 환경 (동기 엔진).

두 가지 실행 경로를 지원한다:
1. CLI (`alembic upgrade head` / `revision --autogenerate` / `check`) — config.DATABASE_URL로
   자체 엔진을 만든다. 로컬 개발·CI에서 사용.
2. 프로그램 실행 (api/db.py:run_migrations) — 이미 만들어진 엔진/커넥션을
   `config.attributes["connection"]`로 주입받아 그대로 쓴다. 프로덕션 부팅 시 db.engine을
   재사용해, 테스트의 StaticPool 인메모리 DB 같은 특수 엔진과 연결이 끊기지 않게 한다.

target_metadata는 api.models_db가 등록한 Base.metadata. sqlite는 ALTER 제약이 있어
batch mode(render_as_batch)를 켠다(postgres는 불필요).
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# alembic.ini 로거 설정
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 모델 메타데이터 등록 (api.models_db import 부수효과로 Base에 테이블 바인딩)
from config import DATABASE_URL  # noqa: E402
from api.db import Base  # noqa: E402
import api.models_db  # noqa: E402,F401 — 모델 등록

target_metadata = Base.metadata

# CLI 실행 시 ini의 플레이스홀더 URL을 실제 DATABASE_URL로 교체.
config.set_main_option("sqlalchemy.url", DATABASE_URL)

_is_sqlite = DATABASE_URL.startswith("sqlite")


def run_migrations_offline() -> None:
    """오프라인(--sql) 모드 — URL만으로 SQL 스크립트 생성."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=_is_sqlite,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """온라인 모드 — 실제 커넥션 위에서 실행.

    run_migrations가 주입한 커넥션이 있으면 재사용(그 엔진의 DB에 적용),
    없으면 config URL로 엔진을 새로 만든다(CLI 경로).
    """
    connectable = config.attributes.get("connection", None)

    if connectable is not None:
        _do_run(connectable)
    else:
        engine = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
        with engine.connect() as connection:
            _do_run(connection)


def _do_run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=_is_sqlite,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
