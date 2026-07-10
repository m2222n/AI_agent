import os
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch
from src.data.loader import load_etf_data, create_documents


# ── API 테스트 공용 픽스처 ─────────────────────────────────────────
# _reset_sse_global(7개 파일 복붙)·client(StaticPool 인메모리 DB, 4개 파일 복붙)를
# 여기로 통합. 각 test 파일 상단의 os.environ["API_SKIP_INIT"]="1" +
# DATABASE_URL="sqlite://" 설정은 import 타이밍 때문에 파일에 남긴다(여기로 옮기면
# 각 파일의 top-level import보다 늦게 실행될 수 있음). auth 헬퍼는 파일마다
# 이름·이메일·시그니처가 달라 공통화하지 않는다.


@pytest.fixture(autouse=True)
def _reset_sse_global():
    """sse-starlette 전역 이벤트를 테스트마다 초기화(테스트 간 누수 방지)."""
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
    yield


@pytest.fixture
def client():
    """StaticPool 단일 공유 인메모리 sqlite로 앱 client 구성.

    db.engine/SessionLocal을 테스트 엔진으로 교체 + get_db 오버라이드.
    TestClient 컨텍스트 진입 시 lifespan(init_models)이 이 엔진에 스키마 생성.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient
    import api.db as db
    from api.db import Base, get_db

    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    TestSession = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)
    db.engine = test_engine
    db.SessionLocal = TestSession
    Base.metadata.create_all(test_engine)

    from api.main import app

    def _override_db():
        s = TestSession()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    Base.metadata.drop_all(test_engine)


@pytest.fixture(scope="session")
def etf_data():
    """하드코딩 샘플 데이터 로드 (기존 테스트 호환용)"""
    from pathlib import Path
    fake_db = Path("/tmp/nonexistent.db")
    with patch("src.data.loader.DB_PATH", fake_db), \
         patch("src.data.loader.get_latest_collected_path", return_value=None):
        return load_etf_data()


@pytest.fixture(scope="session")
def documents(etf_data):
    return create_documents(etf_data)
