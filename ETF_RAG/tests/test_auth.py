"""JWT 이메일 인증 엔드포인트 테스트 (Phase F-1 A).

sqlite :memory:는 커넥션마다 별도 DB → StaticPool로 단일 공유 커넥션.
import 전 API_SKIP_INIT=1 + DATABASE_URL=sqlite://(인메모리).
"""

import os

os.environ["API_SKIP_INIT"] = "1"
os.environ["DATABASE_URL"] = "sqlite://"  # 인메모리; fixture에서 실제 엔진 교체

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.db as db  # noqa: E402
from api.db import Base, get_db  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_sse_global():
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
    yield


@pytest.fixture
def client():
    # 단일 공유 인메모리 커넥션 (StaticPool) — 테이블이 요청 간 유지됨
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    TestSession = sessionmaker(
        bind=test_engine, autoflush=False, expire_on_commit=False
    )
    db.engine = test_engine  # init_models()가 이 엔진에 create_all
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
    with TestClient(app) as c:  # 컨텍스트 매니저 → lifespan(init_models) 실행
        yield c
    app.dependency_overrides.clear()
    Base.metadata.drop_all(test_engine)


def _signup(client, email="a@b.com", pw="pw123456"):
    return client.post("/auth/signup", json={"email": email, "password": pw})


def test_signup_login_me_happy_path(client):
    r = _signup(client)
    assert r.status_code == 201
    token = r.json()["access_token"]
    assert token and r.json()["token_type"] == "bearer"

    # login
    r2 = client.post("/auth/login", json={"email": "a@b.com", "password": "pw123456"})
    assert r2.status_code == 200
    token2 = r2.json()["access_token"]

    # me
    r3 = client.get("/auth/me", headers={"Authorization": f"Bearer {token2}"})
    assert r3.status_code == 200
    body = r3.json()
    assert body["email"] == "a@b.com"
    assert isinstance(body["id"], int)


def test_signup_duplicate_email_400(client):
    assert _signup(client).status_code == 201
    r = _signup(client)  # 같은 이메일
    assert r.status_code == 400


def test_login_wrong_password_401(client):
    _signup(client)
    r = client.post("/auth/login", json={"email": "a@b.com", "password": "wrongpw1"})
    assert r.status_code == 401


def test_login_unknown_email_401(client):
    r = client.post("/auth/login", json={"email": "x@y.com", "password": "pw123456"})
    assert r.status_code == 401


def test_me_without_token_401(client):
    r = client.get("/auth/me")
    assert r.status_code == 401


def test_me_bad_token_401(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_signup_short_password_422(client):
    r = client.post("/auth/signup", json={"email": "a@b.com", "password": "short"})
    assert r.status_code == 422  # min_length=8


def test_signup_invalid_email_422(client):
    r = client.post("/auth/signup", json={"email": "notanemail", "password": "pw123456"})
    assert r.status_code == 422
