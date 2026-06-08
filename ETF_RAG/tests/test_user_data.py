"""유저별 저장(관심종목/대화이력) CRUD 테스트 (Phase F-1 B).

test_auth.py와 동일한 StaticPool 인메모리 sqlite 패턴.
"""

import os

os.environ["API_SKIP_INIT"] = "1"
os.environ["DATABASE_URL"] = "sqlite://"

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


@pytest.fixture
def auth(client):
    """가입 후 Authorization 헤더 반환."""
    r = client.post("/auth/signup", json={"email": "u@b.com", "password": "pw123456"})
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ── 관심종목 ────────────────────────────────────────────
def test_watchlist_add_list_remove(client, auth):
    assert client.get("/me/watchlist", headers=auth).json()["tickers"] == []

    r = client.put("/me/watchlist/005930", headers=auth)
    assert r.status_code == 200
    assert r.json()["tickers"] == ["005930"]

    client.put("/me/watchlist/000660", headers=auth)
    assert set(client.get("/me/watchlist", headers=auth).json()["tickers"]) == {
        "005930", "000660"
    }

    # 중복 추가 멱등
    client.put("/me/watchlist/005930", headers=auth)
    assert len(client.get("/me/watchlist", headers=auth).json()["tickers"]) == 2

    r = client.delete("/me/watchlist/005930", headers=auth)
    assert r.json()["tickers"] == ["000660"]


def test_watchlist_requires_auth(client):
    assert client.get("/me/watchlist").status_code == 401
    assert client.put("/me/watchlist/005930").status_code == 401


def test_watchlist_isolated_per_user(client):
    t1 = client.post("/auth/signup", json={"email": "a@b.com", "password": "pw123456"}).json()["access_token"]
    t2 = client.post("/auth/signup", json={"email": "b@b.com", "password": "pw123456"}).json()["access_token"]
    client.put("/me/watchlist/005930", headers={"Authorization": f"Bearer {t1}"})
    # user2는 비어있어야
    r = client.get("/me/watchlist", headers={"Authorization": f"Bearer {t2}"})
    assert r.json()["tickers"] == []


# ── 대화 이력 ──────────────────────────────────────────
def test_history_append_list_clear(client, auth):
    assert client.get("/me/history", headers=auth).json()["messages"] == []

    payload = {"messages": [
        {"role": "user", "content": "삼성전자 어때?"},
        {"role": "assistant", "content": "PER 49.81배입니다.", "question_type": "simple", "model": "gpt-4o-mini"},
    ]}
    r = client.post("/me/history", headers=auth, json=payload)
    assert r.status_code == 201
    msgs = r.json()["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["question_type"] == "simple"

    # 순서 보존 + 누적
    client.post("/me/history", headers=auth, json={"messages": [{"role": "user", "content": "다음은?"}]})
    got = client.get("/me/history", headers=auth).json()["messages"]
    assert len(got) == 3
    assert got[-1]["content"] == "다음은?"

    assert client.delete("/me/history", headers=auth).status_code == 204
    assert client.get("/me/history", headers=auth).json()["messages"] == []


def test_history_requires_auth(client):
    assert client.get("/me/history").status_code == 401


def test_history_empty_payload_422(client, auth):
    r = client.post("/me/history", headers=auth, json={"messages": []})
    assert r.status_code == 422  # min_length=1
