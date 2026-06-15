"""웹 푸시 구독/발송 테스트 (Phase F 푸시 A).

test_user_data.py와 동일한 StaticPool 인메모리 sqlite 패턴.
"""

import os

os.environ["API_SKIP_INIT"] = "1"
os.environ["DATABASE_URL"] = "sqlite://"

from unittest.mock import patch  # noqa: E402

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
    r = client.post("/auth/signup", json={"email": "p@b.com", "password": "pw123456"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


SUB = {
    "endpoint": "https://push.example.com/abc",
    "keys": {"p256dh": "BPp2...", "auth": "xyz"},
}


# ── VAPID 공개키 (인증 불필요) ────────────────────────────
def test_vapid_public_key_disabled(client):
    with patch("config.VAPID", {"public_key": "", "private_key": "",
                                "subject": "mailto:x", "enabled": False}):
        r = client.get("/push/vapid-public-key")
    assert r.status_code == 200
    assert r.json() == {"public_key": "", "enabled": False}


def test_vapid_public_key_enabled(client):
    with patch("config.VAPID", {"public_key": "PUBKEY", "private_key": "p",
                                "subject": "mailto:x", "enabled": True}):
        r = client.get("/push/vapid-public-key")
    assert r.json()["public_key"] == "PUBKEY"
    assert r.json()["enabled"] is True


# ── 구독/해제 ─────────────────────────────────────────────
def test_subscribe_requires_auth(client):
    assert client.put("/push/subscribe", json=SUB).status_code == 401


def test_subscribe_and_unsubscribe(client, auth):
    r = client.put("/push/subscribe", json=SUB, headers=auth)
    assert r.status_code == 200 and r.json()["ok"] is True
    # 멱등 — 같은 endpoint 재구독 OK (중복 행 없음)
    r2 = client.put("/push/subscribe", json=SUB, headers=auth)
    assert r2.status_code == 200

    from api.models_db import PushSubscription
    s = db.SessionLocal()
    try:
        assert s.query(PushSubscription).count() == 1
    finally:
        s.close()

    r3 = client.post("/push/unsubscribe",
                     json={"endpoint": SUB["endpoint"]}, headers=auth)
    assert r3.status_code == 200
    s = db.SessionLocal()
    try:
        assert s.query(PushSubscription).count() == 0
    finally:
        s.close()


# ── 테스트 발송 ───────────────────────────────────────────
def test_send_test_503_when_disabled(client, auth):
    client.put("/push/subscribe", json=SUB, headers=auth)
    with patch("config.VAPID", {"enabled": False}):
        r = client.post("/push/test", headers=auth)
    assert r.status_code == 503


def test_send_test_calls_webpush(client, auth):
    client.put("/push/subscribe", json=SUB, headers=auth)
    with patch("config.VAPID", {"enabled": True, "private_key": "pk",
                                "public_key": "pub", "subject": "mailto:x"}), \
         patch("api.push.send_push", return_value=True) as sp:
        r = client.post("/push/test", headers=auth)
    assert r.status_code == 200
    assert r.json()["ok"] is True
    sp.assert_called_once()


# ── 발송 헬퍼: 만료 구독 삭제 ─────────────────────────────
def test_send_push_to_user_removes_gone_subscription(client, auth):
    client.put("/push/subscribe", json=SUB, headers=auth)
    from api import push
    from api.models_db import PushSubscription, User

    s = db.SessionLocal()
    try:
        uid = s.query(User).first().id

        def _gone(info, payload):
            raise push._SubscriptionGone()

        with patch("api.push.send_push", side_effect=_gone):
            sent = push.send_push_to_user(s, uid, {"title": "t", "body": "b"})
        assert sent == 0
        assert s.query(PushSubscription).count() == 0  # 만료 → 삭제
    finally:
        s.close()
