"""웹 푸시 구독/발송 테스트 (Phase F 푸시 A).

test_user_data.py와 동일한 StaticPool 인메모리 sqlite 패턴.
"""

import os

os.environ["API_SKIP_INIT"] = "1"
os.environ["DATABASE_URL"] = "sqlite://"

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

import api.db as db  # noqa: E402 — 테스트 본문이 db.SessionLocal()로 직접 세팅

# client / _reset_sse_global 픽스처는 tests/conftest.py에 공통 정의됨.


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


# ── 관심종목 일일 알림 ────────────────────────────────────
def _seed_watchlist(uid, *tickers):
    from api.models_db import Watchlist
    s = db.SessionLocal()
    try:
        for t in tickers:
            s.add(Watchlist(user_id=uid, ticker=t))
        s.commit()
    finally:
        s.close()


def test_run_alerts_403_without_token(client):
    with patch("config.CRON_TOKEN", "secret"):
        r = client.post("/push/run-watchlist-alerts")
    assert r.status_code == 403


def test_run_alerts_403_disabled_when_no_cron_token(client):
    with patch("config.CRON_TOKEN", ""):
        r = client.post("/push/run-watchlist-alerts",
                        headers={"X-Cron-Token": "anything"})
    assert r.status_code == 403


def test_run_alerts_sends_for_movers(client, auth):
    # 구독 + 관심종목 2개 등록
    client.put("/push/subscribe", json=SUB, headers=auth)
    from api.models_db import User
    s = db.SessionLocal()
    try:
        uid = s.query(User).first().id
    finally:
        s.close()
    _seed_watchlist(uid, "005930", "000660")

    # 005930 +6.2%(급등), 000660 -1.0%(임계 미만)
    def _fsd(t):
        return {
            "005930": {"name": "삼성전자", "change_pct": 6.2, "close": 70000},
            "000660": {"name": "SK하이닉스", "change_pct": -1.0, "close": 100000},
        }.get(t)

    with patch("config.CRON_TOKEN", "secret"), \
         patch("config.WATCHLIST_ALERT_THRESHOLD", 5.0), \
         patch("src.llm.tools._find_structured_data", side_effect=_fsd), \
         patch("api.push.send_push", return_value=True) as sp:
        r = client.post("/push/run-watchlist-alerts",
                        headers={"X-Cron-Token": "secret"})
    assert r.status_code == 200
    b = r.json()
    assert b["users_notified"] == 1
    assert b["pushes_sent"] == 1
    assert b["movers"] == 1  # 005930만 임계 초과
    # 발송 payload에 삼성전자 포함
    payload = sp.call_args[0][1]
    assert "삼성전자" in payload["body"]


def test_run_alerts_no_movers_no_send(client, auth):
    client.put("/push/subscribe", json=SUB, headers=auth)
    from api.models_db import User
    s = db.SessionLocal()
    try:
        uid = s.query(User).first().id
    finally:
        s.close()
    _seed_watchlist(uid, "005930")

    with patch("config.CRON_TOKEN", "secret"), \
         patch("config.WATCHLIST_ALERT_THRESHOLD", 5.0), \
         patch("src.llm.tools._find_structured_data",
               return_value={"name": "삼성전자", "change_pct": 1.0}), \
         patch("api.push.send_push", return_value=True) as sp:
        r = client.post("/push/run-watchlist-alerts",
                        headers={"X-Cron-Token": "secret"})
    assert r.json()["users_notified"] == 0
    sp.assert_not_called()
