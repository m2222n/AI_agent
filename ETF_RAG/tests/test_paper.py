"""가상투자(모의투자) 테스트 (2026-06-23).

test_user_data.py와 동일한 StaticPool 인메모리 sqlite + TestClient 패턴.
현재가 조회(_resolve_price)는 patch로 고정 — 매수/매도/평가손익/잔고/리셋/랭킹 로직만 검증.
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
from api.models_db import INITIAL_CASH  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_sse_global():
    from sse_starlette.sse import AppStatus
    AppStatus.should_exit_event = None
    yield


@pytest.fixture
def client():
    test_engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool, future=True,
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


def _auth(client, email="u@b.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "pw123456"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# 고정 현재가 mock — _resolve_price를 종목별 가격 dict로 대체
def _price_patch(prices):
    """prices: {ticker_or_name: price}. _resolve_price를 가로채 고정값 반환."""
    def fake(ticker):
        if ticker not in prices:
            from fastapi import HTTPException
            raise HTTPException(404, "not found")
        return {"ticker": ticker, "name": f"종목{ticker}", "price": prices[ticker],
                "source": "close"}
    return patch("api.paper._resolve_price", side_effect=fake)


def test_portfolio_auto_creates_1eok(client):
    auth = _auth(client)
    r = client.get("/me/paper/portfolio", headers=auth)
    assert r.status_code == 200
    b = r.json()
    assert b["cash"] == INITIAL_CASH
    assert b["total_value"] == INITIAL_CASH
    assert b["holdings"] == []
    assert b["total_pnl"] == 0


def test_buy_decreases_cash_and_adds_holding(client):
    auth = _auth(client)
    with _price_patch({"005930": 70000}):
        r = client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10}, headers=auth)
    assert r.status_code == 200
    b = r.json()
    assert b["side"] == "buy" and b["amount"] == 700000
    assert b["cash"] == INITIAL_CASH - 700000
    # 포트폴리오에 반영
    with _price_patch({"005930": 70000}):
        pf = client.get("/me/paper/portfolio", headers=auth).json()
    assert len(pf["holdings"]) == 1
    h = pf["holdings"][0]
    assert h["qty"] == 10 and h["avg_price"] == 70000 and h["pnl"] == 0


def test_buy_insufficient_cash_400(client):
    auth = _auth(client)
    with _price_patch({"005930": 70000}):
        # 1억으로 70000원짜리 2000주(=1.4억) 매수 → 잔고부족
        r = client.post("/me/paper/buy", json={"ticker": "005930", "qty": 2000}, headers=auth)
    assert r.status_code == 400
    assert "잔고 부족" in r.json()["detail"]


def test_avg_price_weighted_on_additional_buy(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10}, headers=auth)
    with _price_patch({"005930": 200}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10}, headers=auth)
        pf = client.get("/me/paper/portfolio", headers=auth).json()
    # 평단가 = (100*10 + 200*10)/20 = 150
    assert pf["holdings"][0]["avg_price"] == 150
    assert pf["holdings"][0]["qty"] == 20


def test_eval_pnl_reflects_price_change(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 100}, headers=auth)
    # 가격이 100 → 150으로 상승 → 평가손익 +5000 (50%)
    with _price_patch({"005930": 150}):
        pf = client.get("/me/paper/portfolio", headers=auth).json()
    h = pf["holdings"][0]
    assert h["current_price"] == 150
    assert h["pnl"] == 5000
    assert h["pnl_pct"] == 50.0
    assert pf["total_pnl"] == 5000  # 현금 -10000 + 평가 15000 = +5000


def test_sell_realizes_pnl_and_restores_cash(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 100}, headers=auth)
    with _price_patch({"005930": 130}):
        r = client.post("/me/paper/sell", json={"ticker": "005930", "qty": 50}, headers=auth)
    b = r.json()
    assert b["side"] == "sell" and b["qty"] == 50
    # 실현손익 = (130-100)*50 = 1500
    assert b["realized_pnl"] == 1500
    # 현금 = 1억 - 10000(매수) + 6500(매도 50*130)
    assert b["cash"] == INITIAL_CASH - 10000 + 6500


def test_sell_more_than_held_400(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10}, headers=auth)
        r = client.post("/me/paper/sell", json={"ticker": "005930", "qty": 20}, headers=auth)
    assert r.status_code == 400
    assert "보유 수량 부족" in r.json()["detail"]


def test_sell_all_removes_holding(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10}, headers=auth)
        client.post("/me/paper/sell", json={"ticker": "005930", "qty": 10}, headers=auth)
        pf = client.get("/me/paper/portfolio", headers=auth).json()
    assert pf["holdings"] == []


def test_trades_history_records_buy_and_sell(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10}, headers=auth)
        client.post("/me/paper/sell", json={"ticker": "005930", "qty": 5}, headers=auth)
    t = client.get("/me/paper/trades", headers=auth).json()["trades"]
    assert len(t) == 2
    sides = {x["side"] for x in t}
    assert sides == {"buy", "sell"}


def test_reset_restores_initial(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10}, headers=auth)
    r = client.post("/me/paper/reset", headers=auth)
    b = r.json()
    assert b["cash"] == INITIAL_CASH
    assert b["holdings"] == []
    assert client.get("/me/paper/trades", headers=auth).json()["trades"] == []


def test_requires_auth_401(client):
    assert client.get("/me/paper/portfolio").status_code == 401
    assert client.post("/me/paper/buy", json={"ticker": "005930", "qty": 1}).status_code == 401


def test_ranking_orders_by_total_value(client):
    a = _auth(client, "a@b.com")
    b = _auth(client, "b@b.com")
    # b: 가상투자 탭 진입(포트폴리오 조회) → 계좌 생성(현금만)
    client.get("/me/paper/portfolio", headers=b)
    # a: 가격 오른 종목 보유 → 이득
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 100}, headers=a)
    with _price_patch({"005930": 500}):  # a의 평가액 급등
        rk = client.get("/me/paper/ranking", headers=a).json()
    assert rk["total_players"] == 2
    assert rk["rankings"][0]["is_me"] is True  # a가 1등
    assert rk["my_rank"] == 1


def test_ranking_excludes_users_without_account(client):
    """가상투자를 한 번도 안 한 유저(계좌 미생성)는 랭킹에 미포함."""
    a = _auth(client, "a@b.com")
    _auth(client, "b@b.com")  # b는 계좌 안 만듦
    client.get("/me/paper/portfolio", headers=a)  # a만 계좌 생성
    rk = client.get("/me/paper/ranking", headers=a).json()
    assert rk["total_players"] == 1
