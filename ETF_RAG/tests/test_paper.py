"""가상투자(모의투자) 테스트 (2026-06-23).

test_user_data.py와 동일한 StaticPool 인메모리 sqlite + TestClient 패턴.
현재가 조회(_resolve_price)는 patch로 고정 — 매수/매도/평가손익/잔고/리셋/랭킹 로직만 검증.
"""

import os

os.environ["API_SKIP_INIT"] = "1"
os.environ["DATABASE_URL"] = "sqlite://"

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from api.models_db import INITIAL_CASH  # noqa: E402

# client / _reset_sse_global 픽스처는 tests/conftest.py에 공통 정의됨.


def _auth(client, email="u@b.com"):
    r = client.post("/auth/signup", json={"email": email, "password": "pw123456", "gender": "선택안함"})
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
        r = client.post("/me/paper/reset", json={"confirm": "초기화"}, headers=auth)
    b = r.json()
    assert b["cash"] == INITIAL_CASH
    assert b["holdings"] == []
    assert client.get("/me/paper/trades", headers=auth).json()["trades"] == []


def test_reset_wrong_confirm_400(client):
    auth = _auth(client)
    r = client.post("/me/paper/reset", json={"confirm": "ok"}, headers=auth)
    assert r.status_code == 400


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


# ── 수익률 추이 스냅샷 ──────────────────────────────────
def test_history_empty_no_chart(client):
    auth = _auth(client)
    client.get("/me/paper/portfolio", headers=auth)  # 계좌 생성(스냅샷 없음)
    h = client.get("/me/paper/history", headers=auth).json()
    assert h["points"] == []
    assert h["chart_b64"] is None


def test_buy_records_snapshot(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10}, headers=auth)
        h = client.get("/me/paper/history", headers=auth).json()
    # 거래로 당일 스냅샷 1개 생성 — 총자산은 1억 그대로(현금-1000 + 평가1000)
    assert len(h["points"]) == 1
    assert h["points"][0]["total_value"] == INITIAL_CASH
    assert h["points"][0]["pnl_pct"] == 0.0


def test_reset_clears_snapshots(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10}, headers=auth)
        client.post("/me/paper/reset", json={"confirm": "초기화"}, headers=auth)
    h = client.get("/me/paper/history", headers=auth).json()
    # 리셋 후 1억 스냅샷 1개만(초기화 시점)
    assert len(h["points"]) == 1
    assert h["points"][0]["total_value"] == INITIAL_CASH


# ── 라운드 결산 (기록 보존) ──────────────────────────────
def test_reset_records_round_with_symbol_pnl(client):
    auth = _auth(client)
    # 005930 100주 @100 매수 → 50주 @130 매도(실현 +1500) → 가격 150에서 초기화(미실현)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 100}, headers=auth)
    with _price_patch({"005930": 130}):
        client.post("/me/paper/sell", json={"ticker": "005930", "qty": 50}, headers=auth)
    with _price_patch({"005930": 150}):
        client.post("/me/paper/reset", json={"confirm": "초기화"}, headers=auth)
    rounds = client.get("/me/paper/rounds", headers=auth).json()["rounds"]
    assert len(rounds) == 1
    rd = rounds[0]
    assert rd["round_no"] == 1
    assert rd["trade_count"] == 2
    syms = rd["symbols"]
    assert len(syms) == 1
    s = syms[0]
    assert s["ticker"] == "005930"
    # 실현: (130-100)*50 = 1500 / 미실현: (150-100)*50 = 2500 / total 4000
    assert s["realized"] == 1500
    assert s["unrealized"] == 2500
    assert s["total"] == 4000


def test_reset_no_trades_no_round(client):
    """거래 없이 초기화하면 라운드 기록을 남기지 않는다."""
    auth = _auth(client)
    client.get("/me/paper/portfolio", headers=auth)  # 계좌 생성만
    client.post("/me/paper/reset", json={"confirm": "초기화"}, headers=auth)
    assert client.get("/me/paper/rounds", headers=auth).json()["rounds"] == []


def test_round_no_increments(client):
    auth = _auth(client)
    for _ in range(2):
        with _price_patch({"005930": 100}):
            client.post("/me/paper/buy", json={"ticker": "005930", "qty": 1}, headers=auth)
            client.post("/me/paper/reset", json={"confirm": "초기화"}, headers=auth)
    rounds = client.get("/me/paper/rounds", headers=auth).json()["rounds"]
    assert [r["round_no"] for r in rounds] == [2, 1]  # 최신순


def test_snapshot_all_requires_cron_token(client):
    from unittest.mock import patch
    # 토큰 미설정 → 403
    with patch("config.CRON_TOKEN", ""):
        assert client.post("/me/paper/snapshot-all").status_code == 403
    # 잘못된 토큰 → 403
    with patch("config.CRON_TOKEN", "secret"):
        r = client.post("/me/paper/snapshot-all", headers={"X-Cron-Token": "wrong"})
        assert r.status_code == 403


def test_snapshot_all_records_all_accounts(client):
    from unittest.mock import patch
    a = _auth(client, "a@b.com")
    b = _auth(client, "b@b.com")
    client.get("/me/paper/portfolio", headers=a)  # 계좌 생성
    client.get("/me/paper/portfolio", headers=b)
    with patch("config.CRON_TOKEN", "secret"):
        r = client.post("/me/paper/snapshot-all", headers={"X-Cron-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["users_snapshotted"] == 2
    # 각 유저 history에 스냅샷 1개(현금만이라 1억)
    h = client.get("/me/paper/history", headers=a).json()
    assert len(h["points"]) == 1 and h["points"][0]["total_value"] == INITIAL_CASH


def test_stats_empty_account(client):
    auth = _auth(client)
    client.get("/me/paper/portfolio", headers=auth)  # 계좌 생성
    s = client.get("/me/paper/stats", headers=auth).json()
    assert s["total_trades"] == 0 and s["sell_count"] == 0
    assert s["win_rate"] == 0.0 and s["realized_pnl"] == 0
    assert s["profit_factor"] is None and s["best_trade"] is None


def test_stats_win_loss_and_winrate(client):
    auth = _auth(client)
    # 005930 이익 실현(+1500), 000660 손실 실현(-1000)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 100}, headers=auth)
    with _price_patch({"000660": 200}):
        client.post("/me/paper/buy", json={"ticker": "000660", "qty": 100}, headers=auth)
    with _price_patch({"005930": 130}):
        client.post("/me/paper/sell", json={"ticker": "005930", "qty": 50}, headers=auth)  # +1500
    with _price_patch({"000660": 180}):
        client.post("/me/paper/sell", json={"ticker": "000660", "qty": 50}, headers=auth)  # -1000
    s = client.get("/me/paper/stats", headers=auth).json()
    assert s["buy_count"] == 2 and s["sell_count"] == 2
    assert s["win_count"] == 1 and s["loss_count"] == 1
    assert s["win_rate"] == 50.0
    assert s["realized_pnl"] == 500          # +1500 - 1000
    assert s["avg_win"] == 1500 and s["avg_loss"] == -1000
    assert s["best_trade"] == 1500 and s["worst_trade"] == -1000
    assert s["profit_factor"] == 1.5         # 1500 / 1000


def test_dividend_pays_dps_times_qty(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 100}, headers=auth)
    # 보유 100주 × DPS 30 = 3000원 지급
    with _price_patch({"005930": 100}), \
         patch("api.paper._holding_dps", return_value={"005930": 30.0}):
        r = client.post("/me/paper/dividend", headers=auth)
    b = r.json()
    assert b["ok"] and b["paid"] is True
    assert b["total"] == 3000
    assert b["cash"] == INITIAL_CASH - 10000 + 3000
    assert len(b["items"]) == 1 and b["items"][0]["amount"] == 3000


def test_dividend_once_per_round(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 100}, headers=auth)
    with _price_patch({"005930": 100}), \
         patch("api.paper._holding_dps", return_value={"005930": 30.0}):
        client.post("/me/paper/dividend", headers=auth)  # 1차 지급
        r2 = client.post("/me/paper/dividend", headers=auth)  # 2차는 미지급
    b = r2.json()
    assert b["paid"] is False
    # 현금은 1회분만 증가
    pf = client.get("/me/paper/portfolio", headers=auth).json()
    assert pf["cash"] == INITIAL_CASH - 10000 + 3000


def test_dividend_resets_after_round_reset(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 100}, headers=auth)
    with _price_patch({"005930": 100}), \
         patch("api.paper._holding_dps", return_value={"005930": 30.0}):
        client.post("/me/paper/dividend", headers=auth)
    # 초기화 → 새 라운드
    with _price_patch({"005930": 100}):
        client.post("/me/paper/reset", json={"confirm": "초기화"}, headers=auth)
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 100}, headers=auth)
    with _price_patch({"005930": 100}), \
         patch("api.paper._holding_dps", return_value={"005930": 30.0}):
        r = client.post("/me/paper/dividend", headers=auth)  # 새 라운드라 다시 지급
    assert r.json()["paid"] is True


def test_dividend_no_dps_holdings(client):
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 100}, headers=auth)
    with patch("api.paper._holding_dps", return_value={"005930": 0.0}):
        r = client.post("/me/paper/dividend", headers=auth)
    b = r.json()
    assert b["paid"] is False and b["total"] == 0


def test_portfolio_holding_since_and_days(client):
    """보유 종목에 보유 시작일(since)·보유일수(holding_days)가 채워진다."""
    auth = _auth(client)
    with _price_patch({"005930": 100}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10}, headers=auth)
        pf = client.get("/me/paper/portfolio", headers=auth).json()
    h = pf["holdings"][0]
    assert h["since"] is not None          # YYYY-MM-DD
    assert len(h["since"]) == 10
    assert h["holding_days"] is not None and h["holding_days"] >= 0


def test_holding_since_map_reentry():
    """전량 매도 후 재매수 시 since가 '재진입일'로 갱신된다."""
    from datetime import datetime, timezone, timedelta
    from unittest.mock import MagicMock
    from api import paper

    KST = timezone(timedelta(hours=9))

    class T:  # PaperTrade 더미
        def __init__(self, ticker, side, qty, day):
            self.ticker = ticker; self.side = side; self.qty = qty; self.id = day
            self.created_at = datetime(2026, 6, day, 10, 0, tzinfo=KST)

    trades = [
        T("005930", "buy", 10, 1),    # 6/1 진입
        T("005930", "sell", 10, 5),   # 6/5 전량 청산
        T("005930", "buy", 5, 10),    # 6/10 재진입 ← since 기준
        T("000660", "buy", 3, 2),     # 6/2 진입(보유 유지)
    ]
    db = MagicMock()
    db.scalars.return_value = trades
    out = paper._holding_since_map(db, user_id=1)
    assert out["005930"] == "2026-06-10"   # 재진입일
    assert out["000660"] == "2026-06-02"


def test_to_kst_naive_utc_treated_as_utc():
    """created_at이 naive(sqlite 반환)여도 UTC로 간주해 KST 변환.

    회귀: 이전엔 naive에 .astimezone(KST)를 해 시스템 로컬시간으로 오해 → UTC
    자정 근처 체결의 진입일이 하루 어긋났다(#93~108 점검). aware/naive 둘 다
    2026-07-09T23:30 UTC(=KST 07-10 08:30)로 같은 날짜를 내야 한다.
    """
    from datetime import datetime, timezone
    from api.paper import _to_kst

    aware = datetime(2026, 7, 9, 23, 30, tzinfo=timezone.utc)
    naive = datetime(2026, 7, 9, 23, 30)  # sqlite가 tzinfo 벗겨 반환하는 형태
    assert _to_kst(aware).strftime("%Y-%m-%d") == "2026-07-10"
    assert _to_kst(naive).strftime("%Y-%m-%d") == "2026-07-10"
    assert _to_kst(None) is None


def test_holding_since_map_fully_sold_excluded():
    """전량 매도해 보유 0인 종목은 since에서 제외."""
    from datetime import datetime, timezone, timedelta
    from unittest.mock import MagicMock
    from api import paper
    KST = timezone(timedelta(hours=9))

    class T:
        def __init__(self, ticker, side, qty, day):
            self.ticker = ticker; self.side = side; self.qty = qty; self.id = day
            self.created_at = datetime(2026, 6, day, 10, 0, tzinfo=KST)

    db = MagicMock()
    db.scalars.return_value = [
        T("005930", "buy", 10, 1),
        T("005930", "sell", 10, 3),   # 전량 청산
    ]
    out = paper._holding_since_map(db, user_id=1)
    assert "005930" not in out
