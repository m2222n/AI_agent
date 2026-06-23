"""5개 탭 + 자동완성 REST 엔드포인트 테스트 (Phase F).

API_SKIP_INIT=1로 실제 init 우회 → 래핑 대상 함수는 api.tabs import 사이트에서 patch.
"""

import os

os.environ["API_SKIP_INIT"] = "1"

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_sse_global():
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── Technical ──────────────────────────────────────────────────────
def test_technical_returns_summary_and_chart(client):
    structured = {"ticker": "005930", "name": "삼성전자", "per": 12.0}
    summary = {"close": 70000, "rsi": 55.0, "trend": "상승"}
    with patch("api.tabs._find_structured_data", return_value=structured), patch(
        "api.tabs.get_technical_summary", return_value=summary
    ), patch("api.tabs.generate_technical_chart", return_value="BASE64PNG"):
        r = client.get("/tabs/technical", params={"ticker": "삼성전자"})
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "005930"
    assert body["name"] == "삼성전자"
    assert body["summary"]["rsi"] == 55.0
    assert body["chart_b64"] == "BASE64PNG"


def test_technical_404_when_unresolved(client):
    with patch("api.tabs._find_structured_data", return_value=None):
        r = client.get("/tabs/technical", params={"ticker": "ZZZ"})
    assert r.status_code == 404


def test_technical_404_when_no_summary(client):
    with patch(
        "api.tabs._find_structured_data",
        return_value={"ticker": "005930", "name": "삼성전자"},
    ), patch("api.tabs.get_technical_summary", return_value=None):
        r = client.get("/tabs/technical", params={"ticker": "삼성전자"})
    assert r.status_code == 404


# ── Outlook ────────────────────────────────────────────────────────
def test_outlook_assembles_summary_and_structured(client):
    structured = {"ticker": "005930", "name": "삼성전자", "per": 12.0}
    summary = {"close": 70000}
    outlook = {"composite_score": 0.3, "confidence_grade": "B", "scenarios": {}}
    with patch("api.tabs._find_structured_data", return_value=structured), patch(
        "api.tabs.get_technical_summary", return_value=summary
    ), patch("api.tabs.build_price_outlook", return_value=outlook) as m:
        r = client.get(
            "/tabs/outlook", params={"ticker": "삼성전자", "horizon": "1m"}
        )
    assert r.status_code == 200
    assert r.json()["confidence_grade"] == "B"
    # summary / structured_data가 키워드로 전달됐는지
    kwargs = m.call_args.kwargs
    assert kwargs["summary"] == summary
    assert kwargs["structured_data"] == structured


def test_outlook_404_when_no_summary(client):
    with patch("api.tabs._find_structured_data", return_value={"ticker": "X", "name": "X"}), patch(
        "api.tabs.get_technical_summary", return_value=None
    ):
        r = client.get("/tabs/outlook", params={"ticker": "X"})
    assert r.status_code == 404


# ── Intraday ───────────────────────────────────────────────────────
def test_intraday_returns_chart(client):
    data = {"ticker": "005930", "name": "삼성전자", "close": 70000}
    with patch("api.tabs._find_structured_data", return_value=data), patch(
        "api.tabs.generate_intraday_chart", return_value="INTRADAY_PNG"
    ):
        r = client.get("/tabs/intraday", params={"ticker": "삼성전자"})
    assert r.status_code == 200
    assert r.json()["chart_b64"] == "INTRADAY_PNG"


def test_intraday_404_when_no_data(client):
    data = {"ticker": "005930", "name": "삼성전자", "close": 70000}
    with patch("api.tabs._find_structured_data", return_value=data), patch(
        "api.tabs.generate_intraday_chart", return_value=None
    ):
        r = client.get("/tabs/intraday", params={"ticker": "삼성전자"})
    assert r.status_code == 404


# ── Financial ──────────────────────────────────────────────────────
def test_financial_returns_rows_and_chart(client):
    rows = [
        {"fiscal_year": 2025, "fiscal_quarter": 1, "revenue": 1_000_000_000_000, "operating_margin": 10.0}
    ]
    with patch("api.tabs.DB_PATH") as db, patch("api.tabs._find_structured_data", return_value={"ticker": "005930", "name": "삼성전자"}), patch(
        "api.tabs.get_connection", return_value=MagicMock()
    ), patch("api.tabs.get_financial_data", return_value=rows), patch(
        "api.tabs.generate_financial_chart", return_value="PNG"
    ):
        db.exists.return_value = True
        r = client.get("/tabs/financial", params={"ticker": "005930"})
    assert r.status_code == 200
    body = r.json()
    assert body["rows"][0]["fiscal_year"] == 2025
    assert body["chart_b64"] == "PNG"


def test_financial_404_when_db_missing(client):
    with patch("api.tabs.DB_PATH") as db:
        db.exists.return_value = False
        r = client.get("/tabs/financial", params={"ticker": "005930"})
    assert r.status_code == 404


# ── Comparison ─────────────────────────────────────────────────────
def test_comparison_two_tickers(client):
    d1 = {"ticker": "005930", "name": "삼성전자", "per": 12.0, "pbr": 1.2}
    d2 = {"ticker": "000660", "name": "SK하이닉스", "per": 8.0, "pbr": 1.5}
    with patch("api.tabs._find_structured_data", side_effect=[d1, d2]), patch(
        "api.tabs.generate_comparison_chart", return_value="CMP"
    ), patch("api.tabs.generate_valuation_chart", return_value="VAL"):
        r = client.post(
            "/tabs/comparison", json={"tickers": ["삼성전자", "SK하이닉스"]}
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 2
    assert body["comparison_chart_b64"] == "CMP"
    assert body["valuation_chart_b64"] == "VAL"


def test_comparison_404_when_one_unresolved(client):
    d1 = {"ticker": "005930", "name": "삼성전자"}
    with patch("api.tabs._find_structured_data", side_effect=[d1, None]):
        r = client.post("/tabs/comparison", json={"tickers": ["삼성전자", "ZZZ"]})
    assert r.status_code == 404


# ── Sector ─────────────────────────────────────────────────────────
def test_sector_overview(client):
    sector_index = {
        "전기·전자": [
            {"name": "삼성전자", "ticker": "005930", "market_cap": 5e14, "change_pct": 1.0, "per": 12.0},
            {"name": "SK하이닉스", "ticker": "000660", "market_cap": 1e14, "change_pct": -0.5, "per": 8.0},
        ]
    }
    with patch("api.tabs.get_sector_index", return_value=sector_index), patch(
        "api.tabs.generate_sector_overview_chart", return_value="OVR"
    ):
        r = client.get("/tabs/sector")
    assert r.status_code == 200
    body = r.json()
    assert body["overview_chart_b64"] == "OVR"
    assert body["stats"][0]["sector"] == "전기·전자"
    assert body["stats"][0]["count"] == 2
    assert body["stats"][0]["up_count"] == 1


def test_sector_detail(client):
    sector_index = {"전기·전자": [{"name": "삼성전자", "ticker": "005930", "market_cap": 5e14, "change_pct": 1.0, "per": 12.0}]}
    with patch("api.tabs.get_sector_index", return_value=sector_index), patch(
        "api.tabs.generate_sector_overview_chart", return_value="OVR"
    ), patch("api.tabs.generate_sector_detail_chart", return_value="DET"):
        r = client.get("/tabs/sector", params={"sector": "전기·전자"})
    assert r.status_code == 200
    assert r.json()["detail_chart_b64"] == "DET"


def test_sector_404_unknown(client):
    with patch("api.tabs.get_sector_index", return_value={"A": [{"market_cap": 1, "change_pct": 0}]}), patch(
        "api.tabs.generate_sector_overview_chart", return_value="OVR"
    ):
        r = client.get("/tabs/sector", params={"sector": "없는섹터"})
    assert r.status_code == 404


def test_sector_period_invalid_422(client):
    r = client.get("/tabs/sector", params={"sector": "전기·전자", "period": "7d"})
    assert r.status_code == 422  # pattern ^(1d|1w|...|10y)$


def test_sector_1d_has_no_trend(client):
    """기본 1d는 스냅샷 — trend 차트를 만들지 않는다."""
    sector_index = {"전기·전자": [{"name": "삼성전자", "ticker": "005930", "market_cap": 5e14, "change_pct": 1.0, "per": 12.0}]}
    with patch("api.tabs.get_sector_index", return_value=sector_index), patch(
        "api.tabs.generate_sector_overview_chart", return_value="OVR"
    ), patch("api.tabs.generate_sector_detail_chart", return_value="DET"):
        r = client.get("/tabs/sector", params={"sector": "전기·전자", "period": "1d"})
    assert r.status_code == 200
    assert "trend_chart_b64" not in r.json()


def test_sector_period_includes_trend(client):
    """period!=1d면 _sector_trend 결과로 trend 차트/수익률을 포함한다."""
    sector_index = {"전기·전자": [{"name": "삼성전자", "ticker": "005930", "market_cap": 5e14, "change_pct": 1.0, "per": 12.0}]}
    trend = {"dates": ["20250101", "20260101"], "index_values": [100.0, 130.0],
             "return_pct": 30.0, "constituents": 1}
    with patch("api.tabs.get_sector_index", return_value=sector_index), patch(
        "api.tabs.generate_sector_overview_chart", return_value="OVR"
    ), patch("api.tabs.generate_sector_detail_chart", return_value="DET"), patch(
        "api.tabs._sector_trend", return_value=trend
    ), patch("api.tabs.generate_sector_trend_chart", return_value="TREND"):
        r = client.get("/tabs/sector", params={"sector": "전기·전자", "period": "1y"})
    body = r.json()
    assert body["trend_chart_b64"] == "TREND"
    assert body["trend_return_pct"] == 30.0
    assert body["trend_constituents"] == 1


# ── Tickers ────────────────────────────────────────────────────────
def test_tickers_filters_and_caps(client):
    opts = [f"삼성전자{i} (0059{i:02d})" for i in range(40)] + ["LG (003550)"]
    with patch("api.tabs.get_available_tickers", return_value=opts):
        r = client.get("/tabs/tickers", params={"q": "삼성", "limit": 30})
    body = r.json()
    assert len(body["options"]) == 30
    assert all("삼성" in o for o in body["options"])


def test_tickers_asset_type_passed_through(client):
    """asset_type 쿼리가 get_available_tickers에 그대로 전달돼야 한다(재무제표=주식)."""
    with patch("api.tabs.get_available_tickers", return_value=["삼성전자 (005930)"]) as m:
        r = client.get("/tabs/tickers", params={"asset_type": "stock"})
    assert r.status_code == 200
    # run_in_threadpool(get_available_tickers, asset_type) → 첫 위치인자로 전달
    assert m.call_args.args[0] == "stock"


def test_tickers_asset_type_invalid_422(client):
    r = client.get("/tabs/tickers", params={"asset_type": "bond"})
    assert r.status_code == 422  # pattern ^(stock|etf)$


def test_tickers_resolve_404(client):
    with patch("api.tabs._find_structured_data", return_value=None):
        r = client.get("/tabs/tickers/resolve", params={"q": "ZZZ"})
    assert r.status_code == 404


# ── Movers (동적 추천질문) ─────────────────────────────────────────
def test_movers(client):
    etf_idx = {
        "kodex 200": {"name": "KODEX 200", "ticker": "069500", "change_pct": 2.1, "trade_value": 1e11, "close": 45000},
    }
    stock_idx = {
        "삼성전자": {"name": "삼성전자", "ticker": "005930", "change_pct": -1.5, "trade_value": 5e11, "close": 70000},
        "005930": {"name": "삼성전자", "ticker": "005930", "change_pct": -1.5, "trade_value": 5e11, "close": 70000},  # dedup 대상
    }
    with patch("api.tabs.get_data_indices", return_value=(etf_idx, stock_idx)):
        r = client.get("/tabs/movers", params={"n": 3})
    assert r.status_code == 200
    body = r.json()
    # dedup 후 2종목: 삼성전자(-1.5)는 losers·traded, KODEX(+2.1)는 gainers
    assert body["gainers"][0]["ticker"] == "069500"
    assert body["losers"][0]["ticker"] == "005930"
    assert body["most_traded"][0]["ticker"] == "005930"  # 거래대금 최대


# ── Overview (사이드바) ────────────────────────────────────────────
def test_overview(client):
    etf_idx = {
        "kodex 200": {"name": "KODEX 200", "ticker": "069500", "close": 45000, "change_pct": 1.2, "trade_value": 9e10, "date": "20260609"},
        "069500": {"name": "KODEX 200", "ticker": "069500", "close": 45000, "change_pct": 1.2, "trade_value": 9e10, "date": "20260609"},
    }
    stock_idx = {
        "삼성전자": {"name": "삼성전자", "ticker": "005930", "close": 70000, "change_pct": -0.5, "trade_value": 5e11, "sector": "전기·전자", "per": 49.8, "market_cap": 4e14, "date": "20260609"},
        "005930": {"name": "삼성전자", "ticker": "005930", "close": 70000, "change_pct": -0.5, "trade_value": 5e11, "sector": "전기·전자", "per": 49.8, "market_cap": 4e14, "date": "20260609"},
    }
    with patch("api.tabs.get_data_indices", return_value=(etf_idx, stock_idx)):
        r = client.get("/tabs/overview", params={"top": 20})
    assert r.status_code == 200
    b = r.json()
    assert b["etf_count"] == 1  # dedup
    assert b["stock_count"] == 1
    assert b["as_of"] == "20260609"
    assert b["top_etfs"][0]["ticker"] == "069500"
    assert b["top_stocks"][0]["sector"] == "전기·전자"
    assert "전기·전자" in b["sectors"]


def test_overview_sector_filter(client):
    """sector 지정 시 해당 업종 종목만 top_stocks에 포함."""
    etf_idx = {}
    stock_idx = {
        "005930": {"name": "삼성전자", "ticker": "005930", "close": 70000, "change_pct": -0.5, "trade_value": 5e11, "sector": "전기·전자", "date": "20260609"},
        "035720": {"name": "카카오", "ticker": "035720", "close": 50000, "change_pct": 1.0, "trade_value": 3e11, "sector": "서비스업", "date": "20260609"},
    }
    with patch("api.tabs.get_data_indices", return_value=(etf_idx, stock_idx)):
        r = client.get("/tabs/overview", params={"top": 20, "sector": "서비스업"})
    assert r.status_code == 200
    b = r.json()
    # 서비스업만 — 카카오 1종목
    assert [s["ticker"] for s in b["top_stocks"]] == ["035720"]
    # sectors는 항상 전체 업종(필터와 무관)
    assert set(b["sectors"]) == {"전기·전자", "서비스업"}


# ── 실시간 시세 /tabs/price ────────────────────────────────────────
def test_price_live_kis(client):
    """장중 + KIS 실시간 → is_live=True, source=kis."""
    structured = {"ticker": "005930", "name": "삼성전자", "close": 69000,
                  "date": "20260611"}
    rt = {"price": 70000, "prev_close": 69000, "change": 1000,
          "change_pct": 1.45, "volume": 12000000,
          "timestamp": "2026-06-12 10:00", "source": "kis"}
    with patch("api.tabs._find_structured_data", return_value=structured), \
         patch("src.data.realtime.is_market_open", return_value=True), \
         patch("src.data.realtime.get_realtime_price", return_value=rt):
        r = client.get("/tabs/price", params={"ticker": "삼성전자"})
    assert r.status_code == 200
    b = r.json()
    assert b["price"] == 70000
    assert b["source"] == "kis"
    assert b["is_live"] is True
    assert b["market_open"] is True
    assert b["change_pct"] == 1.45


def test_price_fallback_close_when_market_closed(client):
    """장 외 → 실시간 미조회, 수집 종가(source=close, is_live=False)."""
    structured = {"ticker": "069500", "name": "KODEX 200", "close": 80800,
                  "change_pct": 2.91, "volume": 14000000, "nav": 80647,
                  "date": "20260611"}
    with patch("api.tabs._find_structured_data", return_value=structured), \
         patch("src.data.realtime.is_market_open", return_value=False):
        r = client.get("/tabs/price", params={"ticker": "KODEX 200"})
    assert r.status_code == 200
    b = r.json()
    assert b["price"] == 80800
    assert b["source"] == "close"
    assert b["is_live"] is False
    assert b["market_open"] is False
    assert b["timestamp"] == "2026-06-11"  # YYYYMMDD → YYYY-MM-DD


def test_price_fallback_close_when_realtime_fails(client):
    """장중이나 실시간 실패(None) → 종가 fallback."""
    structured = {"ticker": "005930", "name": "삼성전자", "close": 69000,
                  "change_pct": 0.5, "date": "20260611"}
    with patch("api.tabs._find_structured_data", return_value=structured), \
         patch("src.data.realtime.is_market_open", return_value=True), \
         patch("src.data.realtime.get_realtime_price", return_value=None):
        r = client.get("/tabs/price", params={"ticker": "삼성전자"})
    assert r.status_code == 200
    b = r.json()
    assert b["source"] == "close"
    assert b["is_live"] is False
    assert b["market_open"] is True  # 장중이지만 실시간 실패


def test_price_404_when_unresolved(client):
    with patch("api.tabs._find_structured_data", return_value=None):
        r = client.get("/tabs/price", params={"ticker": "ZZZ없는종목"})
    assert r.status_code == 404


# ── 호가 /tabs/orderbook ───────────────────────────────────────────
def _ob():
    return {
        "asks": [{"price": 70000 + i * 100, "qty": i * 10} for i in range(1, 11)],
        "bids": [{"price": 69900 - i * 100, "qty": i * 20} for i in range(1, 11)],
        "total_ask_qty": 5500, "total_bid_qty": 11000,
        "timestamp": "2026-06-12 10:00", "source": "kis",
    }


def test_orderbook_success(client):
    structured = {"ticker": "005930", "name": "삼성전자"}
    with patch("api.tabs._find_structured_data", return_value=structured), \
         patch("src.data.kis_client.is_enabled", return_value=True), \
         patch("src.data.kis_client.get_orderbook", return_value=_ob()):
        r = client.get("/tabs/orderbook", params={"ticker": "삼성전자"})
    assert r.status_code == 200
    b = r.json()
    assert b["ticker"] == "005930"
    assert len(b["asks"]) == 10 and len(b["bids"]) == 10
    assert b["asks"][0] == {"price": 70100, "qty": 10}
    assert b["total_bid_qty"] == 11000
    assert b["source"] == "kis"


def test_orderbook_404_when_kis_disabled(client):
    structured = {"ticker": "005930", "name": "삼성전자"}
    with patch("api.tabs._find_structured_data", return_value=structured), \
         patch("src.data.kis_client.is_enabled", return_value=False):
        r = client.get("/tabs/orderbook", params={"ticker": "삼성전자"})
    assert r.status_code == 404


def test_orderbook_404_when_no_data(client):
    """KIS 활성이나 장 외/조회 실패(None) → 404."""
    structured = {"ticker": "005930", "name": "삼성전자"}
    with patch("api.tabs._find_structured_data", return_value=structured), \
         patch("src.data.kis_client.is_enabled", return_value=True), \
         patch("src.data.kis_client.get_orderbook", return_value=None):
        r = client.get("/tabs/orderbook", params={"ticker": "삼성전자"})
    assert r.status_code == 404


def test_orderbook_404_when_unresolved(client):
    with patch("api.tabs._find_structured_data", return_value=None):
        r = client.get("/tabs/orderbook", params={"ticker": "ZZZ"})
    assert r.status_code == 404


# ── 실시간 체결 SSE /tabs/price/stream ─────────────────────────────
def test_price_stream_unavailable_when_subscribe_none(client):
    """KIS WS 구독 실패(None) → unavailable 이벤트 1건 후 종료."""
    structured = {"ticker": "005930", "name": "삼성전자"}

    async def _no_sub(code):
        return None

    fake_mgr = MagicMock()
    fake_mgr.subscribe = _no_sub
    with patch("api.tabs._find_structured_data", return_value=structured), \
         patch("src.data.kis_ws.get_manager", return_value=fake_mgr):
        r = client.get("/tabs/price/stream", params={"ticker": "삼성전자"})
    assert r.status_code == 200
    assert "unavailable" in r.text


# ── 가드 ───────────────────────────────────────────────────────────
def test_tabs_require_ready_503(client):
    # ready=False면 503 (require_ready 의존성)
    app.state.app_state.ready = False
    try:
        r = client.get("/tabs/tickers")
        assert r.status_code == 503
    finally:
        app.state.app_state.ready = True
