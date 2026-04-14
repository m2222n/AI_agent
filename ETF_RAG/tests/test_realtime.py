"""실시간 시세 모듈 + 도구 테스트"""

import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from src.data.realtime import (
    is_market_open,
    krx_to_yfinance,
    get_realtime_price,
    clear_cache,
    KST,
    _cache,
    _market_suffix_cache,
)


# ── 장 운영 시간 판단 ─────────────────────────────────────

def test_market_open_weekday_during_hours():
    """평일 장중 → True"""
    now = datetime(2026, 4, 8, 10, 30, tzinfo=KST)  # 수요일 10:30
    assert is_market_open(now) is True


def test_market_open_weekday_after_hours():
    """평일 장 마감 후 → False"""
    now = datetime(2026, 4, 8, 16, 0, tzinfo=KST)
    assert is_market_open(now) is False


def test_market_open_weekday_before_hours():
    """평일 장 시작 전 → False"""
    now = datetime(2026, 4, 8, 8, 30, tzinfo=KST)
    assert is_market_open(now) is False


def test_market_open_weekend():
    """주말 → False"""
    now = datetime(2026, 4, 11, 10, 30, tzinfo=KST)  # 토요일
    assert is_market_open(now) is False


def test_market_open_boundary_open():
    """장 시작 정각 09:00 → True"""
    now = datetime(2026, 4, 8, 9, 0, 0, tzinfo=KST)
    assert is_market_open(now) is True


def test_market_open_boundary_close():
    """장 마감 정각 15:30 → True"""
    now = datetime(2026, 4, 8, 15, 30, 0, tzinfo=KST)
    assert is_market_open(now) is True


def test_market_open_just_after_close():
    """15:30:01 → False"""
    now = datetime(2026, 4, 8, 15, 30, 1, tzinfo=KST)
    assert is_market_open(now) is False


# ── 티커 변환 ──────────────────────────────────────────────

def test_krx_to_yfinance_etf():
    """ETF는 항상 .KS"""
    clear_cache()
    result = krx_to_yfinance("069500", "etf")
    assert result == "069500.KS"


def test_krx_to_yfinance_stock_kospi():
    """주식 KOSPI 매핑 — .KS에서 데이터 찾음"""
    clear_cache()
    mock_info = MagicMock()
    mock_info.last_price = 55000
    mock_ticker = MagicMock()
    mock_ticker.fast_info = mock_info

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = krx_to_yfinance("005930", "stock")
    assert result == "005930.KS"


def test_krx_to_yfinance_stock_kosdaq():
    """주식 KOSDAQ — .KS 실패 후 .KQ 성공"""
    clear_cache()
    call_count = 0

    def mock_ticker_factory(yf_ticker):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        if yf_ticker.endswith(".KS"):
            mock.fast_info.last_price = None
        else:
            mock.fast_info.last_price = 30000
        return mock

    with patch("yfinance.Ticker", side_effect=mock_ticker_factory):
        result = krx_to_yfinance("373220", "stock")
    assert result == "373220.KQ"


def test_krx_to_yfinance_uses_cache():
    """캐시된 티커는 yfinance 호출 없이 반환"""
    clear_cache()
    _market_suffix_cache["005930"] = "KS"
    result = krx_to_yfinance("005930", "stock")
    assert result == "005930.KS"


# ── 실시간 가격 조회 ──────────────────────────────────────

def test_get_realtime_price_market_closed():
    """장 외 시간 → None"""
    clear_cache()
    now = datetime(2026, 4, 8, 18, 0, tzinfo=KST)
    with patch("src.data.realtime.is_market_open", return_value=False):
        result = get_realtime_price("069500", "etf")
    assert result is None


def test_get_realtime_price_success():
    """장중 yfinance 성공"""
    clear_cache()
    mock_info = MagicMock()
    mock_info.last_price = 80500.0
    mock_info.previous_close = 80000.0
    mock_info.last_volume = 1500000
    mock_ticker = MagicMock()
    mock_ticker.fast_info = mock_info

    with patch("src.data.realtime.is_market_open", return_value=True), \
         patch("yfinance.Ticker", return_value=mock_ticker):
        result = get_realtime_price("069500", "etf")

    assert result is not None
    assert result["price"] == 80500
    assert result["prev_close"] == 80000
    assert result["change"] == 500
    assert result["change_pct"] == 0.62
    assert result["volume"] == 1500000
    assert result["source"] == "yfinance"


def test_get_realtime_price_cache_hit():
    """캐시 히트 — 두 번째 호출은 yfinance 미호출"""
    clear_cache()
    _cache["069500"] = {
        "data": {"price": 80500, "source": "yfinance"},
        "fetched_at": time.time(),  # 방금 캐시
    }
    with patch("src.data.realtime.is_market_open", return_value=True):
        # yfinance를 mock하지 않아도 캐시에서 반환
        result = get_realtime_price("069500", "etf", cache_ttl=300)

    assert result["price"] == 80500


def test_get_realtime_price_cache_expired():
    """캐시 만료 → 재조회"""
    clear_cache()
    _cache["069500"] = {
        "data": {"price": 80000, "source": "yfinance"},
        "fetched_at": time.time() - 600,  # 10분 전 (만료)
    }
    mock_info = MagicMock()
    mock_info.last_price = 81000.0
    mock_info.previous_close = 80000.0
    mock_info.last_volume = 2000000
    mock_ticker = MagicMock()
    mock_ticker.fast_info = mock_info

    with patch("src.data.realtime.is_market_open", return_value=True), \
         patch("yfinance.Ticker", return_value=mock_ticker):
        result = get_realtime_price("069500", "etf", cache_ttl=300)

    assert result["price"] == 81000  # 새 값


def test_get_realtime_price_yfinance_error():
    """yfinance 예외 → None"""
    clear_cache()
    with patch("src.data.realtime.is_market_open", return_value=True), \
         patch("yfinance.Ticker", side_effect=Exception("API error")):
        result = get_realtime_price("069500", "etf")
    assert result is None


def test_get_realtime_price_no_last_price():
    """last_price가 None → None"""
    clear_cache()
    mock_info = MagicMock()
    mock_info.last_price = None
    mock_ticker = MagicMock()
    mock_ticker.fast_info = mock_info

    with patch("src.data.realtime.is_market_open", return_value=True), \
         patch("yfinance.Ticker", return_value=mock_ticker):
        result = get_realtime_price("069500", "etf")
    assert result is None


def test_clear_cache():
    """캐시 초기화"""
    _cache["test"] = {"data": {}, "fetched_at": 0}
    _market_suffix_cache["test"] = "KS"
    clear_cache()
    assert len(_cache) == 0
    assert len(_market_suffix_cache) == 0


# ── 도구 테스트 ──────────────────────────────────────────

def test_tool_all_tools_count():
    """ALL_TOOLS 8개"""
    from src.llm.tools import ALL_TOOLS
    assert len(ALL_TOOLS) == 12
    names = [t.name for t in ALL_TOOLS]
    assert "get_realtime_price" in names


def test_tool_realtime_price_not_found():
    """없는 종목 → 에러 메시지"""
    from src.llm.tools import get_realtime_price as tool_fn, set_retriever
    set_retriever(None, [], etf_data=[], stock_data=[])
    result = tool_fn.invoke({"name_or_ticker": "없는종목XYZ"})
    assert "찾을 수 없습니다" in result


def test_tool_realtime_price_fallback():
    """실시간 실패 시 pykrx fallback"""
    from src.llm.tools import get_realtime_price as tool_fn, set_retriever

    etf_data = [{
        "name": "KODEX 200", "ticker": "069500", "date": "20260408",
        "close": 80800, "change_pct": 2.91, "nav": 80647,
        "returns": {"1d": 2.91, "1m": 5.0},
        "volume": 14000000, "trade_value": 1000000000000,
    }]
    set_retriever(None, [], etf_data=etf_data, stock_data=[])

    with patch("config.REALTIME_PRICE", {"enabled": False}):
        result = tool_fn.invoke({"name_or_ticker": "KODEX 200"})

    assert "80,800" in result
    assert "KODEX 200" in result


def test_tool_realtime_price_with_realtime_data():
    """장중 실시간 데이터 반환"""
    from src.llm.tools import get_realtime_price as tool_fn, set_retriever

    etf_data = [{
        "name": "KODEX 200", "ticker": "069500", "date": "20260408",
        "close": 80800, "change_pct": 2.91, "nav": 80647,
        "volume": 14000000, "trade_value": 1000000000000,
    }]
    set_retriever(None, [], etf_data=etf_data, stock_data=[])

    rt_data = {
        "price": 81200, "prev_close": 80800,
        "change": 400, "change_pct": 0.50,
        "volume": 5000000, "timestamp": "2026-04-08 14:30",
        "source": "yfinance",
    }

    with patch("config.REALTIME_PRICE", {"enabled": True, "cache_ttl": 300}), \
         patch("src.data.realtime.get_realtime_price", return_value=rt_data):
        result = tool_fn.invoke({"name_or_ticker": "KODEX 200"})

    assert "81,200" in result
    assert "현재가" in result
    assert "yfinance" in result
