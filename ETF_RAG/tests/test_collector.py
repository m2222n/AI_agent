"""ETF 수집기 — pykrx Series 반환 방어 회귀 테스트

2026-06-01 daily-collect 장애 회귀 방지:
pykrx 내부 DataFrame에 KRX ticker가 중복되면 get_etf_ticker_name() /
get_market_ticker_name()이 string 대신 pandas Series를 반환 →
SQLite 바인딩에서 'type Series is not supported' 발생.
모든 종목명 조회는 _coerce_name()을 거쳐 Series면 iloc[0]로 첫 값만 추출해야 한다.
"""
from unittest.mock import patch

import pandas as pd

from src.data.collector import (
    _coerce_name,
    _safe_get_ticker_name,
    _safe_get_etf_name,
)


def test_coerce_name_with_series():
    """Series 반환 시 첫 값만 추출 (티커 중복 시 발생)"""
    series = pd.Series(["KODEX 200", "KODEX 200TR"], index=["069500", "278530"])
    result = _coerce_name(series)
    assert result == "KODEX 200"
    assert isinstance(result, str)


def test_coerce_name_plain_string():
    assert _coerce_name("TIGER 미국S&P500") == "TIGER 미국S&P500"


def test_coerce_name_none():
    assert _coerce_name(None) == ""


def test_coerce_name_empty_series():
    """빈 Series는 빈 문자열로 (iloc[0] IndexError 방어)"""
    assert _coerce_name(pd.Series([], dtype=object)) == ""


def test_safe_get_etf_name_series_response():
    """get_etf_ticker_name이 Series 반환 시 string으로 강제 변환"""
    series = pd.Series(["KODEX 200"], index=["069500"])
    with patch(
        "src.data.collector.stock.get_etf_ticker_name", return_value=series
    ):
        result = _safe_get_etf_name("069500")
    assert result == "KODEX 200"
    assert isinstance(result, str)


def test_safe_get_etf_name_api_error():
    """pykrx API 에러 시 빈 문자열 fallback (크래시 방지)"""
    with patch(
        "src.data.collector.stock.get_etf_ticker_name",
        side_effect=Exception("KRX down"),
    ):
        assert _safe_get_etf_name("999999") == ""


def test_safe_get_ticker_name_series_response():
    series = pd.Series(["삼성전자"], index=["005930"])
    with patch(
        "src.data.collector.stock.get_market_ticker_name", return_value=series
    ):
        result = _safe_get_ticker_name("005930")
    assert result == "삼성전자"
    assert isinstance(result, str)


def test_safe_get_ticker_name_api_error():
    with patch(
        "src.data.collector.stock.get_market_ticker_name",
        side_effect=Exception("KRX down"),
    ):
        assert _safe_get_ticker_name("999999") == ""
