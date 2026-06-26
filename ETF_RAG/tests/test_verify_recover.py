"""verify_and_recover.find_missing_dates — ETF/주식 분리 누락 감지 검증.

회귀 대상: ETF만 있고 주식이 0인 날(수집 중단)을 'COUNT(ticker)<500'만 보던
기존 로직이 정상 처리해 누락을 방치하던 버그(2026-06-26 6/22·6/25 케이스).
"""

import sqlite3

import pytest

from scripts.verify_and_recover import find_missing_dates


def _make_conn(rows):
    """rows: [(ticker, date, type)] → in-memory DB(daily_prices+instruments)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE daily_prices (ticker TEXT, date TEXT, close INTEGER)")
    conn.execute("CREATE TABLE instruments (ticker TEXT PRIMARY KEY, type TEXT)")
    seen = set()
    for ticker, date, typ in rows:
        conn.execute("INSERT INTO daily_prices VALUES (?,?,1)", (ticker, date))
        if ticker not in seen:
            conn.execute("INSERT INTO instruments VALUES (?,?)", (ticker, typ))
            seen.add(ticker)
    conn.commit()
    return conn


def _rows(date, n_etf, n_stock):
    r = [(f"E{i:05d}", date, "etf") for i in range(n_etf)]
    r += [(f"S{i:05d}", date, "stock") for i in range(n_stock)]
    return r


def test_normal_day_not_flagged():
    conn = _make_conn(_rows("20260623", 1100, 2700))
    assert find_missing_dates(conn, ["20260623"]) == []


def test_stock_missing_flagged():
    """ETF만 있고 주식 0 → 부분 누락 감지(기존 버그가 놓치던 케이스)."""
    conn = _make_conn(_rows("20260622", 1140, 0))
    missing = find_missing_dates(conn, ["20260622"])
    assert [m[0] for m in missing] == ["20260622"]


def test_etf_missing_flagged():
    """주식만 있고 ETF 0 → 부분 누락 감지."""
    conn = _make_conn(_rows("20260626", 0, 2700))
    missing = find_missing_dates(conn, ["20260626"])
    assert [m[0] for m in missing] == ["20260626"]


def test_empty_day_flagged_as_holiday_candidate():
    """둘 다 0 → 휴장일 후보로 감지(recover에서 0건=휴장 확인)."""
    conn = _make_conn([])
    missing = find_missing_dates(conn, ["20260620"])
    assert [m[0] for m in missing] == ["20260620"]


def test_stock_below_threshold_flagged():
    """주식이 하한(1500) 미만이면 누락."""
    conn = _make_conn(_rows("20260624", 1100, 800))
    assert [m[0] for m in find_missing_dates(conn, ["20260624"])] == ["20260624"]
