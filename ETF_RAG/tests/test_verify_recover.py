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


# ── 유령 행(휴장일 전-0 저장분) 감지·정리 ─────────────────────────────
# 배경(2026-10-08): KRX는 휴장일을 지정일로 조회하면 종가=0 프레임을 돌려주고,
# 백필 경로가 이를 그대로 저장해 2026-05-01/05-05/05-25/09-24/09-25/10-05 여섯 날에
# 유령 행이 쌓였다. find_missing_dates가 이를 '정상'으로 세던 것을 고친다.

def _make_conn_with_close(rows):
    """rows: [(ticker, date, type, close)]."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE daily_prices (ticker TEXT, date TEXT, close INTEGER)")
    conn.execute("CREATE TABLE instruments (ticker TEXT PRIMARY KEY, type TEXT)")
    seen = set()
    for ticker, date, typ, close in rows:
        conn.execute("INSERT INTO daily_prices VALUES (?,?,?)", (ticker, date, close))
        if ticker not in seen:
            conn.execute("INSERT INTO instruments VALUES (?,?)", (ticker, typ))
            seen.add(ticker)
    conn.commit()
    return conn


def test_phantom_zero_day_is_purged_and_flagged():
    """종가 전부 0인 날 → 유령 행으로 보고 삭제 + '데이터 없음(0)'으로 누락 등록."""
    rows = [(f"E{i:05d}", "20261005", "etf", 0) for i in range(1100)]
    rows += [(f"S{i:05d}", "20261005", "stock", 0) for i in range(2700)]
    conn = _make_conn_with_close(rows)
    missing = find_missing_dates(conn, ["20261005"])
    assert missing == [("20261005", 0)]
    left = conn.execute("SELECT COUNT(*) FROM daily_prices WHERE date='20261005'").fetchone()[0]
    assert left == 0, "유령 행이 삭제돼야 recover가 재수집(휴장이면 0건)으로 깨끗이 끝난다"


def test_real_day_with_some_zero_closes_not_purged():
    """정상 영업일에도 거래정지 종목 등 종가 0이 일부 섞인다 — 전부 0일 때만 유령."""
    rows = [(f"E{i:05d}", "20261006", "etf", 1000) for i in range(1100)]
    rows += [(f"S{i:05d}", "20261006", "stock", 0 if i < 50 else 500) for i in range(2700)]
    conn = _make_conn_with_close(rows)
    assert find_missing_dates(conn, ["20261006"]) == []
    left = conn.execute("SELECT COUNT(*) FROM daily_prices WHERE date='20261006'").fetchone()[0]
    assert left == 3800


# ── recover_missing: 실패(None) vs 휴장(0) 구분 ────────────────────────
# 배경(2026-10-08): collect_*_day가 예외에도 0을 돌려줘 로그인 실패·API 오류가
# "공휴일"로 분류되고 스크립트가 ✅로 끝났다(pykrx 구버전 장애에서 실제 발생).

from scripts.verify_and_recover import recover_missing  # noqa: E402


def _patch_collect(monkeypatch, etf, stk):
    import scripts.backfill_historical as bh
    monkeypatch.setattr(bh, "collect_etf_day", lambda conn, d: etf)
    monkeypatch.setattr(bh, "collect_stock_day", lambda conn, d: stk)


def test_recover_none_is_failure_not_holiday(monkeypatch):
    _patch_collect(monkeypatch, None, None)
    r = recover_missing(None, [("20261007", 0)])
    assert r["failed"] == ["20261007"] and r["holidays"] == []


def test_recover_partial_none_is_failure(monkeypatch):
    """ETF는 됐는데 주식 수집이 예외면 실패로 — 반쪽 복구를 성공으로 치지 않는다."""
    _patch_collect(monkeypatch, 1171, None)
    r = recover_missing(None, [("20261007", 0)])
    assert r["failed"] == ["20261007"]


def test_recover_zero_zero_with_no_existing_is_holiday(monkeypatch):
    _patch_collect(monkeypatch, 0, 0)
    r = recover_missing(None, [("20261005", 0)])
    assert r["holidays"] == ["20261005"] and r["failed"] == []


def test_recover_positive_is_recovered(monkeypatch):
    _patch_collect(monkeypatch, 1171, 2873)
    r = recover_missing(None, [("20261006", 0)])
    assert r["recovered"] == ["20261006"]


# ── backfill_historical 휴장일 가드 ────────────────────────────────────

def test_backfill_holiday_frame_detection():
    import pandas as pd
    from scripts.backfill_historical import _is_holiday_frame
    assert _is_holiday_frame(None)
    assert _is_holiday_frame(pd.DataFrame())
    assert _is_holiday_frame(pd.DataFrame({"종가": [0, 0, 0]}))
    assert not _is_holiday_frame(pd.DataFrame({"종가": [0, 71000, 0]}))


def test_collect_stock_day_skips_holiday_without_writing(monkeypatch):
    """휴장일 전-0 프레임 → 0 반환, DB에 아무것도 쓰지 않는다."""
    import pandas as pd
    from unittest.mock import MagicMock
    import scripts.backfill_historical as bh
    fake = MagicMock()
    fake.get_market_ohlcv_by_ticker.return_value = pd.DataFrame(
        {"시가": [0, 0], "고가": [0, 0], "저가": [0, 0], "종가": [0, 0], "거래량": [0, 0]},
        index=["005930", "000660"],
    )
    monkeypatch.setattr(bh, "stock", fake)
    conn = MagicMock()
    assert bh.collect_stock_day(conn, "20261005") == 0
    fake.get_market_cap_by_ticker.assert_not_called()
    conn.execute.assert_not_called()


def test_collect_stock_day_exception_returns_none(monkeypatch):
    """수집 예외 → None(실패). 0(휴장)과 구분돼야 recover가 오판하지 않는다."""
    from unittest.mock import MagicMock
    import scripts.backfill_historical as bh
    fake = MagicMock()
    fake.get_market_ohlcv_by_ticker.side_effect = ValueError("Expecting value")
    monkeypatch.setattr(bh, "stock", fake)
    assert bh.collect_stock_day(MagicMock(), "20261007") is None


def test_collect_etf_day_skips_holiday_without_writing(monkeypatch):
    import pandas as pd
    from unittest.mock import MagicMock
    import scripts.backfill_historical as bh
    fake = MagicMock()
    fake.get_etf_ticker_list.return_value = ["069500", "102110"]
    fake.get_etf_ohlcv_by_ticker.return_value = pd.DataFrame(
        {"종가": [0, 0], "NAV": [0, 0]}, index=["069500", "102110"],
    )
    monkeypatch.setattr(bh, "stock", fake)
    monkeypatch.setattr(bh.time, "sleep", lambda s: None)
    conn = MagicMock()
    assert bh.collect_etf_day(conn, "20261005") == 0
    fake.get_etf_price_change_by_ticker.assert_not_called()
    conn.execute.assert_not_called()
