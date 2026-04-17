"""SQLite 데이터베이스 테스트"""

import pytest
import sqlite3
from pathlib import Path

from src.data.database import (
    init_db,
    upsert_daily_data,
    get_latest_data,
    get_latest_date,
    get_historical_prices,
    search_instruments,
    prune_old_data,
    import_json_file,
    get_db_stats,
)


# ── Fixtures ──────────────────────────────────────────────────

SAMPLE_DATA = {
    "metadata": {
        "collection_date": "20260407",
        "collected_at": "2026-04-07T18:00:00",
        "total_etfs": 2,
        "holdings_collected": 1,
    },
    "etfs": [
        {
            "ticker": "069500",
            "name": "KODEX 200",
            "date": "20260407",
            "ohlcv": {
                "open": 80210, "high": 81200, "low": 80100, "close": 80800,
                "volume": 14703488, "trade_value": 1184866376189,
                "nav": 80647.71, "base_index": 798.32,
                "change": 735, "change_pct": 2.91,
            },
            "deviation": -0.17,
            "tracking_error": 0.05,
            "returns": {"1d": 2.91, "1w": 5.12, "1m": 8.3},
            "holdings": [
                {"stock_ticker": "005930", "stock_name": "삼성전자",
                 "shares": 8140.0, "amount": 667480000, "weight": 31.77},
            ],
        },
        {
            "ticker": "091160",
            "name": "KODEX 반도체",
            "date": "20260407",
            "ohlcv": {
                "open": 15100, "high": 15300, "low": 15000, "close": 15200,
                "volume": 500000, "trade_value": 7600000000,
                "nav": 15150.0, "base_index": 500.0,
                "change": 100, "change_pct": 0.66,
            },
            "deviation": -0.33,
            "tracking_error": 0.1,
            "returns": {"1d": 0.66, "1w": 3.2},
            "holdings": [],
        },
    ],
}

SAMPLE_DATA_DAY2 = {
    "metadata": {
        "collection_date": "20260408",
        "collected_at": "2026-04-08T18:00:00",
        "total_etfs": 1,
        "holdings_collected": 0,
    },
    "etfs": [
        {
            "ticker": "069500",
            "name": "KODEX 200",
            "date": "20260408",
            "ohlcv": {
                "open": 80800, "high": 81500, "low": 80600, "close": 81100,
                "volume": 12000000, "trade_value": 970000000000,
                "nav": 81050.0, "base_index": 802.0,
                "change": 300, "change_pct": 0.37,
            },
            "deviation": -0.06,
            "tracking_error": 0.03,
            "returns": {"1d": 0.37, "1w": 4.8, "1m": 9.1},
            "holdings": [],
        },
    ],
}


@pytest.fixture
def db(tmp_path):
    """테스트용 임시 DB"""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture
def db_with_data(db):
    """샘플 데이터가 들어간 DB"""
    upsert_daily_data(db, SAMPLE_DATA)
    return db


# ── 초기화 테스트 ─────────────────────────────────────────────

def test_init_db_creates_tables(db):
    """init_db가 5개 테이블을 생성하는지 확인"""
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {t["name"] for t in tables}
    assert "instruments" in names
    assert "daily_prices" in names
    assert "returns" in names
    assert "holdings" in names
    assert "collection_log" in names


def test_init_db_idempotent(tmp_path):
    """init_db를 여러 번 호출해도 안전"""
    db_path = tmp_path / "test.db"
    conn1 = init_db(db_path)
    conn1.close()
    conn2 = init_db(db_path)
    conn2.close()


# ── 쓰기 테스트 ──────────────────────────────────────────────

def test_upsert_daily_data(db):
    """upsert_daily_data가 ETF 수를 반환"""
    count = upsert_daily_data(db, SAMPLE_DATA)
    assert count == 2


def test_upsert_instruments(db_with_data):
    """instruments 테이블에 종목이 저장되는지 확인"""
    rows = db_with_data.execute("SELECT * FROM instruments").fetchall()
    assert len(rows) == 2
    tickers = {r["ticker"] for r in rows}
    assert "069500" in tickers
    assert "091160" in tickers


def test_upsert_daily_prices(db_with_data):
    """daily_prices에 시세 데이터 저장 확인"""
    row = db_with_data.execute(
        "SELECT * FROM daily_prices WHERE ticker = '069500'"
    ).fetchone()
    assert row["close"] == 80800
    assert row["nav"] == 80647.71
    assert row["change_pct"] == 2.91


def test_upsert_returns(db_with_data):
    """returns 테이블에 수익률 저장 확인"""
    rows = db_with_data.execute(
        "SELECT * FROM returns WHERE ticker = '069500'"
    ).fetchall()
    periods = {r["period"]: r["return_pct"] for r in rows}
    assert periods["1d"] == 2.91
    assert periods["1w"] == 5.12
    assert periods["1m"] == 8.3


def test_upsert_holdings(db_with_data):
    """holdings 테이블에 보유종목 저장 확인"""
    rows = db_with_data.execute(
        "SELECT * FROM holdings WHERE ticker = '069500'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["stock_name"] == "삼성전자"
    assert rows[0]["weight"] == 31.77


def test_upsert_collection_log(db_with_data):
    """collection_log에 수집 기록 저장 확인"""
    row = db_with_data.execute(
        "SELECT * FROM collection_log WHERE date = '20260407'"
    ).fetchone()
    assert row["total_count"] == 2


def test_upsert_replace_on_conflict(db_with_data):
    """같은 날짜 데이터 재삽입 시 업데이트"""
    modified = SAMPLE_DATA.copy()
    modified["etfs"] = [SAMPLE_DATA["etfs"][0].copy()]
    modified["etfs"][0]["ohlcv"] = dict(SAMPLE_DATA["etfs"][0]["ohlcv"])
    modified["etfs"][0]["ohlcv"]["close"] = 99999

    upsert_daily_data(db_with_data, modified)

    row = db_with_data.execute(
        "SELECT close FROM daily_prices WHERE ticker = '069500' AND date = '20260407'"
    ).fetchone()
    assert row["close"] == 99999


def test_upsert_empty_data(db):
    """빈 데이터 처리"""
    count = upsert_daily_data(db, {"metadata": {}, "etfs": []})
    assert count == 0


# ── 읽기 테스트 ──────────────────────────────────────────────

def test_get_latest_date(db_with_data):
    """최신 날짜 반환"""
    assert get_latest_date(db_with_data) == "20260407"


def test_get_latest_date_empty(db):
    """빈 DB에서 None 반환"""
    assert get_latest_date(db) is None


def test_get_latest_data(db_with_data):
    """loader 호환 포맷으로 데이터 반환"""
    data = get_latest_data(db_with_data)
    assert len(data) == 2

    # 거래대금 내림차순 정렬 확인
    assert data[0]["ticker"] == "069500"  # 거래대금 더 큼
    assert data[0]["close"] == 80800
    assert data[0]["nav"] == 80647.71
    assert data[0]["change_pct"] == 2.91
    assert data[0]["returns"]["1d"] == 2.91
    assert len(data[0]["holdings"]) == 1
    assert data[0]["holdings"][0]["stock_name"] == "삼성전자"


def test_get_latest_data_empty(db):
    """빈 DB에서 빈 리스트 반환"""
    assert get_latest_data(db) == []


def test_get_latest_data_specific_date(db_with_data):
    """특정 날짜 데이터 조회"""
    upsert_daily_data(db_with_data, SAMPLE_DATA_DAY2)

    # 20260408 날짜 조회
    data = get_latest_data(db_with_data, date="20260408")
    assert len(data) == 1
    assert data[0]["close"] == 81100


def test_get_historical_prices(db_with_data):
    """시계열 가격 조회"""
    upsert_daily_data(db_with_data, SAMPLE_DATA_DAY2)

    prices = get_historical_prices(
        db_with_data, "069500", "20260407", "20260408"
    )
    assert len(prices) == 2
    assert prices[0]["date"] == "20260407"  # 오름차순
    assert prices[1]["date"] == "20260408"
    assert prices[0]["close"] == 80800
    assert prices[1]["close"] == 81100


def test_search_instruments(db_with_data):
    """종목 검색"""
    results = search_instruments(db_with_data, keyword="KODEX")
    assert len(results) == 2

    results = search_instruments(db_with_data, keyword="반도체")
    assert len(results) == 1
    assert results[0]["ticker"] == "091160"


def test_search_instruments_by_ticker(db_with_data):
    """티커로 검색"""
    results = search_instruments(db_with_data, keyword="069500")
    assert len(results) == 1


# ── 유지보수 테스트 ──────────────────────────────────────────

def test_prune_old_data_preserves_prices(db_with_data):
    """daily_prices/returns/stock_fundamentals는 영구 보존"""
    prune_old_data(db_with_data, retention_days=0)

    # daily_prices는 삭제되지 않음 (KRX 슬라이딩 윈도우로 재수집 불가)
    assert db_with_data.execute(
        "SELECT COUNT(*) FROM daily_prices"
    ).fetchone()[0] == 2  # 영구 보존


def test_prune_deletes_old_holdings(db_with_data):
    """holdings만 1년 기준으로 삭제"""
    # 2년 전 holdings 데이터 삽입
    db_with_data.execute(
        "INSERT INTO holdings (ticker, stock_ticker, stock_name, weight, date) "
        "VALUES (?, ?, ?, ?, ?)",
        ("069500", "005930", "삼성전자", 30.0, "20240101"),
    )
    db_with_data.commit()

    prune_old_data(db_with_data)

    # 오래된 holdings는 삭제
    assert db_with_data.execute(
        "SELECT COUNT(*) FROM holdings WHERE date = '20240101'"
    ).fetchone()[0] == 0


def test_get_db_stats(db_with_data):
    """DB 통계"""
    stats = get_db_stats(db_with_data)
    assert stats["instruments"] == 2
    assert stats["daily_prices"] == 2
    assert stats["latest_date"] == "20260407"


def test_import_json_file(db, tmp_path):
    """JSON 파일 import"""
    import json
    json_path = tmp_path / "etf_data_20260407.json"
    with open(json_path, "w") as f:
        json.dump(SAMPLE_DATA, f)

    count = import_json_file(db, json_path)
    assert count == 2
    assert get_latest_date(db) == "20260407"
