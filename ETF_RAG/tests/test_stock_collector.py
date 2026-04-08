"""주식 수집기 + DB 저장/읽기 테스트

pykrx API 호출 없이 수집 로직과 DB CRUD를 검증합니다.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.data.database import (
    init_db,
    upsert_stock_data,
    get_latest_stock_data,
    get_db_stats,
    prune_old_data,
)


# ── 테스트 데이터 ────────────────────────────────────────────

SAMPLE_STOCK_DATA = {
    "metadata": {
        "collection_date": "20260408",
        "collected_at": "2026-04-08T19:00:00",
        "total_stocks": 3,
        "market": "ALL",
        "source": "pykrx",
    },
    "stocks": [
        {
            "ticker": "005930",
            "name": "삼성전자",
            "date": "20260408",
            "ohlcv": {
                "open": 82000, "high": 83000, "low": 81500, "close": 82500,
                "volume": 15000000, "trade_value": 1237500000000,
                "change_pct": 1.23,
            },
            "market_cap": 492_000_000_000_000,
            "shares_outstanding": 5_969_782_550,
            "fundamental": {
                "bps": 50000.0, "per": 12.5, "pbr": 1.65,
                "eps": 6600.0, "div": 2.1, "dps": 1444.0,
            },
            "returns": {"1d": 1.23, "1w": 3.45, "1m": -2.1, "3m": 5.0, "1y": 10.5},
        },
        {
            "ticker": "000660",
            "name": "SK하이닉스",
            "date": "20260408",
            "ohlcv": {
                "open": 180000, "high": 185000, "low": 179000, "close": 183000,
                "volume": 5000000, "trade_value": 915000000000,
                "change_pct": 2.5,
            },
            "market_cap": 133_000_000_000_000,
            "shares_outstanding": 728_002_365,
            "fundamental": {
                "bps": 120000.0, "per": 8.2, "pbr": 1.52,
                "eps": 22317.0, "div": 1.0, "dps": 1200.0,
            },
            "returns": {"1d": 2.5, "1w": 5.0},
        },
        {
            "ticker": "373220",
            "name": "LG에너지솔루션",
            "date": "20260408",
            "ohlcv": {
                "open": 350000, "high": 352000, "low": 348000, "close": 350000,
                "volume": 500000, "trade_value": 175000000000,
                "change_pct": 0.0,
            },
            "market_cap": 82_000_000_000_000,
            "shares_outstanding": 234_000_000,
            "fundamental": {
                "bps": 150000.0, "per": 0.0, "pbr": 2.33,
                "eps": 0.0, "div": 0.0, "dps": 0.0,
            },
            "returns": {},
        },
    ],
}


@pytest.fixture
def stock_db(tmp_path):
    """테스트용 주식 DB"""
    db_path = tmp_path / "test_stock.db"
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture
def stock_db_with_data(stock_db):
    """데이터가 들어간 주식 DB"""
    upsert_stock_data(stock_db, SAMPLE_STOCK_DATA)
    return stock_db


@pytest.fixture
def stock_json_file(tmp_path):
    """임시 주식 수집 JSON 파일"""
    filepath = tmp_path / "stock_data_20260408.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_STOCK_DATA, f, ensure_ascii=False)
    return filepath


# ── DB 쓰기 테스트 ────────────────────────────────────────────

def test_upsert_stock_data(stock_db):
    count = upsert_stock_data(stock_db, SAMPLE_STOCK_DATA)
    assert count == 3


def test_upsert_stock_instruments(stock_db_with_data):
    rows = stock_db_with_data.execute(
        "SELECT * FROM instruments WHERE type = 'stock'"
    ).fetchall()
    assert len(rows) == 3
    names = {r["name"] for r in rows}
    assert "삼성전자" in names
    assert "SK하이닉스" in names


def test_upsert_stock_prices(stock_db_with_data):
    rows = stock_db_with_data.execute(
        "SELECT * FROM daily_prices WHERE ticker = '005930'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["close"] == 82500
    assert rows[0]["trade_value"] == 1237500000000


def test_upsert_stock_fundamentals(stock_db_with_data):
    rows = stock_db_with_data.execute(
        "SELECT * FROM stock_fundamentals WHERE ticker = '005930'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["per"] == 12.5
    assert rows[0]["pbr"] == 1.65
    assert rows[0]["market_cap"] == 492_000_000_000_000


def test_upsert_stock_returns(stock_db_with_data):
    rows = stock_db_with_data.execute(
        "SELECT * FROM returns WHERE ticker = '005930' ORDER BY period"
    ).fetchall()
    periods = {r["period"]: r["return_pct"] for r in rows}
    assert periods["1d"] == 1.23
    assert periods["1y"] == 10.5
    assert len(periods) == 5


def test_upsert_stock_collection_log(stock_db_with_data):
    row = stock_db_with_data.execute(
        "SELECT * FROM collection_log WHERE source = 'pykrx_stock'"
    ).fetchone()
    assert row is not None
    assert row["total_count"] == 3


def test_upsert_stock_empty(stock_db):
    count = upsert_stock_data(stock_db, {"metadata": {}, "stocks": []})
    assert count == 0


def test_upsert_stock_replace_on_conflict(stock_db):
    """동일 (ticker, date) 재삽입 시 덮어쓰기"""
    upsert_stock_data(stock_db, SAMPLE_STOCK_DATA)

    # 가격 변경한 데이터로 재삽입
    modified = json.loads(json.dumps(SAMPLE_STOCK_DATA))
    modified["stocks"][0]["ohlcv"]["close"] = 99999
    upsert_stock_data(stock_db, modified)

    row = stock_db.execute(
        "SELECT close FROM daily_prices WHERE ticker = '005930'"
    ).fetchone()
    assert row["close"] == 99999


# ── DB 읽기 테스트 ────────────────────────────────────────────

def test_get_latest_stock_data(stock_db_with_data):
    data = get_latest_stock_data(stock_db_with_data)
    assert len(data) == 3
    # 거래대금 내림차순 정렬
    assert data[0]["ticker"] == "005930"


def test_get_latest_stock_data_has_fundamentals(stock_db_with_data):
    data = get_latest_stock_data(stock_db_with_data)
    samsung = data[0]
    assert samsung["per"] == 12.5
    assert samsung["pbr"] == 1.65
    assert samsung["market_cap"] == 492_000_000_000_000
    assert samsung["eps"] == 6600.0


def test_get_latest_stock_data_has_returns(stock_db_with_data):
    data = get_latest_stock_data(stock_db_with_data)
    samsung = data[0]
    assert samsung["returns"]["1d"] == 1.23
    assert samsung["returns"]["1y"] == 10.5

    # LG에너지솔루션은 수익률 없음
    lg = data[2]
    assert lg["returns"] == {}


def test_get_latest_stock_data_empty(stock_db):
    data = get_latest_stock_data(stock_db)
    assert data == []


def test_get_latest_stock_data_specific_date(stock_db_with_data):
    data = get_latest_stock_data(stock_db_with_data, date="20260408")
    assert len(data) == 3


def test_get_latest_stock_data_wrong_date(stock_db_with_data):
    data = get_latest_stock_data(stock_db_with_data, date="20260401")
    assert data == []


# ── ETF/주식 분리 테스트 ──────────────────────────────────────

def test_etf_stock_separate(stock_db):
    """ETF와 주식이 type으로 분리됨"""
    from src.data.database import upsert_daily_data, get_latest_data

    # ETF 데이터 삽입
    etf_data = {
        "metadata": {"collection_date": "20260408", "total_etfs": 1, "holdings_collected": 0},
        "etfs": [{
            "ticker": "069500", "name": "KODEX 200", "date": "20260408",
            "ohlcv": {"close": 80000, "trade_value": 1000000000,
                      "open": 0, "high": 0, "low": 0, "volume": 0,
                      "nav": 0, "base_index": 0, "change": 0, "change_pct": 0},
            "returns": {}, "deviation": None, "tracking_error": None, "holdings": [],
        }],
    }
    upsert_daily_data(stock_db, etf_data)

    # 주식 데이터 삽입
    upsert_stock_data(stock_db, SAMPLE_STOCK_DATA)

    # ETF만 조회
    etfs = get_latest_data(stock_db, inst_type="etf")
    assert len(etfs) == 1
    assert etfs[0]["ticker"] == "069500"

    # 주식만 조회
    stocks = get_latest_stock_data(stock_db)
    assert len(stocks) == 3
    assert all(s["ticker"] != "069500" for s in stocks)


# ── DB 통계 테스트 ────────────────────────────────────────────

def test_db_stats_includes_fundamentals(stock_db_with_data):
    stats = get_db_stats(stock_db_with_data)
    assert "stock_fundamentals" in stats
    assert stats["stock_fundamentals"] == 3


# ── 수집기 로직 테스트 (pykrx mock) ───────────────────────────

def test_collect_bulk_ohlcv_structure():
    """collect_bulk_ohlcv 반환 구조 확인"""
    from src.data.stock_collector import collect_bulk_ohlcv

    mock_df = pd.DataFrame({
        "시가": [82000],
        "고가": [83000],
        "저가": [81500],
        "종가": [82500],
        "거래량": [15000000],
        "거래대금": [1237500000000],
        "등락률": [1.23],
    }, index=["005930"])

    with patch("src.data.stock_collector.stock.get_market_ohlcv_by_ticker", return_value=mock_df):
        result = collect_bulk_ohlcv("20260408", "ALL")

    assert "005930" in result
    assert result["005930"]["close"] == 82500
    assert result["005930"]["change_pct"] == 1.23


def test_collect_bulk_market_cap_structure():
    """collect_bulk_market_cap 반환 구조 확인"""
    from src.data.stock_collector import collect_bulk_market_cap

    mock_df = pd.DataFrame({
        "종가": [82500],
        "시가총액": [492000000000000],
        "거래량": [15000000],
        "거래대금": [1237500000000],
        "상장주식수": [5969782550],
    }, index=["005930"])

    with patch("src.data.stock_collector.stock.get_market_cap_by_ticker", return_value=mock_df):
        result = collect_bulk_market_cap("20260408", "ALL")

    assert result["005930"]["market_cap"] == 492000000000000
    assert result["005930"]["shares_outstanding"] == 5969782550


def test_collect_bulk_fundamental_structure():
    """collect_bulk_fundamental 반환 구조 확인"""
    from src.data.stock_collector import collect_bulk_fundamental

    mock_df = pd.DataFrame({
        "BPS": [50000.0],
        "PER": [12.5],
        "PBR": [1.65],
        "EPS": [6600.0],
        "DIV": [2.1],
        "DPS": [1444.0],
    }, index=["005930"])

    with patch("src.data.stock_collector.stock.get_market_fundamental_by_ticker", return_value=mock_df):
        result = collect_bulk_fundamental("20260408", "KOSPI")

    assert result["005930"]["per"] == 12.5
    assert result["005930"]["pbr"] == 1.65
    assert result["005930"]["eps"] == 6600.0


def test_validate_result_clean():
    """정상 데이터는 이슈 없음"""
    from src.data.stock_collector import validate_result
    issues = validate_result(SAMPLE_STOCK_DATA)
    assert issues == []


def test_validate_result_count_mismatch():
    """메타데이터와 실제 수 불일치"""
    from src.data.stock_collector import validate_result
    data = json.loads(json.dumps(SAMPLE_STOCK_DATA))
    data["metadata"]["total_stocks"] = 99
    issues = validate_result(data)
    assert any("불일치" in i for i in issues)


def test_save_result(tmp_path):
    """JSON 저장 확인"""
    from src.data.stock_collector import save_result
    filepath = save_result(SAMPLE_STOCK_DATA, output_dir=tmp_path)
    assert filepath.exists()
    assert filepath.name == "stock_data_20260408.json"

    with open(filepath, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["metadata"]["total_stocks"] == 3
