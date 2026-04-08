"""ETF 데이터 로더 테스트

수집 데이터와 하드코딩 데이터 모두 테스트합니다.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

# DB를 무시하는 공통 패치 (실제 DB가 있으면 테스트가 DB 데이터를 로드하므로)
_FAKE_DB = Path("/tmp/nonexistent_test.db")


# ── 수집 데이터 샘플 ─────────────────────────────────────────

SAMPLE_COLLECTED = {
    "metadata": {
        "collection_date": "20260406",
        "collected_at": "2026-04-06T19:00:00",
        "total_etfs": 2,
        "holdings_collected": 1,
    },
    "etfs": [
        {
            "ticker": "069500",
            "name": "KODEX 200",
            "date": "20260406",
            "ohlcv": {
                "open": 80210, "high": 81200, "low": 80100, "close": 80800,
                "volume": 14703488, "trade_value": 1184866376189,
                "nav": 80647.71, "base_index": 798.32,
                "change": 735, "change_pct": 2.91,
            },
            "returns": {"1d": 2.91, "1w": 5.12, "1m": -1.34, "3m": 3.21, "1y": 8.55},
            "deviation": -0.17,
            "tracking_error": 0.05,
            "holdings": [
                {"stock_ticker": "005930", "stock_name": "삼성전자",
                 "shares": 8140.0, "amount": 667480000, "weight": 31.77},
            ],
        },
        {
            "ticker": "153130",
            "name": "KODEX 단기채권",
            "date": "20260406",
            "ohlcv": {
                "open": 102350, "high": 102360, "low": 102340, "close": 102350,
                "volume": 500, "trade_value": 511750000,
                "nav": 102348.5, "base_index": 100.0,
                "change": 10, "change_pct": 0.01,
            },
            "returns": {},
            "deviation": 0.0,
            "tracking_error": 0.01,
            "holdings": [],
        },
    ],
}


@pytest.fixture
def collected_file(tmp_path):
    """임시 수집 데이터 파일 생성"""
    filepath = tmp_path / "etf_data_20260406.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_COLLECTED, f, ensure_ascii=False)
    return filepath


# ── 수집 데이터 로드 테스트 ──────────────────────────────────

def test_load_collected_data(collected_file):
    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=collected_file):
        from src.data.loader import load_etf_data
        data = load_etf_data()

    assert len(data) == 2
    assert data[0]["ticker"] == "069500"
    assert data[0]["close"] == 80800
    assert data[0]["nav"] == 80647.71
    assert data[0]["deviation"] == -0.17
    assert len(data[0]["holdings"]) == 1


def test_collected_documents(collected_file):
    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=collected_file):
        from src.data.loader import load_etf_data, create_documents
        data = load_etf_data()
        docs = create_documents(data)

    assert len(docs) == 2
    doc = docs[0]
    assert doc.metadata["ticker"] == "069500"
    assert doc.metadata["source"] == "krx_collected"
    assert "KODEX 200" in doc.page_content
    assert "80,800원" in doc.page_content
    assert "삼성전자" in doc.page_content


def test_collected_doc_without_holdings(collected_file):
    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=collected_file):
        from src.data.loader import load_etf_data, create_documents
        data = load_etf_data()
        docs = create_documents(data)

    doc = docs[1]  # 단기채권 (holdings 비어 있음)
    assert "정보 없음" in doc.page_content


def test_collected_doc_has_returns(collected_file):
    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=collected_file):
        from src.data.loader import load_etf_data, create_documents
        data = load_etf_data()
        docs = create_documents(data)

    doc = docs[0]  # KODEX 200 (returns 있음)
    assert "수익률:" in doc.page_content
    assert "1일:" in doc.page_content
    assert "1년:" in doc.page_content

    doc_no_returns = docs[1]  # 단기채권 (returns 비어 있음)
    assert "수익률: 정보 없음" in doc_no_returns.page_content


def test_collected_data_has_required_fields(collected_file):
    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=collected_file):
        from src.data.loader import load_etf_data
        data = load_etf_data()

    required = ["ticker", "name", "date", "close", "nav", "volume", "trade_value"]
    for etf in data:
        for field in required:
            assert field in etf, f"{etf['name']} missing field: {field}"


# ── ETF 필터링 테스트 ─────────────────────────────────────────

def test_filter_excludes_zero_close(tmp_path):
    """종가 0원 ETF는 필터링됨"""
    sample = {
        "metadata": {"collection_date": "20260406", "total_etfs": 2, "holdings_collected": 0},
        "etfs": [
            {"ticker": "069500", "name": "KODEX 200", "date": "20260406",
             "ohlcv": {"close": 80800, "trade_value": 1000000000, "open": 0, "high": 0,
                       "low": 0, "volume": 0, "nav": 0, "base_index": 0, "change": 0, "change_pct": 0},
             "returns": {}, "deviation": None, "tracking_error": None, "holdings": []},
            {"ticker": "999999", "name": "거래정지 ETF", "date": "20260406",
             "ohlcv": {"close": 0, "trade_value": 0, "open": 0, "high": 0,
                       "low": 0, "volume": 0, "nav": 0, "base_index": 0, "change": 0, "change_pct": 0},
             "returns": {}, "deviation": None, "tracking_error": None, "holdings": []},
        ],
    }
    filepath = tmp_path / "etf_data_20260406.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False)

    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=filepath):
        from src.data.loader import load_etf_data
        data = load_etf_data()

    assert len(data) == 1
    assert data[0]["ticker"] == "069500"


def test_filter_excludes_low_trade_value(tmp_path):
    """거래대금 1억 미만 ETF는 필터링됨"""
    sample = {
        "metadata": {"collection_date": "20260406", "total_etfs": 2, "holdings_collected": 0},
        "etfs": [
            {"ticker": "069500", "name": "KODEX 200", "date": "20260406",
             "ohlcv": {"close": 80800, "trade_value": 1000000000, "open": 0, "high": 0,
                       "low": 0, "volume": 0, "nav": 0, "base_index": 0, "change": 0, "change_pct": 0},
             "returns": {}, "deviation": None, "tracking_error": None, "holdings": []},
            {"ticker": "888888", "name": "비유동 ETF", "date": "20260406",
             "ohlcv": {"close": 10000, "trade_value": 50000000, "open": 0, "high": 0,
                       "low": 0, "volume": 0, "nav": 0, "base_index": 0, "change": 0, "change_pct": 0},
             "returns": {}, "deviation": None, "tracking_error": None, "holdings": []},
        ],
    }
    filepath = tmp_path / "etf_data_20260406.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False)

    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=filepath):
        from src.data.loader import load_etf_data
        data = load_etf_data()

    assert len(data) == 1
    assert data[0]["ticker"] == "069500"


# ── 하드코딩 fallback 테스트 ─────────────────────────────────

def test_fallback_to_hardcoded():
    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=None):
        from src.data.loader import load_etf_data
        data = load_etf_data()

    assert len(data) == 8
    assert "id" in data[0]
    assert "category" in data[0]


def test_hardcoded_documents():
    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=None):
        from src.data.loader import load_etf_data, create_documents
        data = load_etf_data()
        docs = create_documents(data)

    assert len(docs) == 8
    doc = docs[0]
    assert doc.metadata["source"] == "hardcoded"
    assert "ETF ID:" in doc.page_content
    assert "카테고리:" in doc.page_content


def test_hardcoded_documents_have_metadata():
    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=None):
        from src.data.loader import load_etf_data, create_documents
        data = load_etf_data()
        docs = create_documents(data)

    for doc in docs:
        assert "name" in doc.metadata
        assert "ticker" in doc.metadata
        assert len(doc.page_content) > 100


# ── config 테스트 ─────────────────────────────────────────────

def test_get_latest_collected_path_picks_newest(tmp_path):
    collected = tmp_path / "collected"
    collected.mkdir()
    (collected / "etf_data_20260403.json").touch()
    (collected / "etf_data_20260406.json").touch()
    (collected / "etf_data_20260404.json").touch()

    with patch("config.COLLECTED_DIR", collected):
        from config import get_latest_collected_path
        result = get_latest_collected_path()

    assert result is not None
    assert result.name == "etf_data_20260406.json"


def test_get_latest_collected_path_empty(tmp_path):
    collected = tmp_path / "collected"
    collected.mkdir()

    with patch("config.COLLECTED_DIR", collected):
        from config import get_latest_collected_path
        result = get_latest_collected_path()

    assert result is None


# ── 주식 데이터 로드 테스트 ──────────────────────────────────

SAMPLE_STOCK = {
    "ticker": "005930",
    "name": "삼성전자",
    "date": "20260408",
    "close": 82500,
    "volume": 15000000,
    "trade_value": 1237500000000,
    "change_pct": 1.23,
    "open": 82000, "high": 83000, "low": 81500,
    "market_cap": 492_000_000_000_000,
    "shares_outstanding": 5_969_782_550,
    "per": 12.5, "pbr": 1.65, "eps": 6600.0,
    "bps": 50000.0, "div": 2.1, "dps": 1444.0,
    "returns": {"1d": 1.23, "1w": 3.45, "1m": -2.1},
}


def test_load_stock_data_from_db(tmp_path):
    """주식 데이터 DB 로드"""
    from src.data.database import init_db, upsert_stock_data
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)

    stock_data = {
        "metadata": {"collection_date": "20260408", "total_stocks": 1, "market": "ALL"},
        "stocks": [{
            "ticker": "005930", "name": "삼성전자", "date": "20260408",
            "ohlcv": {"open": 82000, "high": 83000, "low": 81500, "close": 82500,
                      "volume": 15000000, "trade_value": 1237500000000, "change_pct": 1.23},
            "market_cap": 492_000_000_000_000,
            "shares_outstanding": 5_969_782_550,
            "fundamental": {"bps": 50000.0, "per": 12.5, "pbr": 1.65,
                           "eps": 6600.0, "div": 2.1, "dps": 1444.0},
            "returns": {"1d": 1.23},
        }],
    }
    upsert_stock_data(conn, stock_data)
    conn.close()

    with patch("src.data.loader.DB_PATH", db_path):
        from src.data.loader import load_stock_data
        data = load_stock_data()

    assert len(data) == 1
    assert data[0]["ticker"] == "005930"
    assert data[0]["per"] == 12.5


def test_load_stock_data_no_db():
    """DB 없으면 빈 리스트 반환"""
    with patch("src.data.loader.DB_PATH", _FAKE_DB):
        from src.data.loader import load_stock_data
        data = load_stock_data()
    assert data == []


def test_stock_document_creation():
    """주식 Document 변환"""
    from src.data.loader import create_stock_documents
    docs = create_stock_documents([SAMPLE_STOCK])

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["ticker"] == "005930"
    assert doc.metadata["asset_type"] == "stock"
    assert doc.metadata["source"] == "krx_collected"
    assert "삼성전자" in doc.page_content
    assert "82,500원" in doc.page_content
    assert "PER:" in doc.page_content
    assert "시가총액:" in doc.page_content


def test_stock_document_has_returns():
    """주식 Document에 수익률 포함"""
    from src.data.loader import create_stock_documents
    docs = create_stock_documents([SAMPLE_STOCK])
    doc = docs[0]
    assert "수익률:" in doc.page_content
    assert "1일:" in doc.page_content


def test_stock_document_no_returns():
    """수익률 없는 주식"""
    from src.data.loader import create_stock_documents
    stock_no_returns = {**SAMPLE_STOCK, "returns": {}}
    docs = create_stock_documents([stock_no_returns])
    assert "정보 없음" in docs[0].page_content


def test_format_market_cap():
    """시가총액 포맷팅"""
    from src.data.loader import _format_market_cap
    assert "조원" in _format_market_cap(492_000_000_000_000)
    assert "억원" in _format_market_cap(500_000_000_000)
    assert "원" in _format_market_cap(50_000_000)


def test_etf_document_has_asset_type(collected_file):
    """ETF Document에 asset_type 메타데이터 포함"""
    with patch("src.data.loader.DB_PATH", _FAKE_DB), \
         patch("src.data.loader.get_latest_collected_path", return_value=collected_file):
        from src.data.loader import load_etf_data, create_documents
        data = load_etf_data()
        docs = create_documents(data)

    assert docs[0].metadata["asset_type"] == "etf"
