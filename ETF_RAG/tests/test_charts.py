"""비교 차트/테이블 렌더링 테스트"""

import json

from src.ui.charts import try_parse_comparison, _format_value, _format_market_cap, _fmt_pct


# ── try_parse_comparison 테스트 ──


def test_parse_valid_comparison():
    """유효한 comparison_table JSON 파싱"""
    data = {
        "__type__": "comparison_table",
        "items": [{"name": "A", "close": 100}, {"name": "B", "close": 200}],
    }
    raw = json.dumps(data, ensure_ascii=False) + "\n\n---\n\nA vs B 텍스트"
    result = try_parse_comparison(raw)
    assert result is not None
    assert result["__type__"] == "comparison_table"
    assert len(result["items"]) == 2


def test_parse_json_only():
    """--- 없이 JSON만 있는 경우"""
    data = {
        "__type__": "comparison_table",
        "items": [{"name": "A"}, {"name": "B"}],
    }
    raw = json.dumps(data, ensure_ascii=False)
    result = try_parse_comparison(raw)
    assert result is not None


def test_parse_invalid_json():
    """잘못된 JSON은 None 반환"""
    assert try_parse_comparison("not json at all") is None


def test_parse_wrong_type():
    """__type__이 다르면 None 반환"""
    data = {"__type__": "other", "items": []}
    raw = json.dumps(data)
    assert try_parse_comparison(raw) is None


def test_parse_no_items():
    """items가 비어있으면 None 반환"""
    data = {"__type__": "comparison_table", "items": []}
    raw = json.dumps(data)
    assert try_parse_comparison(raw) is None


# ── 포맷 유틸 테스트 ──


def test_format_value_trillion():
    assert _format_value(1_500_000_000_000) == "1.5조"


def test_format_value_billion():
    assert _format_value(500_000_000) == "5억"


def test_format_value_small():
    assert _format_value(50_000) == "5만"


def test_format_market_cap_trillion():
    assert _format_market_cap(3_200_000_000_000) == "3.2조원"


def test_format_market_cap_billion():
    assert _format_market_cap(8_000_000_000) == "80억원"


def test_fmt_pct_none():
    assert _fmt_pct(None) == "-"


def test_fmt_pct_positive():
    assert _fmt_pct(2.5) == "+2.50%"


def test_fmt_pct_negative():
    assert _fmt_pct(-1.23) == "-1.23%"


# ── 주식 비교 파싱 테스트 ──


def test_parse_stock_comparison():
    """주식 비교 JSON 파싱 — asset_type=stock"""
    data = {
        "__type__": "comparison_table",
        "items": [
            {"name": "삼성전자", "ticker": "005930", "close": 55000,
             "change_pct": 1.5, "volume": 10000000, "trade_value": 500000000000,
             "per": 12.5, "pbr": 1.2, "market_cap": 350000000000000,
             "div": 2.1, "asset_type": "stock"},
            {"name": "SK하이닉스", "ticker": "000660", "close": 180000,
             "change_pct": -0.8, "volume": 3000000, "trade_value": 540000000000,
             "per": 8.3, "pbr": 1.8, "market_cap": 130000000000000,
             "div": 1.5, "asset_type": "stock"},
        ],
    }
    raw = json.dumps(data, ensure_ascii=False) + "\n\n---\n\n텍스트"
    result = try_parse_comparison(raw)
    assert result is not None
    assert result["items"][0]["asset_type"] == "stock"
    assert result["items"][0]["per"] == 12.5


def test_parse_stock_comparison_with_returns():
    """주식 비교 JSON — 수익률 포함"""
    data = {
        "__type__": "comparison_table",
        "items": [
            {"name": "A", "close": 100, "change_pct": 1.0,
             "trade_value": 1000, "asset_type": "stock",
             "per": 10.0, "pbr": 1.0, "market_cap": 1000000000000, "div": 2.0,
             "return_1d": 1.0, "return_1m": 5.0},
            {"name": "B", "close": 200, "change_pct": -1.0,
             "trade_value": 2000, "asset_type": "stock",
             "per": 15.0, "pbr": 2.0, "market_cap": 2000000000000, "div": 1.0,
             "return_1d": -1.0, "return_1m": -3.0},
        ],
    }
    result = try_parse_comparison(json.dumps(data, ensure_ascii=False))
    assert result is not None
    assert result["items"][0]["return_1d"] == 1.0
    assert result["items"][1]["return_1m"] == -3.0
