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


def test_parse_stock_comparison_full_fields():
    """주식 비교 JSON — BPS/DPS/EPS 포함"""
    data = {
        "__type__": "comparison_table",
        "items": [
            {"name": "삼성전자", "close": 55000, "change_pct": 1.0,
             "trade_value": 500000000000, "asset_type": "stock",
             "per": 12.5, "pbr": 1.2, "eps": 4400, "bps": 45000,
             "market_cap": 350000000000000, "div": 2.1, "dps": 1444},
            {"name": "SK하이닉스", "close": 180000, "change_pct": -0.5,
             "trade_value": 540000000000, "asset_type": "stock",
             "per": 8.3, "pbr": 1.8, "eps": 21700, "bps": 100000,
             "market_cap": 130000000000000, "div": 1.5, "dps": 1200},
        ],
    }
    result = try_parse_comparison(json.dumps(data, ensure_ascii=False))
    assert result is not None
    assert result["items"][0]["bps"] == 45000
    assert result["items"][0]["dps"] == 1444
    assert result["items"][1]["eps"] == 21700


# ── enrichment 테스트 ──


def test_enrich_stock_with_full_data():
    """주식 enrichment에 시가총액/PBR/배당/EPS 포함 확인"""
    from src.llm.tools import _enrich_with_structured_data

    index = {
        "005930": {
            "name": "삼성전자", "ticker": "005930",
            "close": 55000, "change_pct": 1.5,
            "per": 12.5, "pbr": 1.2, "eps": 4400,
            "market_cap": 350_0000_0000_0000,
            "div": 2.1, "returns": {"1m": 5.0},
        }
    }
    sources = [{"ticker": "005930", "name": "삼성전자"}]
    result = _enrich_with_structured_data(sources, index)

    assert "PER: 12.50배" in result
    assert "PBR: 1.20배" in result
    assert "시가총액:" in result
    assert "배당수익률: 2.10%" in result
    assert "EPS: 4,400원" in result


def test_enrich_etf_no_stock_fields():
    """ETF enrichment에는 주식 필드(PBR/시가총액 등) 없음"""
    from src.llm.tools import _enrich_with_structured_data

    index = {
        "069500": {
            "name": "KODEX 200", "ticker": "069500",
            "close": 35000, "change_pct": 0.5,
            "nav": 35100, "returns": {},
        }
    }
    sources = [{"ticker": "069500", "name": "KODEX 200"}]
    result = _enrich_with_structured_data(sources, index)

    assert "NAV:" in result
    assert "PER" not in result
    assert "시가총액" not in result


# ── 재무제표 비교 필드 테스트 ──


def test_comparison_with_financial_fields():
    """비교 테이블에 재무제표 필드가 포함된 경우 파싱"""
    data = {
        "__type__": "comparison_table",
        "items": [
            {"name": "삼성전자", "ticker": "005930", "close": 55000,
             "change_pct": 1.5, "volume": 10000000, "trade_value": 500000000000,
             "per": 12.5, "pbr": 1.2, "market_cap": 350000000000000,
             "div": 2.1, "asset_type": "stock",
             "revenue": 79100000000000, "operating_profit": 6700000000000,
             "net_income": 5000000000000, "operating_margin": 8.5,
             "revenue_growth_yoy": 10.8, "op_growth_yoy": 33.2,
             "fiscal_period": "2025Q1"},
            {"name": "SK하이닉스", "ticker": "000660", "close": 180000,
             "change_pct": -0.8, "volume": 3000000, "trade_value": 540000000000,
             "per": 8.3, "pbr": 1.8, "market_cap": 130000000000000,
             "div": 1.5, "asset_type": "stock",
             "revenue": 20000000000000, "operating_profit": 7000000000000,
             "net_income": 6000000000000, "operating_margin": 35.0,
             "revenue_growth_yoy": 46.8, "op_growth_yoy": 120.5,
             "fiscal_period": "2025Q1"},
        ],
    }
    result = try_parse_comparison(json.dumps(data, ensure_ascii=False))
    assert result is not None
    assert result["items"][0]["revenue"] == 79100000000000
    assert result["items"][0]["fiscal_period"] == "2025Q1"
    assert result["items"][1]["operating_margin"] == 35.0
    assert result["items"][1]["revenue_growth_yoy"] == 46.8


def test_comparison_without_financial_fields():
    """재무제표 필드 없는 비교 테이블도 정상 파싱"""
    data = {
        "__type__": "comparison_table",
        "items": [
            {"name": "A", "close": 100, "change_pct": 1.0,
             "trade_value": 1000, "asset_type": "stock",
             "per": 10.0, "pbr": 1.0, "market_cap": 1000000000000, "div": 2.0},
            {"name": "B", "close": 200, "change_pct": -1.0,
             "trade_value": 2000, "asset_type": "stock",
             "per": 15.0, "pbr": 2.0, "market_cap": 2000000000000, "div": 1.0},
        ],
    }
    result = try_parse_comparison(json.dumps(data, ensure_ascii=False))
    assert result is not None
    assert result["items"][0].get("revenue") is None
