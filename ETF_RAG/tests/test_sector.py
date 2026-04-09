"""섹터 분석 도구 + 역인덱스 테스트"""

import pytest

from src.llm.tools import (
    _build_holdings_reverse_index,
    analyze_sector,
    set_retriever,
    _holdings_reverse_index,
    ALL_TOOLS,
)


# ── 테스트 데이터 ─────────────────────────────────────────

SAMPLE_ETF_DATA = [
    {
        "name": "KODEX 200",
        "ticker": "069500",
        "close": 80800,
        "nav": 80647,
        "change_pct": 2.91,
        "volume": 14000000,
        "trade_value": 1000000000000,
        "holdings": [
            {"stock_ticker": "005930", "stock_name": "삼성전자",
             "shares": 913, "amount": 176300300, "weight": 29.29},
            {"stock_ticker": "000660", "stock_name": "SK하이닉스",
             "shares": 117, "amount": 103662000, "weight": 17.49},
            {"stock_ticker": "005380", "stock_name": "현대차",
             "shares": 29, "amount": 13601000, "weight": 2.24},
        ],
    },
    {
        "name": "TIGER 200",
        "ticker": "102110",
        "close": 81000,
        "nav": 80900,
        "change_pct": 1.50,
        "volume": 5000000,
        "trade_value": 500000000000,
        "holdings": [
            {"stock_ticker": "005930", "stock_name": "삼성전자",
             "shares": 800, "amount": 154400000, "weight": 28.50},
            {"stock_ticker": "000660", "stock_name": "SK하이닉스",
             "shares": 100, "amount": 88600000, "weight": 16.80},
        ],
    },
    {
        "name": "KODEX 반도체",
        "ticker": "091160",
        "close": 35000,
        "nav": 34800,
        "change_pct": 3.10,
        "volume": 2000000,
        "trade_value": 70000000000,
        "holdings": [
            {"stock_ticker": "005930", "stock_name": "삼성전자",
             "shares": 500, "amount": 96500000, "weight": 35.00},
            {"stock_ticker": "000660", "stock_name": "SK하이닉스",
             "shares": 200, "amount": 177200000, "weight": 40.00},
        ],
    },
]


@pytest.fixture(autouse=True)
def setup_sector_data():
    """테스트 전 샘플 데이터로 인덱스 초기화"""
    set_retriever(None, [], etf_data=SAMPLE_ETF_DATA, stock_data=[])
    yield
    set_retriever(None, [], etf_data=[], stock_data=[])


# ── 역인덱스 구축 테스트 ──────────────────────────────────

def test_build_reverse_index_basic():
    """역인덱스에 삼성전자(3개 ETF)가 있는지"""
    idx = _build_holdings_reverse_index(SAMPLE_ETF_DATA)
    assert "005930" in idx
    assert len(idx["005930"]) == 3  # 3개 ETF에 편입


def test_build_reverse_index_by_name():
    """종목명(소문자)으로도 조회 가능"""
    idx = _build_holdings_reverse_index(SAMPLE_ETF_DATA)
    assert "삼성전자" in idx


def test_build_reverse_index_empty():
    """보유종목 없는 데이터"""
    idx = _build_holdings_reverse_index([{"name": "KODEX X", "ticker": "999999"}])
    assert len(idx) == 0


def test_build_reverse_index_weight():
    """비중 값이 올바르게 저장되는지"""
    idx = _build_holdings_reverse_index(SAMPLE_ETF_DATA)
    kodex_entry = [e for e in idx["005930"] if e["etf_ticker"] == "069500"][0]
    assert kodex_entry["weight"] == 29.29


# ── analyze_sector 도구 테스트 ────────────────────────────

def test_analyze_sector_exact_ticker():
    """티커로 정확 매칭"""
    result = analyze_sector.invoke({"query": "005930"})
    assert "삼성전자" in result
    assert "3개" in result  # 3개 ETF
    assert "KODEX 200" in result
    assert "TIGER 200" in result


def test_analyze_sector_exact_name():
    """종목명으로 정확 매칭"""
    result = analyze_sector.invoke({"query": "삼성전자"})
    assert "3개" in result
    assert "비중" in result


def test_analyze_sector_sorted_by_weight():
    """비중 높은 순으로 정렬"""
    result = analyze_sector.invoke({"query": "005930"})
    lines = result.split("\n")
    # 첫 번째 ETF가 KODEX 반도체(35.00%)여야 함
    etf_lines = [l for l in lines if l.startswith("- [")]
    assert "KODEX 반도체" in etf_lines[0]
    assert "35.00%" in etf_lines[0]


def test_analyze_sector_sk_hynix():
    """SK하이닉스 — 3개 ETF에 편입"""
    result = analyze_sector.invoke({"query": "SK하이닉스"})
    assert "3개" in result
    assert "통계" in result


def test_analyze_sector_hyundai():
    """현대차 — 1개 ETF에만 편입"""
    result = analyze_sector.invoke({"query": "현대차"})
    assert "1개" in result
    assert "KODEX 200" in result


def test_analyze_sector_not_found():
    """없는 종목"""
    result = analyze_sector.invoke({"query": "테슬라"})
    assert "찾지 못했습니다" in result


def test_analyze_sector_keyword_partial():
    """부분 매칭 — '하이닉스'로 검색"""
    result = analyze_sector.invoke({"query": "하이닉스"})
    assert "SK하이닉스" in result


def test_analyze_sector_statistics():
    """통계 정보 (평균 비중, 최대 비중)"""
    result = analyze_sector.invoke({"query": "005930"})
    assert "평균 비중" in result
    assert "최대 비중" in result


def test_analyze_sector_no_holdings_data():
    """보유종목 데이터 없을 때"""
    set_retriever(None, [], etf_data=[], stock_data=[])
    result = analyze_sector.invoke({"query": "삼성전자"})
    assert "보유종목 데이터가 없습니다" in result


# ── ALL_TOOLS 확인 ────────────────────────────────────────

def test_all_tools_includes_sector():
    """ALL_TOOLS에 analyze_sector 포함 (8개)"""
    assert len(ALL_TOOLS) == 8
    names = [t.name for t in ALL_TOOLS]
    assert "analyze_sector" in names
