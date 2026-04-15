"""섹터 분석 도구 + 역인덱스 + 업종 인덱스 테스트"""

import pytest

from src.llm.tools import (
    _build_holdings_reverse_index,
    _build_sector_index,
    _format_sector_analysis,
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
    assert "데이터가 없습니다" in result


# ── ALL_TOOLS 확인 ────────────────────────────────────────

def test_all_tools_includes_sector():
    """ALL_TOOLS에 analyze_sector 포함 (8개)"""
    assert len(ALL_TOOLS) == 13
    names = [t.name for t in ALL_TOOLS]
    assert "analyze_sector" in names


# ── 업종 인덱스 테스트 ──────────────────────────────────────

SAMPLE_STOCK_DATA = [
    {"name": "삼성전자", "ticker": "005930", "sector": "전기·전자",
     "close": 55000, "change_pct": 1.5, "market_cap": 350_0000_0000_0000,
     "trade_value": 500_0000_0000, "per": 12.5, "pbr": 1.2, "eps": 4400, "div": 2.1},
    {"name": "SK하이닉스", "ticker": "000660", "sector": "전기·전자",
     "close": 180000, "change_pct": -0.8, "market_cap": 130_0000_0000_0000,
     "trade_value": 540_0000_0000, "per": 8.3, "pbr": 1.8, "eps": 21700, "div": 1.5},
    {"name": "LG전자", "ticker": "066570", "sector": "전기·전자",
     "close": 85000, "change_pct": 0.5, "market_cap": 14_0000_0000_0000,
     "trade_value": 50_0000_0000, "per": 15.0, "pbr": 0.8, "eps": 5600, "div": 1.8},
    {"name": "KB금융", "ticker": "105560", "sector": "기타금융",
     "close": 80000, "change_pct": 2.0, "market_cap": 30_0000_0000_0000,
     "trade_value": 200_0000_0000, "per": 6.0, "pbr": 0.5, "eps": 13000, "div": 4.5},
    {"name": "신한지주", "ticker": "055550", "sector": "기타금융",
     "close": 50000, "change_pct": 1.0, "market_cap": 25_0000_0000_0000,
     "trade_value": 150_0000_0000, "per": 5.5, "pbr": 0.4, "eps": 9000, "div": 5.0},
    {"name": "현대차", "ticker": "005380", "sector": "운송장비·부품",
     "close": 200000, "change_pct": -1.0, "market_cap": 42_0000_0000_0000,
     "trade_value": 300_0000_0000, "per": 7.0, "pbr": 0.9, "eps": 28500, "div": 3.0},
]


def test_build_sector_index_basic():
    """업종 인덱스에 전기·전자 업종 3종목"""
    idx = _build_sector_index(SAMPLE_STOCK_DATA)
    assert "전기·전자" in idx
    assert len(idx["전기·전자"]) == 3


def test_build_sector_index_sorted_by_market_cap():
    """업종 내 종목이 시가총액 기준 내림차순"""
    idx = _build_sector_index(SAMPLE_STOCK_DATA)
    caps = [s["market_cap"] for s in idx["전기·전자"]]
    assert caps == sorted(caps, reverse=True)


def test_build_sector_index_empty_sector():
    """업종 없는 종목은 인덱스에 포함 안됨"""
    data = [{"name": "X", "ticker": "999999", "sector": ""}]
    idx = _build_sector_index(data)
    assert len(idx) == 0


def test_build_sector_index_per_values():
    """업종 인덱스에 PER 값이 올바르게 저장"""
    idx = _build_sector_index(SAMPLE_STOCK_DATA)
    samsung = [s for s in idx["전기·전자"] if s["ticker"] == "005930"][0]
    assert samsung["per"] == 12.5


def test_analyze_sector_by_sector_name():
    """업종명으로 업종 분석 — '전기·전자'"""
    set_retriever(None, [], etf_data=SAMPLE_ETF_DATA, stock_data=SAMPLE_STOCK_DATA)
    result = analyze_sector.invoke({"query": "전기·전자"})
    assert "업종 분석" in result
    assert "삼성전자" in result
    assert "SK하이닉스" in result
    assert "PER" in result


def test_analyze_sector_partial_sector_match():
    """업종 부분 매칭 — '전기'로 '전기·전자' 매칭"""
    set_retriever(None, [], etf_data=SAMPLE_ETF_DATA, stock_data=SAMPLE_STOCK_DATA)
    result = analyze_sector.invoke({"query": "전기"})
    assert "전기·전자" in result
    assert "업종 분석" in result


def test_analyze_sector_finance_sector():
    """기타금융 업종 분석"""
    set_retriever(None, [], etf_data=SAMPLE_ETF_DATA, stock_data=SAMPLE_STOCK_DATA)
    result = analyze_sector.invoke({"query": "기타금융"})
    assert "KB금융" in result
    assert "신한지주" in result


def test_analyze_sector_valuation_stats():
    """업종 밸류에이션 통계 출력 확인"""
    set_retriever(None, [], etf_data=SAMPLE_ETF_DATA, stock_data=SAMPLE_STOCK_DATA)
    result = analyze_sector.invoke({"query": "전기·전자"})
    assert "평균" in result
    assert "배당수익률" in result
    assert "시가총액 합계" in result


def test_format_sector_analysis():
    """_format_sector_analysis 직접 테스트"""
    stocks = [_build_sector_index(SAMPLE_STOCK_DATA)["전기·전자"][i] for i in range(3)]
    result = _format_sector_analysis("전기·전자", stocks)
    assert "[전기·전자]" in result
    assert "3종목" in result
    assert "시가총액 상위" in result


def test_analyze_sector_stock_with_sector_info():
    """종목 검색 시 업종 정보도 함께 표시"""
    set_retriever(None, [], etf_data=SAMPLE_ETF_DATA, stock_data=SAMPLE_STOCK_DATA)
    result = analyze_sector.invoke({"query": "삼성전자"})
    assert "전기·전자" in result
    assert "동일 업종" in result
