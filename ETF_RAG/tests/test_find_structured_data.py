"""_find_structured_data 종목 해석 테스트 (2026-06-24).

실버그: 일부 신형 ETF 티커는 영문 대문자 포함(0162Z0, 0192L0)인데, 인덱스는
티커를 원본 대소문자로 키에 넣고 조회는 .lower()로 해서 대문자 티커 조회가
None이 되던 문제(기술분석/전망 "데이터 없음"의 원인). 인덱스 키도 .lower() 통일로 수정.
"""

from src.llm.tools import _find_structured_data, set_retriever

_ETF = [
    {"name": "KODEX 200", "ticker": "069500", "close": 80800, "nav": 80647},
    {"name": "RISE 삼성전자SK하이닉스채권혼합50", "ticker": "0162Z0",
     "close": 10765, "nav": 10760},
]
_STOCK = [
    {"name": "삼성전자", "ticker": "005930", "close": 70000},
]


def _setup():
    set_retriever(None, [], etf_data=_ETF, stock_data=_STOCK)


def test_lookup_by_uppercase_ticker():
    """영문 대문자 포함 티커도 조회돼야 한다(버그 케이스)."""
    _setup()
    d = _find_structured_data("0162Z0")
    assert d is not None
    assert d["ticker"] == "0162Z0"


def test_lookup_by_lowercase_ticker_input():
    """사용자가 소문자로 입력해도 동일하게 조회."""
    _setup()
    d = _find_structured_data("0162z0")
    assert d is not None and d["ticker"] == "0162Z0"


def test_lookup_by_numeric_ticker_still_works():
    """기존 숫자 티커 회귀 없음."""
    _setup()
    assert _find_structured_data("069500")["ticker"] == "069500"
    assert _find_structured_data("005930")["ticker"] == "005930"


def test_lookup_by_name():
    """이름 조회 회귀 없음."""
    _setup()
    assert _find_structured_data("삼성전자")["ticker"] == "005930"
    assert _find_structured_data("RISE 삼성전자SK하이닉스채권혼합50")["ticker"] == "0162Z0"


def test_lookup_unknown_returns_none():
    _setup()
    assert _find_structured_data("ZZZZ999") is None
