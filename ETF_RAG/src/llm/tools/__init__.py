"""
ETF/주식 RAG 도구 패키지 — LangGraph Function Calling용

기존 tools.py의 단일 파일을 아래 모듈로 분할:
- _state.py   : 공유 상태 (retriever, 인덱스, set_retriever)
- _helpers.py : 공통 헬퍼 (종목 조회, enrichment, 비교 필드)
- search.py   : search_etf, search_stock, get_etf_list, get_stock_list
- compare.py  : compare_etfs, compare_stocks
- analysis.py : get_realtime_price, analyze_sector, get_technical_indicators
- portfolio.py: get_stock_correlation, simulate_portfolio
- financial.py: get_financial_statements, predict_price_outlook, get_stock_news

외부 모듈은 이 __init__.py만 import하면 된다.
"""

import sys as _sys

from src.llm.tools import _state as _state_mod

# ── 상태 관리 (public API) ──
from src.llm.tools._state import (
    set_retriever,
    get_available_tickers,
    get_data_indices,
    get_sector_index,
)

# ── 헬퍼 (tabs.py 등 UI에서 사용하는 public API) ──
from src.llm.tools._helpers import (
    _find_structured_data,
    _find_similar_names,
    _enrich_with_structured_data,
    _extract_comparison_fields,
    _fmt_date,
)

# ── 내부 헬퍼 (테스트에서 사용) ──
from src.llm.tools._state import _build_data_index, _build_holdings_reverse_index, _build_sector_index
from src.llm.tools.analysis import (
    _calc_percentile, _format_valuation_position, _format_sector_analysis,
)

# ── 도구 함수들 ──
from src.llm.tools.search import search_etf, search_stock, get_etf_list, get_stock_list
from src.llm.tools.compare import compare_etfs, compare_stocks
from src.llm.tools.analysis import get_realtime_price, analyze_sector, get_technical_indicators
from src.llm.tools.portfolio import get_stock_correlation, simulate_portfolio
from src.llm.tools.financial import get_financial_statements, predict_price_outlook, get_stock_news

# 에이전트에 바인딩할 도구 목록
ALL_TOOLS = [
    search_etf, compare_etfs, get_etf_list,
    search_stock, compare_stocks, get_stock_list,
    get_realtime_price, analyze_sector, get_technical_indicators,
    get_stock_correlation, simulate_portfolio, get_financial_statements,
    predict_price_outlook, get_stock_news,
]

# ── _state 모듈 변수에 대한 동적 위임 ──
# 외부 코드에서 `import src.llm.tools as m; m._retriever` 또는
# `from src.llm.tools import _etf_data_index` 접근을 지원한다.
# _state.py의 globals가 set_retriever()에서 재할당되므로,
# 항상 _state 모듈에서 최신 값을 읽어야 한다.

_STATE_VARS = frozenset({
    "_retriever", "_stock_retriever", "_documents",
    "_etf_data_index", "_stock_data_index", "_data_initialized",
    "_holdings_reverse_index", "_sector_index",
})


def __getattr__(name: str):
    if name in _STATE_VARS:
        return getattr(_state_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# tools_mod._retriever = None 같은 직접 할당도 _state에 위임
_this = _sys.modules[__name__]
_original_setattr = type(_this).__setattr__


class _ToolsModule(type(_this)):
    """_state 변수 할당을 위임하는 모듈 타입."""

    def __setattr__(self, name, value):
        if name in _STATE_VARS:
            setattr(_state_mod, name, value)
            return
        _original_setattr(self, name, value)


_this.__class__ = _ToolsModule
