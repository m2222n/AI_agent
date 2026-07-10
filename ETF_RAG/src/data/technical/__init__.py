"""
기술적 지표 계산 패키지 — SQLite 일봉 데이터 기반 (yfinance fallback)

서브모듈:
- _data.py       : DB 커넥션, TTL 캐시, OHLCV/종가 조회
- _indicators.py : 기본 지표 (MA, EMA, RSI, MACD, 볼린저, 크로스)
- _advanced.py   : 고급 지표 (스토캐스틱, 일목균형표, CCI, ADX, OBV, ATR)
- _portfolio.py  : 상관관계, 베타, 포트폴리오 시뮬레이션
- _summary.py    : 종합 요약 (get_technical_summary)
"""

# ── 상수 ──
from src.data.technical._data import MARKET_BENCHMARK, BENCHMARK_TICKER

# ── 데이터 접근 (internal, chart_generator/predictor 등에서 사용) ──
from src.data.technical._data import (
    _get_closes,
    _get_ohlcv,
    _yfinance_ohlcv,
    _db_available,
    _get_db_conn,
    _ohlcv_cache,
    _closes_cache,
    reset_db_connection,
)

# ── 기본 지표 ──
from src.data.technical._indicators import (
    calc_ma,
    calc_ema,
    calc_rsi,
    calc_macd,
    calc_bollinger,
    detect_cross,
)

# ── 고급 지표 ──
from src.data.technical._advanced import (
    calc_stochastic,
    calc_ichimoku,
    calc_cci,
    calc_adx,
    calc_obv,
    calc_atr,
)

# ── 포트폴리오 ──
from src.data.technical._portfolio import (
    _daily_returns,
    calc_correlation,
    calc_beta,
    simulate_portfolio,
    _calc_benchmark,
)

# ── 종합 요약 ──
from src.data.technical._summary import get_technical_summary

__all__ = [
    # 상수
    "MARKET_BENCHMARK", "BENCHMARK_TICKER",
    # 데이터 접근
    "_get_closes", "_get_ohlcv", "_yfinance_ohlcv", "_db_available",
    "_get_db_conn", "_ohlcv_cache", "_closes_cache", "reset_db_connection",
    # 기본 지표
    "calc_ma", "calc_ema", "calc_rsi", "calc_macd", "calc_bollinger",
    "detect_cross",
    # 고급 지표
    "calc_stochastic", "calc_ichimoku", "calc_cci", "calc_adx",
    "calc_obv", "calc_atr",
    # 포트폴리오
    "_daily_returns", "calc_correlation", "calc_beta",
    "simulate_portfolio", "_calc_benchmark",
    # 종합
    "get_technical_summary",
]
