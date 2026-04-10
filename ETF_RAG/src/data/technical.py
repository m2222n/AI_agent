"""
기술적 지표 계산 모듈 — SQLite 일봉 데이터 기반

MA(이동평균), RSI, MACD, 볼린저 밴드, 골든/데드크로스 판정 등
DB에서 종가 시계열을 읽어 계산하고, 도구(tools.py)에서 호출.
"""

import sqlite3
from typing import Optional

from src.data.database import DB_PATH, get_historical_prices

# KOSPI 대표 ETF (베타 계산 시 시장 벤치마크)
MARKET_BENCHMARK = "069500"  # KODEX 200


def _get_closes(ticker: str, days: int = 250,
                conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """최근 N영업일 종가 조회 (날짜 오름차순).

    Returns:
        [{"date": "20260408", "close": 210500}, ...]
    """
    should_close = conn is None
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT date, close FROM daily_prices
        WHERE ticker = ? AND close > 0
        ORDER BY date DESC
        LIMIT ?
    """, (ticker, days)).fetchall()

    if should_close:
        conn.close()

    # 날짜 오름차순으로 뒤집기
    return [{"date": r["date"], "close": r["close"]} for r in reversed(rows)]


def calc_ma(closes: list[int], period: int) -> Optional[float]:
    """단순이동평균(SMA) 계산. 데이터 부족 시 None."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def calc_ema(closes: list[int], period: int) -> Optional[float]:
    """지수이동평균(EMA) 계산."""
    if len(closes) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period  # 초기값은 SMA
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calc_rsi(closes: list[int], period: int = 14) -> Optional[float]:
    """RSI (Relative Strength Index) 계산. Wilder's smoothing 방식."""
    if len(closes) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    # 첫 period개의 평균
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(closes: list[int],
              fast: int = 12, slow: int = 26, signal: int = 9
              ) -> Optional[dict]:
    """MACD 계산. Returns {"macd": float, "signal": float, "histogram": float}."""
    if len(closes) < slow + signal:
        return None

    # EMA 시리즈 계산
    def ema_series(data, period):
        mult = 2 / (period + 1)
        ema = sum(data[:period]) / period
        result = [ema]
        for val in data[period:]:
            ema = (val - ema) * mult + ema
            result.append(ema)
        return result

    fast_ema = ema_series(closes, fast)
    slow_ema = ema_series(closes, slow)

    # MACD 라인 = fast EMA - slow EMA (길이 맞추기)
    offset = slow - fast
    macd_line = [f - s for f, s in zip(fast_ema[offset:], slow_ema)]

    if len(macd_line) < signal:
        return None

    # 시그널 라인 = MACD의 EMA
    signal_line = ema_series(macd_line, signal)

    macd_val = macd_line[-1]
    signal_val = signal_line[-1]
    histogram = macd_val - signal_val

    return {
        "macd": round(macd_val, 2),
        "signal": round(signal_val, 2),
        "histogram": round(histogram, 2),
    }


def calc_bollinger(closes: list[int], period: int = 20, num_std: float = 2.0
                   ) -> Optional[dict]:
    """볼린저 밴드 계산. Returns {"upper", "middle", "lower", "width", "pct_b"}."""
    if len(closes) < period:
        return None

    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = variance ** 0.5

    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle * 100 if middle else 0
    pct_b = (closes[-1] - lower) / (upper - lower) * 100 if upper != lower else 50

    return {
        "upper": round(upper, 0),
        "middle": round(middle, 0),
        "lower": round(lower, 0),
        "width": round(width, 2),
        "pct_b": round(pct_b, 2),
    }


def detect_cross(closes: list[int],
                 short_period: int = 5, long_period: int = 20
                 ) -> Optional[str]:
    """골든크로스/데드크로스 판정.

    최근 2일간 단기MA와 장기MA의 교차를 확인.
    Returns: "golden_cross", "dead_cross", or None.
    """
    if len(closes) < long_period + 1:
        return None

    # 오늘과 어제의 MA
    ma_short_today = sum(closes[-short_period:]) / short_period
    ma_long_today = sum(closes[-long_period:]) / long_period

    ma_short_yesterday = sum(closes[-short_period - 1:-1]) / short_period
    ma_long_yesterday = sum(closes[-long_period - 1:-1]) / long_period

    # 어제: 단기 < 장기, 오늘: 단기 >= 장기 → 골든크로스
    if ma_short_yesterday < ma_long_yesterday and ma_short_today >= ma_long_today:
        return "golden_cross"
    # 어제: 단기 > 장기, 오늘: 단기 <= 장기 → 데드크로스
    if ma_short_yesterday > ma_long_yesterday and ma_short_today <= ma_long_today:
        return "dead_cross"

    return None


def get_technical_summary(ticker: str) -> Optional[dict]:
    """종목의 기술적 지표 종합 요약.

    Returns:
        {
            "ticker": str,
            "date": str,
            "close": int,
            "ma": {"ma5": float, "ma20": float, "ma60": float, "ma120": float},
            "rsi": float,
            "macd": {"macd": float, "signal": float, "histogram": float},
            "bollinger": {"upper", "middle", "lower", "width", "pct_b"},
            "cross": {"5_20": str|None, "20_60": str|None, "60_120": str|None},
            "trend": str,  # "상승", "하락", "횡보"
        }
    """
    data = _get_closes(ticker, days=250)
    if len(data) < 20:
        return None

    closes = [d["close"] for d in data]
    latest = data[-1]

    # 이동평균
    ma5 = calc_ma(closes, 5)
    ma20 = calc_ma(closes, 20)
    ma60 = calc_ma(closes, 60)
    ma120 = calc_ma(closes, 120)

    # RSI
    rsi = calc_rsi(closes, 14)

    # MACD
    macd = calc_macd(closes)

    # 볼린저 밴드
    bollinger = calc_bollinger(closes)

    # 크로스 판정
    cross_5_20 = detect_cross(closes, 5, 20)
    cross_20_60 = detect_cross(closes, 20, 60)
    cross_60_120 = detect_cross(closes, 60, 120)

    # 추세 판정 (간단한 규칙 기반)
    trend = "횡보"
    if ma5 and ma20 and ma60:
        if ma5 > ma20 > ma60:
            trend = "상승"
        elif ma5 < ma20 < ma60:
            trend = "하락"

    return {
        "ticker": ticker,
        "date": latest["date"],
        "close": latest["close"],
        "data_days": len(data),
        "ma": {
            "ma5": round(ma5) if ma5 else None,
            "ma20": round(ma20) if ma20 else None,
            "ma60": round(ma60) if ma60 else None,
            "ma120": round(ma120) if ma120 else None,
        },
        "rsi": round(rsi, 1) if rsi else None,
        "macd": macd,
        "bollinger": bollinger,
        "cross": {
            "5_20": cross_5_20,
            "20_60": cross_20_60,
            "60_120": cross_60_120,
        },
        "trend": trend,
    }


def _daily_returns(closes: list[int]) -> list[float]:
    """종가 리스트 → 일간 수익률 리스트."""
    return [(closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes)) if closes[i - 1] != 0]


def calc_correlation(ticker1: str, ticker2: str,
                     days: int = 120) -> Optional[dict]:
    """두 종목의 일간 수익률 상관계수 계산.

    Returns:
        {"correlation": float, "data_days": int, "period": str}
    """
    data1 = _get_closes(ticker1, days=days)
    data2 = _get_closes(ticker2, days=days)

    if len(data1) < 20 or len(data2) < 20:
        return None

    # 공통 날짜만 매칭
    dates1 = {d["date"]: d["close"] for d in data1}
    dates2 = {d["date"]: d["close"] for d in data2}
    common_dates = sorted(set(dates1.keys()) & set(dates2.keys()))

    if len(common_dates) < 20:
        return None

    closes1 = [dates1[d] for d in common_dates]
    closes2 = [dates2[d] for d in common_dates]

    ret1 = _daily_returns(closes1)
    ret2 = _daily_returns(closes2)

    n = min(len(ret1), len(ret2))
    if n < 10:
        return None

    r1, r2 = ret1[:n], ret2[:n]
    mean1 = sum(r1) / n
    mean2 = sum(r2) / n

    cov = sum((a - mean1) * (b - mean2) for a, b in zip(r1, r2)) / n
    std1 = (sum((a - mean1) ** 2 for a in r1) / n) ** 0.5
    std2 = (sum((b - mean2) ** 2 for b in r2) / n) ** 0.5

    if std1 == 0 or std2 == 0:
        return None

    corr = cov / (std1 * std2)

    return {
        "correlation": round(corr, 4),
        "data_days": n,
        "period": f"{common_dates[0]}~{common_dates[-1]}",
    }


def calc_beta(ticker: str, benchmark: str = None,
              days: int = 250) -> Optional[dict]:
    """종목의 베타 계수 계산 (시장 대비 민감도).

    beta = Cov(종목, 시장) / Var(시장)

    Returns:
        {"beta": float, "data_days": int, "benchmark": str}
    """
    if benchmark is None:
        benchmark = MARKET_BENCHMARK

    data_stock = _get_closes(ticker, days=days)
    data_market = _get_closes(benchmark, days=days)

    if len(data_stock) < 20 or len(data_market) < 20:
        return None

    # 공통 날짜 매칭
    dates_s = {d["date"]: d["close"] for d in data_stock}
    dates_m = {d["date"]: d["close"] for d in data_market}
    common = sorted(set(dates_s.keys()) & set(dates_m.keys()))

    if len(common) < 20:
        return None

    closes_s = [dates_s[d] for d in common]
    closes_m = [dates_m[d] for d in common]

    ret_s = _daily_returns(closes_s)
    ret_m = _daily_returns(closes_m)

    n = min(len(ret_s), len(ret_m))
    if n < 10:
        return None

    rs, rm = ret_s[:n], ret_m[:n]
    mean_s = sum(rs) / n
    mean_m = sum(rm) / n

    cov = sum((a - mean_s) * (b - mean_m) for a, b in zip(rs, rm)) / n
    var_m = sum((b - mean_m) ** 2 for b in rm) / n

    if var_m == 0:
        return None

    beta = cov / var_m

    return {
        "beta": round(beta, 3),
        "data_days": n,
        "benchmark": benchmark,
    }


def simulate_portfolio(tickers: list[str], weights: list[float],
                       days: int = 250) -> Optional[dict]:
    """포트폴리오 백테스트 시뮬레이션.

    Args:
        tickers: 종목 티커 리스트
        weights: 비중 리스트 (합계 1.0으로 정규화됨)
        days: 시뮬레이션 기간 (영업일)

    Returns:
        {
            "portfolio": {"total_return", "annualized_return", "volatility",
                          "sharpe_ratio", "max_drawdown"},
            "individual": [{"ticker", "weight", "total_return"}],
            "period": str, "data_days": int,
        }
    """
    if not tickers or not weights or len(tickers) != len(weights):
        return None

    # 비중 정규화
    w_sum = sum(weights)
    if w_sum <= 0:
        return None
    weights = [w / w_sum for w in weights]

    # 각 종목 종가 조회
    all_data = {}
    for ticker in tickers:
        data = _get_closes(ticker, days=days + 10)
        if len(data) < 20:
            return None
        all_data[ticker] = {d["date"]: d["close"] for d in data}

    # 공통 날짜 매칭
    common_dates = sorted(
        set.intersection(*(set(d.keys()) for d in all_data.values()))
    )
    if len(common_dates) < 20:
        return None

    common_dates = common_dates[-days:] if len(common_dates) > days else common_dates

    # 개별 종목 일간 수익률
    individual_returns = {}
    for ticker in tickers:
        closes = [all_data[ticker][d] for d in common_dates]
        individual_returns[ticker] = _daily_returns(closes)

    n = min(len(r) for r in individual_returns.values())
    if n < 10:
        return None

    # 포트폴리오 일간 수익률 = 가중합
    port_daily = []
    for i in range(n):
        r = sum(w * individual_returns[t][i]
                for t, w in zip(tickers, weights))
        port_daily.append(r)

    # 누적 수익률 (wealth curve)
    wealth = [1.0]
    for r in port_daily:
        wealth.append(wealth[-1] * (1 + r))

    total_return = wealth[-1] - 1
    trading_days = len(port_daily)

    # 연환산 수익률
    ann_return = (wealth[-1] ** (250 / trading_days) - 1
                  if trading_days > 0 and wealth[-1] > 0 else 0.0)

    # 변동성 (연환산)
    mean_r = sum(port_daily) / trading_days
    variance = sum((r - mean_r) ** 2 for r in port_daily) / trading_days
    ann_vol = (variance ** 0.5) * (250 ** 0.5)

    # 샤프 비율 (무위험 수익률 3.5%)
    sharpe = (ann_return - 0.035) / ann_vol if ann_vol > 0 else 0.0

    # 최대 낙폭 (MDD)
    peak = wealth[0]
    max_dd = 0.0
    for w in wealth:
        if w > peak:
            peak = w
        dd = (w - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # 개별 종목 수익률
    individual = []
    for ticker, weight in zip(tickers, weights):
        closes = [all_data[ticker][d] for d in common_dates]
        ret = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0.0
        individual.append({
            "ticker": ticker,
            "weight": round(weight, 4),
            "total_return": round(ret, 4),
        })

    return {
        "portfolio": {
            "total_return": round(total_return, 4),
            "annualized_return": round(ann_return, 4),
            "volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_dd, 4),
        },
        "individual": individual,
        "period": f"{common_dates[0]}~{common_dates[-1]}",
        "data_days": trading_days,
    }
