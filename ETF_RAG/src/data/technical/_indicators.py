"""
기본 기술적 지표 — 종가(closes) 기반

MA, EMA, RSI, MACD, 볼린저 밴드, 골든/데드크로스.
"""

from typing import Optional


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
