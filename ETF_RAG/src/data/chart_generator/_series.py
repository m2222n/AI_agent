"""
차트용 시리즈 계산 헬퍼 — MA, RSI, MACD, 볼린저 밴드
"""


def calc_ma_series(closes: list, period: int) -> list:
    """MA 시리즈 (None 패딩, closes와 동일 길이)."""
    result = [None] * (period - 1)
    for i in range(period - 1, len(closes)):
        result.append(sum(closes[i - period + 1:i + 1]) / period)
    return result


def calc_rsi_series(closes: list, period: int = 14) -> list:
    """RSI 시리즈 (None 패딩)."""
    if len(closes) < period + 1:
        return []

    result = [None] * period
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result.append(100.0)
    else:
        result.append(100 - (100 / (1 + avg_gain / avg_loss)))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            result.append(100 - (100 / (1 + avg_gain / avg_loss)))

    return result


def calc_macd_series(closes: list,
                     fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 시리즈 (macd_line, signal_line, histogram) — closes와 동일 길이."""
    if len(closes) < slow + signal:
        return None

    def ema_s(data, period):
        mult = 2 / (period + 1)
        ema = sum(data[:period]) / period
        out = [ema]
        for val in data[period:]:
            ema = (val - ema) * mult + ema
            out.append(ema)
        return out

    fast_ema = ema_s(closes, fast)
    slow_ema = ema_s(closes, slow)
    offset = slow - fast
    macd_raw = [f - s for f, s in zip(fast_ema[offset:], slow_ema)]

    if len(macd_raw) < signal:
        return None

    signal_raw = ema_s(macd_raw, signal)

    pad_m = slow - 1
    pad_s = pad_m + signal - 1
    t = len(closes)

    macd_full = ([None] * pad_m + macd_raw + [None] * t)[:t]
    signal_full = ([None] * pad_s + signal_raw + [None] * t)[:t]
    histogram = [(m - s) if m is not None and s is not None else None
                 for m, s in zip(macd_full, signal_full)]

    return macd_full, signal_full, histogram


def calc_bb_series(closes: list, period: int = 20, num_std: float = 2.0):
    """볼린저 밴드 상단/하단 시리즈."""
    upper, lower = [None] * (period - 1), [None] * (period - 1)
    for i in range(period - 1, len(closes)):
        w = closes[i - period + 1:i + 1]
        mid = sum(w) / period
        std = (sum((v - mid) ** 2 for v in w) / period) ** 0.5
        upper.append(mid + num_std * std)
        lower.append(mid - num_std * std)
    return upper, lower
