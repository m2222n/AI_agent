"""
고급 기술적 지표 — OHLCV 기반

스토캐스틱, 일목균형표, CCI, ADX, OBV, ATR.
"""

from typing import Optional


def calc_stochastic(highs: list[int], lows: list[int], closes: list[int],
                    k_period: int = 14, d_period: int = 3) -> Optional[dict]:
    """스토캐스틱 계산.

    Returns: {"k": float, "d": float, "signal": str}
    """
    min_len = k_period + d_period
    if len(closes) < min_len or len(highs) < min_len or len(lows) < min_len:
        return None

    # %K 시리즈 계산
    k_values = []
    for i in range(k_period - 1, len(closes)):
        window_highs = highs[i - k_period + 1:i + 1]
        window_lows = lows[i - k_period + 1:i + 1]
        highest = max(window_highs)
        lowest = min(window_lows)
        if highest == lowest:
            k_values.append(50.0)
        else:
            k_values.append((closes[i] - lowest) / (highest - lowest) * 100)

    if len(k_values) < d_period:
        return None

    # %D = %K의 d_period일 SMA
    k = k_values[-1]
    d = sum(k_values[-d_period:]) / d_period

    signal = "중립"
    if k > 80:
        signal = "과매수"
    elif k < 20:
        signal = "과매도"

    return {"k": round(k, 1), "d": round(d, 1), "signal": signal}


def calc_ichimoku(highs: list[int], lows: list[int], closes: list[int]
                  ) -> Optional[dict]:
    """일목균형표 계산.

    Returns:
        {"tenkan": float, "kijun": float, "senkou_a": float,
         "senkou_b": float, "chikou": float, "cloud_status": str}
    """
    if len(closes) < 52 or len(highs) < 52 or len(lows) < 52:
        return None

    def mid_value(data, period):
        """최근 period일간 (최고+최저)/2."""
        window = data[-period:]
        return (max(window) + min(window)) / 2

    # 전환선 (9일)
    tenkan = mid_value(highs, 9) / 2 + mid_value(lows, 9) / 2
    tenkan = (max(highs[-9:]) + min(lows[-9:])) / 2

    # 기준선 (26일)
    kijun = (max(highs[-26:]) + min(lows[-26:])) / 2

    # 선행스팬1 = (전환선 + 기준선) / 2
    senkou_a = (tenkan + kijun) / 2

    # 선행스팬2 = (52일 최고 + 52일 최저) / 2
    senkou_b = (max(highs[-52:]) + min(lows[-52:])) / 2

    # 후행스팬 = 현재 종가 (26일 전에 표시)
    chikou = closes[-1]

    # 구름대 위/아래 판정 (현재가 vs 구름대)
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    current = closes[-1]

    if current > cloud_top:
        cloud_status = "구름대 위"
    elif current < cloud_bottom:
        cloud_status = "구름대 아래"
    else:
        cloud_status = "구름대 안"

    return {
        "tenkan": round(tenkan),
        "kijun": round(kijun),
        "senkou_a": round(senkou_a),
        "senkou_b": round(senkou_b),
        "chikou": round(chikou),
        "cloud_status": cloud_status,
    }


def calc_cci(highs: list[int], lows: list[int], closes: list[int],
             period: int = 20) -> Optional[dict]:
    """CCI (Commodity Channel Index) 계산.

    Returns: {"cci": float, "signal": str}
    """
    if len(closes) < period or len(highs) < period or len(lows) < period:
        return None

    # Typical Price 시리즈
    tp_series = [(h + l + c) / 3
                 for h, l, c in zip(highs[-period:], lows[-period:], closes[-period:])]

    tp_mean = sum(tp_series) / period
    mean_dev = sum(abs(tp - tp_mean) for tp in tp_series) / period

    if mean_dev == 0:
        return {"cci": 0.0, "signal": "중립"}

    cci = (tp_series[-1] - tp_mean) / (0.015 * mean_dev)

    signal = "중립"
    if cci > 100:
        signal = "과매수"
    elif cci < -100:
        signal = "과매도"

    return {"cci": round(cci, 1), "signal": signal}


def calc_adx(highs: list[int], lows: list[int], closes: list[int],
             period: int = 14) -> Optional[dict]:
    """ADX (Average Directional Index) 계산.

    Returns: {"adx": float, "plus_di": float, "minus_di": float, "trend_strength": str}
    """
    n = len(closes)
    if n < period * 2 + 1 or len(highs) < n or len(lows) < n:
        return None

    # True Range, +DM, -DM 시리즈
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)

        plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    if len(tr_list) < period * 2:
        return None

    # Wilder's smoothing
    def wilder_smooth(data, p):
        smoothed = sum(data[:p])
        result = [smoothed]
        for val in data[p:]:
            smoothed = smoothed - smoothed / p + val
            result.append(smoothed)
        return result

    smooth_tr = wilder_smooth(tr_list, period)
    smooth_plus_dm = wilder_smooth(plus_dm_list, period)
    smooth_minus_dm = wilder_smooth(minus_dm_list, period)

    # +DI, -DI 시리즈
    dx_list = []
    for i in range(len(smooth_tr)):
        if smooth_tr[i] == 0:
            continue
        plus_di = smooth_plus_dm[i] / smooth_tr[i] * 100
        minus_di = smooth_minus_dm[i] / smooth_tr[i] * 100
        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx_list.append(abs(plus_di - minus_di) / di_sum * 100)

    if len(dx_list) < period:
        return None

    # ADX = DX의 Wilder 평균
    adx = sum(dx_list[:period]) / period
    for dx in dx_list[period:]:
        adx = (adx * (period - 1) + dx) / period

    # 최종 +DI, -DI
    last_tr = smooth_tr[-1]
    plus_di = smooth_plus_dm[-1] / last_tr * 100 if last_tr > 0 else 0
    minus_di = smooth_minus_dm[-1] / last_tr * 100 if last_tr > 0 else 0

    trend_strength = "추세 없음"
    if adx >= 25:
        trend_strength = "강한 추세"
    elif adx >= 20:
        trend_strength = "약한 추세"

    return {
        "adx": round(adx, 1),
        "plus_di": round(plus_di, 1),
        "minus_di": round(minus_di, 1),
        "trend_strength": trend_strength,
    }


def calc_obv(closes: list[int], volumes: list[int]) -> Optional[dict]:
    """OBV (On Balance Volume) 계산.

    Returns: {"obv": int, "obv_ma20": float, "trend": str}
    """
    if len(closes) < 20 or len(volumes) < 20 or len(closes) != len(volumes):
        return None

    obv = 0
    obv_series = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
        obv_series.append(obv)

    # OBV 20일 이동평균
    obv_ma20 = sum(obv_series[-20:]) / 20

    trend = "중립"
    if obv > obv_ma20:
        trend = "매집"
    elif obv < obv_ma20:
        trend = "분산"

    return {
        "obv": obv,
        "obv_ma20": round(obv_ma20),
        "trend": trend,
    }


def calc_atr(highs: list[int], lows: list[int], closes: list[int],
             period: int = 14) -> Optional[dict]:
    """ATR (Average True Range) 계산.

    Returns: {"atr": float, "atr_pct": float, "volatility": str}
    """
    if len(closes) < period + 1 or len(highs) < period + 1 or len(lows) < period + 1:
        return None

    # True Range 시리즈
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)

    if len(tr_list) < period:
        return None

    # Wilder's smoothing ATR
    atr = sum(tr_list[:period]) / period
    for tr in tr_list[period:]:
        atr = (atr * (period - 1) + tr) / period

    # ATR% (현재가 대비)
    current_close = closes[-1]
    atr_pct = atr / current_close * 100 if current_close > 0 else 0

    volatility = "보통"
    if atr_pct > 3:
        volatility = "높은 변동성"
    elif atr_pct < 1:
        volatility = "낮은 변동성"

    return {
        "atr": round(atr, 0),
        "atr_pct": round(atr_pct, 2),
        "volatility": volatility,
    }
