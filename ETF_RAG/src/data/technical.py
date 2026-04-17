"""
기술적 지표 계산 모듈 — SQLite 일봉 데이터 기반 (yfinance fallback)

MA(이동평균), RSI, MACD, 볼린저 밴드, 골든/데드크로스 판정,
일목균형표, 스토캐스틱, CCI, ADX, OBV, ATR 등
DB에서 OHLCV 시계열을 읽어 계산하고, 도구(tools.py)에서 호출.
DB 미존재 시 yfinance에서 과거 데이터를 가져옴 (Streamlit Cloud 대응).
"""

import logging
import sqlite3
from typing import Optional

from src.data.database import DB_PATH, get_historical_prices

logger = logging.getLogger(__name__)

# KOSPI 대표 ETF (베타 계산 시 시장 벤치마크)
MARKET_BENCHMARK = "069500"  # KODEX 200


def _yfinance_ohlcv(ticker: str, days: int = 250) -> list[dict]:
    """yfinance에서 과거 OHLCV 데이터 조회 (DB 없을 때 fallback).

    Returns:
        [{"date": "20260408", "open": ..., "high": ..., "low": ...,
          "close": ..., "volume": ...}, ...]
    """
    try:
        import yfinance as yf
        from src.data.realtime import krx_to_yfinance
        yf_ticker = krx_to_yfinance(ticker, "stock")
        # days+여유분 (영업일 변환)
        period_map = {250: "1y", 150: "9mo", 60: "3mo"}
        period = "1y"
        for threshold, p in sorted(period_map.items()):
            if days <= threshold:
                period = p
                break
        if days > 250:
            period = "2y"

        df = yf.download(yf_ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return []

        # MultiIndex 컬럼 처리 (yfinance >= 0.2.31)
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df = df.droplevel("Ticker", axis=1)

        result = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y%m%d")
            c = int(round(float(row["Close"])))
            h = int(round(float(row["High"])))
            l = int(round(float(row["Low"])))
            o = int(round(float(row["Open"])))
            v = int(float(row["Volume"]))
            if c > 0 and h > 0 and l > 0:
                result.append({
                    "date": date_str, "open": o, "high": h,
                    "low": l, "close": c, "volume": v,
                })
        return result[-days:] if len(result) > days else result
    except Exception as e:
        logger.warning(f"yfinance OHLCV 조회 실패 ({ticker}): {e}")
        return []


def _db_available() -> bool:
    """SQLite DB 파일이 존재하는지 확인."""
    return DB_PATH.exists()


def _get_closes(ticker: str, days: int = 250,
                conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """최근 N영업일 종가 조회 (날짜 오름차순).

    DB 미존재 시 yfinance fallback.

    Returns:
        [{"date": "20260408", "close": 210500}, ...]
    """
    if conn is None and not _db_available():
        ohlcv = _yfinance_ohlcv(ticker, days)
        return [{"date": d["date"], "close": d["close"]} for d in ohlcv]

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


def _get_ohlcv(ticker: str, days: int = 250,
               conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """최근 N영업일 OHLCV 조회 (날짜 오름차순).

    DB 미존재 시 yfinance fallback.

    Returns:
        [{"date": "20260408", "open": 210000, "high": 212000,
          "low": 209000, "close": 210500, "volume": 1234567}, ...]
    """
    if conn is None and not _db_available():
        return _yfinance_ohlcv(ticker, days)

    should_close = conn is None
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT date, open, high, low, close, volume FROM daily_prices
        WHERE ticker = ? AND close > 0 AND high > 0 AND low > 0
        ORDER BY date DESC
        LIMIT ?
    """, (ticker, days)).fetchall()

    if should_close:
        conn.close()

    return [
        {"date": r["date"], "open": r["open"], "high": r["high"],
         "low": r["low"], "close": r["close"], "volume": r["volume"]}
        for r in reversed(rows)
    ]


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


# ── 고급 기술적 지표 (OHLCV 기반) ─────────────────────────


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


def get_technical_summary(ticker: str) -> Optional[dict]:
    """종목의 기술적 지표 종합 요약.

    Returns:
        {
            "ticker": str, "date": str, "close": int, "data_days": int,
            "ma": {"ma5", "ma20", "ma60", "ma120"},
            "rsi": float,
            "macd": {"macd", "signal", "histogram"},
            "bollinger": {"upper", "middle", "lower", "width", "pct_b"},
            "cross": {"5_20", "20_60", "60_120"},
            "trend": str,
            "stochastic": {"k", "d", "signal"} | None,
            "ichimoku": {"tenkan", "kijun", "senkou_a", "senkou_b", "chikou", "cloud_status"} | None,
            "cci": {"cci", "signal"} | None,
            "adx": {"adx", "plus_di", "minus_di", "trend_strength"} | None,
            "obv": {"obv", "obv_ma20", "trend"} | None,
            "atr": {"atr", "atr_pct", "volatility"} | None,
        }
    """
    # OHLCV 데이터 조회 (고급 지표용)
    ohlcv = _get_ohlcv(ticker, days=250)
    if len(ohlcv) < 20:
        return None

    closes = [d["close"] for d in ohlcv]
    highs = [d["high"] for d in ohlcv]
    lows = [d["low"] for d in ohlcv]
    volumes = [d["volume"] for d in ohlcv]
    latest = ohlcv[-1]

    # ── 기본 지표 (종가 기반) ──
    ma5 = calc_ma(closes, 5)
    ma20 = calc_ma(closes, 20)
    ma60 = calc_ma(closes, 60)
    ma120 = calc_ma(closes, 120)
    rsi = calc_rsi(closes, 14)
    macd = calc_macd(closes)
    bollinger = calc_bollinger(closes)
    cross_5_20 = detect_cross(closes, 5, 20)
    cross_20_60 = detect_cross(closes, 20, 60)
    cross_60_120 = detect_cross(closes, 60, 120)

    trend = "횡보"
    if ma5 and ma20 and ma60:
        if ma5 > ma20 > ma60:
            trend = "상승"
        elif ma5 < ma20 < ma60:
            trend = "하락"

    # ── 고급 지표 (OHLCV 기반) ──
    stochastic = calc_stochastic(highs, lows, closes)
    ichimoku = calc_ichimoku(highs, lows, closes)
    cci = calc_cci(highs, lows, closes)
    adx = calc_adx(highs, lows, closes)
    obv = calc_obv(closes, volumes)
    atr = calc_atr(highs, lows, closes)

    return {
        "ticker": ticker,
        "date": latest["date"],
        "close": latest["close"],
        "data_days": len(ohlcv),
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
        "stochastic": stochastic,
        "ichimoku": ichimoku,
        "cci": cci,
        "adx": adx,
        "obv": obv,
        "atr": atr,
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

    # 벤치마크 (KODEX 200) 비교
    benchmark = _calc_benchmark(common_dates, port_daily)

    return {
        "portfolio": {
            "total_return": round(total_return, 4),
            "annualized_return": round(ann_return, 4),
            "volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_dd, 4),
        },
        "benchmark": benchmark,
        "individual": individual,
        "period": f"{common_dates[0]}~{common_dates[-1]}",
        "data_days": trading_days,
    }


BENCHMARK_TICKER = "069500"  # KODEX 200


def _calc_benchmark(common_dates: list, port_daily: list) -> Optional[dict]:
    """벤치마크(KODEX 200) 동일 기간 성과 비교.

    Returns:
        {"ticker": str, "name": str, "total_return": float,
         "annualized_return": float, "volatility": float,
         "sharpe_ratio": float, "max_drawdown": float,
         "alpha": float, "tracking_error": float}
        또는 데이터 부족 시 None
    """
    bm_data = _get_closes(BENCHMARK_TICKER, days=len(common_dates) + 10)
    if len(bm_data) < 20:
        return None

    bm_map = {d["date"]: d["close"] for d in bm_data}

    # 공통 날짜 필터
    bm_closes = []
    valid_dates = []
    for d in common_dates:
        if d in bm_map:
            bm_closes.append(bm_map[d])
            valid_dates.append(d)

    if len(bm_closes) < 20:
        return None

    bm_daily = _daily_returns(bm_closes)
    n = min(len(bm_daily), len(port_daily))
    if n < 10:
        return None

    bm_daily = bm_daily[:n]

    # 벤치마크 wealth
    bm_wealth = [1.0]
    for r in bm_daily:
        bm_wealth.append(bm_wealth[-1] * (1 + r))

    bm_total = bm_wealth[-1] - 1
    bm_ann = (bm_wealth[-1] ** (250 / n) - 1
              if n > 0 and bm_wealth[-1] > 0 else 0.0)

    bm_mean = sum(bm_daily) / n
    bm_var = sum((r - bm_mean) ** 2 for r in bm_daily) / n
    bm_vol = (bm_var ** 0.5) * (250 ** 0.5)
    bm_sharpe = (bm_ann - 0.035) / bm_vol if bm_vol > 0 else 0.0

    # MDD
    peak = bm_wealth[0]
    bm_mdd = 0.0
    for w in bm_wealth:
        if w > peak:
            peak = w
        dd = (w - peak) / peak
        if dd < bm_mdd:
            bm_mdd = dd

    # 알파 (포트폴리오 초과 수익률)
    port_ann = sum(port_daily[:n]) / n * 250
    bm_ann_simple = bm_mean * 250
    alpha = port_ann - bm_ann_simple

    # 트래킹 에러 (초과 수익률의 표준편차)
    excess = [p - b for p, b in zip(port_daily[:n], bm_daily)]
    ex_mean = sum(excess) / n
    te_var = sum((e - ex_mean) ** 2 for e in excess) / n
    tracking_error = (te_var ** 0.5) * (250 ** 0.5)

    return {
        "ticker": BENCHMARK_TICKER,
        "name": "KODEX 200",
        "total_return": round(bm_total, 4),
        "annualized_return": round(bm_ann, 4),
        "volatility": round(bm_vol, 4),
        "sharpe_ratio": round(bm_sharpe, 2),
        "max_drawdown": round(bm_mdd, 4),
        "alpha": round(alpha, 4),
        "tracking_error": round(tracking_error, 4),
    }
