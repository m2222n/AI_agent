"""
기술적 지표 종합 요약 — get_technical_summary()

모든 지표 모듈을 통합하여 종목의 전체 기술적 분석 결과를 반환.
"""

from typing import Optional

from src.data.technical import _data
from src.data.technical._indicators import (
    calc_ma, calc_rsi, calc_macd, calc_bollinger, detect_cross,
)
from src.data.technical._advanced import (
    calc_stochastic, calc_ichimoku, calc_cci, calc_adx, calc_obv, calc_atr,
)


def get_technical_summary(ticker: str, days: int = 250) -> Optional[dict]:
    """종목의 기술적 지표 종합 요약.

    Args:
        ticker: 종목 티커
        days: 분석 기간 (영업일 수). 기본 250일(약 1년).
              지표 계산에 최소 120일 이상 필요 (MA120 등).

    Returns:
        {
            "ticker": str, "date": str, "close": int, "data_days": int,
            "first_date": str, "last_date": str,
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
    # OHLCV 데이터 조회 — 지표 계산에 최소 여유분 확보
    fetch_days = max(days, 250)
    ohlcv = _data._get_ohlcv(ticker, days=fetch_days)
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

    # 기간 시작점: 지표 계산용으로 fetch는 250일 이상 당기지만, '기간 대비 등락률'은
    # 사용자가 고른 days 기준이어야 한다(아니면 6개월=1년 first_close가 같아짐).
    # 최근 days개 중 첫 항목을 시작점으로(데이터가 days보다 적으면 가장 오래된 것).
    period_start = ohlcv[-days] if len(ohlcv) >= days else ohlcv[0]
    return {
        "ticker": ticker,
        "date": latest["date"],
        "close": latest["close"],
        "first_close": period_start["close"],  # 선택 기간 시작 종가(기간 대비 등락률용)
        "data_days": len(ohlcv),
        "first_date": period_start["date"],
        "last_date": latest["date"],
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
