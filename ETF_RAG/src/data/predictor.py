"""
가격 전망 예측 모듈 — 3축 종합 분석

기술적 분석 + 펀더멘털 분석 + 통계 모델을 종합하여
상승/횡보/하락 시나리오별 확률과 리스크를 제공.

tools.py의 predict_price_outlook 도구에서 호출.
"""

import logging
import math

logger = logging.getLogger(__name__)

# ── 기간 매핑 ──
HORIZON_MAP = {
    "1w": 5,
    "2w": 10,
    "1m": 20,
    "3m": 60,
    "6m": 120,
    "1y": 240,
}


# ══════════════════════════════════════════════════════════════
# (A) 기술적 신호 스코어
# ══════════════════════════════════════════════════════════════

def _calc_technical_score(summary: dict) -> dict:
    """기존 get_technical_summary() 결과를 점수화.

    Returns:
        {"score": float(-1~+1), "signal": str, "key_factors": [str]}
    """
    if not summary:
        return {"score": 0.0, "signal": "데이터 부족", "key_factors": ["기술적 데이터 부족"]}

    raw_score = 0.0
    factors = []

    # 1) MA 정배열/역배열 (가중치 15%)
    ma = summary.get("ma", {})
    ma5 = ma.get("ma5")
    ma20 = ma.get("ma20")
    ma60 = ma.get("ma60")
    if ma5 and ma20 and ma60:
        if ma5 > ma20 > ma60:
            raw_score += 2.0
            factors.append("✅ MA 정배열 (5일>20일>60일, 강세 구조)")
        elif ma5 < ma20 < ma60:
            raw_score -= 2.0
            factors.append("❌ MA 역배열 (5일<20일<60일, 약세 구조)")
        else:
            factors.append("➡️ MA 혼조 (정배열/역배열 아님)")

    # 2) 골든/데드크로스 (10%)
    cross = summary.get("cross", {})
    cross_type = cross.get("type")
    if cross_type == "golden":
        raw_score += 2.0
        factors.append(f"✅ 골든크로스 발생 ({cross.get('label', '')})")
    elif cross_type == "dead":
        raw_score -= 2.0
        factors.append(f"❌ 데드크로스 발생 ({cross.get('label', '')})")

    # 3) RSI (15%)
    rsi = summary.get("rsi")
    if rsi is not None:
        if rsi < 30:
            raw_score += 1.5
            factors.append(f"✅ RSI {rsi:.1f} — 과매도 구간 (반등 가능성)")
        elif rsi > 70:
            raw_score -= 1.5
            factors.append(f"❌ RSI {rsi:.1f} — 과매수 구간 (조정 가능성)")
        elif rsi > 55:
            raw_score += 0.5
            factors.append(f"➡️ RSI {rsi:.1f} — 중립~강세")
        elif rsi < 45:
            raw_score -= 0.5
            factors.append(f"➡️ RSI {rsi:.1f} — 중립~약세")

    # 4) MACD (10%)
    macd = summary.get("macd", {})
    histogram = macd.get("histogram")
    if histogram is not None:
        if histogram > 0:
            raw_score += 1.5
            factors.append("✅ MACD 히스토그램 양수 (매수 우위)")
        else:
            raw_score -= 1.5
            factors.append("❌ MACD 히스토그램 음수 (매도 우위)")

    # 5) 볼린저 밴드 (5%)
    bb = summary.get("bollinger", {})
    pct_b = bb.get("pct_b")
    if pct_b is not None:
        if pct_b > 100:
            raw_score -= 1.0
            factors.append(f"⚠️ 볼린저 상단 돌파 (%B {pct_b:.0f}%, 과매수)")
        elif pct_b < 0:
            raw_score += 1.0
            factors.append(f"✅ 볼린저 하단 이탈 (%B {pct_b:.0f}%, 반등 가능)")
        elif pct_b > 80:
            raw_score -= 0.5
            factors.append(f"⚠️ 볼린저 상단 근접 (%B {pct_b:.0f}%)")

    # 6) 스토캐스틱 (10%)
    stoch = summary.get("stochastic", {})
    stoch_k = stoch.get("k")
    if stoch_k is not None:
        if stoch_k < 20:
            raw_score += 1.0
            factors.append(f"✅ 스토캐스틱 %K {stoch_k:.0f} — 과매도")
        elif stoch_k > 80:
            raw_score -= 1.0
            factors.append(f"❌ 스토캐스틱 %K {stoch_k:.0f} — 과매수")

    # 7) 일목균형표 (10%)
    ichimoku = summary.get("ichimoku", {})
    cloud_status = ichimoku.get("cloud_status")
    if cloud_status:
        if cloud_status == "구름대 위":
            raw_score += 2.0
            factors.append("✅ 일목균형표 구름대 위 (강세)")
        elif cloud_status == "구름대 아래":
            raw_score -= 2.0
            factors.append("❌ 일목균형표 구름대 아래 (약세)")
        else:
            factors.append("➡️ 일목균형표 구름대 안 (방향 모색 중)")

    # 8) ADX (10%)
    adx = summary.get("adx", {})
    adx_val = adx.get("adx")
    plus_di = adx.get("plus_di")
    minus_di = adx.get("minus_di")
    if adx_val is not None and plus_di is not None and minus_di is not None:
        if adx_val >= 25:
            if plus_di > minus_di:
                raw_score += 1.5
                factors.append(f"✅ ADX {adx_val:.0f} — 강한 상승 추세")
            else:
                raw_score -= 1.5
                factors.append(f"❌ ADX {adx_val:.0f} — 강한 하락 추세")
        else:
            factors.append(f"➡️ ADX {adx_val:.0f} — 추세 약함/횡보")

    # 9) OBV (10%)
    obv = summary.get("obv", {})
    obv_trend = obv.get("trend")
    if obv_trend:
        if obv_trend == "매집":
            raw_score += 1.0
            factors.append("✅ OBV 매집 구간 (기관/외인 매수)")
        elif obv_trend == "분산":
            raw_score -= 1.0
            factors.append("❌ OBV 분산 구간 (매도 압력)")

    # 10) CCI (5%)
    cci = summary.get("cci", {})
    cci_val = cci.get("cci")
    if cci_val is not None:
        if cci_val > 100:
            raw_score -= 0.5
            factors.append(f"⚠️ CCI {cci_val:+.0f} — 과매수 영역")
        elif cci_val < -100:
            raw_score += 0.5
            factors.append(f"✅ CCI {cci_val:+.0f} — 과매도 영역")

    # 정규화: 이론적 최대 ±15 → ±1.0
    max_possible = 15.0
    score = max(-1.0, min(1.0, raw_score / max_possible))

    if score > 0.2:
        signal = "강세" if score > 0.5 else "약한 강세"
    elif score < -0.2:
        signal = "약세" if score < -0.5 else "약한 약세"
    else:
        signal = "중립"

    # 상위 5개 팩터만
    return {"score": round(score, 3), "signal": signal, "key_factors": factors[:5]}


# ══════════════════════════════════════════════════════════════
# (B) 펀더멘털 신호 스코어
# ══════════════════════════════════════════════════════════════

def _calc_fundamental_score(ticker: str, structured_data: dict = None) -> dict:
    """펀더멘털 데이터를 점수화.

    Returns:
        {"score": float(-1~+1), "signal": str, "key_factors": [str]}
    """
    factors = []
    raw_score = 0.0
    has_data = False

    # 재무제표 조회
    financials = _get_financials(ticker)

    if financials:
        has_data = True
        latest = financials[0]  # 가장 최근 분기

        # 영업이익률
        op_margin = latest.get("operating_margin")
        if op_margin is not None:
            if op_margin >= 20:
                raw_score += 1.5
                factors.append(f"✅ 영업이익률 {op_margin:.1f}% — 우수")
            elif op_margin >= 10:
                raw_score += 0.5
                factors.append(f"➡️ 영업이익률 {op_margin:.1f}% — 양호")
            elif op_margin > 0:
                factors.append(f"⚠️ 영업이익률 {op_margin:.1f}% — 낮음")
            else:
                raw_score -= 1.5
                factors.append(f"❌ 영업이익률 {op_margin:.1f}% — 적자")

        # 매출 성장률 (YoY)
        rev_growth = latest.get("revenue_growth_yoy")
        if rev_growth is not None:
            if rev_growth > 20:
                raw_score += 1.5
                factors.append(f"✅ 매출 성장률 YoY {rev_growth:+.1f}% — 고성장")
            elif rev_growth > 5:
                raw_score += 0.5
                factors.append(f"➡️ 매출 성장률 YoY {rev_growth:+.1f}% — 안정 성장")
            elif rev_growth > -5:
                factors.append(f"➡️ 매출 성장률 YoY {rev_growth:+.1f}% — 보합")
            else:
                raw_score -= 1.0
                factors.append(f"❌ 매출 성장률 YoY {rev_growth:+.1f}% — 역성장")

        # 영업이익 성장률 추세 (최근 4분기)
        if len(financials) >= 2:
            op_growths = [f.get("op_growth_yoy") for f in financials[:4]
                         if f.get("op_growth_yoy") is not None]
            if len(op_growths) >= 2:
                if op_growths[0] is not None and op_growths[0] > 0:
                    if len(op_growths) >= 2 and op_growths[0] > op_growths[1]:
                        raw_score += 1.0
                        factors.append("✅ 영업이익 성장 가속 중")
                    else:
                        raw_score += 0.3
                elif op_growths[0] is not None and op_growths[0] < -10:
                    raw_score -= 1.0
                    factors.append("❌ 영업이익 감소 추세")

    # PER/PBR (structured_data에서)
    if structured_data:
        per = structured_data.get("per")
        pbr = structured_data.get("pbr")

        if per is not None and per > 0:
            if per < 10:
                raw_score += 1.0
                factors.append(f"✅ PER {per:.1f}배 — 저평가 영역")
            elif per < 15:
                raw_score += 0.3
                factors.append(f"➡️ PER {per:.1f}배 — 적정 수준")
            elif per > 30:
                raw_score -= 0.5
                factors.append(f"⚠️ PER {per:.1f}배 — 고평가 우려")

        if pbr is not None and pbr > 0:
            if pbr < 1.0:
                raw_score += 0.5
                factors.append(f"✅ PBR {pbr:.2f}배 — 자산가치 대비 저평가")
            elif pbr > 3.0:
                raw_score -= 0.3
                factors.append(f"⚠️ PBR {pbr:.2f}배 — 고평가")

    if not has_data and not structured_data:
        return {"score": 0.0, "signal": "데이터 없음", "key_factors": ["재무 데이터 없음 (ETF 또는 미수집)"]}

    # 정규화: 최대 ±6 → ±1.0
    score = max(-1.0, min(1.0, raw_score / 6.0))

    if score > 0.15:
        signal = "강세" if score > 0.4 else "약한 강세"
    elif score < -0.15:
        signal = "약세" if score < -0.4 else "약한 약세"
    else:
        signal = "중립"

    return {"score": round(score, 3), "signal": signal, "key_factors": factors[:5]}


def _get_financials(ticker: str) -> list:
    """DB에서 최근 재무제표 조회."""
    try:
        from src.data.database import DB_PATH, get_financial_data
        import sqlite3
        if not DB_PATH.exists():
            return []
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = get_financial_data(conn, ticker, quarters=8)
        conn.close()
        return rows
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════
# (C) 통계적 모델 예측
# ══════════════════════════════════════════════════════════════

def _bootstrap_ci(predicted: float, residuals: list,
                   n_bootstrap: int = 500, alpha: float = 0.10) -> tuple:
    """Bootstrap percentile 신뢰 구간.

    예측값에 잔차를 리샘플링하여 분포를 만들고 alpha/2 ~ 1-alpha/2 구간 반환.
    """
    import random
    if len(residuals) < 10:
        std = (sum(r**2 for r in residuals) / max(len(residuals), 1)) ** 0.5
        return (round(predicted - std, 2), round(predicted + std, 2))

    bootstrap_preds = []
    for _ in range(n_bootstrap):
        sampled_residual = random.choice(residuals)
        bootstrap_preds.append(predicted + sampled_residual)

    bootstrap_preds.sort()
    lo_idx = int(n_bootstrap * (alpha / 2))
    hi_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
    return (round(bootstrap_preds[lo_idx], 2), round(bootstrap_preds[hi_idx], 2))


def _calc_statistical_prediction(ticker: str, horizon_days: int) -> dict:
    """Ridge 회귀 + 히스토리컬 아날로그.

    Returns:
        {"predicted_return": float, "confidence_interval": (lo, hi),
         "historical_analog": {...}, "model_r2": float}
    """
    from src.data.technical import _get_ohlcv

    # 장기 예측은 더 많은 데이터 필요
    data_days = max(500, horizon_days * 5 + 60)
    ohlcv = _get_ohlcv(ticker, days=data_days)
    min_required = max(60, horizon_days + 60)
    if len(ohlcv) < min_required:
        return _empty_statistical()

    closes = [d["close"] for d in ohlcv]
    volumes = [d["volume"] for d in ohlcv]

    # 피처/타겟 생성
    features, targets, conditions = _build_features_targets(closes, volumes, horizon_days)

    if len(features) < 30:
        return _empty_statistical()

    # Ridge 회귀
    ridge_result = _fit_ridge(features, targets)

    # 최신 시점 피처로 예측
    latest_features = features[-1]
    predicted_return = ridge_result["model"].predict([latest_features])[0]

    # Bootstrap percentile 신뢰 구간 (90%)
    residuals = ridge_result["residuals"]
    ci_lo, ci_hi = _bootstrap_ci(predicted_return, residuals)

    # 히스토리컬 아날로그: 현재와 유사한 조건의 과거 수익률 분포
    current_cond = conditions[-1] if conditions else None
    analog = _historical_analog(conditions, targets, current_cond)

    return {
        "predicted_return": round(predicted_return, 2),
        "confidence_interval": (round(ci_lo, 2), round(ci_hi, 2)),
        "historical_analog": analog,
        "model_r2": round(ridge_result["r2"], 4),
    }


def _calc_ema_at(closes: list, idx: int, period: int) -> float:
    """인덱스 idx 시점에서의 EMA(period) 계산 (0번부터 순차 누적)."""
    k = 2 / (period + 1)
    # SMA로 시드 (period개 평균)
    start = max(0, idx - period * 3)  # 충분한 히스토리 확보
    ema = sum(closes[start:start + period]) / period
    for j in range(start + period, idx + 1):
        ema = closes[j] * k + ema * (1 - k)
    return ema


def _build_features_targets(closes: list, volumes: list, horizon: int):
    """슬라이딩 윈도우로 피처/타겟 생성."""
    n = len(closes)
    features = []
    targets = []
    conditions = []  # 히스토리컬 아날로그용

    for i in range(60, n - horizon):
        c = closes[i]
        if c <= 0:
            continue

        # 피처
        ret_5 = (c - closes[i - 5]) / closes[i - 5] * 100
        ret_20 = (c - closes[i - 20]) / closes[i - 20] * 100
        ret_60 = (c - closes[i - 60]) / closes[i - 60] * 100

        # RSI 간이 계산
        gains = sum(max(closes[j] - closes[j-1], 0) for j in range(i-13, i+1))
        losses = sum(max(closes[j-1] - closes[j], 0) for j in range(i-13, i+1))
        rsi = 100 - (100 / (1 + gains / losses)) if losses > 0 else 50

        # MA 비율
        ma5 = sum(closes[i-4:i+1]) / 5
        ma20 = sum(closes[i-19:i+1]) / 20
        ma5_ma20 = (ma5 / ma20 - 1) * 100 if ma20 > 0 else 0

        ma60 = sum(closes[i-59:i+1]) / 60
        ma20_ma60 = (ma20 / ma60 - 1) * 100 if ma60 > 0 else 0

        # 볼린저 %B
        window = closes[i-19:i+1]
        mid = sum(window) / 20
        std_dev = (sum((v - mid)**2 for v in window) / 20) ** 0.5
        pct_b = ((c - (mid - 2*std_dev)) / (4*std_dev) * 100) if std_dev > 0 else 50

        # 거래량 비율
        avg_vol = sum(volumes[i-19:i+1]) / 20
        vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1.0

        # 변동성 (ATR 간이)
        atr_pct = 0
        if i >= 14:
            trs = []
            for j in range(i-13, i+1):
                tr = max(closes[j], closes[j-1]) - min(closes[j], closes[j-1])
                trs.append(tr)
            atr = sum(trs) / 14
            atr_pct = atr / c * 100 if c > 0 else 0

        # MACD 부호 (진짜 EMA 기반)
        if i >= 33:
            ema12 = _calc_ema_at(closes, i, 12)
            ema26 = _calc_ema_at(closes, i, 26)
            macd_sign = 1 if ema12 > ema26 else -1
        else:
            macd_sign = 0

        feat = [ret_5, ret_20, ret_60, rsi, ma5_ma20, ma20_ma60,
                pct_b, vol_ratio, atr_pct, macd_sign]
        features.append(feat)

        # 조건 (히스토리컬 아날로그용: RSI 범위 + 추세 방향)
        rsi_band = "low" if rsi < 40 else "high" if rsi > 60 else "mid"
        trend = "up" if ma5_ma20 > 0 else "down"
        conditions.append((rsi_band, trend))

        # 타겟: horizon일 후 수익률
        future_price = closes[i + horizon]
        target = (future_price - c) / c * 100
        targets.append(target)

    return features, targets, conditions


def _fit_ridge(features, targets):
    """sklearn Ridge 학습."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        y = targets

        model = Ridge(alpha=1.0)
        model.fit(X[:-1], y[:-1])

        # R2
        predictions = model.predict(X[:-1])
        ss_res = sum((t - p)**2 for t, p in zip(y[:-1], predictions))
        ss_tot = sum((t - sum(y[:-1])/len(y[:-1]))**2 for t in y[:-1])
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # 잔차
        residuals = [t - p for t, p in zip(y[:-1], predictions)]

        # 최신 피처 변환을 위해 scaler 저장
        class _WrappedModel:
            def __init__(self, model, scaler):
                self._model = model
                self._scaler = scaler
            def predict(self, X):
                return self._model.predict(self._scaler.transform(X))

        return {
            "model": _WrappedModel(model, scaler),
            "r2": max(0, r2),
            "residuals": residuals,
        }
    except ImportError:
        # sklearn 없으면 단순 평균으로 fallback
        avg = sum(targets) / len(targets) if targets else 0
        return {
            "model": type("M", (), {"predict": lambda self, x: [avg]})(),
            "r2": 0.0,
            "residuals": [t - avg for t in targets],
        }


def _historical_analog(conditions, targets, current_cond) -> dict:
    """현재와 유사한 과거 조건에서의 수익률 분포."""
    if not current_cond or not conditions:
        return {"sample_count": 0, "median_return": 0, "win_rate": 0.5}

    # 동일 조건 필터
    similar_returns = [
        t for cond, t in zip(conditions[:-1], targets[:-1])
        if cond == current_cond
    ]

    if len(similar_returns) < 5:
        # 조건 완화: RSI 범위만
        similar_returns = [
            t for cond, t in zip(conditions[:-1], targets[:-1])
            if cond[0] == current_cond[0]
        ]

    if not similar_returns:
        return {"sample_count": 0, "median_return": 0, "win_rate": 0.5}

    similar_returns.sort()
    n = len(similar_returns)
    median = similar_returns[n // 2]
    win_rate = sum(1 for r in similar_returns if r > 0) / n

    p25 = similar_returns[max(0, n // 4)]
    p75 = similar_returns[min(n - 1, 3 * n // 4)]

    return {
        "sample_count": n,
        "median_return": round(median, 2),
        "win_rate": round(win_rate, 3),
        "percentile_25": round(p25, 2),
        "percentile_75": round(p75, 2),
    }


def _calc_prophet_prediction(ticker: str, horizon_days: int) -> dict:
    """Prophet 시계열 예측.

    Returns:
        {"predicted_return": float, "confidence_interval": (lo, hi),
         "trend": str, "available": bool}
    """
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*cmdstan.*")

        from prophet import Prophet
        import pandas as pd
        from src.data.technical import _get_ohlcv

        data_days = max(500, horizon_days * 5 + 60)
        ohlcv = _get_ohlcv(ticker, days=data_days)
        if len(ohlcv) < 120:
            return _empty_prophet()

        # Prophet 입력 형식: ds (날짜), y (종가)
        df = pd.DataFrame([
            {"ds": pd.Timestamp(d["date"]), "y": d["close"]}
            for d in ohlcv if d["close"] > 0
        ])
        if len(df) < 120:
            return _empty_prophet()

        current_price = df["y"].iloc[-1]

        # Prophet 학습 (빠른 실행: 짧은 체인, 주식용 설정)
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.15,  # 주식 변동성 반영
            seasonality_mode="multiplicative",
        )
        m.fit(df)

        # 예측
        future = m.make_future_dataframe(periods=horizon_days, freq="B")  # 영업일
        forecast = m.predict(future)

        # 예측 기간의 마지막 값
        pred_row = forecast.iloc[-1]
        pred_price = pred_row["yhat"]
        pred_lo = pred_row["yhat_lower"]
        pred_hi = pred_row["yhat_upper"]

        predicted_return = (pred_price - current_price) / current_price * 100
        ci_lo = (pred_lo - current_price) / current_price * 100
        ci_hi = (pred_hi - current_price) / current_price * 100

        # 추세 방향
        trend_start = forecast["trend"].iloc[-horizon_days] if len(forecast) > horizon_days else forecast["trend"].iloc[0]
        trend_end = forecast["trend"].iloc[-1]
        if trend_end > trend_start * 1.01:
            trend = "상승"
        elif trend_end < trend_start * 0.99:
            trend = "하락"
        else:
            trend = "횡보"

        return {
            "predicted_return": round(predicted_return, 2),
            "confidence_interval": (round(ci_lo, 2), round(ci_hi, 2)),
            "trend": trend,
            "available": True,
        }

    except Exception as e:
        logger.warning(f"Prophet 예측 실패 ({ticker}): {e}")
        return _empty_prophet()


def _empty_prophet() -> dict:
    return {
        "predicted_return": 0.0,
        "confidence_interval": (0.0, 0.0),
        "trend": "분석 불가",
        "available": False,
    }


def _empty_statistical() -> dict:
    return {
        "predicted_return": 0.0,
        "confidence_interval": (0.0, 0.0),
        "historical_analog": {"sample_count": 0, "median_return": 0, "win_rate": 0.5},
        "model_r2": 0.0,
    }


# ══════════════════════════════════════════════════════════════
# 종합 분석
# ══════════════════════════════════════════════════════════════

def build_price_outlook(ticker: str, name: str, horizon: str = "1m",
                        summary: dict = None,
                        structured_data: dict = None) -> dict:
    """3축 종합 분석 → 시나리오별 확률 + 리스크.

    Args:
        ticker: 종목 티커
        name: 종목명
        horizon: "1w"/"1m"/"3m"
        summary: get_technical_summary() 결과
        structured_data: 종목 구조화 데이터 (PER/PBR 등)

    Returns:
        종합 전망 dict
    """
    horizon_days = HORIZON_MAP.get(horizon, 20)

    # 현재가
    close = summary.get("close", 0) if summary else 0

    # (A) 기술적 분석
    tech = _calc_technical_score(summary)

    # (B) 펀더멘털 분석
    fund = _calc_fundamental_score(ticker, structured_data)

    # (C) 통계 모델 (Ridge 회귀)
    stat = _calc_statistical_prediction(ticker, horizon_days)

    # (D) Prophet 시계열 예측
    prophet = _calc_prophet_prediction(ticker, horizon_days)

    # 가중치 결정: 재무/Prophet 데이터 유무에 따라 조정
    has_fund = fund["signal"] != "데이터 없음"
    has_prophet = prophet["available"]

    if has_fund and has_prophet:
        w_tech, w_fund, w_stat, w_prophet = 0.30, 0.20, 0.25, 0.25
    elif has_fund:
        w_tech, w_fund, w_stat, w_prophet = 0.40, 0.25, 0.35, 0.0
    elif has_prophet:
        w_tech, w_fund, w_stat, w_prophet = 0.40, 0.0, 0.30, 0.30
    else:
        w_tech, w_fund, w_stat, w_prophet = 0.55, 0.0, 0.45, 0.0

    # 통계 모델 스코어 변환 (수익률 → -1~+1)
    stat_score = max(-1.0, min(1.0, stat["predicted_return"] / 10.0))
    prophet_score = max(-1.0, min(1.0, prophet["predicted_return"] / 10.0)) if has_prophet else 0.0

    composite = (w_tech * tech["score"] + w_fund * fund["score"]
                 + w_stat * stat_score + w_prophet * prophet_score)
    composite = round(max(-1.0, min(1.0, composite)), 3)

    # 시나리오 확률
    scenarios = _calc_scenarios(composite, stat)

    # 신뢰도 등급
    confidence = _calc_confidence(tech, fund, stat, summary)

    # 리스크 요인
    risks = _identify_risks(summary, stat, fund)

    # 데이터 품질
    data_days = summary.get("data_days", 0) if summary else 0

    return {
        "ticker": ticker,
        "name": name,
        "horizon": horizon,
        "horizon_days": horizon_days,
        "current_price": close,
        "technical": tech,
        "fundamental": fund,
        "statistical": {
            "predicted_return": stat["predicted_return"],
            "confidence_interval": stat["confidence_interval"],
            "historical_win_rate": stat["historical_analog"].get("win_rate", 0.5),
            "historical_sample_count": stat["historical_analog"].get("sample_count", 0),
            "model_r2": stat["model_r2"],
            "model_reliability": "높음" if stat["model_r2"] > 0.3 else "보통" if stat["model_r2"] > 0.1 else "낮음",
        },
        "prophet": {
            "available": prophet["available"],
            "predicted_return": prophet["predicted_return"],
            "confidence_interval": prophet["confidence_interval"],
            "trend": prophet["trend"],
        },
        "composite_score": composite,
        "scenarios": scenarios,
        "confidence_grade": confidence,
        "risk_factors": risks,
        "data_quality": {
            "price_days": data_days,
            "has_financials": has_fund,
        },
    }


def _calc_scenarios(composite: float, stat: dict) -> dict:
    """종합 점수를 시나리오별 확률로 변환.

    히스토리컬 아날로그 win_rate 반영 + sigmoid 기반 기본 확률.
    """
    # 기본 확률: sigmoid (기울기 3으로 완화 — 극단값 방지)
    bullish_raw = 1 / (1 + math.exp(-3 * composite))

    # 히스토리컬 아날로그 win_rate 반영 (있으면 30% 가중)
    analog = stat.get("historical_analog", {})
    win_rate = analog.get("win_rate", 0.5)
    sample_count = analog.get("sample_count", 0)

    if sample_count >= 10:
        # 표본 충분: sigmoid 70% + win_rate 30%
        bullish_blend = bullish_raw * 0.7 + win_rate * 0.3
    else:
        bullish_blend = bullish_raw

    # 확률 배분 (최소 10%)
    bullish_prob = max(0.10, min(0.65, bullish_blend * 0.7))
    bearish_prob = max(0.10, min(0.65, (1 - bullish_blend) * 0.7))
    neutral_prob = 1.0 - bullish_prob - bearish_prob

    # 정규화 (합 = 1)
    total = bullish_prob + neutral_prob + bearish_prob
    bullish_prob = round(bullish_prob / total, 2)
    bearish_prob = round(bearish_prob / total, 2)
    neutral_prob = round(1.0 - bullish_prob - bearish_prob, 2)

    # 시나리오별 목표 수익률
    ci = stat.get("confidence_interval", (-5, 5))
    pred = stat.get("predicted_return", 0)

    return {
        "bullish": {
            "probability": bullish_prob,
            "target_return": round(max(pred, ci[1], 2), 1),
            "description": "기술적 모멘텀 + 펀더멘털 지지 시" if composite > 0.2 else "시장 전반 강세 시",
        },
        "neutral": {
            "probability": neutral_prob,
            "target_return": round(pred, 1),
            "description": "현 수준 등락 반복",
        },
        "bearish": {
            "probability": bearish_prob,
            "target_return": round(min(pred, ci[0], -2), 1),
            "description": "기술적 약세 전환 시" if composite < -0.2 else "외부 충격 또는 실적 부진 시",
        },
    }


def _calc_confidence(tech: dict, fund: dict, stat: dict, summary: dict) -> str:
    """신뢰도 등급 (A/B/C/D)."""
    score = 0

    # 3축 방향 일치
    directions = []
    if tech["score"] > 0.1:
        directions.append(1)
    elif tech["score"] < -0.1:
        directions.append(-1)

    if fund["signal"] != "데이터 없음":
        if fund["score"] > 0.1:
            directions.append(1)
        elif fund["score"] < -0.1:
            directions.append(-1)

    if stat["model_r2"] > 0.03:
        if stat["predicted_return"] > 1:
            directions.append(1)
        elif stat["predicted_return"] < -1:
            directions.append(-1)

    if len(directions) >= 2 and len(set(directions)) == 1:
        score += 2  # 같은 방향
    elif len(directions) >= 2:
        score += 1

    # R2
    if stat["model_r2"] > 0.3:
        score += 2
    elif stat["model_r2"] > 0.1:
        score += 1

    # 재무 데이터
    if fund["signal"] != "데이터 없음":
        score += 1

    # 데이터 양
    data_days = summary.get("data_days", 0) if summary else 0
    if data_days >= 120:
        score += 1

    if score >= 5:
        return "A"
    elif score >= 4:
        return "B"
    elif score >= 2:
        return "C"
    return "D"


def _identify_risks(summary: dict, stat: dict, fund: dict) -> list:
    """리스크 요인 자동 생성."""
    risks = []

    if summary:
        # 변동성
        atr = summary.get("atr", {})
        atr_pct = atr.get("atr_pct")
        if atr_pct and atr_pct > 3:
            risks.append(f"⚠️ 높은 변동성 (ATR {atr_pct:.1f}%) — 급등락 가능")

        # RSI 극단
        rsi = summary.get("rsi")
        if rsi and rsi > 75:
            risks.append("⚠️ RSI 과매수 극단 — 단기 조정 가능성 높음")
        elif rsi and rsi < 25:
            risks.append("⚠️ RSI 과매도 극단 — 추가 하락 또는 급반등")

        # ADX 약세
        adx = summary.get("adx", {})
        if adx.get("adx") and adx["adx"] < 20:
            risks.append("⚠️ 추세 부재 (ADX<20) — 방향성 불확실")

        # 볼린저 극단
        bb = summary.get("bollinger", {})
        pct_b = bb.get("pct_b")
        if pct_b and pct_b > 95:
            risks.append("⚠️ 볼린저 상단 극접근 — 되돌림 압력")

    # 모델 신뢰도
    r2 = stat.get("model_r2", 0)
    if r2 < 0.05:
        risks.append("⚠️ 통계 모델 예측력 매우 낮음 (R²<0.05) — 수치 참고용")
    elif r2 < 0.1:
        risks.append("⚠️ 통계 모델 예측력 낮음 (R²<0.1) — 방향성 참고만")

    # 재무
    if fund.get("signal") == "데이터 없음":
        risks.append("⚠️ 재무 데이터 미수집 — 펀더멘털 평가 불가")

    # 과거 유사 패턴 부족
    analog = stat.get("historical_analog", {})
    if analog.get("sample_count", 0) < 10:
        risks.append("⚠️ 유사 패턴 표본 부족 — 통계적 신뢰도 제한")

    return risks[:6]
