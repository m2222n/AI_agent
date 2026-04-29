"""
가격 전망 예측 모듈 테스트 — predictor.py
"""

import math
import pytest
from unittest.mock import patch, MagicMock


# ── 기술적 스코어 테스트 ──

class TestCalcTechnicalScore:
    """_calc_technical_score 테스트"""

    def test_empty_summary(self):
        from src.data.predictor import _calc_technical_score
        result = _calc_technical_score({})
        assert result["score"] == 0.0
        assert result["signal"] == "데이터 부족"

    def test_none_summary(self):
        from src.data.predictor import _calc_technical_score
        result = _calc_technical_score(None)
        assert result["score"] == 0.0

    def test_strong_bullish(self):
        """정배열 + 골든크로스 + RSI 과매도 + MACD 양수 + 구름대 위 → 강세"""
        from src.data.predictor import _calc_technical_score
        summary = {
            "ma": {"ma5": 100, "ma20": 95, "ma60": 90},
            "cross": {"type": "golden", "label": "5/20"},
            "rsi": 25,
            "macd": {"histogram": 500},
            "bollinger": {"pct_b": -5},
            "stochastic": {"k": 15},
            "ichimoku": {"cloud_status": "구름대 위"},
            "adx": {"adx": 30, "plus_di": 35, "minus_di": 15},
            "obv": {"trend": "매집"},
            "cci": {"cci": -120},
        }
        result = _calc_technical_score(summary)
        assert result["score"] > 0.5
        assert result["signal"] == "강세"
        assert len(result["key_factors"]) > 0

    def test_strong_bearish(self):
        """역배열 + 데드크로스 + RSI 과매수 + MACD 음수 + 구름대 아래 → 약세"""
        from src.data.predictor import _calc_technical_score
        summary = {
            "ma": {"ma5": 90, "ma20": 95, "ma60": 100},
            "cross": {"type": "dead", "label": "5/20"},
            "rsi": 75,
            "macd": {"histogram": -500},
            "bollinger": {"pct_b": 105},
            "stochastic": {"k": 85},
            "ichimoku": {"cloud_status": "구름대 아래"},
            "adx": {"adx": 30, "plus_di": 15, "minus_di": 35},
            "obv": {"trend": "분산"},
            "cci": {"cci": 120},
        }
        result = _calc_technical_score(summary)
        assert result["score"] < -0.5
        assert result["signal"] == "약세"

    def test_neutral(self):
        """혼조 → 중립"""
        from src.data.predictor import _calc_technical_score
        summary = {
            "ma": {"ma5": 100, "ma20": 100, "ma60": 100},
            "rsi": 50,
            "macd": {"histogram": 0},
        }
        result = _calc_technical_score(summary)
        assert -0.2 <= result["score"] <= 0.2
        assert result["signal"] == "중립"

    def test_score_bounded(self):
        """점수가 -1~+1 범위 내"""
        from src.data.predictor import _calc_technical_score
        # 극단적 강세 데이터
        summary = {
            "ma": {"ma5": 200, "ma20": 150, "ma60": 100},
            "cross": {"type": "golden", "label": "5/20"},
            "rsi": 10,
            "macd": {"histogram": 10000},
            "bollinger": {"pct_b": -50},
            "stochastic": {"k": 5},
            "ichimoku": {"cloud_status": "구름대 위"},
            "adx": {"adx": 40, "plus_di": 50, "minus_di": 5},
            "obv": {"trend": "매집"},
            "cci": {"cci": -200},
        }
        result = _calc_technical_score(summary)
        assert -1.0 <= result["score"] <= 1.0

    def test_max_5_factors(self):
        """key_factors는 최대 5개"""
        from src.data.predictor import _calc_technical_score
        summary = {
            "ma": {"ma5": 100, "ma20": 95, "ma60": 90},
            "cross": {"type": "golden", "label": "5/20"},
            "rsi": 25,
            "macd": {"histogram": 500},
            "bollinger": {"pct_b": -5},
            "stochastic": {"k": 15},
            "ichimoku": {"cloud_status": "구름대 위"},
            "adx": {"adx": 30, "plus_di": 35, "minus_di": 15},
            "obv": {"trend": "매집"},
            "cci": {"cci": -120},
        }
        result = _calc_technical_score(summary)
        assert len(result["key_factors"]) <= 5


# ── 펀더멘털 스코어 테스트 ──

class TestCalcFundamentalScore:
    """_calc_fundamental_score 테스트"""

    @patch("src.data.predictor._get_financials")
    def test_no_data(self, mock_fin):
        from src.data.predictor import _calc_fundamental_score
        mock_fin.return_value = []
        result = _calc_fundamental_score("005930", None)
        assert result["signal"] == "데이터 없음"

    @patch("src.data.predictor._get_financials")
    def test_good_fundamentals(self, mock_fin):
        from src.data.predictor import _calc_fundamental_score
        mock_fin.return_value = [
            {"operating_margin": 25.0, "revenue_growth_yoy": 30.0,
             "op_growth_yoy": 40.0},
            {"op_growth_yoy": 20.0},
        ]
        result = _calc_fundamental_score("005930", {"per": 8, "pbr": 0.8})
        assert result["score"] > 0.3
        assert any("영업이익률" in f for f in result["key_factors"])

    @patch("src.data.predictor._get_financials")
    def test_bad_fundamentals(self, mock_fin):
        from src.data.predictor import _calc_fundamental_score
        mock_fin.return_value = [
            {"operating_margin": -5.0, "revenue_growth_yoy": -15.0,
             "op_growth_yoy": -20.0},
        ]
        result = _calc_fundamental_score("005930", {"per": 50, "pbr": 4.0})
        assert result["score"] < 0

    @patch("src.data.predictor._get_financials")
    def test_per_pbr_only(self, mock_fin):
        """재무제표 없고 PER/PBR만 있는 경우"""
        from src.data.predictor import _calc_fundamental_score
        mock_fin.return_value = []
        result = _calc_fundamental_score("005930", {"per": 8, "pbr": 0.7})
        assert result["score"] > 0
        assert result["signal"] != "데이터 없음"

    @patch("src.data.predictor._get_financials")
    def test_score_bounded(self, mock_fin):
        from src.data.predictor import _calc_fundamental_score
        mock_fin.return_value = [
            {"operating_margin": 30.0, "revenue_growth_yoy": 50.0,
             "op_growth_yoy": 100.0},
            {"op_growth_yoy": 50.0},
        ]
        result = _calc_fundamental_score("005930", {"per": 5, "pbr": 0.3})
        assert -1.0 <= result["score"] <= 1.0


# ── 피처/타겟 빌더 테스트 ──

class TestBuildFeaturesTargets:
    """_build_features_targets 테스트"""

    def test_insufficient_data(self):
        from src.data.predictor import _build_features_targets
        closes = list(range(1000, 1050))
        volumes = [100000] * 50
        features, targets, conditions = _build_features_targets(closes, volumes, 5)
        assert len(features) == 0  # 60일 미만

    def test_sufficient_data(self):
        from src.data.predictor import _build_features_targets
        # 100일 데이터, horizon=5
        closes = [10000 + i * 10 for i in range(100)]
        volumes = [100000 + i * 100 for i in range(100)]
        features, targets, conditions = _build_features_targets(closes, volumes, 5)
        assert len(features) > 0
        assert len(features) == len(targets) == len(conditions)
        # 피처 개수: 10개
        assert len(features[0]) == 10

    def test_conditions_format(self):
        from src.data.predictor import _build_features_targets
        closes = [10000 + i * 10 for i in range(100)]
        volumes = [100000] * 100
        _, _, conditions = _build_features_targets(closes, volumes, 5)
        if conditions:
            rsi_band, trend = conditions[0]
            assert rsi_band in ("low", "mid", "high")
            assert trend in ("up", "down")


# ── Ridge 회귀 테스트 ──

class TestFitRidge:
    """_fit_ridge 테스트"""

    def test_basic_fit(self):
        from src.data.predictor import _fit_ridge
        # 간단한 선형 관계
        features = [[float(i)] * 10 for i in range(50)]
        targets = [float(i) * 0.5 for i in range(50)]
        result = _fit_ridge(features, targets)
        assert "model" in result
        assert "r2" in result
        assert result["r2"] >= 0
        # 예측 가능
        pred = result["model"].predict([features[-1]])
        assert len(pred) == 1

    def test_r2_bounded(self):
        from src.data.predictor import _fit_ridge
        import random
        random.seed(42)
        features = [[random.random() for _ in range(10)] for _ in range(50)]
        targets = [random.random() for _ in range(50)]
        result = _fit_ridge(features, targets)
        assert 0 <= result["r2"] <= 1.0


# ── 히스토리컬 아날로그 테스트 ──

class TestHistoricalAnalog:
    """_historical_analog 테스트"""

    def test_no_conditions(self):
        from src.data.predictor import _historical_analog
        result = _historical_analog([], [], None)
        assert result["sample_count"] == 0
        assert result["win_rate"] == 0.5

    def test_exact_match(self):
        from src.data.predictor import _historical_analog
        conditions = [("mid", "up")] * 20 + [("mid", "up")]
        targets = [2.0] * 15 + [-1.0] * 5 + [0]
        result = _historical_analog(conditions, targets, ("mid", "up"))
        assert result["sample_count"] == 20
        assert result["win_rate"] == 0.75  # 15/20

    def test_relaxed_match(self):
        """정확한 조건 부족 시 RSI 범위만으로 완화 매칭"""
        from src.data.predictor import _historical_analog
        # 정확 매칭 < 5건, RSI 범위만 매칭
        conditions = [("mid", "up")] * 3 + [("mid", "down")] * 10 + [("mid", "up")]
        targets = [1.0] * 3 + [2.0] * 10 + [0]
        result = _historical_analog(conditions, targets, ("mid", "up"))
        # relaxed: "mid" RSI → 13건
        assert result["sample_count"] == 13


# ── 시나리오 계산 테스트 ──

class TestCalcScenarios:
    """_calc_scenarios 테스트"""

    def test_bullish_composite(self):
        from src.data.predictor import _calc_scenarios
        stat = {"predicted_return": 5.0, "confidence_interval": (2.0, 8.0)}
        result = _calc_scenarios(0.5, stat)
        assert result["bullish"]["probability"] > result["bearish"]["probability"]
        # 확률 합 = 1
        total = sum(s["probability"] for s in result.values())
        assert abs(total - 1.0) < 0.05

    def test_bearish_composite(self):
        from src.data.predictor import _calc_scenarios
        stat = {"predicted_return": -5.0, "confidence_interval": (-8.0, -2.0)}
        result = _calc_scenarios(-0.5, stat)
        assert result["bearish"]["probability"] > result["bullish"]["probability"]

    def test_neutral_composite(self):
        from src.data.predictor import _calc_scenarios
        stat = {"predicted_return": 0.0, "confidence_interval": (-3.0, 3.0)}
        result = _calc_scenarios(0.0, stat)
        # 중립에서는 bull ≈ bear
        assert abs(result["bullish"]["probability"] - result["bearish"]["probability"]) < 0.15

    def test_probabilities_bounded(self):
        from src.data.predictor import _calc_scenarios
        stat = {"predicted_return": 20.0, "confidence_interval": (10.0, 30.0)}
        result = _calc_scenarios(1.0, stat)
        for s in result.values():
            assert 0 <= s["probability"] <= 1.0


# ── 신뢰도 등급 테스트 ──

class TestCalcConfidence:
    """_calc_confidence 테스트"""

    def test_high_confidence(self):
        from src.data.predictor import _calc_confidence
        tech = {"score": 0.5}
        fund = {"score": 0.4, "signal": "강세"}
        stat = {"model_r2": 0.15, "predicted_return": 5.0}
        summary = {"data_days": 200}
        grade = _calc_confidence(tech, fund, stat, summary)
        assert grade in ("A", "B")

    def test_low_confidence(self):
        from src.data.predictor import _calc_confidence
        tech = {"score": 0.1}
        fund = {"score": 0.0, "signal": "데이터 없음"}
        stat = {"model_r2": 0.01, "predicted_return": 0.5}
        summary = {"data_days": 30}
        grade = _calc_confidence(tech, fund, stat, summary)
        assert grade in ("C", "D")

    def test_returns_valid_grade(self):
        from src.data.predictor import _calc_confidence
        tech = {"score": 0.0}
        fund = {"score": 0.0, "signal": "중립"}
        stat = {"model_r2": 0.0, "predicted_return": 0.0}
        summary = {}
        grade = _calc_confidence(tech, fund, stat, summary)
        assert grade in ("A", "B", "C", "D")


# ── 리스크 식별 테스트 ──

class TestIdentifyRisks:
    """_identify_risks 테스트"""

    def test_high_volatility_risk(self):
        from src.data.predictor import _identify_risks
        summary = {"atr": {"atr_pct": 5.0}, "rsi": 50}
        stat = {"model_r2": 0.1, "historical_analog": {"sample_count": 20}}
        fund = {"signal": "중립"}
        risks = _identify_risks(summary, stat, fund)
        assert any("변동성" in r for r in risks)

    def test_rsi_extreme_risk(self):
        from src.data.predictor import _identify_risks
        summary = {"rsi": 80, "atr": {}}
        stat = {"model_r2": 0.1, "historical_analog": {"sample_count": 20}}
        fund = {"signal": "중립"}
        risks = _identify_risks(summary, stat, fund)
        assert any("RSI" in r for r in risks)

    def test_no_financial_data_risk(self):
        from src.data.predictor import _identify_risks
        summary = {"rsi": 50, "atr": {}}
        stat = {"model_r2": 0.1, "historical_analog": {"sample_count": 20}}
        fund = {"signal": "데이터 없음"}
        risks = _identify_risks(summary, stat, fund)
        assert any("재무" in r for r in risks)

    def test_low_model_r2_risk(self):
        from src.data.predictor import _identify_risks
        summary = {"rsi": 50, "atr": {}}
        stat = {"model_r2": 0.02, "historical_analog": {"sample_count": 20}}
        fund = {"signal": "중립"}
        risks = _identify_risks(summary, stat, fund)
        assert any("R²" in r for r in risks)

    def test_max_6_risks(self):
        from src.data.predictor import _identify_risks
        summary = {
            "rsi": 80, "atr": {"atr_pct": 5.0},
            "adx": {"adx": 15}, "bollinger": {"pct_b": 98},
        }
        stat = {"model_r2": 0.02, "historical_analog": {"sample_count": 3}}
        fund = {"signal": "데이터 없음"}
        risks = _identify_risks(summary, stat, fund)
        assert len(risks) <= 6


# ── build_price_outlook 통합 테스트 ──

class TestBuildPriceOutlook:
    """build_price_outlook 통합 테스트"""

    @patch("src.data.predictor._get_financials")
    @patch("src.data.predictor._calc_statistical_prediction")
    def test_basic_output_structure(self, mock_stat, mock_fin):
        from src.data.predictor import build_price_outlook
        mock_fin.return_value = []
        mock_stat.return_value = {
            "predicted_return": 2.0,
            "confidence_interval": (-1.0, 5.0),
            "historical_analog": {"sample_count": 15, "median_return": 1.5, "win_rate": 0.6},
            "model_r2": 0.08,
        }
        summary = {
            "close": 70000,
            "data_days": 120,
            "ma": {"ma5": 71000, "ma20": 69000, "ma60": 68000},
            "cross": {},
            "rsi": 55,
            "macd": {"histogram": 200},
            "trend": "상승 추세",
        }
        result = build_price_outlook("005930", "삼성전자", "1m", summary, {"per": 12})

        assert result["ticker"] == "005930"
        assert result["name"] == "삼성전자"
        assert result["horizon"] == "1m"
        assert result["horizon_days"] == 20
        assert result["current_price"] == 70000
        assert "technical" in result
        assert "fundamental" in result
        assert "statistical" in result
        assert "scenarios" in result
        assert "confidence_grade" in result
        assert "risk_factors" in result
        assert -1.0 <= result["composite_score"] <= 1.0

    @patch("src.data.predictor._get_financials")
    @patch("src.data.predictor._calc_statistical_prediction")
    def test_different_horizons(self, mock_stat, mock_fin):
        from src.data.predictor import build_price_outlook
        mock_fin.return_value = []
        mock_stat.return_value = {
            "predicted_return": 0.0,
            "confidence_interval": (-2.0, 2.0),
            "historical_analog": {"sample_count": 10, "win_rate": 0.5},
            "model_r2": 0.05,
        }
        summary = {"close": 50000, "data_days": 100, "ma": {}, "cross": {}}

        for h, expected_days in [("1w", 5), ("2w", 10), ("1m", 20), ("3m", 60), ("6m", 120), ("1y", 240)]:
            result = build_price_outlook("005930", "삼성전자", h, summary)
            assert result["horizon_days"] == expected_days

    @patch("src.data.predictor._get_financials")
    @patch("src.data.predictor._calc_statistical_prediction")
    def test_no_summary(self, mock_stat, mock_fin):
        """summary=None일 때도 동작"""
        from src.data.predictor import build_price_outlook
        mock_fin.return_value = []
        mock_stat.return_value = {
            "predicted_return": 0.0,
            "confidence_interval": (0.0, 0.0),
            "historical_analog": {"sample_count": 0, "win_rate": 0.5},
            "model_r2": 0.0,
        }
        result = build_price_outlook("005930", "삼성전자", "1m", None, None)
        assert result["technical"]["signal"] == "데이터 부족"
        assert result["fundamental"]["signal"] == "데이터 없음"

    @patch("src.data.predictor._get_financials")
    @patch("src.data.predictor._calc_statistical_prediction")
    def test_scenario_probabilities_sum_to_1(self, mock_stat, mock_fin):
        from src.data.predictor import build_price_outlook
        mock_fin.return_value = []
        mock_stat.return_value = {
            "predicted_return": 3.0,
            "confidence_interval": (0.0, 6.0),
            "historical_analog": {"sample_count": 20, "win_rate": 0.65},
            "model_r2": 0.1,
        }
        summary = {
            "close": 50000, "data_days": 120,
            "ma": {"ma5": 51000, "ma20": 49000, "ma60": 48000},
            "cross": {}, "rsi": 60,
            "macd": {"histogram": 300},
        }
        result = build_price_outlook("005930", "삼성전자", "1m", summary)
        s = result["scenarios"]
        total = s["bullish"]["probability"] + s["neutral"]["probability"] + s["bearish"]["probability"]
        assert abs(total - 1.0) < 0.05


# ── HORIZON_MAP 테스트 ──

class TestHorizonMap:
    def test_valid_horizons(self):
        from src.data.predictor import HORIZON_MAP
        assert HORIZON_MAP["1w"] == 5
        assert HORIZON_MAP["2w"] == 10
        assert HORIZON_MAP["1m"] == 20
        assert HORIZON_MAP["3m"] == 60
        assert HORIZON_MAP["6m"] == 120
        assert HORIZON_MAP["1y"] == 240


# ── _empty_statistical 테스트 ──

class TestEmptyStatistical:
    def test_structure(self):
        from src.data.predictor import _empty_statistical
        result = _empty_statistical()
        assert result["predicted_return"] == 0.0
        assert result["model_r2"] == 0.0
        assert result["historical_analog"]["sample_count"] == 0


# ── EMA 헬퍼 테스트 ──

class TestCalcEmaAt:
    """_calc_ema_at 테스트"""

    def test_constant_series(self):
        """상수 시계열 → EMA = 상수"""
        from src.data.predictor import _calc_ema_at
        closes = [100.0] * 50
        assert abs(_calc_ema_at(closes, 49, 12) - 100.0) < 0.01

    def test_trending_up(self):
        """상승 시계열 → EMA < 최근 가격 (후행)"""
        from src.data.predictor import _calc_ema_at
        closes = [float(1000 + i * 10) for i in range(50)]
        ema = _calc_ema_at(closes, 49, 12)
        assert ema < closes[49]
        assert ema > closes[30]  # 너무 뒤처지지 않음

    def test_ema12_vs_ema26(self):
        """상승 추세에서 EMA(12) > EMA(26) (단기 EMA가 더 빠르게 반응)"""
        from src.data.predictor import _calc_ema_at
        closes = [float(1000 + i * 10) for i in range(60)]
        ema12 = _calc_ema_at(closes, 59, 12)
        ema26 = _calc_ema_at(closes, 59, 26)
        assert ema12 > ema26

    def test_different_from_sma(self):
        """EMA ≠ SMA (변동 시계열에서)"""
        from src.data.predictor import _calc_ema_at
        import math
        closes = [100 + 10 * math.sin(i * 0.3) for i in range(60)]
        ema = _calc_ema_at(closes, 59, 12)
        sma = sum(closes[48:60]) / 12
        assert ema != sma  # 정확히 같을 수 없음


# ── Bootstrap CI 테스트 ──

class TestBootstrapCI:
    """_bootstrap_ci 테스트"""

    def test_basic_ci(self):
        """CI가 예측값을 포함"""
        from src.data.predictor import _bootstrap_ci
        residuals = [r * 0.1 for r in range(-50, 51)]
        lo, hi = _bootstrap_ci(5.0, residuals)
        assert lo < 5.0 < hi

    def test_narrow_residuals_narrow_ci(self):
        """잔차 작으면 CI도 좁음"""
        from src.data.predictor import _bootstrap_ci
        small_residuals = [0.01 * i for i in range(-10, 11)]
        large_residuals = [1.0 * i for i in range(-10, 11)]
        lo_s, hi_s = _bootstrap_ci(0.0, small_residuals)
        lo_l, hi_l = _bootstrap_ci(0.0, large_residuals)
        assert (hi_s - lo_s) < (hi_l - lo_l)

    def test_few_residuals_fallback(self):
        """잔차 10개 미만이면 std 기반 fallback"""
        from src.data.predictor import _bootstrap_ci
        residuals = [1.0, -1.0, 0.5]
        lo, hi = _bootstrap_ci(3.0, residuals)
        assert lo < 3.0
        assert hi > 3.0

    def test_symmetric_residuals(self):
        """대칭 잔차 → CI 대략 대칭"""
        from src.data.predictor import _bootstrap_ci
        import random
        random.seed(42)
        residuals = [r * 0.5 for r in range(-100, 101)]
        lo, hi = _bootstrap_ci(0.0, residuals)
        # 대략 대칭 (완벽하진 않음 — 리샘플링 노이즈)
        assert abs(abs(lo) - abs(hi)) < 5.0


# ── 장기 예측 기간 테스트 ──

class TestLongHorizon:
    """6m/1y 장기 예측 테스트"""

    @patch("src.data.predictor._get_financials")
    @patch("src.data.predictor._calc_statistical_prediction")
    def test_6m_horizon(self, mock_stat, mock_fin):
        from src.data.predictor import build_price_outlook
        mock_fin.return_value = []
        mock_stat.return_value = {
            "predicted_return": 8.0,
            "confidence_interval": (2.0, 14.0),
            "historical_analog": {"sample_count": 10, "win_rate": 0.6},
            "model_r2": 0.06,
        }
        summary = {"close": 50000, "data_days": 300, "ma": {}, "cross": {}}
        result = build_price_outlook("005930", "삼성전자", "6m", summary)
        assert result["horizon_days"] == 120
        assert result["horizon"] == "6m"

    @patch("src.data.predictor._get_financials")
    @patch("src.data.predictor._calc_statistical_prediction")
    def test_1y_horizon(self, mock_stat, mock_fin):
        from src.data.predictor import build_price_outlook
        mock_fin.return_value = []
        mock_stat.return_value = {
            "predicted_return": 12.0,
            "confidence_interval": (0.0, 24.0),
            "historical_analog": {"sample_count": 8, "win_rate": 0.55},
            "model_r2": 0.04,
        }
        summary = {"close": 50000, "data_days": 500, "ma": {}, "cross": {}}
        result = build_price_outlook("005930", "삼성전자", "1y", summary)
        assert result["horizon_days"] == 240
        assert result["horizon"] == "1y"


# ── 시나리오 win_rate 반영 테스트 ──

class TestScenarioWinRate:
    """히스토리컬 win_rate 반영 테스트"""

    def test_high_win_rate_boosts_bullish(self):
        from src.data.predictor import _calc_scenarios
        stat_high_wr = {
            "predicted_return": 0.0,
            "confidence_interval": (-3, 3),
            "historical_analog": {"sample_count": 30, "win_rate": 0.8},
        }
        stat_low_wr = {
            "predicted_return": 0.0,
            "confidence_interval": (-3, 3),
            "historical_analog": {"sample_count": 30, "win_rate": 0.2},
        }
        result_high = _calc_scenarios(0.0, stat_high_wr)
        result_low = _calc_scenarios(0.0, stat_low_wr)
        assert result_high["bullish"]["probability"] > result_low["bullish"]["probability"]

    def test_low_sample_ignores_win_rate(self):
        """표본 < 10이면 win_rate 무시"""
        from src.data.predictor import _calc_scenarios
        stat = {
            "predicted_return": 0.0,
            "confidence_interval": (-3, 3),
            "historical_analog": {"sample_count": 3, "win_rate": 0.9},
        }
        result = _calc_scenarios(0.0, stat)
        # win_rate 미반영 → 중립에 가까움
        assert abs(result["bullish"]["probability"] - result["bearish"]["probability"]) < 0.15


# ── R² 신뢰도 기준 테스트 ──

class TestModelReliability:
    """model_reliability 및 R² 관련 신뢰도 기준 테스트"""

    def test_r2_high_reliability(self):
        """R² > 0.3 → 높음"""
        from src.data.predictor import _calc_confidence
        tech = {"score": 0.5}
        fund = {"score": 0.4, "signal": "강세"}
        stat = {"model_r2": 0.35, "predicted_return": 5.0}
        summary = {"data_days": 200}
        # R² 0.35 → score +2 (높음 등급)
        grade = _calc_confidence(tech, fund, stat, summary)
        assert grade in ("A", "B")

    def test_r2_medium_reliability(self):
        """0.1 < R² <= 0.3 → 보통 (score +1)"""
        from src.data.predictor import _calc_confidence
        tech = {"score": 0.0}
        fund = {"score": 0.0, "signal": "데이터 없음"}
        stat = {"model_r2": 0.15, "predicted_return": 0.5}
        summary = {"data_days": 50}
        grade = _calc_confidence(tech, fund, stat, summary)
        # R²=0.15 → +1, 다른 요인 부족 → C or D
        assert grade in ("C", "D")

    def test_r2_low_no_score(self):
        """R² <= 0.1 → score 0"""
        from src.data.predictor import _calc_confidence
        tech = {"score": 0.0}
        fund = {"score": 0.0, "signal": "데이터 없음"}
        stat = {"model_r2": 0.05, "predicted_return": 0.0}
        summary = {"data_days": 30}
        grade = _calc_confidence(tech, fund, stat, summary)
        assert grade == "D"

    def test_low_r2_risk_tiering(self):
        """R² < 0.05 vs 0.05~0.1 리스크 메시지 구분"""
        from src.data.predictor import _identify_risks
        summary = {"rsi": 50, "atr": {}}
        fund = {"signal": "중립"}

        # R² < 0.05 → "매우 낮음"
        risks_very_low = _identify_risks(summary, {"model_r2": 0.02, "historical_analog": {"sample_count": 20}}, fund)
        assert any("매우 낮음" in r for r in risks_very_low)

        # 0.05 <= R² < 0.1 → "낮음"
        risks_low = _identify_risks(summary, {"model_r2": 0.07, "historical_analog": {"sample_count": 20}}, fund)
        assert any("낮음" in r and "매우" not in r for r in risks_low)

        # R² >= 0.1 → 리스크 없음
        risks_ok = _identify_risks(summary, {"model_r2": 0.15, "historical_analog": {"sample_count": 20}}, fund)
        assert not any("R²" in r for r in risks_ok)


# ── Prophet 예측 테스트 ──

class TestProphetPrediction:
    """_calc_prophet_prediction 테스트"""

    def test_empty_prophet(self):
        from src.data.predictor import _empty_prophet
        result = _empty_prophet()
        assert result["available"] is False
        assert result["predicted_return"] == 0.0

    @patch("src.data.technical._get_ohlcv")
    def test_prophet_insufficient_data(self, mock_ohlcv):
        """데이터 부족 시 empty 반환"""
        from src.data.predictor import _calc_prophet_prediction
        mock_ohlcv.return_value = [{"date": "2026-01-01", "close": 100}] * 50
        result = _calc_prophet_prediction("005930", 20)
        assert result["available"] is False

    @patch("src.data.technical._get_ohlcv")
    def test_prophet_with_mock_data(self, mock_ohlcv):
        """충분한 데이터로 Prophet 실행"""
        from src.data.predictor import _calc_prophet_prediction
        import random
        random.seed(42)

        # 200일치 mock 데이터 (약간 상승 추세)
        base_price = 70000
        ohlcv = []
        for i in range(200):
            price = base_price + i * 50 + random.randint(-500, 500)
            ohlcv.append({
                "date": f"2025-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}",
                "open": price,
                "high": price + 500,
                "low": price - 500,
                "close": price,
                "volume": 1000000,
            })
        mock_ohlcv.return_value = ohlcv

        result = _calc_prophet_prediction("005930", 20)
        assert result["available"] is True
        assert isinstance(result["predicted_return"], float)
        assert len(result["confidence_interval"]) == 2
        assert result["trend"] in ("상승", "하락", "횡보")

    @patch("src.data.technical._get_ohlcv")
    def test_prophet_exception_handling(self, mock_ohlcv):
        """Prophet 실패 시 graceful fallback"""
        from src.data.predictor import _calc_prophet_prediction
        mock_ohlcv.side_effect = Exception("DB error")
        result = _calc_prophet_prediction("005930", 20)
        assert result["available"] is False


class TestBuildPriceOutlookWithProphet:
    """build_price_outlook에 Prophet 통합 확인"""

    @patch("src.data.predictor._calc_prophet_prediction")
    @patch("src.data.predictor._calc_statistical_prediction")
    @patch("src.data.predictor._calc_fundamental_score")
    def test_outlook_includes_prophet(self, mock_fund, mock_stat, mock_prophet):
        """Prophet 결과가 outlook에 포함되는지"""
        from src.data.predictor import build_price_outlook

        mock_fund.return_value = {"score": 0.0, "signal": "데이터 없음", "key_factors": []}
        mock_stat.return_value = {
            "predicted_return": 2.0,
            "confidence_interval": (-1.0, 5.0),
            "historical_analog": {"sample_count": 50, "median_return": 1.5, "win_rate": 0.6},
            "model_r2": 0.15,
        }
        mock_prophet.return_value = {
            "predicted_return": 3.5,
            "confidence_interval": (1.0, 6.0),
            "trend": "상승",
            "available": True,
        }

        summary = {"close": 70000, "ma": {}, "cross": {}, "data_days": 200}
        result = build_price_outlook("005930", "삼성전자", "1m", summary=summary)

        assert "prophet" in result
        assert result["prophet"]["available"] is True
        assert result["prophet"]["predicted_return"] == 3.5
        assert result["prophet"]["trend"] == "상승"

    @patch("src.data.predictor._calc_prophet_prediction")
    @patch("src.data.predictor._calc_statistical_prediction")
    @patch("src.data.predictor._calc_fundamental_score")
    def test_outlook_without_prophet(self, mock_fund, mock_stat, mock_prophet):
        """Prophet 불가 시에도 정상 작동"""
        from src.data.predictor import build_price_outlook

        mock_fund.return_value = {"score": 0.2, "signal": "중립", "key_factors": []}
        mock_stat.return_value = {
            "predicted_return": 1.0,
            "confidence_interval": (-2.0, 4.0),
            "historical_analog": {"sample_count": 30, "median_return": 0.5, "win_rate": 0.55},
            "model_r2": 0.12,
        }
        mock_prophet.return_value = {
            "predicted_return": 0.0,
            "confidence_interval": (0.0, 0.0),
            "trend": "분석 불가",
            "available": False,
        }

        summary = {"close": 50000, "ma": {}, "cross": {}, "data_days": 100}
        result = build_price_outlook("000660", "SK하이닉스", "1m", summary=summary)

        assert result["prophet"]["available"] is False
        # composite 계산은 기존 3축으로 (Prophet 제외)
        assert "composite_score" in result


# ── Prophet 엣지 케이스 테스트 ──

class TestProphetEdgeCases:
    """Prophet 예측 추가 엣지 케이스"""

    @patch("src.data.technical._get_ohlcv")
    def test_prophet_all_same_price(self, mock_ohlcv):
        """가격이 모두 동일할 때 (변동성 0)"""
        from src.data.predictor import _calc_prophet_prediction
        ohlcv = [{"date": f"2025-01-{i+1:02d}", "close": 50000} for i in range(200)]
        mock_ohlcv.return_value = ohlcv
        result = _calc_prophet_prediction("005930", 20)
        # Prophet이 에러 없이 처리되어야 함
        assert isinstance(result, dict)
        assert "available" in result

    @patch("src.data.technical._get_ohlcv")
    def test_prophet_empty_ohlcv(self, mock_ohlcv):
        """빈 OHLCV 데이터"""
        from src.data.predictor import _calc_prophet_prediction
        mock_ohlcv.return_value = []
        result = _calc_prophet_prediction("005930", 20)
        assert result["available"] is False

    @patch("src.data.technical._get_ohlcv")
    def test_prophet_ohlcv_returns_none(self, mock_ohlcv):
        """OHLCV가 None 반환"""
        from src.data.predictor import _calc_prophet_prediction
        mock_ohlcv.return_value = None
        result = _calc_prophet_prediction("005930", 20)
        assert result["available"] is False

    def test_empty_prophet_structure(self):
        """_empty_prophet 반환 구조 검증"""
        from src.data.predictor import _empty_prophet
        result = _empty_prophet()
        assert set(result.keys()) == {"available", "predicted_return", "confidence_interval", "trend"}
        assert result["confidence_interval"] == (0.0, 0.0)
        assert result["trend"] == "분석 불가"


# ── 통계 예측 엣지 케이스 테스트 ──

class TestStatisticalEdgeCases:

    def test_calc_scenarios_extreme_positive(self):
        """극단적 양수 composite_score"""
        from src.data.predictor import _calc_scenarios
        stat = {
            "predicted_return": 30.0,
            "confidence_interval": (20.0, 40.0),
            "historical_analog": {"sample_count": 50, "win_rate": 0.9},
        }
        result = _calc_scenarios(1.0, stat)
        assert result["bullish"]["probability"] > 0.5
        assert result["bearish"]["probability"] < 0.3
        # 확률 합이 1.0
        total = sum(s["probability"] for s in result.values())
        assert abs(total - 1.0) < 0.01

    def test_calc_scenarios_extreme_negative(self):
        """극단적 음수 composite_score"""
        from src.data.predictor import _calc_scenarios
        stat = {
            "predicted_return": -20.0,
            "confidence_interval": (-30.0, -10.0),
            "historical_analog": {"sample_count": 50, "win_rate": 0.1},
        }
        result = _calc_scenarios(-1.0, stat)
        assert result["bearish"]["probability"] > 0.5
        assert result["bullish"]["probability"] < 0.3
        total = sum(s["probability"] for s in result.values())
        assert abs(total - 1.0) < 0.01

    def test_calc_scenarios_zero_score(self):
        """composite_score=0 → 중립에 가까움"""
        from src.data.predictor import _calc_scenarios
        stat = {
            "predicted_return": 0.0,
            "confidence_interval": (-5.0, 5.0),
            "historical_analog": {"sample_count": 5, "win_rate": 0.5},
        }
        result = _calc_scenarios(0.0, stat)
        assert result["neutral"]["probability"] > 0.2
