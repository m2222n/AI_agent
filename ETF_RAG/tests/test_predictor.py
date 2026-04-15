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

        for h, expected_days in [("1w", 5), ("2w", 10), ("1m", 20), ("3m", 60)]:
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


# ── _empty_statistical 테스트 ──

class TestEmptyStatistical:
    def test_structure(self):
        from src.data.predictor import _empty_statistical
        result = _empty_statistical()
        assert result["predicted_return"] == 0.0
        assert result["model_r2"] == 0.0
        assert result["historical_analog"]["sample_count"] == 0
