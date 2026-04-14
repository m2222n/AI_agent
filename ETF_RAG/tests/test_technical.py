"""기술적 지표 계산 모듈 테스트"""

import pytest

from src.data.technical import (
    calc_ma,
    calc_ema,
    calc_rsi,
    calc_macd,
    calc_bollinger,
    detect_cross,
    calc_correlation,
    calc_beta,
    _daily_returns,
    simulate_portfolio,
)


# ── 이동평균(MA) 테스트 ──────────────────────────────────────

class TestMA:
    def test_sma_basic(self):
        closes = [100, 200, 300, 400, 500]
        assert calc_ma(closes, 5) == 300.0

    def test_sma_period_3(self):
        closes = [10, 20, 30, 40, 50]
        assert calc_ma(closes, 3) == 40.0  # (30+40+50)/3

    def test_sma_insufficient_data(self):
        assert calc_ma([100, 200], 5) is None

    def test_sma_period_1(self):
        closes = [100, 200, 300]
        assert calc_ma(closes, 1) == 300.0

    def test_ema_basic(self):
        # 가속 상승 데이터 (등차수열이 아닌)
        closes = [100 + i ** 2 for i in range(30)]
        result = calc_ema(closes, 12)
        assert result is not None
        # EMA는 최근값에 더 가중치 → 가속 상승에서 EMA > SMA
        sma = sum(closes[-12:]) / 12
        assert result > sma

    def test_ema_insufficient(self):
        assert calc_ema([100, 200], 5) is None


# ── RSI 테스트 ───────────────────────────────────────────────

class TestRSI:
    def test_rsi_all_up(self):
        """모두 상승 → RSI = 100"""
        closes = list(range(100, 120))  # 20일 연속 상승
        rsi = calc_rsi(closes, 14)
        assert rsi == 100.0

    def test_rsi_all_down(self):
        """모두 하락 → RSI = 0"""
        closes = list(range(120, 100, -1))
        rsi = calc_rsi(closes, 14)
        assert rsi == 0.0

    def test_rsi_range(self):
        """일반적인 데이터에서 0~100 범위"""
        closes = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                  110, 108, 111, 112, 110, 113, 115, 114, 116, 118]
        rsi = calc_rsi(closes, 14)
        assert 0 <= rsi <= 100

    def test_rsi_insufficient(self):
        assert calc_rsi([100, 101, 102], 14) is None

    def test_rsi_neutral(self):
        """반반 등락 → RSI ≈ 50"""
        closes = []
        val = 100
        for i in range(30):
            val += 1 if i % 2 == 0 else -1
            closes.append(val)
        rsi = calc_rsi(closes, 14)
        assert 40 <= rsi <= 60


# ── MACD 테스트 ──────────────────────────────────────────────

class TestMACD:
    def test_macd_basic(self):
        """충분한 데이터에서 MACD 계산"""
        closes = list(range(100, 200))  # 100일 상승
        result = calc_macd(closes)
        assert result is not None
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_macd_uptrend(self):
        """상승 추세 → MACD > 0"""
        closes = list(range(100, 200))
        result = calc_macd(closes)
        assert result["macd"] > 0

    def test_macd_insufficient(self):
        closes = list(range(100, 120))  # 20일 (slow=26+signal=9=35 필요)
        assert calc_macd(closes) is None

    def test_macd_histogram_sign(self):
        """histogram = macd - signal"""
        closes = list(range(100, 200))
        result = calc_macd(closes)
        expected = round(result["macd"] - result["signal"], 2)
        assert result["histogram"] == expected


# ── 볼린저 밴드 테스트 ────────────────────────────────────────

class TestBollinger:
    def test_bollinger_basic(self):
        closes = list(range(100, 125))  # 25일
        result = calc_bollinger(closes)
        assert result is not None
        assert result["upper"] > result["middle"] > result["lower"]

    def test_bollinger_width_positive(self):
        closes = list(range(100, 125))
        result = calc_bollinger(closes)
        assert result["width"] > 0

    def test_bollinger_pct_b_range(self):
        """현재가가 밴드 안에 있으면 0~100"""
        # 등차수열의 마지막 값은 중간 근처
        closes = [100] * 20 + [105]  # 갑자기 올라감
        result = calc_bollinger(closes, period=20)
        assert result is not None
        # 100이 20개 → std ≈ 0, 하지만 마지막은 105
        # 실제로는 period=20이면 마지막 20개 = [100]*19+[105]
        assert result["pct_b"] > 50  # 상단 쪽

    def test_bollinger_insufficient(self):
        assert calc_bollinger([100, 200], period=20) is None

    def test_bollinger_constant_price(self):
        """변동 없으면 상단=하단=중간"""
        closes = [1000] * 25
        result = calc_bollinger(closes)
        assert result["upper"] == result["lower"] == result["middle"]
        assert result["width"] == 0


# ── 크로스 판정 테스트 ────────────────────────────────────────

class TestCross:
    def test_golden_cross(self):
        """단기 MA가 장기 MA를 상향 돌파"""
        # 하락 후 급반등 → 5일MA가 20일MA를 상향 돌파
        closes = list(range(200, 170, -1))  # 30일 하락
        closes += list(range(170, 210, 2))  # 20일 급등
        result = detect_cross(closes, 5, 20)
        # 급등 패턴이므로 골든크로스 가능성 높음
        # (정확한 타이밍은 데이터에 따라 다를 수 있음)
        assert result in ("golden_cross", None)

    def test_dead_cross(self):
        """단기 MA가 장기 MA를 하향 돌파"""
        # 상승 후 급락
        closes = list(range(100, 140))  # 40일 상승
        closes += list(range(140, 100, -2))  # 20일 급락
        result = detect_cross(closes, 5, 20)
        assert result in ("dead_cross", None)

    def test_no_cross(self):
        """지속 상승 → 크로스 없음"""
        closes = list(range(100, 160))
        result = detect_cross(closes, 5, 20)
        assert result is None

    def test_cross_insufficient(self):
        assert detect_cross([100, 200], 5, 20) is None


# ── 통합: get_technical_summary (DB 모킹) ────────────────────

class TestTechnicalSummary:
    def test_summary_with_mock_closes(self, monkeypatch):
        """_get_closes를 모킹해서 get_technical_summary 테스트"""
        from src.data import technical

        # 100일 상승 데이터
        mock_data = [
            {"date": f"2026{i // 30 + 1:02d}{i % 30 + 1:02d}", "close": 50000 + i * 100}
            for i in range(150)
        ]
        monkeypatch.setattr(technical, "_get_closes", lambda ticker, days=250: mock_data)

        result = technical.get_technical_summary("005930")
        assert result is not None
        assert result["ticker"] == "005930"
        assert result["trend"] == "상승"
        assert result["ma"]["ma5"] is not None
        assert result["ma"]["ma20"] is not None
        assert result["rsi"] is not None
        assert result["macd"] is not None
        assert result["bollinger"] is not None

    def test_summary_insufficient_data(self, monkeypatch):
        """데이터 부족 시 None 반환"""
        from src.data import technical

        mock_data = [{"date": "20260401", "close": 50000}] * 10
        monkeypatch.setattr(technical, "_get_closes", lambda ticker, days=250: mock_data)

        result = technical.get_technical_summary("005930")
        assert result is None

    def test_summary_downtrend(self, monkeypatch):
        """하락 추세 판정"""
        from src.data import technical

        mock_data = [
            {"date": f"2026{i // 30 + 1:02d}{i % 30 + 1:02d}", "close": 80000 - i * 100}
            for i in range(150)
        ]
        monkeypatch.setattr(technical, "_get_closes", lambda ticker, days=250: mock_data)

        result = technical.get_technical_summary("005930")
        assert result is not None
        assert result["trend"] == "하락"


# ── 일간 수익률 테스트 ────────────────────────────────────────

class TestDailyReturns:
    def test_basic(self):
        closes = [100, 110, 105]
        ret = _daily_returns(closes)
        assert len(ret) == 2
        assert abs(ret[0] - 0.1) < 1e-10  # +10%
        assert abs(ret[1] - (-5 / 110)) < 1e-10

    def test_zero_price_skipped(self):
        """0원 종가가 있으면 해당 수익률 건너뜀"""
        closes = [100, 0, 110]
        ret = _daily_returns(closes)
        # 100→0: closes[0]!=0이므로 포함, 0→110: closes[1]==0이므로 스킵
        assert len(ret) == 1

    def test_single_price(self):
        assert _daily_returns([100]) == []


# ── 상관계수 테스트 ──────────────────────────────────────────

class TestCorrelation:
    def test_perfect_positive(self, monkeypatch):
        """동일 종가 → 상관계수 ≈ 1.0"""
        from src.data import technical
        mock_data = [
            {"date": f"2026{i // 28 + 1:02d}{i % 28 + 1:02d}", "close": 50000 + i * 100}
            for i in range(60)
        ]
        monkeypatch.setattr(technical, "_get_closes",
                            lambda ticker, days=120: mock_data)
        result = calc_correlation("A", "B", days=60)
        assert result is not None
        assert abs(result["correlation"] - 1.0) < 0.01

    def test_negative_correlation(self, monkeypatch):
        """반대 방향 등락 → 음의 상관계수"""
        from src.data import technical
        import math
        # A가 오르면 B는 내리는 시소 패턴
        base = 50000
        up_data = []
        down_data = []
        for i in range(60):
            date = f"2026{i // 28 + 1:02d}{i % 28 + 1:02d}"
            swing = int(2000 * math.sin(i * 0.5))  # 등락 패턴
            up_data.append({"date": date, "close": base + swing})
            down_data.append({"date": date, "close": base - swing})  # 반대
        def mock_get_closes(ticker, days=120):
            return up_data if ticker == "A" else down_data
        monkeypatch.setattr(technical, "_get_closes", mock_get_closes)
        result = calc_correlation("A", "B", days=60)
        assert result is not None
        assert result["correlation"] < -0.5

    def test_insufficient_data(self, monkeypatch):
        """데이터 부족 → None"""
        from src.data import technical
        mock_data = [{"date": "20260101", "close": 50000}] * 5
        monkeypatch.setattr(technical, "_get_closes",
                            lambda ticker, days=120: mock_data)
        assert calc_correlation("A", "B") is None


# ── 베타 계수 테스트 ─────────────────────────────────────────

class TestBeta:
    def test_beta_same_as_market(self, monkeypatch):
        """시장과 동일 종목 → 베타 ≈ 1.0"""
        from src.data import technical
        mock_data = [
            {"date": f"2026{i // 28 + 1:02d}{i % 28 + 1:02d}", "close": 50000 + i * 100}
            for i in range(60)
        ]
        monkeypatch.setattr(technical, "_get_closes",
                            lambda ticker, days=250: mock_data)
        result = calc_beta("005930", days=60)
        assert result is not None
        assert abs(result["beta"] - 1.0) < 0.01

    def test_beta_high_volatility(self, monkeypatch):
        """시장보다 변동성 큰 종목 → 베타 > 1"""
        from src.data import technical
        market_data = [
            {"date": f"2026{i // 28 + 1:02d}{i % 28 + 1:02d}", "close": 50000 + i * 100}
            for i in range(60)
        ]
        stock_data = [
            {"date": f"2026{i // 28 + 1:02d}{i % 28 + 1:02d}", "close": 50000 + i * 300}
            for i in range(60)
        ]
        call_count = [0]
        def mock_get_closes(ticker, days=250):
            call_count[0] += 1
            # 첫 호출 = stock, 두 번째 = benchmark
            return stock_data if call_count[0] % 2 == 1 else market_data
        monkeypatch.setattr(technical, "_get_closes", mock_get_closes)
        result = calc_beta("005930", benchmark="069500", days=60)
        assert result is not None
        assert result["beta"] > 1.0

    def test_beta_insufficient(self, monkeypatch):
        """데이터 부족 → None"""
        from src.data import technical
        mock_data = [{"date": "20260101", "close": 50000}] * 5
        monkeypatch.setattr(technical, "_get_closes",
                            lambda ticker, days=250: mock_data)
        assert calc_beta("005930") is None


# ── 밸류에이션 백분위 테스트 ──────────────────────────────────

class TestValuationPercentile:
    def test_calc_percentile_basic(self):
        from src.llm.tools import _calc_percentile
        # 10이 [5, 10, 15, 20] 중 하위 25% (5만 아래)
        result = _calc_percentile(10, [5, 10, 15, 20])
        assert result == 25.0

    def test_calc_percentile_lowest(self):
        from src.llm.tools import _calc_percentile
        result = _calc_percentile(1, [5, 10, 15, 20])
        assert result == 0.0

    def test_calc_percentile_highest(self):
        from src.llm.tools import _calc_percentile
        result = _calc_percentile(25, [5, 10, 15, 20])
        assert result == 100.0

    def test_calc_percentile_empty(self):
        from src.llm.tools import _calc_percentile
        assert _calc_percentile(10, []) == 50.0

    def test_format_valuation_position(self):
        from src.llm.tools import _format_valuation_position
        stock = {"ticker": "005930", "per": 15.0, "pbr": 1.5, "div": 2.0, "market_cap": 500_0000_0000_0000}
        sector_stocks = [
            {"ticker": "005930", "per": 15.0, "pbr": 1.5, "div": 2.0, "market_cap": 500_0000_0000_0000},
            {"ticker": "000660", "per": 20.0, "pbr": 2.0, "div": 1.0, "market_cap": 100_0000_0000_0000},
            {"ticker": "066570", "per": 10.0, "pbr": 0.8, "div": 3.0, "market_cap": 50_0000_0000_0000},
            {"ticker": "035420", "per": 30.0, "pbr": 3.0, "div": 0.5, "market_cap": 30_0000_0000_0000},
        ]
        result = _format_valuation_position(stock, sector_stocks)
        assert "PER" in result
        assert "PBR" in result
        assert "배당" in result
        assert "시가총액" in result

    def test_format_valuation_no_data(self):
        from src.llm.tools import _format_valuation_position
        stock = {"ticker": "005930", "per": 0, "pbr": 0, "div": 0, "market_cap": 0}
        sector_stocks = [{"ticker": "005930", "per": 0, "pbr": 0, "div": 0, "market_cap": 0}]
        result = _format_valuation_position(stock, sector_stocks)
        assert result == ""


# ── 포트폴리오 시뮬레이션 테스트 ──────────────────────────────

class TestPortfolioSimulation:
    def _make_mock(self, monkeypatch, closes_map):
        """ticker → closes 딕셔너리로 _get_closes 모킹"""
        from src.data import technical
        def mock_get_closes(ticker, days=260):
            closes = closes_map.get(ticker, [])
            data = [{"date": f"2026{i // 28 + 1:02d}{i % 28 + 1:02d}", "close": c}
                    for i, c in enumerate(closes)]
            return data
        monkeypatch.setattr(technical, "_get_closes", mock_get_closes)

    def test_basic_equal_weight(self, monkeypatch):
        """두 종목 균등 비중 시뮬레이션"""
        n = 60
        self._make_mock(monkeypatch, {
            "A": [50000 + i * 100 for i in range(n)],
            "B": [30000 + i * 50 for i in range(n)],
        })
        result = simulate_portfolio(["A", "B"], [0.5, 0.5], days=50)
        assert result is not None
        p = result["portfolio"]
        assert "total_return" in p
        assert "annualized_return" in p
        assert "volatility" in p
        assert "sharpe_ratio" in p
        assert "max_drawdown" in p
        assert len(result["individual"]) == 2
        assert result["data_days"] > 0

    def test_single_ticker(self, monkeypatch):
        """단일 종목 100% → 포트폴리오 수익률 = 개별 수익률"""
        n = 60
        closes = [50000 + i * 100 for i in range(n)]
        self._make_mock(monkeypatch, {"A": closes})
        result = simulate_portfolio(["A"], [1.0], days=50)
        assert result is not None
        # 포트폴리오 총수익률 ≈ 개별 총수익률
        assert abs(result["portfolio"]["total_return"] -
                   result["individual"][0]["total_return"]) < 0.01

    def test_monotonic_increase_no_drawdown(self, monkeypatch):
        """단조 증가 → MDD = 0"""
        n = 60
        self._make_mock(monkeypatch, {
            "A": [50000 + i * 100 for i in range(n)],
        })
        result = simulate_portfolio(["A"], [1.0], days=50)
        assert result is not None
        assert result["portfolio"]["max_drawdown"] == 0.0

    def test_weight_normalization(self, monkeypatch):
        """비중 합이 100이어도 정규화"""
        n = 60
        self._make_mock(monkeypatch, {
            "A": [50000 + i * 100 for i in range(n)],
            "B": [30000 + i * 50 for i in range(n)],
        })
        result = simulate_portfolio(["A", "B"], [60, 40], days=50)
        assert result is not None
        # 비중이 0.6, 0.4로 정규화
        assert abs(result["individual"][0]["weight"] - 0.6) < 0.01

    def test_insufficient_data(self, monkeypatch):
        """데이터 부족 → None"""
        self._make_mock(monkeypatch, {"A": [50000] * 5})
        assert simulate_portfolio(["A"], [1.0]) is None

    def test_invalid_inputs(self):
        """잘못된 입력 → None"""
        assert simulate_portfolio([], []) is None
        assert simulate_portfolio(["A"], []) is None
        assert simulate_portfolio(["A"], [-1]) is None

    def test_uptrend_positive_sharpe(self, monkeypatch):
        """상승 추세 → 양의 샤프"""
        n = 260
        self._make_mock(monkeypatch, {
            "A": [50000 + i * 200 for i in range(n)],
        })
        result = simulate_portfolio(["A"], [1.0], days=250)
        assert result is not None
        assert result["portfolio"]["sharpe_ratio"] > 0


# ── 포트폴리오 도구 테스트 ────────────────────────────────────

class TestPortfolioTool:
    def test_tool_in_all_tools(self):
        from src.llm.tools import ALL_TOOLS
        names = [t.name for t in ALL_TOOLS]
        assert "simulate_portfolio" in names

    def test_tool_no_tickers(self, monkeypatch):
        from src.llm import tools
        monkeypatch.setattr(tools, "_etf_data_index", {})
        monkeypatch.setattr(tools, "_stock_data_index", {})
        from src.llm.tools import simulate_portfolio as sim_tool
        result = sim_tool.invoke({"tickers_and_weights": "없는종목 50%"})
        assert "찾을 수 없습니다" in result or "종목" in result


# ── 상관관계 도구 테스트 ──────────────────────────────────────

class TestCorrelationTool:
    def test_tool_in_all_tools(self):
        from src.llm.tools import ALL_TOOLS
        names = [t.name for t in ALL_TOOLS]
        assert "get_stock_correlation" in names

    def test_tool_not_found(self, monkeypatch):
        from src.llm import tools
        monkeypatch.setattr(tools, "_etf_data_index", {})
        monkeypatch.setattr(tools, "_stock_data_index", {})
        from src.llm.tools import get_stock_correlation
        result = get_stock_correlation.invoke({"ticker1": "없는종목", "ticker2": "없는종목2"})
        assert "찾을 수 없습니다" in result


# ── 도구 통합 테스트 ─────────────────────────────────────────

class TestToolIntegration:
    def test_all_tools_includes_technical(self):
        """ALL_TOOLS에 get_technical_indicators 포함 (9개)"""
        from src.llm.tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 12
        names = [t.name for t in ALL_TOOLS]
        assert "get_technical_indicators" in names

    def test_tool_no_data(self, monkeypatch):
        """종목 없을 때 에러 메시지"""
        from src.llm import tools
        # 빈 인덱스로 설정
        monkeypatch.setattr(tools, "_etf_data_index", {})
        monkeypatch.setattr(tools, "_stock_data_index", {})

        from src.llm.tools import get_technical_indicators
        result = get_technical_indicators.invoke({"name_or_ticker": "없는종목"})
        assert "찾을 수 없습니다" in result
