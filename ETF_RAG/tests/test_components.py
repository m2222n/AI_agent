"""tests/test_components.py — 동적 예시 질문 생성 테스트."""

import pytest

from src.ui.components import generate_dynamic_examples


# ---------------------------------------------------------------------------
# 헬퍼: 테스트용 데이터 생성
# ---------------------------------------------------------------------------

def _make_stock(name: str, change_pct: float, close: int = 50000,
                trade_value: int = 1_000_000_000, ticker: str = "005930") -> dict:
    return {
        "name": name,
        "ticker": ticker,
        "close": close,
        "change_pct": change_pct,
        "trade_value": trade_value,
    }


def _make_data(n: int = 20) -> list:
    """n개의 다양한 테스트 데이터 생성."""
    data = []
    for i in range(n):
        data.append(_make_stock(
            name=f"테스트종목{i}",
            change_pct=(i - n // 2) * 0.5,  # -5 ~ +5% 범위
            close=50000 + i * 1000,
            trade_value=(n - i) * 1_000_000_000,
            ticker=f"{100000 + i}",
        ))
    return data


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

class TestGenerateDynamicExamples:
    """generate_dynamic_examples() 단위 테스트."""

    def test_returns_none_when_no_data(self):
        assert generate_dynamic_examples(None, None) is None
        assert generate_dynamic_examples([], []) is None

    def test_returns_none_when_insufficient_data(self):
        """데이터 10개 미만이면 None."""
        small = [_make_stock(f"종목{i}", i * 0.1) for i in range(5)]
        assert generate_dynamic_examples(small, None) is None

    def test_returns_dict_with_enough_data(self):
        """충분한 데이터면 카테고리 dict 반환."""
        data = _make_data(20)
        result = generate_dynamic_examples(stock_data=data)
        assert result is not None
        assert isinstance(result, dict)

    def test_contains_gainer_category(self):
        """급등주 카테고리 존재."""
        data = _make_data(20)
        result = generate_dynamic_examples(stock_data=data)
        assert "오늘의 급등주" in result
        questions = result["오늘의 급등주"]["questions"]
        assert len(questions) >= 1
        # 질문에 양수 % 포함
        assert "+" in questions[0]

    def test_contains_loser_category(self):
        """급락주 카테고리 존재."""
        data = _make_data(20)
        result = generate_dynamic_examples(stock_data=data)
        assert "오늘의 급락주" in result
        questions = result["오늘의 급락주"]["questions"]
        assert len(questions) >= 1
        assert "-" in questions[0]

    def test_contains_volume_category(self):
        """거래대금 TOP 카테고리 존재."""
        data = _make_data(20)
        result = generate_dynamic_examples(stock_data=data)
        assert "거래대금 TOP" in result
        questions = result["거래대금 TOP"]["questions"]
        assert len(questions) >= 1

    def test_comparison_category_when_different_names(self):
        """급등 1위와 거래대금 1위가 다르면 비교 카테고리 생성."""
        data = [
            _make_stock("급등종목", 10.0, trade_value=100, ticker="000001"),
            _make_stock("거래대금종목", 0.5, trade_value=99_999_999_999, ticker="000002"),
        ]
        # 나머지 채우기
        for i in range(18):
            data.append(_make_stock(f"기타{i}", -1.0, trade_value=1000, ticker=f"0100{i:02d}"))
        result = generate_dynamic_examples(stock_data=data)
        assert result is not None
        assert "비교 분석" in result

    def test_no_comparison_when_same_top(self):
        """급등 1위 = 거래대금 1위면 비교 카테고리 없음."""
        data = [
            _make_stock("같은종목", 10.0, trade_value=99_999_999_999, ticker="000001"),
        ]
        for i in range(19):
            data.append(_make_stock(f"기타{i}", -1.0, trade_value=1000, ticker=f"0100{i:02d}"))
        result = generate_dynamic_examples(stock_data=data)
        assert result is not None
        assert "비교 분석" not in result

    def test_skips_zero_close(self):
        """종가 0인 종목은 급등/급락에서 제외."""
        data = [_make_stock("상장폐지", 50.0, close=0, ticker="999999")]
        data += _make_data(19)
        result = generate_dynamic_examples(stock_data=data)
        if result and "오늘의 급등주" in result:
            for q in result["오늘의 급등주"]["questions"]:
                assert "상장폐지" not in q

    def test_etf_and_stock_combined(self):
        """ETF + 주식 데이터 합산 처리."""
        etf = [_make_stock(f"ETF{i}", i * 0.3, ticker=f"0699{i:02d}") for i in range(10)]
        stock = [_make_stock(f"주식{i}", -i * 0.3, ticker=f"0059{i:02d}") for i in range(10)]
        result = generate_dynamic_examples(etf_data=etf, stock_data=stock)
        assert result is not None

    def test_long_name_truncated(self):
        """15자 초과 이름은 잘림."""
        data = [_make_stock("KODEX 미국S&P500선물인버스2X", 8.0, ticker="000001")]
        data += _make_data(19)
        result = generate_dynamic_examples(stock_data=data)
        if result and "오늘의 급등주" in result:
            q = result["오늘의 급등주"]["questions"][0]
            # 원래 이름(26자)보다 짧아야 함
            assert "KODEX 미국S&P500선물인버스2X" not in q

    def test_all_flat_no_gainers_or_losers(self):
        """등락률이 전부 0이면 급등/급락 카테고리 없음."""
        data = [_make_stock(f"종목{i}", 0.0, ticker=f"0100{i:02d}") for i in range(20)]
        result = generate_dynamic_examples(stock_data=data)
        if result is not None:
            assert "오늘의 급등주" not in result
            assert "오늘의 급락주" not in result

    def test_example_categories_format(self):
        """반환 형식이 EXAMPLE_CATEGORIES와 동일 (icon, questions 키)."""
        data = _make_data(20)
        result = generate_dynamic_examples(stock_data=data)
        for cat_name, cat_info in result.items():
            assert "icon" in cat_info
            assert "questions" in cat_info
            assert isinstance(cat_info["questions"], list)
            assert len(cat_info["questions"]) >= 1
