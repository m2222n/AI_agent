"""섹터 기간 추이 지수 계산(_sector_trend) 단위 테스트 (2026-06-19, UI #4).

시총 상위 N종목의 '기준일 대비 수익률'을 현재 시총으로 가중 평균 → 섹터 지수(시작=100).
DB는 patch로 합성 데이터 주입(가중평균 정합성·기준일 고정·중간상장 제외 검증).
"""

from unittest.mock import patch

import api.tabs as tabs


def _run_trend(stocks, by_date):
    """get_connection/get_closes_batch를 mock하고 _sector_trend 실행."""
    class _FakeConn:
        def execute(self, *a, **k):
            class _R:
                def fetchone(self_inner):
                    # MAX(date) → by_date의 마지막 날
                    return [sorted(by_date.keys())[-1]]
            return _R()
        def close(self):
            pass

    with patch("api.tabs.get_connection", return_value=_FakeConn()), patch(
        "api.tabs.get_closes_batch", return_value=by_date
    ):
        return tabs._sector_trend("섹터", stocks, days=366)


def test_trend_starts_at_100_and_weighted():
    """기준일=100, 가중평균이 구성종목 수익률 사이값."""
    stocks = [
        {"ticker": "A", "market_cap": 9e14},  # 가중치 큼
        {"ticker": "B", "market_cap": 1e14},  # 가중치 작음
    ]
    # A: 100→200 (+100%), B: 100→110 (+10%)
    by_date = {
        "20250101": {"A": 100, "B": 100},
        "20260101": {"A": 200, "B": 110},
    }
    r = _run_trend(stocks, by_date)
    assert r is not None
    assert r["index_values"][0] == 100.0  # 기준일=100
    assert r["constituents"] == 2
    # 가중평균 = (0.9*200% + 0.1*110%) = 0.9*2 + 0.1*1.1 = 1.91 → 191
    assert abs(r["index_values"][-1] - 191.0) < 0.01
    assert abs(r["return_pct"] - 91.0) < 0.01
    # 구성종목 수익률(10%~100%) 사이에 있어야
    assert 10.0 <= r["return_pct"] <= 100.0


def test_trend_excludes_midlisted_stock():
    """기준일에 종가가 없는 종목(중간 상장)은 지수에서 제외 — base 왜곡 방지."""
    stocks = [
        {"ticker": "A", "market_cap": 5e14},
        {"ticker": "NEW", "market_cap": 5e14},  # 기준일에 없음
    ]
    by_date = {
        "20250101": {"A": 100},               # NEW 없음
        "20260101": {"A": 150, "NEW": 999},   # NEW 등장
    }
    r = _run_trend(stocks, by_date)
    assert r is not None
    assert r["constituents"] == 1  # A만 구성
    # A만 100→150 → +50%
    assert abs(r["return_pct"] - 50.0) < 0.01


def test_trend_top_n_limit():
    """구성종목은 시총 상위 N개로 제한된다."""
    big = [{"ticker": f"T{i}", "market_cap": (100 - i) * 1e12} for i in range(50)]
    by_date = {
        "20250101": {f"T{i}": 100 for i in range(50)},
        "20260101": {f"T{i}": 110 for i in range(50)},
    }
    r = _run_trend(big, by_date)
    assert r is not None
    assert r["constituents"] == tabs._SECTOR_TREND_TOP_N  # 20


def test_trend_insufficient_data_returns_none():
    """유효일이 2개 미만이면 None."""
    stocks = [{"ticker": "A", "market_cap": 1e14}]
    by_date = {"20260101": {"A": 100}}  # 하루뿐
    assert _run_trend(stocks, by_date) is None


def test_trend_no_market_cap_returns_none():
    """시총 0 종목만 있으면 None."""
    stocks = [{"ticker": "A", "market_cap": 0}]
    by_date = {"20250101": {"A": 100}, "20260101": {"A": 110}}
    assert _run_trend(stocks, by_date) is None
