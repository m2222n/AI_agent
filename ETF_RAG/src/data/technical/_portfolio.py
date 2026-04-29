"""
포트폴리오 분석 — 상관관계, 베타, 백테스트 시뮬레이션

DB 종가 데이터를 사용하여 종목 간 상관계수, 시장 베타,
포트폴리오 성과(수익률/MDD/샤프)를 계산.
"""

from typing import Optional

from src.data.technical import _data
from src.data.technical._data import MARKET_BENCHMARK, BENCHMARK_TICKER


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
    data1 = _data._get_closes(ticker1, days=days)
    data2 = _data._get_closes(ticker2, days=days)

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

    data_stock = _data._get_closes(ticker, days=days)
    data_market = _data._get_closes(benchmark, days=days)

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


def _calc_benchmark(common_dates: list, port_daily: list) -> Optional[dict]:
    """벤치마크(KODEX 200) 동일 기간 성과 비교.

    Returns:
        {"ticker": str, "name": str, "total_return": float,
         "annualized_return": float, "volatility": float,
         "sharpe_ratio": float, "max_drawdown": float,
         "alpha": float, "tracking_error": float}
        또는 데이터 부족 시 None
    """
    bm_data = _data._get_closes(BENCHMARK_TICKER, days=len(common_dates) + 10)
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
        data = _data._get_closes(ticker, days=days + 10)
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

    # 벤치마크 wealth curve (차트용)
    bm_wealth_curve = None
    if benchmark:
        bm_data = _data._get_closes(BENCHMARK_TICKER, days=len(common_dates) + 10)
        bm_map = {d["date"]: d["close"] for d in bm_data}
        bm_closes = [bm_map[d] for d in common_dates if d in bm_map]
        if len(bm_closes) >= 20:
            bm_daily_r = _daily_returns(bm_closes)
            bm_n = min(len(bm_daily_r), trading_days)
            bm_wealth_curve = [1.0]
            for r in bm_daily_r[:bm_n]:
                bm_wealth_curve.append(bm_wealth_curve[-1] * (1 + r))

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
        # 차트용 시계열 데이터
        "dates": common_dates[1:],  # wealth[0]=1.0 은 기준일, dates는 그 이후
        "wealth": wealth,
        "bm_wealth": bm_wealth_curve,
    }
