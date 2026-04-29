"""
포트폴리오 도구 — get_stock_correlation, simulate_portfolio
"""

import json
import logging
import re

from langchain_core.tools import tool

from src.llm.tools._helpers import _find_structured_data, _not_found_message, _fmt_date

logger = logging.getLogger(__name__)


@tool
def get_stock_correlation(ticker1: str, ticker2: str) -> str:
    """두 종목 간의 상관관계와 베타 계수를 분석합니다.
    "삼성전자와 SK하이닉스 상관관계", "KODEX 200과 삼성전자 베타", "두 종목 연동성" 등의 질문에 사용합니다.

    Args:
        ticker1: 첫 번째 종목 이름 또는 티커 (예: "삼성전자", "005930")
        ticker2: 두 번째 종목 이름 또는 티커 (예: "SK하이닉스", "000660", "KODEX 200")
    """
    # 종목 조회
    d1 = _find_structured_data(ticker1)
    d2 = _find_structured_data(ticker2)

    if not d1:
        return _not_found_message(ticker1)
    if not d2:
        return _not_found_message(ticker2)

    t1 = d1.get("ticker", "")
    t2 = d2.get("ticker", "")
    n1 = d1.get("name", "")
    n2 = d2.get("name", "")

    try:
        from src.data.technical import calc_correlation, calc_beta, MARKET_BENCHMARK

        lines = [f"[{n1} ({t1}) vs {n2} ({t2})] 상관관계 분석\n"]

        # 상관계수 계산
        corr = calc_correlation(t1, t2, days=120)
        if corr:
            c = corr["correlation"]
            if c >= 0.7:
                label = "강한 양의 상관관계 (동반 등락 경향)"
            elif c >= 0.3:
                label = "약한 양의 상관관계"
            elif c >= -0.3:
                label = "거의 무관 (분산투자 효과)"
            elif c >= -0.7:
                label = "약한 음의 상관관계"
            else:
                label = "강한 음의 상관관계 (반대 방향 등락)"
            lines.append(f"**상관계수:** {c:.4f} — {label}")
            lines.append(f"  분석 기간: {corr['period']} ({corr['data_days']}일)")
        else:
            lines.append("**상관계수:** 데이터 부족으로 계산 불가")

        # 베타 계수 — 각 종목에 대해 시장 벤치마크 기준
        lines.append("")
        for ticker, name in [(t1, n1), (t2, n2)]:
            beta = calc_beta(ticker, days=250)
            if beta:
                b = beta["beta"]
                if b > 1.2:
                    b_label = "공격적 (시장 대비 변동성 큼)"
                elif b > 0.8:
                    b_label = "시장 평균 수준"
                elif b > 0:
                    b_label = "방어적 (시장 대비 변동성 작음)"
                else:
                    b_label = "역방향 (시장과 반대 움직임)"
                lines.append(f"**{name} 베타:** {b:.3f} — {b_label}")
                lines.append(f"  벤치마크: KODEX 200 ({beta['benchmark']}), "
                             f"분석 기간: {beta['data_days']}일")
            else:
                lines.append(f"**{name} 베타:** 데이터 부족으로 계산 불가")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"상관관계/베타 계산 실패: {e}")
        return f"상관관계 분석에 실패했습니다. (데이터 부족 또는 오류)"


@tool
def simulate_portfolio(tickers_and_weights: str, period: str = "1y") -> str:
    """포트폴리오를 구성하여 과거 데이터 기반 시뮬레이션(백테스트)을 수행합니다.
    총 수익률, 연환산 수익률, 최대 낙폭(MDD), 샤프 비율, 변동성을 계산합니다.
    "삼성전자 50% SK하이닉스 50% 1년 백테스트", "포트폴리오 시뮬레이션" 등의 질문에 사용합니다.

    Args:
        tickers_and_weights: 종목과 비중 (예: "삼성전자 50%, SK하이닉스 50%")
        period: 시뮬레이션 기간 (예: "6m", "1y", "2y", "3y", "5y"). 기본값 1y
    """
    # 기간 파싱
    period_map = {"6m": 125, "1y": 250, "2y": 500, "3y": 750, "5y": 1250}
    days = period_map.get(period.lower().strip(), 250)

    # 종목+비중 파싱: "삼성전자 50%, SK하이닉스 50%" 또는 "삼성전자:50 SK하이닉스:50"
    text = tickers_and_weights.replace(":", " ").replace(",", " ")
    parts = re.split(r'(\d+(?:\.\d+)?)\s*%?\s*', text)

    resolved_tickers = []
    resolved_weights = []
    resolved_names = []

    i = 0
    while i < len(parts):
        name_part = parts[i].strip()
        if name_part and i + 1 < len(parts) and parts[i + 1].strip():
            # 종목명 뒤에 비중
            data = _find_structured_data(name_part)
            if data:
                resolved_tickers.append(data["ticker"])
                resolved_names.append(data["name"])
                resolved_weights.append(float(parts[i + 1].strip()))
                i += 2
                continue
        elif name_part:
            # 비중 없이 종목만 있는 경우
            data = _find_structured_data(name_part)
            if data:
                resolved_tickers.append(data["ticker"])
                resolved_names.append(data["name"])
                resolved_weights.append(0)  # 나중에 균등 배분
        i += 1

    if not resolved_tickers:
        return "종목을 찾을 수 없습니다. '삼성전자 50%, SK하이닉스 50%' 형식으로 입력해주세요."

    # 비중 없으면 균등 배분
    if all(w == 0 for w in resolved_weights):
        resolved_weights = [100 / len(resolved_tickers)] * len(resolved_tickers)

    # 비중 정규화 (합=1)
    w_sum = sum(resolved_weights)
    norm_weights = [w / w_sum for w in resolved_weights]

    try:
        from src.data.technical import simulate_portfolio as _sim

        result = _sim(resolved_tickers, norm_weights, days=days)
        if not result:
            return "시뮬레이션 데이터가 부족합니다 (최소 20영업일 필요)."

        p = result["portfolio"]
        lines = [f"[포트폴리오 시뮬레이션] 기간: {_fmt_date(result['period'].split('~')[0])}"
                 f" ~ {_fmt_date(result['period'].split('~')[1])}"
                 f" ({result['data_days']}영업일)\n"]

        # 구성
        lines.append("**포트폴리오 구성:**")
        for name, ticker, w in zip(resolved_names, resolved_tickers, norm_weights):
            lines.append(f"  - {name} ({ticker}): {w * 100:.1f}%")

        # 성과
        lines.append(f"\n**포트폴리오 성과:**")
        lines.append(f"  - 총 수익률: {p['total_return'] * 100:+.2f}%")
        lines.append(f"  - 연환산 수익률: {p['annualized_return'] * 100:+.2f}%")
        lines.append(f"  - 변동성 (연환산): {p['volatility'] * 100:.2f}%")
        lines.append(f"  - 샤프 비율: {p['sharpe_ratio']:.2f}")
        lines.append(f"  - 최대 낙폭(MDD): {p['max_drawdown'] * 100:.2f}%")

        # 벤치마크 비교
        bm = result.get("benchmark")
        if bm:
            lines.append(f"\n**벤치마크 비교 ({bm['name']}):**")
            lines.append(f"  - 총 수익률: {bm['total_return'] * 100:+.2f}%")
            lines.append(f"  - 연환산 수익률: {bm['annualized_return'] * 100:+.2f}%")
            lines.append(f"  - 변동성: {bm['volatility'] * 100:.2f}%")
            lines.append(f"  - 샤프 비율: {bm['sharpe_ratio']:.2f}")
            lines.append(f"  - MDD: {bm['max_drawdown'] * 100:.2f}%")
            excess = p['total_return'] - bm['total_return']
            lines.append(f"  - **포트폴리오 초과 수익:** {excess * 100:+.2f}%p")
            if bm.get('tracking_error'):
                lines.append(f"  - 트래킹 에러: {bm['tracking_error'] * 100:.2f}%")

        # 개별
        lines.append(f"\n**개별 종목 수익률:**")
        for item, name in zip(result["individual"], resolved_names):
            lines.append(f"  - {name}: {item['total_return'] * 100:+.2f}%")

        lines.append("\n※ 과거 수익률은 미래 수익을 보장하지 않습니다. 참고용 시뮬레이션입니다.")
        text_result = "\n".join(lines)

        # 포트폴리오 차트 생성
        try:
            from src.data.chart_generator import generate_portfolio_chart
            wealth = result.get("wealth")
            bm_wealth = result.get("bm_wealth")
            dates = result.get("dates")
            if wealth and dates:
                chart_b64 = generate_portfolio_chart(
                    wealth=wealth,
                    bm_wealth=bm_wealth,
                    dates=dates,
                    names=resolved_names,
                )
                if chart_b64:
                    chart_json = json.dumps({
                        "__type__": "portfolio_chart",
                        "image_b64": chart_b64,
                        "names": resolved_names,
                    })
                    return f"{chart_json}\n\n---\n\n{text_result}"
        except Exception as e:
            logger.warning(f"포트폴리오 차트 생성 실패: {e}")

        return text_result

    except Exception as e:
        logger.warning(f"포트폴리오 시뮬레이션 실패: {e}")
        return f"포트폴리오 시뮬레이션에 실패했습니다. (데이터 부족 또는 오류)"
