"""
재무/예측/뉴스 도구 — get_financial_statements, predict_price_outlook, get_stock_news
"""

import json
import logging

from langchain_core.tools import tool

from src.llm.tools import _state
from src.llm.tools._helpers import _find_structured_data, _not_found_message

logger = logging.getLogger(__name__)


@tool
def get_financial_statements(name_or_ticker: str, quarters: int = 4) -> str:
    """기업의 분기별 재무제표(매출액, 영업이익, 당기순이익)를 조회합니다.
    실적, 매출, 영업이익, 순이익, 영업이익률, 성장률 관련 질문에 사용합니다.

    Args:
        name_or_ticker: 기업명 또는 종목코드 (예: "삼성전자", "005930")
        quarters: 조회할 분기 수 (기본 4분기)
    """
    # 종목 식별
    data = _state._stock_data_index.get(name_or_ticker) or _state._stock_data_index.get(name_or_ticker.lower())
    if not data:
        # 부분 매칭
        for key, val in _state._stock_data_index.items():
            if name_or_ticker.lower() in key:
                data = val
                break
    if not data:
        return _not_found_message(name_or_ticker)

    ticker = data.get("ticker", "")
    name = data.get("name", "")

    try:
        from src.data.database import get_financial_data, get_connection
        conn = get_connection()
        fin_data = get_financial_data(conn, ticker, quarters=quarters)
        conn.close()
    except Exception:
        fin_data = []

    if not fin_data:
        # deploy 데이터의 financial_summary fallback
        fs = data.get("financial_summary")
        if fs and (fs.get("revenue") or fs.get("operating_profit")):
            fy = fs.get("fiscal_year", "")
            fq = fs.get("fiscal_quarter", "")
            rev = fs.get("revenue")
            op = fs.get("operating_profit")
            ni = fs.get("net_income")
            om = fs.get("operating_margin")

            def _fmt(v):
                if v is None:
                    return "-"
                if abs(v) >= 1_0000_0000_0000:
                    return f"{v / 1_0000_0000_0000:.1f}조"
                if abs(v) >= 1_0000_0000:
                    return f"{v / 1_0000_0000:,.0f}억"
                return f"{v:,}"

            lines = [f"## {name}({ticker}) 최근 분기 재무제표\n"]
            lines.append("| 분기 | 매출액 | 영업이익 | 순이익 | 영업이익률 |")
            lines.append("|------|--------|----------|--------|-----------|")
            om_str = f"{om:+.1f}%" if om is not None else "-"
            lines.append(f"| {fy}Q{fq} | {_fmt(rev)} | {_fmt(op)} | {_fmt(ni)} | {om_str} |")
            lines.append(f"\n*deploy 데이터 기준 (최근 1분기만 표시)*")
            return "\n".join(lines)

        return (
            f"{name}({ticker})의 재무제표 데이터가 아직 수집되지 않았습니다.\n"
            f"(OpenDart API 키 설정 후 `python -m src.data.dart_collector`로 수집 가능)"
        )

    # 포맷팅
    lines = [f"## {name}({ticker}) 분기별 재무제표\n"]
    lines.append("| 분기 | 매출액 | 영업이익 | 순이익 | 영업이익률 | 매출 YoY | 영업이익 YoY |")
    lines.append("|------|--------|----------|--------|-----------|----------|-------------|")

    for d in fin_data:
        year = d.get("fiscal_year", "")
        q = d.get("fiscal_quarter", "")

        rev = d.get("revenue")
        op = d.get("operating_profit")
        ni = d.get("net_income")
        om = d.get("operating_margin")
        rg = d.get("revenue_growth_yoy")
        og = d.get("op_growth_yoy")

        def fmt_amount(v):
            if v is None:
                return "-"
            if abs(v) >= 1_0000_0000_0000:
                return f"{v / 1_0000_0000_0000:.1f}조"
            if abs(v) >= 1_0000_0000:
                return f"{v / 1_0000_0000:,.0f}억"
            return f"{v:,}"

        def fmt_pct(v):
            if v is None:
                return "-"
            return f"{v:+.1f}%"

        lines.append(
            f"| {year}Q{q} | {fmt_amount(rev)} | {fmt_amount(op)} | "
            f"{fmt_amount(ni)} | {fmt_pct(om)} | {fmt_pct(rg)} | {fmt_pct(og)} |"
        )

    return "\n".join(lines)


@tool
def predict_price_outlook(name_or_ticker: str, horizon: str = "1m") -> str:
    """종목의 가격 전망을 기술적 분석 + 펀더멘털 분석 + 통계 모델로 종합 예측합니다.
    상승/횡보/하락 시나리오별 확률, 신뢰도 등급, 리스크 요인을 제공합니다.
    "삼성전자 앞으로 어떨까", "SK하이닉스 전망", "KODEX 200 1개월 예측", "삼성전자 오를까" 등의 질문에 사용합니다.

    Args:
        name_or_ticker: ETF/주식 이름 또는 티커 (예: "삼성전자", "005930", "KODEX 200")
        horizon: 예측 기간 ("1w": 1주, "2w": 2주, "1m": 1개월, "3m": 3개월, "6m": 6개월, "1y": 1년). 기본값 1m
    """
    data = _find_structured_data(name_or_ticker)
    if not data:
        return _not_found_message(name_or_ticker)

    ticker = data.get("ticker", "")
    name = data.get("name", "")

    # 기술적 요약 조회
    try:
        from src.data.technical import get_technical_summary
        summary = get_technical_summary(ticker)
    except Exception as e:
        logger.warning(f"기술적 지표 조회 실패: {e}")
        summary = None

    # 예측 모델 실행
    try:
        from src.data.predictor import build_price_outlook
        outlook = build_price_outlook(
            ticker, name, horizon=horizon,
            summary=summary, structured_data=data,
        )
    except Exception as e:
        logger.error(f"가격 전망 생성 실패: {e}")
        return f"'{name}'의 가격 전망 분석에 실패했습니다. ({type(e).__name__})"

    # 포맷팅
    horizon_labels = {"1w": "1주", "2w": "2주", "1m": "1개월", "3m": "3개월", "6m": "6개월", "1y": "1년"}
    h_label = horizon_labels.get(outlook["horizon"], outlook["horizon"])
    price = outlook.get("current_price", 0)

    lines = [f"[{name} ({ticker})] 가격 전망 분석 — {h_label} ({outlook['horizon_days']}영업일)\n"]

    if price:
        lines.append(f"**현재가:** {price:,}원\n")

    # 종합 점수
    cs = outlook["composite_score"]
    if cs > 0.3:
        outlook_label = "강세 전망"
    elif cs > 0.1:
        outlook_label = "약한 강세"
    elif cs > -0.1:
        outlook_label = "중립"
    elif cs > -0.3:
        outlook_label = "약한 약세"
    else:
        outlook_label = "약세 전망"
    lines.append(f"**종합 판단:** {outlook_label} (점수: {cs:+.3f}, 신뢰도: {outlook['confidence_grade']}등급)\n")

    # 시나리오별 확률
    scenarios = outlook["scenarios"]
    lines.append("**시나리오별 확률:**")
    bull = scenarios["bullish"]
    lines.append(f"  📈 상승: {bull['probability']*100:.0f}% (목표 {bull['target_return']:+.1f}%) — {bull['description']}")
    neut = scenarios["neutral"]
    lines.append(f"  ➡️ 횡보: {neut['probability']*100:.0f}% (예상 {neut['target_return']:+.1f}%)")
    bear = scenarios["bearish"]
    lines.append(f"  📉 하락: {bear['probability']*100:.0f}% (목표 {bear['target_return']:+.1f}%) — {bear['description']}")

    # 기술적 분석 요약
    tech = outlook["technical"]
    lines.append(f"\n**기술적 분석:** {tech['signal']} (점수: {tech['score']:+.3f})")
    for f in tech["key_factors"]:
        lines.append(f"  {f}")

    # 펀더멘털 분석 요약
    fund = outlook["fundamental"]
    lines.append(f"\n**펀더멘털 분석:** {fund['signal']} (점수: {fund['score']:+.3f})")
    for f in fund["key_factors"]:
        lines.append(f"  {f}")

    # 통계 모델
    stat = outlook["statistical"]
    lines.append(f"\n**통계 모델:**")
    lines.append(f"  - 예측 수익률: {stat['predicted_return']:+.2f}% "
                 f"(신뢰구간: {stat['confidence_interval'][0]:+.2f}% ~ {stat['confidence_interval'][1]:+.2f}%)")
    lines.append(f"  - 과거 유사 패턴 승률: {stat['historical_win_rate']*100:.0f}% "
                 f"(표본 {stat['historical_sample_count']}건)")
    lines.append(f"  - 모델 설명력: R²={stat['model_r2']:.4f} ({stat['model_reliability']})")

    # Prophet 시계열 예측
    prophet = outlook.get("prophet", {})
    if prophet.get("available"):
        lines.append(f"\n**Prophet 시계열 예측:**")
        lines.append(f"  - 예측 수익률: {prophet['predicted_return']:+.2f}% "
                     f"(신뢰구간: {prophet['confidence_interval'][0]:+.2f}% ~ {prophet['confidence_interval'][1]:+.2f}%)")
        lines.append(f"  - 추세 방향: {prophet['trend']}")

    # 리스크 요인
    risks = outlook.get("risk_factors", [])
    if risks:
        lines.append(f"\n**리스크 요인:**")
        for r in risks:
            lines.append(f"  {r}")

    # 면책
    lines.append(f"\n⚠️ 본 분석은 과거 데이터 기반 통계적 참고 자료이며, 미래 수익을 보장하지 않습니다. "
                 f"투자 판단은 본인의 책임입니다.")

    text_result = "\n".join(lines)

    # 차트도 함께 제공
    try:
        from src.data.chart_generator import generate_technical_chart
        chart_b64 = generate_technical_chart(ticker, name, days=120)
        if chart_b64:
            chart_json = json.dumps(
                {"__type__": "technical_chart", "image_b64": chart_b64, "name": name},
                ensure_ascii=False,
            )
            return f"{chart_json}\n\n---\n\n{text_result}"
    except Exception as e:
        logger.warning(f"차트 생성 실패: {e}")

    return text_result


@tool
def get_stock_news(name_or_ticker: str, max_articles: int = 8) -> str:
    """종목 관련 최근 뉴스 수집 + 감성 분석. 뉴스, 이슈, 시장 반응, 여론 등을 확인할 때 사용.

    Args:
        name_or_ticker: 종목명 또는 티커 (예: "삼성전자", "005930")
        max_articles: 최대 기사 수 (기본 8)
    """
    from src.data.news import get_stock_news_summary

    # 종목명 확인
    data = _find_structured_data(name_or_ticker)
    if data:
        name = data.get("name", name_or_ticker)
    else:
        name = name_or_ticker

    result = get_stock_news_summary(name, max_articles=max_articles)

    # 포맷팅
    lines = [f"## 📰 {name} 최근 뉴스 분석\n"]

    # 전체 감성
    sentiment_emoji = {
        "긍정": "🟢", "부정": "🔴", "중립": "⚪", "혼재": "🟡",
    }
    emoji = sentiment_emoji.get(result["overall_sentiment"], "⚪")
    src_tag = " (로컬 모델)" if result.get("sentiment_source") == "local" else ""
    lines.append(f"**전체 감성:** {emoji} {result['overall_sentiment']}{src_tag}")
    lines.append(
        f"  긍정 {result['positive_count']}건 / "
        f"부정 {result['negative_count']}건 / "
        f"중립 {result['neutral_count']}건"
    )

    if result["key_topics"]:
        lines.append(f"\n**주요 키워드:** {', '.join(result['key_topics'])}")

    if result["summary"]:
        lines.append(f"\n**뉴스 흐름 요약:** {result['summary']}")

    # 개별 기사
    articles = result.get("articles", [])
    if articles:
        lines.append(f"\n### 최근 기사 ({len(articles)}건)\n")
        for i, a in enumerate(articles, 1):
            sent = a.get("sentiment", "")
            sent_mark = {"긍정": "🟢", "부정": "🔴", "중립": "⚪"}.get(sent, "")
            source_str = f" ({a['source']})" if a.get("source") else ""
            lines.append(f"{i}. {sent_mark} **{a['title']}**{source_str}")
            if a.get("published"):
                lines.append(f"   📅 {a['published']}")

    return "\n".join(lines)
