import streamlit as st

from config import is_langsmith_enabled
from src.utils.logging import get_performance_stats, get_feedback_stats


def render_sidebar(etf_data: list, stock_data: list = None):
    """사이드바 전체 렌더링"""
    with st.sidebar:
        _render_service_info(has_stocks=bool(stock_data))
        st.divider()
        _render_data_summary(etf_data, stock_data or [])
        _render_market_data(etf_data, stock_data or [])
        st.divider()
        _render_investment_warning()
        st.divider()
        _render_performance_dashboard()
        if is_langsmith_enabled():
            st.divider()
            st.caption("🔗 LangSmith 트레이싱 활성화됨")


def _render_service_info(has_stocks: bool = False):
    st.header("ℹ️ 서비스 안내")
    features = [
        "ETF 상품 정보 검색",
        "ETF 비교 분석 (차트 자동 생성)",
        "투자 전략/위험 분석",
    ]
    if has_stocks:
        features.extend([
            "개별 주식 정보 검색",
            "주식 펀더멘털 (PER/PBR/배당)",
        ])

    st.markdown("이 챗봇은 **ETF/주식 투자 정보**를 제공합니다.")
    for f in features:
        st.markdown(f"- {f}")


def _render_data_summary(etf_data: list, stock_data: list):
    """데이터 현황 요약 메트릭"""
    cols = st.columns(2 if stock_data else 1)
    with cols[0]:
        st.metric("ETF", f"{len(etf_data)}종목")
    if stock_data:
        with cols[1]:
            st.metric("주식", f"{len(stock_data)}종목")

    # 데이터 기준일
    if etf_data and "date" in etf_data[0]:
        date_str = etf_data[0]["date"]
        if len(date_str) == 8:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        st.caption(f"📅 기준일: {date_str}")


def _render_market_data(etf_data: list, stock_data: list):
    """ETF/주식 탭 분리 표시"""
    if stock_data:
        tab_etf, tab_stock = st.tabs(["📊 ETF", "📈 주식"])
        with tab_etf:
            _render_etf_list(etf_data)
        with tab_stock:
            _render_stock_list(stock_data)
    else:
        _render_etf_list(etf_data)


def _format_change(pct: float) -> str:
    """등락률을 색상 이모지와 함께 포맷"""
    if pct > 0:
        return f"🔴 +{pct:.2f}%"
    elif pct < 0:
        return f"🔵 {pct:.2f}%"
    return f"⚪ {pct:.2f}%"


def _format_trade_value(value: int) -> str:
    """거래대금을 읽기 쉬운 형태로"""
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}조"
    elif value >= 100_000_000:
        return f"{value / 100_000_000:.0f}억"
    elif value >= 10_000:
        return f"{value / 10_000:.0f}만"
    return f"{value:,}"


def _render_etf_list(etf_data: list):
    # 수집 데이터: 거래대금 상위 20개만 표시
    display_data = etf_data
    if len(etf_data) > 20 and "trade_value" in etf_data[0]:
        display_data = sorted(etf_data, key=lambda e: e.get("trade_value", 0), reverse=True)[:20]
        st.caption(f"거래대금 상위 20개 (전체 {len(etf_data)}종목)")

    for etf in display_data:
        change_pct = etf.get("change_pct", 0)
        change_indicator = _format_change(change_pct) if "close" in etf else ""
        label = f"{etf['name']}  {change_indicator}" if change_indicator else etf["name"]

        with st.expander(label):
            if "category" in etf:
                # 하드코딩 포맷
                st.write(f"**카테고리:** {etf['category']}")
                st.write(f"**위험등급:** {etf['risk_level']}")
                st.write(f"**총보수:** {etf['total_expense_ratio']}")
            else:
                # 수집 데이터 포맷
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**종가:** {etf.get('close', 0):,}원")
                with col2:
                    tv = etf.get("trade_value", 0)
                    st.write(f"**거래대금:** {_format_trade_value(tv)}")


def _render_stock_list(stock_data: list):
    # 거래대금 상위 20개
    display_data = sorted(stock_data, key=lambda s: s.get("trade_value", 0), reverse=True)[:20]
    st.caption(f"거래대금 상위 20개 (전체 {len(stock_data)}종목)")

    for s in display_data:
        change_pct = s.get("change_pct", 0)
        change_indicator = _format_change(change_pct)
        label = f"{s['name']}  {change_indicator}"

        with st.expander(label):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**종가:** {s.get('close', 0):,}원")
            with col2:
                tv = s.get("trade_value", 0)
                st.write(f"**거래대금:** {_format_trade_value(tv)}")
            # 펀더멘털
            per = s.get("per", 0)
            pbr = s.get("pbr", 0)
            if per:
                st.write(f"**PER:** {per:.1f}배 | **PBR:** {pbr:.2f}배")
            # 시가총액
            market_cap = s.get("market_cap", 0)
            if market_cap >= 1_000_000_000_000:
                st.write(f"**시가총액:** {market_cap / 1_000_000_000_000:.1f}조원")
            elif market_cap >= 100_000_000:
                st.write(f"**시가총액:** {market_cap / 100_000_000:.0f}억원")


def _render_investment_warning():
    st.warning(
        "⚠️ **투자 유의사항**\n\n"
        "본 서비스는 정보 제공 목적이며, "
        "투자 권유가 아닙니다. "
        "투자 결정은 본인의 판단과 "
        "책임 하에 이루어져야 합니다."
    )


def _render_performance_dashboard():
    st.header("📊 성능 모니터링")
    stats = get_performance_stats()
    if stats:
        st.metric("총 질의 수", stats["total_queries"])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("평균 응답시간", f"{stats['avg_total_time_ms']:.0f}ms")
        with col2:
            st.metric("평균 검색시간", f"{stats['avg_search_time_ms']:.0f}ms")

        if stats["question_types"]:
            st.markdown("**질문 유형 분포:**")
            for q_type, count in stats["question_types"].items():
                pct = count / stats["total_queries"] * 100
                st.progress(pct / 100, text=f"{q_type}: {count}건 ({pct:.0f}%)")
    else:
        st.info("아직 통계 데이터가 없습니다.")

    # 피드백 통계
    fb_stats = get_feedback_stats()
    if fb_stats:
        st.markdown("**사용자 만족도:**")
        st.metric("만족도", f"{fb_stats['satisfaction_rate']}%",
                  f"{fb_stats['positive']}👍 / {fb_stats['negative']}👎")
