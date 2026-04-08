import streamlit as st

from config import is_langsmith_enabled
from src.utils.logging import get_performance_stats


def render_sidebar(etf_data: list, stock_data: list = None):
    """사이드바 전체 렌더링"""
    with st.sidebar:
        _render_service_info(has_stocks=bool(stock_data))
        st.divider()
        _render_market_data(etf_data, stock_data or [])
        st.divider()
        _render_investment_warning()
        st.divider()
        _render_performance_dashboard()
        if is_langsmith_enabled():
            st.divider()
            st.caption("LangSmith 트레이싱 활성화됨")


def _render_service_info(has_stocks: bool = False):
    st.header("ℹ️ 서비스 안내")
    stock_info = """
        - 개별 주식 정보 검색
        - 주식 펀더멘털 (PER/PBR/배당)
    """ if has_stocks else ""

    st.markdown(f"""
        이 챗봇은 **ETF/주식 투자 정보**를 제공합니다.

        **주요 기능:**
        - ETF 상품 정보 검색
        - ETF 비교 분석
        - 투자 전략/위험 분석{stock_info}

        **지원 ETF:**
        - 국내 주식형 (KODEX 200 등)
        - 해외 주식형 (S&P500, 나스닥100)
        - 섹터/테마형 (2차전지, 반도체)
        - 채권형, 배당형, 인버스형
        """)


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


def _render_etf_list(etf_data: list):
    # 수집 데이터: 거래대금 상위 20개만 표시
    display_data = etf_data
    if len(etf_data) > 20 and "trade_value" in etf_data[0]:
        display_data = sorted(etf_data, key=lambda e: e.get("trade_value", 0), reverse=True)[:20]
        st.caption(f"거래대금 상위 20개 (전체 {len(etf_data)}종목)")

    for etf in display_data:
        with st.expander(f"{etf['name']} ({etf['ticker']})"):
            if "category" in etf:
                # 하드코딩 포맷
                st.write(f"**카테고리:** {etf['category']}")
                st.write(f"**위험등급:** {etf['risk_level']}")
                st.write(f"**총보수:** {etf['total_expense_ratio']}")
            else:
                # 수집 데이터 포맷
                st.write(f"**종가:** {etf.get('close', 0):,}원")
                st.write(f"**등락률:** {etf.get('change_pct', 0):+.2f}%")
                st.write(f"**거래대금:** {etf.get('trade_value', 0):,}원")


def _render_stock_list(stock_data: list):
    # 거래대금 상위 20개
    display_data = sorted(stock_data, key=lambda s: s.get("trade_value", 0), reverse=True)[:20]
    st.caption(f"거래대금 상위 20개 (전체 {len(stock_data)}종목)")

    for s in display_data:
        with st.expander(f"{s['name']} ({s['ticker']})"):
            st.write(f"**종가:** {s.get('close', 0):,}원")
            change = s.get("change_pct", 0)
            st.write(f"**등락률:** {change:+.2f}%")
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
    st.warning("""
        ⚠️ **투자 유의사항**

        본 서비스는 정보 제공 목적이며,
        투자 권유가 아닙니다.
        투자 결정은 본인의 판단과
        책임 하에 이루어져야 합니다.
        """)


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
