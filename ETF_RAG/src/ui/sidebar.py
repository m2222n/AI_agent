import streamlit as st

from src.utils.logging import get_performance_stats


def render_sidebar(etf_data: list):
    """사이드바 전체 렌더링"""
    with st.sidebar:
        _render_service_info()
        st.divider()
        _render_etf_list(etf_data)
        st.divider()
        _render_investment_warning()
        st.divider()
        _render_performance_dashboard()


def _render_service_info():
    st.header("ℹ️ 서비스 안내")
    st.markdown("""
        이 챗봇은 **ETF 투자 정보**를 제공합니다.

        **주요 기능:**
        - ETF 상품 정보 검색
        - 투자 전략 설명
        - 위험도/수수료 비교
        - 배당 정책 안내

        **지원 ETF:**
        - 국내 주식형 (KODEX 200 등)
        - 해외 주식형 (S&P500, 나스닥100)
        - 섹터/테마형 (2차전지, 전기차)
        - 채권형 (단기채권)
        - 배당형, 인버스형
        """)


def _render_etf_list(etf_data: list):
    st.header("📊 ETF 목록")
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
