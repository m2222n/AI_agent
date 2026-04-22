"""
탭별 전용 UI 렌더러

각 탭은 LLM 에이전트 없이 직접 데이터 함수를 호출하여 결과를 보여준다.
- 기술적 분석: 종목 입력 → 11개 지표 + 차트
- 재무제표: 종목 입력 → 분기별 재무 테이블
- 비교 분석: 2개 종목 입력 → 비교 테이블 + 차트
- 가격 전망: 종목 입력 → 3축 예측 결과
"""

import base64
import logging
from typing import Optional

import streamlit as st

from src.data.chart_generator import generate_technical_chart, generate_comparison_chart
from src.data.technical import get_technical_summary
from src.data.predictor import build_price_outlook
from src.data.database import get_connection, get_financial_data, DB_PATH
from src.llm.tools import _find_structured_data, _find_similar_names, _etf_data_index, _stock_data_index

logger = logging.getLogger(__name__)


# ── 공통 헬퍼 ─────────────────────────────────────────────

def _build_autocomplete_options() -> list[str]:
    """인덱스에서 자동완성 옵션 목록 생성. '종목명 (티커)' 형식."""
    cached = st.session_state.get("_autocomplete_options")
    if cached:
        return cached

    seen = set()
    options = []
    for index in (_etf_data_index, _stock_data_index):
        for _key, data in index.items():
            ticker = data.get("ticker", "")
            name = data.get("name", "")
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            options.append(f"{name} ({ticker})")

    options.sort()
    if options:  # 비어있으면 캐시하지 않음 (인덱스 미초기화 방어)
        st.session_state["_autocomplete_options"] = options
    return options


def _resolve_ticker(name_or_ticker: str) -> Optional[dict]:
    """종목명/티커 → 구조화 데이터 조회. 실패 시 유사 종목 제안."""
    data = _find_structured_data(name_or_ticker)
    if data:
        return data

    # 유사 종목 제안
    similar = _find_similar_names(name_or_ticker, max_results=5)
    if similar:
        names = ", ".join(f"**{s['name']}**({s['ticker']})" for s in similar)
        st.warning(f"'{name_or_ticker}'을(를) 찾을 수 없습니다. 유사 종목: {names}")
    else:
        st.warning(f"'{name_or_ticker}'을(를) 찾을 수 없습니다.")
    return None


def _filter_options(query: str) -> list[str]:
    """부분 매칭 필터링. 숫자 검색 시 '티커 (종목명)' 형식으로 표시."""
    if not query or len(query.strip()) == 0:
        return []
    options = _build_autocomplete_options()
    q = query.strip().lower()
    matched = [opt for opt in options if q in opt.lower()][:50]

    # 숫자로 검색 시 "티커 (종목명)" 형식으로 재포맷
    if q.isdigit() and matched:
        reformatted = []
        for opt in matched:
            if " (" in opt and opt.endswith(")"):
                name, ticker = opt.rsplit(" (", 1)
                ticker = ticker.rstrip(")")
                reformatted.append(f"{ticker} ({name})")
            else:
                reformatted.append(opt)
        return reformatted

    return matched


def _extract_name(display: str) -> str:
    """'종목명 (티커)' 또는 '티커 (종목명)' → 종목명 추출."""
    if " (" in display and display.endswith(")"):
        left, right = display.rsplit(" (", 1)
        right = right.rstrip(")")
        if left.isdigit():
            return right  # "005930 (삼성전자)" → 삼성전자
        return left       # "삼성전자 (005930)" → 삼성전자
    return display


def _ticker_input(label: str, key: str, placeholder: str = "종목명 또는 티커 입력") -> str:
    """종목 입력 위젯 (text_input + selectbox 실시간 필터링)"""
    query = st.text_input(label, placeholder=placeholder, key=f"{key}_input")

    if not query or not query.strip():
        return ""

    filtered = _filter_options(query.strip())
    if not filtered:
        st.caption("검색 결과 없음")
        return ""

    if len(filtered) == 1:
        # 정확히 1개 매칭 → 바로 선택
        st.caption(f"✅ {filtered[0]}")
        return _extract_name(filtered[0])

    selected = st.selectbox(
        "종목 선택",
        filtered,
        key=f"{key}_select",
        label_visibility="collapsed",
    )
    if selected:
        return _extract_name(selected)
    return ""


# ── 기술적 분석 탭 ─────────────────────────────────────────

def render_technical_tab():
    """기술적 분석 탭: 종목 입력 → 11개 지표 + 차트"""
    st.markdown("##### 📊 기술적 분석")
    st.caption("종목명 또는 티커를 입력하면 11개 기술적 지표와 차트를 바로 확인합니다.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = _ticker_input("종목", "tech_ticker", "예: 삼성전자, 005930")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("분석", key="tech_run", use_container_width=True)

    if not run or not query:
        return

    data = _resolve_ticker(query)
    if not data:
        return

    ticker = data["ticker"]
    name = data.get("name", query)

    with st.spinner(f"{name} 기술적 지표 계산 중..."):
        summary = get_technical_summary(ticker)

    if not summary:
        st.error("기술적 지표 데이터가 부족합니다.")
        return

    # 차트 생성
    with st.spinner("차트 생성 중..."):
        chart_b64 = generate_technical_chart(ticker, name, days=120)

    if chart_b64:
        st.image(base64.b64decode(chart_b64), use_container_width=True)
        st.caption(
            "📊 상단: 종가 + MA(5/20/60) + 볼린저 밴드 | "
            "중단: RSI(14) — 70↑ 과매수, 30↓ 과매도 | "
            "하단: 거래량 + MACD 히스토그램"
        )

    # 지표 요약
    st.markdown(f"#### {name} ({ticker}) 기술적 지표")

    close = summary.get("close", 0)
    trend = summary.get("trend", "-")
    date = summary.get("date", "-")
    st.markdown(f"**기준일:** {date} | **종가:** {close:,}원 | **추세:** {trend}")

    # 이동평균선
    ma = summary.get("ma", {})
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("MA5", f"{ma.get('ma5', '-'):,}" if ma.get('ma5') else "-")
    col_b.metric("MA20", f"{ma.get('ma20', '-'):,}" if ma.get('ma20') else "-")
    col_c.metric("MA60", f"{ma.get('ma60', '-'):,}" if ma.get('ma60') else "-")
    col_d.metric("MA120", f"{ma.get('ma120', '-'):,}" if ma.get('ma120') else "-")

    # 크로스 신호
    cross = summary.get("cross", {})
    cross_texts = []
    for pair, label in [("5_20", "5/20"), ("20_60", "20/60"), ("60_120", "60/120")]:
        signal = cross.get(pair)
        if signal == "golden_cross":
            cross_texts.append(f"🟢 {label} 골든크로스")
        elif signal == "dead_cross":
            cross_texts.append(f"🔴 {label} 데드크로스")
    if cross_texts:
        st.info(" | ".join(cross_texts))

    # 보조 지표 (2열)
    left, right = st.columns(2)

    with left:
        st.markdown("**모멘텀 지표**")
        rsi = summary.get("rsi")
        if rsi is not None:
            rsi_signal = "과매수" if rsi > 70 else ("과매도" if rsi < 30 else "중립")
            st.metric("RSI(14)", f"{rsi:.1f}", delta=rsi_signal)

        macd = summary.get("macd")
        if macd:
            st.metric("MACD", f"{macd['macd']:.1f}",
                      delta=f"Signal {macd['signal']:.1f}")
            hist = macd.get("histogram", 0)
            st.caption(f"히스토그램: {hist:+.1f} ({'상승' if hist > 0 else '하락'} 모멘텀)")

        stoch = summary.get("stochastic")
        if stoch:
            st.metric("스토캐스틱 %K/%D", f"{stoch['k']:.1f} / {stoch['d']:.1f}",
                      delta=stoch.get("signal", ""))

        cci = summary.get("cci")
        if cci:
            st.metric("CCI(20)", f"{cci['cci']:.1f}", delta=cci.get("signal", ""))

    with right:
        st.markdown("**추세/변동성 지표**")
        bb = summary.get("bollinger")
        if bb:
            st.metric("볼린저 밴드", f"{bb['lower']:,.0f} ~ {bb['upper']:,.0f}",
                      delta=f"폭 {bb['width']:.1f}%")
            st.caption(f"%B: {bb['pct_b']:.2f}")

        adx = summary.get("adx")
        if adx:
            st.metric("ADX(14)", f"{adx['adx']:.1f}",
                      delta=adx.get("trend_strength", ""))
            st.caption(f"+DI {adx['plus_di']:.1f} / -DI {adx['minus_di']:.1f}")

        ichimoku = summary.get("ichimoku")
        if ichimoku:
            st.metric("일목균형표", ichimoku.get("cloud_status", "-"),
                      delta=f"전환 {ichimoku['tenkan']:,} / 기준 {ichimoku['kijun']:,}")

        obv = summary.get("obv")
        if obv:
            st.metric("OBV", obv.get("trend", "-"))

        atr = summary.get("atr")
        if atr:
            st.metric("ATR(14)", f"{atr['atr']:,.0f}원",
                      delta=f"{atr['atr_pct']:.1f}% ({atr.get('volatility', '')})")


# ── 재무제표 탭 ────────────────────────────────────────────

def render_financial_tab():
    """재무제표 탭: 종목 입력 → 분기별 재무 테이블"""
    st.markdown("##### 📑 재무제표")
    st.caption("2015년부터 재무제표를 조회합니다. (DART 공시 기준)")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = _ticker_input("종목", "fin_ticker", "예: 삼성전자, 005930")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("조회", key="fin_run", use_container_width=True)

    if not run or not query:
        return

    data = _resolve_ticker(query)
    if not data:
        return

    ticker = data["ticker"]
    name = data.get("name", query)

    if not DB_PATH.exists():
        st.error("데이터베이스 파일이 없습니다.")
        return

    conn = get_connection()
    try:
        # 전체 데이터 조회 (충분히 큰 수로 전체 가져옴)
        all_rows = get_financial_data(conn, ticker, quarters=200)
    finally:
        conn.close()

    if not all_rows:
        st.warning(f"{name}의 재무제표 데이터가 없습니다. (DART 미공시 종목이거나 ETF일 수 있습니다)")
        return

    # 분기 수 선택 (1 ~ 전체)
    max_q = len(all_rows)
    # 기본값: 8분기 또는 최대 분기 중 작은 값
    default_q = min(8, max_q)
    quarters = st.slider(
        "조회 분기 수", min_value=1, max_value=max_q,
        value=default_q, key="fin_quarters",
    )
    rows = all_rows[:quarters]

    if not rows:
        st.warning(f"{name}의 재무제표 데이터가 없습니다. (DART 미공시 종목이거나 ETF일 수 있습니다)")
        return

    st.markdown(f"#### {name} ({ticker}) 재무제표")

    # 테이블 렌더링
    import pandas as pd

    def _fmt_억(v):
        if v is None:
            return "-"
        return f"{v / 100_000_000:,.0f}억"

    def _fmt_pct(v):
        if v is None:
            return "-"
        return f"{v:+.1f}%"

    table_data = []
    for r in rows:
        table_data.append({
            "분기": f"{r['fiscal_year']}Q{r['fiscal_quarter']}",
            "매출액": _fmt_억(r.get("revenue")),
            "영업이익": _fmt_억(r.get("operating_profit")),
            "순이익": _fmt_억(r.get("net_income")),
            "영업이익률": _fmt_pct(r.get("operating_margin")),
            "매출 YoY": _fmt_pct(r.get("revenue_growth_yoy")),
            "영업이익 YoY": _fmt_pct(r.get("op_growth_yoy")),
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 추이 차트 (매출/영업이익)
    chart_rows = [r for r in reversed(rows) if r.get("revenue")]
    if len(chart_rows) >= 2:
        st.markdown("#### 실적 추이")
        chart_df = pd.DataFrame([
            {
                "분기": f"{r['fiscal_year']}Q{r['fiscal_quarter']}",
                "매출액(억)": r["revenue"] / 100_000_000 if r.get("revenue") else 0,
                "영업이익(억)": r["operating_profit"] / 100_000_000 if r.get("operating_profit") else 0,
            }
            for r in chart_rows
        ]).set_index("분기")
        st.bar_chart(chart_df)


# ── 비교 분석 탭 ───────────────────────────────────────────

def render_comparison_tab():
    """비교 분석 탭: 2개 종목 입력 → 비교 테이블 + 차트"""
    st.markdown("##### ⚖️ 비교 분석")
    st.caption("2개 종목을 입력하면 가격, 수익률, 밸류에이션을 비교합니다.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        q1 = _ticker_input("종목 1", "cmp_ticker1", "예: 삼성전자")
    with col2:
        q2 = _ticker_input("종목 2", "cmp_ticker2", "예: SK하이닉스")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("비교", key="cmp_run", use_container_width=True)

    if not run or not q1 or not q2:
        return

    data1 = _resolve_ticker(q1)
    data2 = _resolve_ticker(q2)
    if not data1 or not data2:
        return

    name1, name2 = data1.get("name", q1), data2.get("name", q2)
    ticker1, ticker2 = data1["ticker"], data2["ticker"]

    st.markdown(f"#### {name1} vs {name2}")

    # 기본 정보 비교 테이블
    def _val(d, key, fmt="{:,}"):
        v = d.get(key)
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.2f}"
        return fmt.format(v)

    def _pct(d, key):
        v = d.get(key)
        if v is None:
            return "-"
        return f"{v:+.2f}%"

    rows = [
        ("종가", f"{data1.get('close', 0):,}원", f"{data2.get('close', 0):,}원"),
        ("등락률", _pct(data1, "change_pct"), _pct(data2, "change_pct")),
    ]

    # 수익률
    returns1 = data1.get("returns", {})
    returns2 = data2.get("returns", {})
    for period, label in [("1d", "1일"), ("1w", "1주"), ("1m", "1개월"), ("3m", "3개월"), ("1y", "1년")]:
        r1 = returns1.get(period)
        r2 = returns2.get(period)
        if r1 is not None or r2 is not None:
            rows.append((f"{label} 수익률",
                         f"{r1:+.2f}%" if r1 is not None else "-",
                         f"{r2:+.2f}%" if r2 is not None else "-"))

    # 주식 전용 (PER/PBR/시가총액 등)
    for key, label, fmt in [
        ("market_cap", "시가총액", None),
        ("per", "PER", "{:.2f}배"),
        ("pbr", "PBR", "{:.2f}배"),
        ("eps", "EPS", "{:,.0f}원"),
        ("div", "배당수익률", "{:.2f}%"),
    ]:
        v1, v2 = data1.get(key), data2.get(key)
        if v1 is not None or v2 is not None:
            if key == "market_cap":
                from src.ui.charts import _format_market_cap
                rows.append((label, _format_market_cap(v1 or 0), _format_market_cap(v2 or 0)))
            else:
                rows.append((label,
                             fmt.format(v1) if v1 is not None else "-",
                             fmt.format(v2) if v2 is not None else "-"))

    # ETF 전용 (NAV)
    for key, label, fmt in [("nav", "NAV", "{:,.0f}원")]:
        v1, v2 = data1.get(key), data2.get(key)
        if v1 is not None or v2 is not None:
            rows.append((label,
                         fmt.format(v1) if v1 is not None else "-",
                         fmt.format(v2) if v2 is not None else "-"))

    header = f"| 항목 | {name1} | {name2} |"
    sep = "|------|------|------|"
    body = "\n".join(f"| {label} | {va} | {vb} |" for label, va, vb in rows)
    st.markdown(f"{header}\n{sep}\n{body}")

    # 수익률 바 차트
    import pandas as pd
    return_periods = [("1d", "1일"), ("1w", "1주"), ("1m", "1개월"), ("3m", "3개월"), ("1y", "1년")]
    chart_data = {"기간": [], name1: [], name2: []}
    for p, label in return_periods:
        r1 = returns1.get(p)
        r2 = returns2.get(p)
        if r1 is not None or r2 is not None:
            chart_data["기간"].append(label)
            chart_data[name1].append(r1 or 0)
            chart_data[name2].append(r2 or 0)
    if chart_data["기간"]:
        st.markdown("#### 수익률 비교")
        df = pd.DataFrame(chart_data).set_index("기간")
        st.bar_chart(df)

    # 상대 수익률 추이 차트
    with st.spinner("비교 차트 생성 중..."):
        chart_b64 = generate_comparison_chart([ticker1, ticker2], [name1, name2], days=120)
    if chart_b64:
        st.markdown("#### 기간별 상대 수익률 추이")
        st.image(base64.b64decode(chart_b64), use_container_width=True)

    # 재무제표 비교 (DB 있을 때만)
    if DB_PATH.exists():
        conn = get_connection()
        try:
            fin1 = get_financial_data(conn, ticker1, quarters=1)
            fin2 = get_financial_data(conn, ticker2, quarters=1)
        finally:
            conn.close()

        if fin1 and fin2:
            f1, f2 = fin1[0], fin2[0]
            st.markdown("#### 최근 분기 실적 비교")
            def _fmt_억(v):
                if v is None:
                    return "-"
                return f"{v / 100_000_000:,.0f}억"
            fin_rows = [
                ("기준", f"{f1['fiscal_year']}Q{f1['fiscal_quarter']}",
                         f"{f2['fiscal_year']}Q{f2['fiscal_quarter']}"),
                ("매출액", _fmt_억(f1.get("revenue")), _fmt_억(f2.get("revenue"))),
                ("영업이익", _fmt_억(f1.get("operating_profit")), _fmt_억(f2.get("operating_profit"))),
                ("순이익", _fmt_억(f1.get("net_income")), _fmt_억(f2.get("net_income"))),
                ("영업이익률",
                 f"{f1.get('operating_margin', 0):.1f}%" if f1.get("operating_margin") is not None else "-",
                 f"{f2.get('operating_margin', 0):.1f}%" if f2.get("operating_margin") is not None else "-"),
            ]
            header = f"| 항목 | {name1} | {name2} |"
            sep = "|------|------|------|"
            body = "\n".join(f"| {l} | {a} | {b} |" for l, a, b in fin_rows)
            st.markdown(f"{header}\n{sep}\n{body}")


# ── 가격 전망 탭 ──────────────────────────────────────────

def render_outlook_tab():
    """가격 전망 탭: 종목 입력 → 3축 예측 결과"""
    st.markdown("##### 🔮 가격 전망")
    st.caption("기술적 분석 + 펀더멘털 + Ridge 회귀 모델을 종합한 전망입니다.")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = _ticker_input("종목", "outlook_ticker", "예: 삼성전자, 005930")
    with col2:
        horizon = st.selectbox("예측 기간", ["1m", "3m", "6m", "1y"],
                               format_func=lambda x: {"1m": "1개월", "3m": "3개월",
                                                       "6m": "6개월", "1y": "1년"}[x],
                               key="outlook_horizon")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("전망", key="outlook_run", use_container_width=True)

    if not run or not query:
        return

    data = _resolve_ticker(query)
    if not data:
        return

    ticker = data["ticker"]
    name = data.get("name", query)

    with st.spinner(f"{name} 전망 분석 중..."):
        summary = get_technical_summary(ticker)
        outlook = build_price_outlook(ticker, name, horizon=horizon,
                                       summary=summary, structured_data=data)

    if not outlook:
        st.error("전망 데이터를 생성할 수 없습니다.")
        return

    st.markdown(f"#### {name} ({ticker}) — {_horizon_label(horizon)} 전망")

    # 종합 점수
    score = outlook.get("composite_score", 0)
    grade = outlook.get("confidence_grade", "-")
    current = outlook.get("current_price", 0)

    score_color = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")
    st.markdown(f"**현재가:** {current:,}원 | **종합 점수:** {score_color} {score:+.2f} | **신뢰도:** {grade}")

    # 3축 분석
    col_t, col_f, col_s = st.columns(3)

    tech = outlook.get("technical", {})
    with col_t:
        st.markdown("**📊 기술적 분석**")
        st.metric("신호", tech.get("signal", "-"),
                  delta=f"점수 {tech.get('score', 0):+.2f}")
        factors = tech.get("key_factors", [])
        if factors:
            for f in factors[:4]:
                st.caption(f)

    fund = outlook.get("fundamental", {})
    with col_f:
        st.markdown("**📑 펀더멘털**")
        st.metric("신호", fund.get("signal", "-"),
                  delta=f"점수 {fund.get('score', 0):+.2f}")
        factors = fund.get("key_factors", [])
        if factors:
            for f in factors[:4]:
                st.caption(f)

    stat = outlook.get("statistical", {})
    with col_s:
        st.markdown("**📈 통계 모델**")
        pred_ret = stat.get("predicted_return", 0)
        ci = stat.get("confidence_interval", (0, 0))
        st.metric("예상 수익률", f"{pred_ret:+.1f}%",
                  delta=f"R² {stat.get('model_r2', 0):.2f}")
        st.caption(f"90% CI: {ci[0]:+.1f}% ~ {ci[1]:+.1f}%")
        st.caption(f"과거 적중률: {stat.get('historical_win_rate', 0):.0%} "
                   f"({stat.get('historical_sample_count', 0)}회)")
        st.caption(f"모델 신뢰도: {stat.get('model_reliability', '-')}")

    # 시나리오
    scenarios = outlook.get("scenarios", {})
    if scenarios:
        st.markdown("#### 시나리오 분석")
        cols = st.columns(3)
        for col, (key, label, emoji) in zip(cols, [
            ("bullish", "강세", "📈"), ("neutral", "중립", "➡️"), ("bearish", "약세", "📉")
        ]):
            sc = scenarios.get(key, {})
            with col:
                prob = sc.get("probability", 0)
                ret = sc.get("target_return", 0)
                st.markdown(f"**{emoji} {label}** ({prob:.0%})")
                st.caption(f"목표 수익률: {ret:+.1f}%")
                desc = sc.get("description", "")
                if desc:
                    st.caption(desc)

    # 리스크 요인
    risks = outlook.get("risk_factors", [])
    if risks:
        st.markdown("#### ⚠️ 리스크 요인")
        for r in risks:
            st.caption(f"• {r}")

    # 면책
    st.divider()
    st.caption("⚠️ 본 전망은 과거 데이터 기반 통계적 분석이며, 투자 권유가 아닙니다. "
               "투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.")


def _horizon_label(h: str) -> str:
    return {"1w": "1주", "2w": "2주", "1m": "1개월", "3m": "3개월",
            "6m": "6개월", "1y": "1년"}.get(h, h)
