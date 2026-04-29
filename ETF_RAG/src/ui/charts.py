"""
비교 차트/테이블 렌더링 모듈

compare_etfs 도구가 반환한 구조화 데이터를 Streamlit 테이블 + 바 차트로 시각화.
기술적 분석 차트(matplotlib 이미지)도 렌더링.
"""

import base64
import json
from typing import Optional

import streamlit as st


def try_parse_comparison(raw: str) -> Optional[dict]:
    """structured_data 이벤트에서 comparison_table JSON 추출"""
    try:
        json_part = raw.split("\n\n---\n\n")[0] if "\n\n---\n\n" in raw else raw
        data = json.loads(json_part)
        if data.get("__type__") == "comparison_table" and data.get("items"):
            return data
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return None


def try_parse_structured_data(raw: str) -> Optional[dict]:
    """structured_data 이벤트에서 모든 타입의 구조화 데이터 추출.

    지원 타입: comparison_table, technical_chart, portfolio_chart
    """
    try:
        json_part = raw.split("\n\n---\n\n")[0] if "\n\n---\n\n" in raw else raw
        data = json.loads(json_part)
        dtype = data.get("__type__")
        if dtype == "comparison_table" and data.get("items"):
            return data
        if dtype == "technical_chart" and data.get("image_b64"):
            return data
        if dtype == "portfolio_chart" and data.get("image_b64"):
            return data
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return None


def render_structured_data(data: dict):
    """구조화 데이터를 타입에 따라 렌더링."""
    dtype = data.get("__type__")
    if dtype == "comparison_table":
        render_comparison(data)
    elif dtype == "technical_chart":
        render_technical_chart(data)
    elif dtype == "portfolio_chart":
        render_portfolio_chart(data)


def render_portfolio_chart(data: dict):
    """포트폴리오 시뮬레이션 차트 이미지 렌더링."""
    image_b64 = data.get("image_b64")
    names = data.get("names", [])
    if not image_b64:
        return
    try:
        image_bytes = base64.b64decode(image_b64)
        title = " + ".join(names) if names else "포트폴리오"
        st.markdown(f"#### {title} 시뮬레이션 차트")
        st.image(image_bytes, use_container_width=True)
        st.caption(
            "📈 상단: 포트폴리오 vs 벤치마크 자산가치 추이 (시작=100) | "
            "하단: 고점 대비 낙폭(Drawdown)"
        )
    except Exception:
        pass


def render_technical_chart(data: dict):
    """기술적 분석 차트 이미지 렌더링 + 해석 캡션."""
    image_b64 = data.get("image_b64")
    name = data.get("name", "")
    if not image_b64:
        return
    try:
        image_bytes = base64.b64decode(image_b64)
        if name:
            st.markdown(f"#### {name} 기술적 분석 차트")
        st.image(image_bytes, use_container_width=True)
        st.caption(
            "📊 상단: 종가 + 이동평균선(MA5/20/60) + 볼린저 밴드 | "
            "중단: RSI(14) — 70↑ 과매수, 30↓ 과매도 | "
            "하단: 거래량 + MACD 히스토그램"
        )
    except Exception:
        pass


def render_comparison(data: dict):
    """비교 데이터를 테이블 + 차트로 렌더링

    Args:
        data: {"__type__": "comparison_table", "items": [{...}, {...}]}
    """
    items = data.get("items", [])
    if len(items) < 2:
        return

    a, b = items[0], items[1]
    name_a, name_b = a["name"], b["name"]

    # ── 기본 정보 테이블 ──
    st.markdown("#### 비교 요약")

    asset_type = a.get("asset_type", "etf")

    rows = [
        ("종가", f"{a['close']:,}원", f"{b['close']:,}원"),
        ("등락률", f"{a['change_pct']:+.2f}%", f"{b['change_pct']:+.2f}%"),
        ("거래대금", _format_value(a["trade_value"]), _format_value(b["trade_value"])),
    ]

    # ETF 전용 행
    if asset_type == "etf":
        if a.get("nav") or b.get("nav"):
            rows.append(("NAV", f"{a.get('nav', 0):,.0f}원", f"{b.get('nav', 0):,.0f}원"))
        if a.get("deviation") is not None or b.get("deviation") is not None:
            rows.append(("괴리율", _fmt_pct(a.get("deviation")), _fmt_pct(b.get("deviation"))))

    # 주식 전용 행
    if asset_type == "stock":
        rows.extend([
            ("시가총액", _format_market_cap(a.get("market_cap", 0)),
             _format_market_cap(b.get("market_cap", 0))),
            ("PER", f"{a.get('per', 0):.2f}배", f"{b.get('per', 0):.2f}배"),
            ("PBR", f"{a.get('pbr', 0):.2f}배", f"{b.get('pbr', 0):.2f}배"),
            ("EPS", f"{a.get('eps', 0):,.0f}원", f"{b.get('eps', 0):,.0f}원"),
            ("BPS", f"{a.get('bps', 0):,.0f}원", f"{b.get('bps', 0):,.0f}원"),
            ("배당수익률", f"{a.get('div', 0):.2f}%", f"{b.get('div', 0):.2f}%"),
            ("DPS", f"{a.get('dps', 0):,.0f}원", f"{b.get('dps', 0):,.0f}원"),
        ])

        # 재무제표 행 (데이터 있을 때만)
        if a.get("revenue") is not None or b.get("revenue") is not None:
            period_a = a.get("fiscal_period", "")
            period_b = b.get("fiscal_period", "")
            period_label = period_a or period_b
            if period_label:
                rows.append(("**실적 기준**", period_a, period_b))
            rows.append(("매출액", _format_value(a.get("revenue") or 0),
                         _format_value(b.get("revenue") or 0)))
            rows.append(("영업이익", _format_value(a.get("operating_profit") or 0),
                         _format_value(b.get("operating_profit") or 0)))
            rows.append(("순이익", _format_value(a.get("net_income") or 0),
                         _format_value(b.get("net_income") or 0)))
            rows.append(("영업이익률",
                         f"{a.get('operating_margin', 0):.1f}%" if a.get("operating_margin") is not None else "-",
                         f"{b.get('operating_margin', 0):.1f}%" if b.get("operating_margin") is not None else "-"))
            rows.append(("매출 YoY",
                         f"{a.get('revenue_growth_yoy', 0):+.1f}%" if a.get("revenue_growth_yoy") is not None else "-",
                         f"{b.get('revenue_growth_yoy', 0):+.1f}%" if b.get("revenue_growth_yoy") is not None else "-"))
            rows.append(("영업이익 YoY",
                         f"{a.get('op_growth_yoy', 0):+.1f}%" if a.get("op_growth_yoy") is not None else "-",
                         f"{b.get('op_growth_yoy', 0):+.1f}%" if b.get("op_growth_yoy") is not None else "-"))

    # 테이블 렌더링
    header = f"| 항목 | {name_a} | {name_b} |"
    sep = "|------|------|------|"
    body = "\n".join(f"| {label} | {va} | {vb} |" for label, va, vb in rows)
    st.markdown(f"{header}\n{sep}\n{body}")

    # ── 수익률 바 차트 ──
    return_periods = [
        ("1d", "1일"),
        ("1w", "1주"),
        ("1m", "1개월"),
        ("3m", "3개월"),
        ("1y", "1년"),
    ]

    has_returns = any(
        a.get(f"return_{p}") is not None or b.get(f"return_{p}") is not None
        for p, _ in return_periods
    )

    if has_returns:
        st.markdown("#### 수익률 비교")

        chart_data = {
            "기간": [],
            name_a: [],
            name_b: [],
        }

        for period_key, period_label in return_periods:
            val_a = a.get(f"return_{period_key}")
            val_b = b.get(f"return_{period_key}")
            if val_a is not None or val_b is not None:
                chart_data["기간"].append(period_label)
                chart_data[name_a].append(val_a or 0)
                chart_data[name_b].append(val_b or 0)

        if chart_data["기간"]:
            import pandas as pd
            df = pd.DataFrame(chart_data).set_index("기간")
            st.bar_chart(df)

    # ── 상대 수익률 추이 차트 (matplotlib) ──
    chart_b64 = data.get("comparison_chart_b64")
    if chart_b64:
        try:
            st.markdown("#### 기간별 상대 수익률 추이")
            image_bytes = base64.b64decode(chart_b64)
            st.image(image_bytes, use_container_width=True)
        except Exception:
            pass

    # ── 주식 밸류에이션 바 차트 ──
    if asset_type == "stock":
        val_metrics = [
            ("PER", "per", "배"),
            ("PBR", "pbr", "배"),
            ("배당수익률", "div", "%"),
        ]
        val_data = {"지표": [], name_a: [], name_b: []}
        for label, key, _ in val_metrics:
            va = a.get(key, 0)
            vb = b.get(key, 0)
            if va or vb:
                val_data["지표"].append(label)
                val_data[name_a].append(va or 0)
                val_data[name_b].append(vb or 0)

        if val_data["지표"]:
            st.markdown("#### 밸류에이션 비교")
            import pandas as pd
            df_val = pd.DataFrame(val_data).set_index("지표")
            st.bar_chart(df_val)


from src.utils.formatters import (
    format_large_number as _format_value,
    format_market_cap as _format_market_cap,
    format_percentage as _fmt_pct,
)
