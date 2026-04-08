"""
비교 차트/테이블 렌더링 모듈

compare_etfs 도구가 반환한 구조화 데이터를 Streamlit 테이블 + 바 차트로 시각화.
"""

import json
from typing import Optional

import streamlit as st


def try_parse_comparison(raw: str) -> Optional[dict]:
    """structured_data 이벤트에서 comparison_table JSON 추출"""
    try:
        # JSON은 첫 줄에 있고 "---" 이후 텍스트가 이어짐
        json_part = raw.split("\n\n---\n\n")[0] if "\n\n---\n\n" in raw else raw
        data = json.loads(json_part)
        if data.get("__type__") == "comparison_table" and data.get("items"):
            return data
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return None


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
            ("PER", f"{a.get('per', 0):.2f}배", f"{b.get('per', 0):.2f}배"),
            ("PBR", f"{a.get('pbr', 0):.2f}배", f"{b.get('pbr', 0):.2f}배"),
            ("시가총액", _format_market_cap(a.get("market_cap", 0)),
             _format_market_cap(b.get("market_cap", 0))),
            ("배당수익률", f"{a.get('div', 0):.2f}%", f"{b.get('div', 0):.2f}%"),
        ])

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


def _format_value(value: int) -> str:
    """거래대금 등 큰 숫자를 읽기 쉬운 형태로"""
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}조"
    elif value >= 100_000_000:
        return f"{value / 100_000_000:.0f}억"
    elif value >= 10_000:
        return f"{value / 10_000:.0f}만"
    return f"{value:,}"


def _format_market_cap(value: int) -> str:
    """시가총액 포맷"""
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}조원"
    elif value >= 100_000_000:
        return f"{value / 100_000_000:.0f}억원"
    return f"{value:,}원"


def _fmt_pct(value) -> str:
    """퍼센트 포맷 (None 처리)"""
    if value is None:
        return "-"
    return f"{value:+.2f}%"
