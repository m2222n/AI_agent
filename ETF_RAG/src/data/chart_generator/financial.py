"""
재무제표 + 밸류에이션 + 포트폴리오 차트
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt

from src.data.chart_generator._style import (
    BG_COLOR, GRID_COLOR, TEXT_COLOR,
    REV_COLOR, OP_COLOR, NI_COLOR, MARGIN_COLOR,
    VAL_COLORS, PORT_COLOR, BM_COLOR, DD_COLOR, DD_FILL,
    setup_font, font_kw, build_xlabels, to_base64_tight, FONT_PROP,
)

logger = logging.getLogger(__name__)


def generate_financial_chart(
    rows: list,
    name: str,
) -> Optional[str]:
    """재무제표 실적 추이 차트 생성 → base64 PNG."""
    if not rows or len(rows) < 2:
        return None

    setup_font()

    try:
        labels = [f"{r['fiscal_year']}Q{r['fiscal_quarter']}" for r in rows]
        n = len(labels)
        x = list(range(n))

        revenue = [(r.get("revenue") or 0) / 1e8 for r in rows]
        op_profit = [(r.get("operating_profit") or 0) / 1e8 for r in rows]
        net_income = [(r.get("net_income") or 0) / 1e8 for r in rows]
        margin = [r.get("operating_margin") for r in rows]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(max(10, n * 0.8), 7), facecolor=BG_COLOR,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.25},
        )

        fp = font_kw()
        bar_width = 0.25

        ax1.set_facecolor(BG_COLOR)
        x_rev = [i - bar_width for i in x]
        x_op = x
        x_ni = [i + bar_width for i in x]

        ax1.bar(x_rev, revenue, width=bar_width, color=REV_COLOR,
                label="매출액", alpha=0.85, zorder=3)
        ax1.bar(x_op, op_profit, width=bar_width, color=OP_COLOR,
                label="영업이익", alpha=0.85, zorder=3)
        ax1.bar(x_ni, net_income, width=bar_width, color=NI_COLOR,
                label="순이익", alpha=0.85, zorder=3)

        ax1.set_ylabel("억 원", fontsize=10, color=TEXT_COLOR, **fp)
        ax1.set_title(f"{name} 분기별 실적 추이", fontsize=13,
                      fontweight="bold", color=TEXT_COLOR, pad=10, **fp)
        ax1.legend(fontsize=8, loc="upper left", framealpha=0.8)
        ax1.grid(True, axis="y", alpha=0.3, color=GRID_COLOR, zorder=0)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=7, color=TEXT_COLOR, rotation=45,
                            ha="right")
        ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.axhline(y=0, color="#999999", linewidth=0.5, zorder=2)

        ax2.set_facecolor(BG_COLOR)
        valid_margin = [(i, m) for i, m in zip(x, margin) if m is not None]
        if valid_margin:
            mx, my = zip(*valid_margin)
            ax2.plot(mx, my, color=MARGIN_COLOR, linewidth=2.0, marker="o",
                     markersize=4, alpha=0.9, zorder=3)
            ax2.fill_between(mx, my, 0, color=MARGIN_COLOR, alpha=0.1, zorder=2)

        ax2.axhline(y=0, color="#999999", linewidth=0.5)
        ax2.set_ylabel("영업이익률 (%)", fontsize=9, color=TEXT_COLOR, **fp)
        ax2.grid(True, alpha=0.3, color=GRID_COLOR, zorder=0)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=7, color=TEXT_COLOR, rotation=45,
                            ha="right")
        ax2.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.12)
        return to_base64_tight(fig)

    except Exception as e:
        logger.error(f"재무제표 차트 생성 실패: {e}")
        return None


def generate_valuation_chart(
    name1: str, name2: str,
    metrics: dict,
) -> Optional[str]:
    """2종목 밸류에이션 비교 바 차트 → base64 PNG."""
    labels = [k for k, (v1, v2) in metrics.items() if v1 or v2]
    if len(labels) < 2:
        return None

    setup_font()

    try:
        vals1 = [metrics[l][0] or 0 for l in labels]
        vals2 = [metrics[l][1] or 0 for l in labels]
        n = len(labels)
        x = list(range(n))
        bar_w = 0.35

        fig, ax = plt.subplots(figsize=(max(7, n * 1.5), 4.5), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        fp = font_kw()

        x1 = [i - bar_w / 2 for i in x]
        x2 = [i + bar_w / 2 for i in x]

        bars1 = ax.bar(x1, vals1, width=bar_w, color=VAL_COLORS[0],
                       label=name1, alpha=0.85, zorder=3)
        bars2 = ax.bar(x2, vals2, width=bar_w, color=VAL_COLORS[1],
                       label=name2, alpha=0.85, zorder=3)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if h != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h,
                            f"{h:.1f}", ha="center", va="bottom",
                            fontsize=7, color=TEXT_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, color=TEXT_COLOR, **fp)
        ax.set_title(f"{name1} vs {name2} 밸류에이션 비교", fontsize=12,
                     fontweight="bold", color=TEXT_COLOR, pad=10, **fp)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(True, axis="y", alpha=0.3, color=GRID_COLOR, zorder=0)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.15)
        return to_base64_tight(fig)

    except Exception as e:
        logger.error(f"밸류에이션 차트 생성 실패: {e}")
        return None


def generate_portfolio_chart(
    wealth: list,
    bm_wealth: Optional[list],
    dates: list,
    names: list,
    bm_name: str = "KODEX 200",
) -> Optional[str]:
    """포트폴리오 시뮬레이션 차트 생성 → base64 PNG."""
    if not wealth or len(wealth) < 10:
        return None

    setup_font()

    try:
        n = len(wealth)
        x = list(range(n))

        port_norm = [w * 100 for w in wealth]

        peak = wealth[0]
        drawdown = []
        for w in wealth:
            if w > peak:
                peak = w
            drawdown.append((w - peak) / peak * 100)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), facecolor=BG_COLOR,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.20},
        )

        fp = font_kw()

        ax1.set_facecolor(BG_COLOR)
        port_label = f"포트폴리오 ({', '.join(names[:3])}{'...' if len(names) > 3 else ''})"
        ax1.plot(x, port_norm, color=PORT_COLOR, linewidth=1.8,
                 label=port_label, alpha=0.9)

        if bm_wealth and len(bm_wealth) >= 10:
            bm_norm = [w * 100 for w in bm_wealth[:n]]
            if len(bm_norm) < n:
                bm_norm.extend([bm_norm[-1]] * (n - len(bm_norm)))
            ax1.plot(x[:len(bm_norm)], bm_norm, color=BM_COLOR,
                     linewidth=1.5, label=bm_name, alpha=0.7, linestyle="--")

        ax1.axhline(y=100, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.set_ylabel("누적 수익률 (시작=100)", fontsize=10, color=TEXT_COLOR, **fp)
        ax1.set_title("포트폴리오 시뮬레이션", fontsize=13, fontweight="bold",
                       color=TEXT_COLOR, pad=10, **fp)
        ax1.legend(fontsize=8, loc="upper left", framealpha=0.8)
        ax1.grid(True, alpha=0.3, color=GRID_COLOR)
        ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax1.spines.values():
            spine.set_visible(False)

        ax2.set_facecolor(BG_COLOR)
        ax2.fill_between(x, drawdown, 0, color=DD_FILL, alpha=0.7)
        ax2.plot(x, drawdown, color=DD_COLOR, linewidth=1.0, alpha=0.8)
        ax2.axhline(y=0, color="#999999", linewidth=0.5)
        ax2.set_ylabel("Drawdown (%)", fontsize=9, color=TEXT_COLOR, **fp)
        ax2.grid(True, alpha=0.3, color=GRID_COLOR)
        ax2.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        if dates and len(dates) >= n - 1:
            chart_dates = [""] + dates[:n - 1]
        else:
            chart_dates = [str(i) for i in x]

        step = max(1, n // 6)
        xtick_pos, xtick_labels = build_xlabels(chart_dates, step)
        ax2.set_xticks(xtick_pos)
        ax2.set_xticklabels(xtick_labels, fontsize=8, color=TEXT_COLOR)

        fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.10)
        return to_base64_tight(fig)

    except Exception as e:
        logger.error(f"포트폴리오 차트 생성 실패: {e}")
        return None


def generate_paper_trend_chart(
    dates: list,
    pnl_pcts: list,
) -> Optional[str]:
    """가상투자 수익률 추이 라인 차트(0%=원금) → base64 PNG.

    dates: ["YYYYMMDD", ...] 오름차순, pnl_pcts: 동일 길이 수익률(%) 시계열.
    """
    if not dates or len(dates) < 2 or len(dates) != len(pnl_pcts):
        return None

    setup_font()
    try:
        x = list(range(len(dates)))
        last = pnl_pcts[-1]
        color = "#E8453C" if last >= 0 else "#1A73E8"

        fig, ax = plt.subplots(figsize=(11, 4), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        ax.plot(x, pnl_pcts, color=color, linewidth=1.8, zorder=3)
        ax.fill_between(x, 0.0, pnl_pcts, color=color, alpha=0.08, zorder=2)
        ax.axhline(0.0, color=TEXT_COLOR, linewidth=0.6, alpha=0.4, zorder=1)

        fp = font_kw()
        n = len(dates)
        step = max(1, n // 8)
        ticks = list(range(0, n, step))
        if ticks and ticks[-1] != n - 1:
            ticks.append(n - 1)

        def _fmt(d: str) -> str:
            return f"{d[4:6]}.{d[6:8]}" if len(d) == 8 else d

        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt(dates[i]) for i in ticks],
                           fontsize=7, color=TEXT_COLOR, **fp)
        ax.set_ylabel("수익률 (%)", fontsize=8, color=TEXT_COLOR, **fp)
        ax.grid(True, alpha=0.3, color=GRID_COLOR, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"가상투자 수익률 추이 (현재 {last:+.2f}%)",
                     fontsize=12, fontweight="bold", color=TEXT_COLOR, **fp)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.12)
        return to_base64_tight(fig)
    except Exception as e:
        logger.error(f"가상투자 추이 차트 생성 실패: {e}")
        return None
