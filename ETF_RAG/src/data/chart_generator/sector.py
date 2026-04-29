"""
섹터(업종) 분석 차트
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt

from src.data.chart_generator._style import (
    BG_COLOR, GRID_COLOR, TEXT_COLOR,
    setup_font, font_kw, to_base64_tight, FONT_PROP,
)

logger = logging.getLogger(__name__)


def generate_sector_overview_chart(
    sector_stats: list,
) -> Optional[str]:
    """업종별 등락률 + 시가총액 수평 바 차트 → base64 PNG."""
    if not sector_stats or len(sector_stats) < 2:
        return None

    setup_font()

    try:
        top = sector_stats[:20]
        sectors = [s["sector"] for s in top]
        changes = [s["change_pct"] for s in top]
        n = len(sectors)
        y = list(range(n))

        fig, ax = plt.subplots(figsize=(10, max(5, n * 0.35)), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        fp = font_kw()

        colors = ["#E8453C" if c >= 0 else "#1A73E8" for c in changes]
        bars = ax.barh(y, changes, color=colors, alpha=0.8, height=0.6, zorder=3)

        for i, (bar, val) in enumerate(zip(bars, changes)):
            ha = "left" if val >= 0 else "right"
            offset = 0.1 if val >= 0 else -0.1
            ax.text(val + offset, i, f"{val:+.2f}%", va="center", ha=ha,
                    fontsize=8, color=TEXT_COLOR)

        ax.set_yticks(y)
        ax.set_yticklabels(sectors, fontsize=9, color=TEXT_COLOR, **fp)
        ax.invert_yaxis()
        ax.set_xlabel("등락률 (%)", fontsize=9, color=TEXT_COLOR, **fp)
        ax.set_title("업종별 등락률 (시가총액 순)", fontsize=12,
                     fontweight="bold", color=TEXT_COLOR, pad=10, **fp)
        ax.axvline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.5)
        ax.grid(True, axis="x", alpha=0.3, color=GRID_COLOR, zorder=0)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.subplots_adjust(left=0.22, right=0.92, top=0.92, bottom=0.10)
        return to_base64_tight(fig)

    except Exception as e:
        logger.error(f"섹터 개요 차트 생성 실패: {e}")
        return None


def generate_sector_detail_chart(
    sector: str,
    stocks: list,
) -> Optional[str]:
    """업종 내 종목 등락률 트리맵 스타일 수평 바 차트 → base64 PNG."""
    if not stocks or len(stocks) < 2:
        return None

    setup_font()

    try:
        top = stocks[:15]
        names = [s["name"] for s in top]
        changes = [s.get("change_pct", 0) for s in top]
        caps = [s.get("market_cap", 0) for s in top]
        n = len(names)
        y = list(range(n))

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, max(4, n * 0.35)), facecolor=BG_COLOR,
            gridspec_kw={"width_ratios": [3, 2], "wspace": 0.30},
        )
        fp = font_kw()

        # ── 왼쪽: 등락률 바 ──
        ax1.set_facecolor(BG_COLOR)
        colors = ["#E8453C" if c >= 0 else "#1A73E8" for c in changes]
        bars1 = ax1.barh(y, changes, color=colors, alpha=0.8, height=0.6, zorder=3)
        for i, (bar, val) in enumerate(zip(bars1, changes)):
            ha = "left" if val >= 0 else "right"
            offset = 0.05 if val >= 0 else -0.05
            ax1.text(val + offset, i, f"{val:+.2f}%", va="center", ha=ha,
                     fontsize=7, color=TEXT_COLOR)

        ax1.set_yticks(y)
        ax1.set_yticklabels(names, fontsize=8, color=TEXT_COLOR, **fp)
        ax1.invert_yaxis()
        ax1.set_xlabel("등락률 (%)", fontsize=8, color=TEXT_COLOR, **fp)
        ax1.axvline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.5)
        ax1.grid(True, axis="x", alpha=0.3, color=GRID_COLOR, zorder=0)
        for spine in ax1.spines.values():
            spine.set_visible(False)

        # ── 오른쪽: 시가총액 바 ──
        ax2.set_facecolor(BG_COLOR)
        caps_조 = [c / 1_000_000_000_000 for c in caps]
        bars2 = ax2.barh(y, caps_조, color="#78909C", alpha=0.7, height=0.6, zorder=3)
        for i, val in enumerate(caps_조):
            if val >= 0.1:
                ax2.text(val + max(caps_조) * 0.02, i, f"{val:.1f}조",
                         va="center", fontsize=7, color=TEXT_COLOR)
            elif caps[i] >= 100_000_000:
                ax2.text(val + max(caps_조) * 0.02, i,
                         f"{caps[i] / 100_000_000:.0f}억",
                         va="center", fontsize=7, color=TEXT_COLOR)

        ax2.set_yticks(y)
        ax2.set_yticklabels([""] * n)
        ax2.invert_yaxis()
        ax2.set_xlabel("시가총액", fontsize=8, color=TEXT_COLOR, **fp)
        ax2.grid(True, axis="x", alpha=0.3, color=GRID_COLOR, zorder=0)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        fig.suptitle(f"{sector} 업종 — 종목별 등락률 & 시가총액",
                     fontsize=12, fontweight="bold", color=TEXT_COLOR,
                     y=0.98, **fp)
        fig.subplots_adjust(left=0.18, right=0.95, top=0.90, bottom=0.10)
        return to_base64_tight(fig)

    except Exception as e:
        logger.error(f"섹터 상세 차트 생성 실패: {e}")
        return None
