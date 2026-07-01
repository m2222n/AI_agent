"""
뉴스 감성 시계열 차트 — 일별 감성 점수 라인 + 긍정/부정/중립 건수 막대.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt

from src.data.chart_generator._style import (
    BG_COLOR, GRID_COLOR, TEXT_COLOR,
    setup_font, font_kw, to_base64_tight,
)

logger = logging.getLogger(__name__)

_POS = "#e8453c"   # 긍정 = 빨강(상승 톤, 국내 관례)
_NEG = "#1a73e8"   # 부정 = 파랑(하락 톤)
_NEU = "#9ca3af"   # 중립 = 회색
_SCORE = "#8b5cf6"  # 감성 점수 라인


def generate_news_sentiment_chart(
    series: list, name: str,
) -> Optional[str]:
    """일별 감성 시계열 차트 → base64 PNG.

    Args:
        series: build_sentiment_timeseries 결과
                [{date, positive, negative, neutral, total, score}]
        name: 종목명
    """
    if not series or len(series) < 2:
        return None  # 하루치뿐이면 시계열 의미 없음

    setup_font()
    try:
        dates = [r["date"][5:] for r in series]  # MM-DD
        n = len(dates)
        x = list(range(n))
        pos = [r["positive"] for r in series]
        neg = [r["negative"] for r in series]
        neu = [r["neutral"] for r in series]
        score = [r["score"] for r in series]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(max(8, n * 0.9), 6), facecolor=BG_COLOR,
            gridspec_kw={"height_ratios": [2, 2], "hspace": 0.3},
        )
        fp = font_kw()

        # 상단: 일별 감성 점수 라인 (-1~+1)
        ax1.set_facecolor(BG_COLOR)
        ax1.plot(x, score, color=_SCORE, linewidth=2.0, marker="o",
                 markersize=5, zorder=3)
        ax1.fill_between(x, score, 0, color=_SCORE, alpha=0.12, zorder=2)
        ax1.axhline(y=0, color="#999999", linewidth=0.7, zorder=2)
        ax1.set_ylabel("감성 점수", fontsize=10, color=TEXT_COLOR, **fp)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_title(f"{name} 뉴스 감성 추이", fontsize=13, fontweight="bold",
                      color=TEXT_COLOR, pad=10, **fp)
        ax1.grid(True, axis="y", alpha=0.3, color=GRID_COLOR, zorder=0)
        ax1.set_xticks(x)
        ax1.set_xticklabels(dates, fontsize=8, color=TEXT_COLOR)
        ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
        for s in ax1.spines.values():
            s.set_visible(False)

        # 하단: 긍정/부정/중립 스택 막대
        ax2.set_facecolor(BG_COLOR)
        ax2.bar(x, pos, color=_POS, label="긍정", alpha=0.85, zorder=3)
        ax2.bar(x, neg, bottom=pos, color=_NEG, label="부정", alpha=0.85, zorder=3)
        bottom2 = [p + ng for p, ng in zip(pos, neg)]
        ax2.bar(x, neu, bottom=bottom2, color=_NEU, label="중립", alpha=0.7, zorder=3)
        ax2.set_ylabel("기사 수", fontsize=10, color=TEXT_COLOR, **fp)
        ax2.legend(fontsize=8, loc="upper left", framealpha=0.8)
        ax2.grid(True, axis="y", alpha=0.3, color=GRID_COLOR, zorder=0)
        ax2.set_xticks(x)
        ax2.set_xticklabels(dates, fontsize=8, color=TEXT_COLOR)
        ax2.tick_params(colors=TEXT_COLOR, labelsize=8)
        for s in ax2.spines.values():
            s.set_visible(False)

        fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.10)
        return to_base64_tight(fig)

    except Exception as e:
        logger.error(f"뉴스 감성 차트 생성 실패: {e}")
        return None
