"""
기술적 분석 + 비교 + 장중 시세 차트
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.data.chart_generator._style import (
    BG_COLOR, GRID_COLOR, TEXT_COLOR, PRICE_COLOR, MA_COLORS, BB_COLOR,
    RSI_COLOR, VOL_UP, VOL_DOWN, MACD_LINE, MACD_SIGNAL, COMPARE_COLORS,
    setup_font, apply_style, font_kw, fmt_date_full, build_xlabels,
    to_base64, to_base64_tight,
)
from src.data.chart_generator._series import (
    calc_ma_series, calc_rsi_series, calc_macd_series, calc_bb_series,
)

logger = logging.getLogger(__name__)

# 장중 차트용 상수
_INTRA_COLOR = "#1A73E8"
_INTRA_PREV = "#999999"


def generate_technical_chart(ticker: str, name: str,
                             days: int = 120) -> Optional[str]:
    """기술적 분석 차트 생성 → base64 PNG 문자열 반환.

    3단 구성:
    1. 상단: 종가 + MA(5/20/60) + 볼린저 밴드 + 고가/저가 밴드
    2. 중단: RSI(14) + 과매수/과매도 구간
    3. 하단: 거래량 바 + MACD 히스토그램/시그널
    """
    try:
        setup_font()
        from src.data.technical import _get_ohlcv

        ohlcv = _get_ohlcv(ticker, days=days + 30)
        if len(ohlcv) < 40:
            return None

        closes = [d["close"] for d in ohlcv]
        highs = [d["high"] for d in ohlcv]
        lows = [d["low"] for d in ohlcv]
        volumes = [d["volume"] for d in ohlcv]
        dates = [d["date"] for d in ohlcv]

        ma5 = calc_ma_series(closes, 5)
        ma20 = calc_ma_series(closes, 20)
        ma60 = calc_ma_series(closes, 60)
        rsi = calc_rsi_series(closes, 14)
        macd_data = calc_macd_series(closes)
        bb_upper, bb_lower = calc_bb_series(closes, 20, 2.0)

        n = min(days, len(dates))
        sl = slice(-n, None)
        x = list(range(n))

        d_closes = closes[sl]
        d_highs = highs[sl]
        d_lows = lows[sl]
        d_volumes = volumes[sl]
        d_dates = dates[sl]

        step = max(1, n // 6)
        xticks, xlabels = build_xlabels(d_dates, step)

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 7.5),
            gridspec_kw={"height_ratios": [3, 1, 1.3]},
            sharex=True,
        )
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(hspace=0.06, left=0.08, right=0.95, top=0.88, bottom=0.06)

        fp = font_kw()
        fig.text(0.08, 0.98, f"{name} ({ticker})", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR, va="top", **fp)
        fig.text(0.08, 0.955,
                 f"기준일: {fmt_date_full(d_dates[-1])}  |  종가: {d_closes[-1]:,}원  |  {n}일",
                 fontsize=9, color="#888888", va="top", **fp)

        # ── 상단: 가격 ──
        apply_style(ax1, ylabel="")
        ax1.fill_between(x, d_lows, d_highs, alpha=0.06, color="#9E9E9E", linewidth=0)

        if bb_upper and bb_lower:
            d_bbu = bb_upper[sl]
            d_bbl = bb_lower[sl]
            ax1.fill_between(x, d_bbl, d_bbu, alpha=0.06, color=BB_COLOR, linewidth=0)
            ax1.plot(x, d_bbu, color=BB_COLOR, linewidth=0.6, alpha=0.5, linestyle="--")
            ax1.plot(x, d_bbl, color=BB_COLOR, linewidth=0.6, alpha=0.5, linestyle="--")

        for label, series, color in [("MA5", ma5, MA_COLORS["MA5"]),
                                     ("MA20", ma20, MA_COLORS["MA20"]),
                                     ("MA60", ma60, MA_COLORS["MA60"])]:
            if series:
                ax1.plot(x, series[sl], color=color, linewidth=0.9, alpha=0.85, label=label)

        ax1.plot(x, d_closes, color=PRICE_COLOR, linewidth=1.5, label="종가", zorder=5)
        ax1.annotate(f"{d_closes[-1]:,}", xy=(x[-1], d_closes[-1]),
                     fontsize=8, color=PRICE_COLOR, fontweight="bold",
                     xytext=(5, 0), textcoords="offset points", va="center")
        ax1.legend(loc="upper right", fontsize=7.5, ncol=4, framealpha=0.8,
                   edgecolor=GRID_COLOR, fancybox=False)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))

        # ── 중단: RSI ──
        apply_style(ax2, ylabel="RSI")
        if rsi:
            d_rsi = rsi[sl]
            ax2.axhspan(70, 100, alpha=0.08, color="#F44336", linewidth=0)
            ax2.axhspan(0, 30, alpha=0.08, color="#4CAF50", linewidth=0)
            ax2.axhline(70, color="#F44336", linewidth=0.5, linestyle=":", alpha=0.6)
            ax2.axhline(30, color="#4CAF50", linewidth=0.5, linestyle=":", alpha=0.6)
            ax2.axhline(50, color=GRID_COLOR, linewidth=0.5, linestyle="-", alpha=0.4)
            ax2.plot(x, d_rsi, color=RSI_COLOR, linewidth=1.0)
            ax2.set_ylim(0, 100)
            last_rsi = d_rsi[-1]
            if last_rsi is not None:
                ax2.annotate(f"{last_rsi:.0f}", xy=(x[-1], last_rsi),
                             fontsize=7, color=RSI_COLOR, fontweight="bold",
                             xytext=(5, 0), textcoords="offset points", va="center")

        # ── 하단: 거래량 + MACD ──
        apply_style(ax3, ylabel="거래량", hide_xticklabels=False)
        vol_colors = [VOL_UP if i == 0 or d_closes[i] >= d_closes[i - 1] else VOL_DOWN
                      for i in range(len(d_closes))]
        ax3.bar(x, d_volumes, color=vol_colors, alpha=0.35, width=0.7)
        ax3.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda v, p: f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v / 1e3:.0f}K" if v >= 1e3 else ""
            )
        )

        if macd_data:
            macd_line, signal_line, histogram = macd_data
            d_macd = macd_line[sl]
            d_signal = signal_line[sl]
            d_hist = histogram[sl]

            ax3b = ax3.twinx()
            ax3b.spines["top"].set_visible(False)
            ax3b.spines["left"].set_visible(False)
            ax3b.spines["right"].set_color(GRID_COLOR)
            ax3b.spines["bottom"].set_visible(False)
            ax3b.tick_params(axis="y", labelsize=7, colors="#999999")

            hist_c = [VOL_UP if h and h >= 0 else VOL_DOWN for h in d_hist]
            ax3b.bar(x, [h or 0 for h in d_hist], color=hist_c, alpha=0.25, width=0.5)
            ax3b.plot(x, [m or 0 for m in d_macd], color=MACD_LINE, linewidth=0.9,
                      alpha=0.9, label="MACD")
            ax3b.plot(x, [s or 0 for s in d_signal], color=MACD_SIGNAL, linewidth=0.9,
                      alpha=0.9, label="Signal")
            ax3b.axhline(0, color=GRID_COLOR, linewidth=0.4)
            ax3b.set_ylabel("MACD", fontsize=8, color="#999999", labelpad=8)
            ax3b.legend(loc="upper right", fontsize=7, framealpha=0.8,
                        edgecolor=GRID_COLOR, fancybox=False)

        ax3.set_xticks(xticks)
        ax3.set_xticklabels(xlabels, fontsize=8, color=TEXT_COLOR)

        return to_base64(fig)

    except Exception as e:
        logger.warning(f"차트 생성 실패 ({ticker}): {e}")
        return None


def generate_comparison_chart(
    tickers: list, names: list, days: int = 120,
) -> Optional[str]:
    """종목 간 상대 수익률 차트 생성 → base64 PNG."""
    from src.data.technical import _get_closes

    if len(tickers) < 2 or len(tickers) != len(names):
        return None

    setup_font()

    all_data = {}
    for ticker in tickers:
        data = _get_closes(ticker, days=days + 10)
        if len(data) < 20:
            logger.warning(f"비교 차트: {ticker} 데이터 부족 ({len(data)}일)")
            return None
        all_data[ticker] = {d["date"]: d["close"] for d in data}

    common_dates = sorted(
        set.intersection(*(set(d.keys()) for d in all_data.values()))
    )
    if len(common_dates) < 20:
        return None
    common_dates = common_dates[-days:] if len(common_dates) > days else common_dates

    try:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        for i, (ticker, name) in enumerate(zip(tickers, names)):
            closes = [all_data[ticker][d] for d in common_dates]
            if closes[0] <= 0:
                continue
            normalized = [c / closes[0] * 100 for c in closes]
            color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
            ax.plot(range(len(normalized)), normalized,
                    color=color, linewidth=1.8, label=name, alpha=0.9)

        ax.axhline(y=100, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)

        n_points = len(common_dates)
        step = max(1, n_points // 6)
        xtick_pos, xtick_labels = build_xlabels(common_dates, step)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, fontsize=8, color=TEXT_COLOR)

        fp = font_kw()
        ax.set_ylabel("상대 수익률 (기준=100)", fontsize=10, color=TEXT_COLOR, **fp)
        ax.set_title("기간별 상대 수익률 비교", fontsize=13, fontweight="bold",
                      color=TEXT_COLOR, pad=12, **fp)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        return to_base64_tight(fig)

    except Exception as e:
        logger.error(f"비교 차트 생성 실패: {e}")
        return None


def generate_intraday_chart(
    ticker: str,
    name: str,
    prev_close: Optional[float] = None,
) -> Optional[str]:
    """yfinance에서 당일 장중 데이터 조회 + 차트 생성 → base64 PNG."""
    setup_font()

    try:
        import yfinance as yf
        from src.data.realtime import krx_to_yfinance

        yf_ticker = krx_to_yfinance(ticker, "stock")
        df = yf.download(yf_ticker, period="1d", interval="15m", progress=False)

        if df.empty or len(df) < 3:
            return None

        if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
            df = df.droplevel("Ticker", axis=1)

        closes = df["Close"].values.flatten()
        times = [idx.strftime("%H:%M") for idx in df.index]
        volumes = df["Volume"].values.flatten()

        n = len(closes)
        x = list(range(n))

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 5), facecolor=BG_COLOR,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.20},
        )
        fp = font_kw()

        ax1.set_facecolor(BG_COLOR)
        ax1.plot(x, closes, color=_INTRA_COLOR, linewidth=1.8, alpha=0.9)
        ax1.fill_between(x, closes, closes[0], alpha=0.08, color=_INTRA_COLOR)

        if prev_close:
            ax1.axhline(y=prev_close, color=_INTRA_PREV, linewidth=1.0,
                        linestyle="--", alpha=0.7)
            ax1.text(0, prev_close, f" 전일종가 {prev_close:,.0f}",
                     fontsize=7, color=_INTRA_PREV, va="bottom", **fp)

        ax1.set_title(f"{name} 장중 시세 (15분봉)", fontsize=12,
                      fontweight="bold", color=TEXT_COLOR, pad=8, **fp)
        ax1.set_ylabel("가격 (원)", fontsize=9, color=TEXT_COLOR, **fp)
        ax1.grid(True, alpha=0.3, color=GRID_COLOR)
        ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax1.spines.values():
            spine.set_visible(False)

        ax2.set_facecolor(BG_COLOR)
        colors = [VOL_UP if i > 0 and closes[i] >= closes[i - 1] else VOL_DOWN
                  for i in range(n)]
        ax2.bar(x, volumes, color=colors, alpha=0.6, width=0.8, zorder=3)
        ax2.set_ylabel("거래량", fontsize=8, color=TEXT_COLOR, **fp)
        ax2.grid(True, alpha=0.3, color=GRID_COLOR, zorder=0)
        ax2.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        step = max(1, n // 6)
        xtick_pos = list(range(0, n, step))
        xtick_labels = [times[i] for i in xtick_pos]
        for ax in (ax1, ax2):
            ax.set_xticks(xtick_pos)
            ax.set_xticklabels(xtick_labels, fontsize=7, color=TEXT_COLOR)

        fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.10)
        return to_base64_tight(fig)

    except ImportError:
        logger.warning("yfinance 미설치 — 장중 차트 생성 불가")
        return None
    except Exception as e:
        logger.error(f"장중 시세 차트 생성 실패: {e}")
        return None
