"""
기술적 분석 차트 생성 모듈

matplotlib로 3단 차트(가격+MA+볼린저 / RSI / 거래량+MACD)를
생성하여 base64 PNG로 반환. Streamlit st.image()에서 직접 사용.
"""

import base64
import logging
from io import BytesIO
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # GUI 백엔드 없이 사용
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.data.technical import _get_ohlcv

logger = logging.getLogger(__name__)

# ── 디자인 상수 ──
_BG_COLOR = "#FAFAFA"
_GRID_COLOR = "#E0E0E0"
_TEXT_COLOR = "#333333"
_PRICE_COLOR = "#1A1A2E"
_MA_COLORS = {"MA5": "#FF6B35", "MA20": "#2196F3", "MA60": "#7B1FA2"}
_BB_COLOR = "#64B5F6"
_RSI_COLOR = "#E91E63"
_RSI_OVER = "#FFCDD2"
_RSI_UNDER = "#C8E6C9"
_VOL_UP = "#EF5350"
_VOL_DOWN = "#42A5F5"
_MACD_LINE = "#FF9800"
_MACD_SIGNAL = "#7B1FA2"

_FONT_SET = False


_FONT_PROP = None  # FontProperties 객체 (한글 렌더링용)


def _setup_font():
    global _FONT_SET, _FONT_PROP
    if _FONT_SET:
        return
    import matplotlib.font_manager as fm
    import glob

    plt.rcParams["axes.unicode_minus"] = False

    # 0) matplotlib 폰트 캐시 삭제 + 리빌드 (Streamlit Cloud에서 packages.txt 설치 후 필수)
    try:
        cache_dir = matplotlib.get_cachedir()
        if cache_dir:
            import os
            for f in os.listdir(cache_dir):
                if f.startswith("fontlist") and f.endswith(".json"):
                    os.remove(os.path.join(cache_dir, f))
                    logger.info(f"폰트 캐시 삭제: {f}")
            fm.fontManager.__init__()  # 폰트 매니저 재초기화
    except Exception as e:
        logger.warning(f"폰트 캐시 리빌드 실패 (무시): {e}")

    # 1) TTF 파일 직접 탐색 + FontProperties 저장 (가장 확실한 방법)
    # Streamlit Cloud (Debian): fonts-nanum → /usr/share/fonts/truetype/nanum/
    search_patterns = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    ]
    for pattern in search_patterns:
        found = glob.glob(pattern)
        if found:
            ttf_path = found[0]
            try:
                fm.fontManager.addfont(ttf_path)
                _FONT_PROP = fm.FontProperties(fname=ttf_path)
                font_name = _FONT_PROP.get_name()
                plt.rcParams["font.family"] = font_name
                plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams.get("font.sans-serif", [])
                _FONT_SET = True
                logger.info(f"한글 폰트 로드 (TTF 직접): {ttf_path} → {font_name}")
                return
            except Exception as e:
                logger.warning(f"폰트 등록 실패 ({ttf_path}): {e}")
                continue

    # 2) 이름 기반 매칭 (macOS AppleGothic 등)
    for font_name in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
        try:
            if any(font_name in f.name for f in fm.fontManager.ttflist):
                plt.rcParams["font.family"] = font_name
                _FONT_SET = True
                logger.info(f"한글 폰트 로드 (시스템): {font_name}")
                return
        except Exception:
            continue

    # 3) 최종 fallback
    plt.rcParams["font.family"] = "sans-serif"
    _FONT_SET = True
    logger.warning("한글 폰트를 찾지 못함 — sans-serif fallback")


def _apply_style(ax, ylabel: str = "", hide_xticklabels: bool = True):
    """공통 축 스타일 적용."""
    ax.set_facecolor(_BG_COLOR)
    ax.grid(True, alpha=0.4, color=_GRID_COLOR, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=8, colors=_TEXT_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_GRID_COLOR)
    ax.spines["bottom"].set_color(_GRID_COLOR)
    if ylabel:
        fp_kw = {"fontproperties": _FONT_PROP} if _FONT_PROP else {}
        ax.set_ylabel(ylabel, fontsize=9, color=_TEXT_COLOR, labelpad=8, **fp_kw)
    if hide_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)


def generate_technical_chart(ticker: str, name: str,
                             days: int = 120) -> Optional[str]:
    """기술적 분석 차트 생성 → base64 PNG 문자열 반환.

    3단 구성:
    1. 상단: 종가 + MA(5/20/60) + 볼린저 밴드 + 고가/저가 밴드
    2. 중단: RSI(14) + 과매수/과매도 구간
    3. 하단: 거래량 바 + MACD 히스토그램/시그널

    Returns:
        base64 인코딩된 PNG 문자열, 실패 시 None
    """
    try:
        _setup_font()

        ohlcv = _get_ohlcv(ticker, days=days + 30)
        if len(ohlcv) < 40:
            return None

        # 전체 데이터
        closes = [d["close"] for d in ohlcv]
        highs = [d["high"] for d in ohlcv]
        lows = [d["low"] for d in ohlcv]
        volumes = [d["volume"] for d in ohlcv]
        dates = [d["date"] for d in ohlcv]

        # 지표 시리즈 계산 (전체 데이터)
        ma5 = _calc_ma_series(closes, 5)
        ma20 = _calc_ma_series(closes, 20)
        ma60 = _calc_ma_series(closes, 60)
        rsi = _calc_rsi_series(closes, 14)
        macd_data = _calc_macd_series(closes)
        bb_upper, bb_lower = _calc_bb_series(closes, 20, 2.0)

        # 표시 범위
        n = min(days, len(dates))
        sl = slice(-n, None)
        x = list(range(n))

        d_closes = closes[sl]
        d_highs = highs[sl]
        d_lows = lows[sl]
        d_volumes = volumes[sl]
        d_dates = dates[sl]

        # X축 날짜 라벨 (연도 변경 시 YYYY/MM/DD 표시)
        step = max(1, n // 6)
        xticks, xlabels = _build_xlabels(d_dates, step)

        # ── Figure 생성 ──
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 7.5),
            gridspec_kw={"height_ratios": [3, 1, 1.3]},
            sharex=True,
        )
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(hspace=0.06, left=0.08, right=0.95, top=0.88, bottom=0.06)

        # 타이틀 — top 마진을 확보하여 범례와 겹치지 않게
        fp = _FONT_PROP or {}
        fp_kw = {"fontproperties": fp} if fp else {}
        fig.text(0.08, 0.98, f"{name} ({ticker})", fontsize=14, fontweight="bold",
                 color=_TEXT_COLOR, va="top", **fp_kw)
        fig.text(0.08, 0.955,
                 f"기준일: {_fmt_date_full(d_dates[-1])}  |  종가: {d_closes[-1]:,}원  |  {n}일",
                 fontsize=9, color="#888888", va="top", **fp_kw)

        # ── 상단: 가격 ──
        _apply_style(ax1, ylabel="")

        # 고가/저가 밴드
        ax1.fill_between(x, d_lows, d_highs, alpha=0.06, color="#9E9E9E", linewidth=0)

        # 볼린저 밴드
        if bb_upper and bb_lower:
            d_bbu = bb_upper[sl]
            d_bbl = bb_lower[sl]
            ax1.fill_between(x, d_bbl, d_bbu, alpha=0.06, color=_BB_COLOR, linewidth=0)
            ax1.plot(x, d_bbu, color=_BB_COLOR, linewidth=0.6, alpha=0.5, linestyle="--")
            ax1.plot(x, d_bbl, color=_BB_COLOR, linewidth=0.6, alpha=0.5, linestyle="--")

        # 이동평균선
        for label, series, color in [("MA5", ma5, _MA_COLORS["MA5"]),
                                     ("MA20", ma20, _MA_COLORS["MA20"]),
                                     ("MA60", ma60, _MA_COLORS["MA60"])]:
            if series:
                ax1.plot(x, series[sl], color=color, linewidth=0.9, alpha=0.85, label=label)

        # 종가 라인 (가장 위에)
        ax1.plot(x, d_closes, color=_PRICE_COLOR, linewidth=1.5, label="종가", zorder=5)

        # 최근 종가 표시
        ax1.annotate(f"{d_closes[-1]:,}", xy=(x[-1], d_closes[-1]),
                     fontsize=8, color=_PRICE_COLOR, fontweight="bold",
                     xytext=(5, 0), textcoords="offset points", va="center")

        ax1.legend(loc="upper right", fontsize=7.5, ncol=4, framealpha=0.8,
                   edgecolor=_GRID_COLOR, fancybox=False)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))

        # ── 중단: RSI ──
        _apply_style(ax2, ylabel="RSI")

        if rsi:
            d_rsi = rsi[sl]
            # 과매수/과매도 배경
            ax2.axhspan(70, 100, alpha=0.08, color="#F44336", linewidth=0)
            ax2.axhspan(0, 30, alpha=0.08, color="#4CAF50", linewidth=0)
            ax2.axhline(70, color="#F44336", linewidth=0.5, linestyle=":", alpha=0.6)
            ax2.axhline(30, color="#4CAF50", linewidth=0.5, linestyle=":", alpha=0.6)
            ax2.axhline(50, color=_GRID_COLOR, linewidth=0.5, linestyle="-", alpha=0.4)

            ax2.plot(x, d_rsi, color=_RSI_COLOR, linewidth=1.0)
            ax2.set_ylim(0, 100)

            # 현재 RSI 값
            last_rsi = d_rsi[-1]
            if last_rsi is not None:
                ax2.annotate(f"{last_rsi:.0f}", xy=(x[-1], last_rsi),
                             fontsize=7, color=_RSI_COLOR, fontweight="bold",
                             xytext=(5, 0), textcoords="offset points", va="center")

        # ── 하단: 거래량 + MACD ──
        _apply_style(ax3, ylabel="거래량", hide_xticklabels=False)

        # 거래량 바
        vol_colors = [_VOL_UP if i == 0 or d_closes[i] >= d_closes[i - 1] else _VOL_DOWN
                      for i in range(len(d_closes))]
        ax3.bar(x, d_volumes, color=vol_colors, alpha=0.35, width=0.7)
        ax3.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda v, p: f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v / 1e3:.0f}K" if v >= 1e3 else ""
            )
        )

        # MACD 오버레이
        if macd_data:
            macd_line, signal_line, histogram = macd_data
            d_macd = macd_line[sl]
            d_signal = signal_line[sl]
            d_hist = histogram[sl]

            ax3b = ax3.twinx()
            ax3b.spines["top"].set_visible(False)
            ax3b.spines["left"].set_visible(False)
            ax3b.spines["right"].set_color(_GRID_COLOR)
            ax3b.spines["bottom"].set_visible(False)
            ax3b.tick_params(axis="y", labelsize=7, colors="#999999")

            # MACD 히스토그램
            hist_c = [_VOL_UP if h and h >= 0 else _VOL_DOWN for h in d_hist]
            ax3b.bar(x, [h or 0 for h in d_hist], color=hist_c, alpha=0.25, width=0.5)

            # MACD / Signal 라인
            ax3b.plot(x, [m or 0 for m in d_macd], color=_MACD_LINE, linewidth=0.9,
                      alpha=0.9, label="MACD")
            ax3b.plot(x, [s or 0 for s in d_signal], color=_MACD_SIGNAL, linewidth=0.9,
                      alpha=0.9, label="Signal")
            ax3b.axhline(0, color=_GRID_COLOR, linewidth=0.4)
            ax3b.set_ylabel("MACD", fontsize=8, color="#999999", labelpad=8)
            ax3b.legend(loc="upper right", fontsize=7, framealpha=0.8,
                        edgecolor=_GRID_COLOR, fancybox=False)

        # X축 라벨
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(xlabels, fontsize=8, color=_TEXT_COLOR)

        # ── base64 PNG 변환 ──
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        logger.warning(f"차트 생성 실패 ({ticker}): {e}")
        return None


# ── 시리즈 계산 헬퍼 ──


def _calc_ma_series(closes: list[int], period: int) -> list:
    """MA 시리즈 (None 패딩, closes와 동일 길이)."""
    result = [None] * (period - 1)
    for i in range(period - 1, len(closes)):
        result.append(sum(closes[i - period + 1:i + 1]) / period)
    return result


def _calc_rsi_series(closes: list[int], period: int = 14) -> list:
    """RSI 시리즈 (None 패딩)."""
    if len(closes) < period + 1:
        return []

    result = [None] * period
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result.append(100.0)
    else:
        result.append(100 - (100 / (1 + avg_gain / avg_loss)))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            result.append(100 - (100 / (1 + avg_gain / avg_loss)))

    return result


def _calc_macd_series(closes: list[int],
                      fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 시리즈 (macd_line, signal_line, histogram) — closes와 동일 길이."""
    if len(closes) < slow + signal:
        return None

    def ema_s(data, period):
        mult = 2 / (period + 1)
        ema = sum(data[:period]) / period
        out = [ema]
        for val in data[period:]:
            ema = (val - ema) * mult + ema
            out.append(ema)
        return out

    fast_ema = ema_s(closes, fast)
    slow_ema = ema_s(closes, slow)
    offset = slow - fast
    macd_raw = [f - s for f, s in zip(fast_ema[offset:], slow_ema)]

    if len(macd_raw) < signal:
        return None

    signal_raw = ema_s(macd_raw, signal)

    pad_m = slow - 1
    pad_s = pad_m + signal - 1
    t = len(closes)

    macd_full = ([None] * pad_m + macd_raw + [None] * t)[:t]
    signal_full = ([None] * pad_s + signal_raw + [None] * t)[:t]
    histogram = [(m - s) if m is not None and s is not None else None
                 for m, s in zip(macd_full, signal_full)]

    return macd_full, signal_full, histogram


def _calc_bb_series(closes: list[int], period: int = 20, num_std: float = 2.0):
    """볼린저 밴드 상단/하단 시리즈."""
    upper, lower = [None] * (period - 1), [None] * (period - 1)
    for i in range(period - 1, len(closes)):
        w = closes[i - period + 1:i + 1]
        mid = sum(w) / period
        std = (sum((v - mid) ** 2 for v in w) / period) ** 0.5
        upper.append(mid + num_std * std)
        lower.append(mid - num_std * std)
    return upper, lower


def _fmt_date(date_str: str) -> str:
    """YYYYMMDD → MM/DD"""
    if len(date_str) == 8:
        return f"{date_str[4:6]}/{date_str[6:]}"
    return date_str


def _fmt_date_full(date_str: str) -> str:
    """YYYYMMDD → YYYY/MM/DD (차트 제목용)"""
    if len(date_str) == 8:
        return f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
    return date_str


def _build_xlabels(dates: list[str], step: int) -> tuple[list[int], list[str]]:
    """X축 라벨 생성 — 연도 변경 시 'YYYY/MM/DD' 표시, 나머지는 MM/DD.

    각 연도의 첫 데이터 지점에 연도를 표시. 데이터의 첫 날짜에도 항상 연도 표시.
    연도 변경점은 반드시 tick에 포함되며, 너무 가까운 기존 tick은 제거.

    Returns:
        (xtick 위치 리스트, 라벨 문자열 리스트)
    """
    n = len(dates)
    if n == 0:
        return [], []

    xticks = set(range(0, n, step))

    # 각 연도가 처음 나타나는 인덱스 수집
    year_change_indices = set()
    seen_years = set()
    for i in range(n):
        if len(dates[i]) == 8:
            yr = dates[i][:4]
            if yr not in seen_years:
                seen_years.add(yr)
                year_change_indices.add(i)  # index 0도 포함

    # 연도 변경점을 xticks에 강제 삽입, 너무 가까운 기존 tick 제거
    min_gap = max(step // 3, 3)
    for yci in year_change_indices:
        # yci와 너무 가까운 기존 tick 제거 (yci 자체는 남김)
        to_remove = {t for t in xticks if t != yci and abs(t - yci) < min_gap}
        xticks -= to_remove
        xticks.add(yci)

    xticks_sorted = sorted(xticks)

    labels = []
    for i in xticks_sorted:
        if i >= n:
            continue
        d = dates[i]
        if len(d) == 8:
            if i in year_change_indices:
                labels.append(f"{d[:4]}/{d[4:6]}/{d[6:]}")
            else:
                labels.append(f"{d[4:6]}/{d[6:]}")
        else:
            labels.append(d)

    xticks_sorted = [t for t in xticks_sorted if t < n]
    return xticks_sorted, labels


# ══════════════════════════════════════════════════════════════
# 비교 상대 수익률 차트
# ══════════════════════════════════════════════════════════════

_COMPARE_COLORS = ["#1A73E8", "#E8453C", "#34A853", "#FBBC04"]


def generate_comparison_chart(
    tickers: list[str], names: list[str], days: int = 120,
) -> Optional[str]:
    """종목 간 상대 수익률 차트 생성 → base64 PNG.

    시작일 = 100 기준으로 정규화하여 기간별 수익률 추이를 비교.

    Args:
        tickers: 비교 대상 티커 리스트 (2~4개)
        names: 종목명 리스트
        days: 비교 기간 (영업일)

    Returns:
        base64 인코딩 PNG 문자열, 실패 시 None
    """
    from src.data.technical import _get_closes

    if len(tickers) < 2 or len(tickers) != len(names):
        return None

    _setup_font()

    # 종가 데이터 수집
    all_data = {}
    for ticker in tickers:
        data = _get_closes(ticker, days=days + 10)
        if len(data) < 20:
            logger.warning(f"비교 차트: {ticker} 데이터 부족 ({len(data)}일)")
            return None
        all_data[ticker] = {d["date"]: d["close"] for d in data}

    # 공통 날짜
    common_dates = sorted(
        set.intersection(*(set(d.keys()) for d in all_data.values()))
    )
    if len(common_dates) < 20:
        return None
    common_dates = common_dates[-days:] if len(common_dates) > days else common_dates

    try:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=_BG_COLOR)
        ax.set_facecolor(_BG_COLOR)

        for i, (ticker, name) in enumerate(zip(tickers, names)):
            closes = [all_data[ticker][d] for d in common_dates]
            if closes[0] <= 0:
                continue
            # 기준일 = 100 정규화
            normalized = [c / closes[0] * 100 for c in closes]
            color = _COMPARE_COLORS[i % len(_COMPARE_COLORS)]
            ax.plot(range(len(normalized)), normalized,
                    color=color, linewidth=1.8, label=name, alpha=0.9)

        # 100 기준선
        ax.axhline(y=100, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)

        # X축 날짜 라벨 (연도 변경 시 YYYY/MM/DD 표시)
        n_points = len(common_dates)
        step = max(1, n_points // 6)
        xtick_pos, xtick_labels = _build_xlabels(common_dates, step)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, fontsize=8, color=_TEXT_COLOR)

        fp_kw = {"fontproperties": _FONT_PROP} if _FONT_PROP else {}
        ax.set_ylabel("상대 수익률 (기준=100)", fontsize=10, color=_TEXT_COLOR, **fp_kw)
        ax.set_title("기간별 상대 수익률 비교", fontsize=13, fontweight="bold",
                      color=_TEXT_COLOR, pad=12, **fp_kw)

        ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
        ax.grid(True, alpha=0.3, color=_GRID_COLOR)
        ax.tick_params(colors=_TEXT_COLOR, labelsize=8)

        # 스파인 제거
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=_BG_COLOR)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        logger.error(f"비교 차트 생성 실패: {e}")
        return None
