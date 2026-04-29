"""
차트 공통 스타일 — 디자인 상수, 폰트 설정, 축 스타일, 날짜 포맷
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ── 디자인 상수 ──
BG_COLOR = "#FAFAFA"
GRID_COLOR = "#E0E0E0"
TEXT_COLOR = "#333333"
PRICE_COLOR = "#1A1A2E"
MA_COLORS = {"MA5": "#FF6B35", "MA20": "#2196F3", "MA60": "#7B1FA2"}
BB_COLOR = "#64B5F6"
RSI_COLOR = "#E91E63"
RSI_OVER = "#FFCDD2"
RSI_UNDER = "#C8E6C9"
VOL_UP = "#EF5350"
VOL_DOWN = "#42A5F5"
MACD_LINE = "#FF9800"
MACD_SIGNAL = "#7B1FA2"

COMPARE_COLORS = ["#1A73E8", "#E8453C", "#34A853", "#FBBC04"]

PORT_COLOR = "#1A73E8"
BM_COLOR = "#E8453C"
DD_COLOR = "#EF5350"
DD_FILL = "#FFCDD2"

REV_COLOR = "#1A73E8"
OP_COLOR = "#34A853"
NI_COLOR = "#FBBC04"
MARGIN_COLOR = "#E8453C"

VAL_COLORS = ["#1A73E8", "#E8453C"]

SECTOR_PALETTE = [
    "#1A73E8", "#E8453C", "#34A853", "#FBBC04", "#9C27B0",
    "#FF6D00", "#00BCD4", "#795548", "#607D8B", "#E91E63",
    "#3F51B5", "#009688", "#CDDC39", "#FF5722", "#673AB7",
    "#4CAF50", "#FFC107", "#03A9F4", "#8BC34A", "#F44336",
]

_FONT_SET = False
FONT_PROP = None  # FontProperties 객체 (한글 렌더링용)


def setup_font():
    global _FONT_SET, FONT_PROP
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
                FONT_PROP = fm.FontProperties(fname=ttf_path)
                font_name = FONT_PROP.get_name()
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


def apply_style(ax, ylabel: str = "", hide_xticklabels: bool = True):
    """공통 축 스타일 적용."""
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, alpha=0.4, color=GRID_COLOR, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=8, colors=TEXT_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    if ylabel:
        fp_kw = {"fontproperties": FONT_PROP} if FONT_PROP else {}
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_COLOR, labelpad=8, **fp_kw)
    if hide_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)


def font_kw() -> dict:
    """fontproperties kwarg dict (비어있으면 빈 dict)."""
    return {"fontproperties": FONT_PROP} if FONT_PROP else {}


def fmt_date(date_str: str) -> str:
    """YYYYMMDD → MM/DD"""
    if len(date_str) == 8:
        return f"{date_str[4:6]}/{date_str[6:]}"
    return date_str


def fmt_date_full(date_str: str) -> str:
    """YYYYMMDD → YYYY/MM/DD (차트 제목용)"""
    if len(date_str) == 8:
        return f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
    return date_str


def build_xlabels(dates: list, step: int) -> tuple:
    """X축 라벨 생성 — 연도 변경 시 'YYYY/MM/DD' 표시, 나머지는 MM/DD."""
    n = len(dates)
    if n == 0:
        return [], []

    xticks = set(range(0, n, step))

    year_change_indices = set()
    seen_years = set()
    for i in range(n):
        if len(dates[i]) == 8:
            yr = dates[i][:4]
            if yr not in seen_years:
                seen_years.add(yr)
                year_change_indices.add(i)

    min_gap = max(step // 3, 3)
    for yci in year_change_indices:
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


def to_base64(fig) -> str:
    """Figure → base64 PNG 문자열 변환 후 close."""
    import base64
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def to_base64_tight(fig, facecolor=None) -> str:
    """Figure → base64 PNG (tight_layout 버전)."""
    import base64
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=facecolor or BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
