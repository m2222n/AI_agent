"""
숫자/퍼센트 포맷 유틸리티

프로젝트 전반에서 사용하는 공통 포맷 함수를 한 곳에서 관리한다.
"""


def format_market_cap(value: int, suffix: bool = True) -> str:
    """시가총액을 조/억 단위로 포맷.

    Args:
        value: 시가총액 (원)
        suffix: True면 '원' 접미사 포함 (UI용), False면 생략 (도구 텍스트용)
    """
    won = "원" if suffix else ""
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}조{won}"
    elif value >= 100_000_000:
        return f"{value / 100_000_000:,.0f}억{won}"
    return f"{value:,}{won}" if suffix else f"{value:,}"


def format_large_number(value: int) -> str:
    """큰 숫자를 조/억/만 단위로 포맷 (거래대금, 매출 등)."""
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}조"
    elif value >= 100_000_000:
        return f"{value / 100_000_000:.0f}억"
    elif value >= 10_000:
        return f"{value / 10_000:.0f}만"
    return f"{value:,}"


def format_change(pct: float) -> str:
    """등락률을 색상 이모지와 함께 포맷."""
    if pct > 0:
        return f"🔴 +{pct:.2f}%"
    elif pct < 0:
        return f"🔵 {pct:.2f}%"
    return f"⚪ {pct:.2f}%"


def format_percentage(value, default: str = "-") -> str:
    """퍼센트 포맷 (None 처리 포함)."""
    if value is None:
        return default
    return f"{value:+.2f}%"
