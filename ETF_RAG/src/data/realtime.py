"""
실시간 시세 조회 모듈 — yfinance 기반 (15분 지연)

장중(평일 09:00-15:30 KST) 실시간 가격 조회, 5분 캐싱.
장 외 시간에는 None 반환 → 호출자가 pykrx 종가 데이터 사용.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# 가격 캐시: {krx_ticker: {"data": dict, "fetched_at": float}}
_cache: dict = {}

# 티커 마켓 매핑 캐시: {krx_ticker: "KS" | "KQ"}
_market_suffix_cache: dict = {}


def is_market_open(now: Optional[datetime] = None) -> bool:
    """KST 기준 장 운영 시간 판단 (평일 09:00~15:30)"""
    if now is None:
        now = datetime.now(KST)
    if now.weekday() >= 5:  # 토/일
        return False
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def krx_to_yfinance(ticker: str, asset_type: str = "etf") -> str:
    """KRX 6자리 티커 → yfinance 포맷 변환 (.KS/.KQ)

    ETF는 항상 KOSPI(.KS), 주식은 .KS 시도 후 .KQ fallback.
    """
    if ticker in _market_suffix_cache:
        return f"{ticker}.{_market_suffix_cache[ticker]}"

    # ETF는 모두 KOSPI
    if asset_type == "etf":
        _market_suffix_cache[ticker] = "KS"
        return f"{ticker}.KS"

    # 주식: KS 먼저 시도, 실패하면 KQ
    try:
        import yfinance as yf
        for suffix in ("KS", "KQ"):
            yf_ticker = f"{ticker}.{suffix}"
            try:
                info = yf.Ticker(yf_ticker).fast_info
                if info and getattr(info, "last_price", None):
                    _market_suffix_cache[ticker] = suffix
                    return yf_ticker
            except Exception:
                continue
    except ImportError:
        logger.warning("yfinance 패키지가 설치되지 않았습니다.")

    # 기본: KS
    _market_suffix_cache[ticker] = "KS"
    return f"{ticker}.KS"


def get_realtime_price(ticker: str, asset_type: str = "etf",
                       cache_ttl: int = 300) -> Optional[dict]:
    """yfinance에서 현재 가격 조회 (장중만, 5분 캐시)

    Returns:
        성공 시 {"price", "prev_close", "change", "change_pct",
                 "volume", "timestamp", "source"} dict.
        장 외 시간이거나 실패 시 None.
    """
    if not is_market_open():
        return None

    # 캐시 확인
    now = time.time()
    cached = _cache.get(ticker)
    if cached and (now - cached["fetched_at"]) < cache_ttl:
        return cached["data"]

    # yfinance 조회
    try:
        import yfinance as yf
        yf_ticker = krx_to_yfinance(ticker, asset_type)
        t = yf.Ticker(yf_ticker)
        info = t.fast_info

        last_price = getattr(info, "last_price", None)
        if last_price is None:
            return None

        prev_close = getattr(info, "previous_close", None)
        change = (last_price - prev_close) if prev_close else None
        change_pct = (change / prev_close * 100) if (prev_close and prev_close != 0) else None

        data = {
            "price": round(last_price),
            "prev_close": round(prev_close) if prev_close else None,
            "change": round(change) if change is not None else None,
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "volume": getattr(info, "last_volume", None),
            "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
            "source": "yfinance",
        }

        _cache[ticker] = {"data": data, "fetched_at": now}
        return data

    except ImportError:
        logger.warning("yfinance 패키지가 설치되지 않았습니다.")
        return None
    except Exception as e:
        logger.warning(f"yfinance 조회 실패 ({ticker}): {e}")
        return None


def clear_cache():
    """캐시 전체 초기화"""
    _cache.clear()
    _market_suffix_cache.clear()
