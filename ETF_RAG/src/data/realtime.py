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

# DB instruments.market 맵 (lazy 1회 로드): {ticker: "KOSPI"|"KOSDAQ"}
_db_market_map: Optional[dict] = None


def _market_suffix_from_db(ticker: str) -> Optional[str]:
    """DB 시장구분으로 정확한 yfinance suffix. 추측 없는 정공법(전 종목 동일 기준)."""
    global _db_market_map
    if _db_market_map is None:
        try:
            from src.data.database import get_connection, get_market_map, DB_PATH
            conn = get_connection(DB_PATH)
            try:
                _db_market_map = get_market_map(conn)
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 — DB 없으면 빈 맵(아래 yfinance fallback)
            _db_market_map = {}
    mkt = _db_market_map.get(ticker, "")
    if mkt == "KOSPI":
        return "KS"
    if mkt == "KOSDAQ":
        return "KQ"
    return None


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

    # 주식: 1순위 DB 시장구분(정확, 전 종목 동일 기준). 없으면 yfinance 거래량 검증.
    db_suffix = _market_suffix_from_db(ticker)
    if db_suffix:
        _market_suffix_cache[ticker] = db_suffix
        return f"{ticker}.{db_suffix}"

    # 2순위: KS/KQ 둘 다 확인 후 '거래량이 있는' 쪽 선택.
    # ⚠️ 같은 6자리 코드가 KOSPI·KOSDAQ 양쪽에 last_price를 반환할 수 있어
    # (예: 코스닥 039440 에스티아이를 .KS로 조회하면 엉뚱한 값+volume 0),
    # last_price만 보면 잘못된 시장을 고른다 → 거래량(volume>0)까지 검증.
    try:
        import yfinance as yf
        candidates = []  # (suffix, has_volume)
        for suffix in ("KS", "KQ"):
            yf_ticker = f"{ticker}.{suffix}"
            try:
                info = yf.Ticker(yf_ticker).fast_info
                if not info or not getattr(info, "last_price", None):
                    continue
                # last_volume만 본다 — ten_day_average_volume은 양 시장이 같은 값을
                # 줘서(엉뚱한 시장도 통과) 변별력 없음. 당일 실거래(last_volume>0)가
                # 있는 시장이 진짜 상장 시장.
                vol = getattr(info, "last_volume", None)
                candidates.append((suffix, bool(vol)))
            except Exception:
                continue
        # 거래량 있는 시장 우선, 없으면 last_price라도 있는 첫 시장
        for want_vol in (True, False):
            for suffix, has_vol in candidates:
                if has_vol == want_vol:
                    _market_suffix_cache[ticker] = suffix
                    return f"{ticker}.{suffix}"
    except ImportError:
        logger.warning("yfinance 패키지가 설치되지 않았습니다.")

    # 기본: KS (단, 캐시에 저장하지 않음 — 일시적 네트워크 실패로 인한
    # 잘못된 suffix 영구 고정 방지. 다음 호출에서 재시도 가능.)
    return f"{ticker}.KS"


def get_realtime_price(ticker: str, asset_type: str = "etf",
                       cache_ttl: int = 300) -> Optional[dict]:
    """현재 가격 조회 (장중만, 5분 캐시).

    KIS Open API(실시간)를 우선 사용하고, 비활성/실패 시 yfinance(15분 지연)로
    fallback 한다. 둘 다 실패하면 None.

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

    # 1) KIS 실시간 시세 우선 (활성화된 경우)
    try:
        from src.data import kis_client
        if kis_client.is_enabled():
            kis_data = kis_client.get_current_price(ticker, cache_ttl=cache_ttl)
            if kis_data:
                _cache[ticker] = {"data": kis_data, "fetched_at": now}
                return kis_data
            # KIS 실패 시 yfinance로 fallback (아래로 진행)
    except Exception as e:
        logger.warning(f"KIS 조회 실패, yfinance fallback ({ticker}): {e}")

    # 2) yfinance fallback (15분 지연)
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
