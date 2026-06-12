"""
한국투자증권(KIS) Open API REST 클라이언트 — 국내 주식/ETF 현재가 조회 (F-2)

- OAuth2 접근토큰 발급/캐싱/자동 갱신 (KIS는 토큰 재발급을 분당 1회로 제한하고
  유효기간 내 동일 토큰을 재사용하라고 안내 → 디스크 캐시로 프로세스 재시작에도 생존)
- 국내주식 현재가 시세 (FHKST01010100)
- 키 미설정 시 비활성 → 호출자(realtime.py)가 yfinance로 fallback

설계: realtime.py와 동일한 dict 스키마({"price","prev_close","change","change_pct",
"volume","timestamp","source"})를 반환해 상위 코드 변경을 최소화한다.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# 토큰 디스크 캐시 (프로세스 재시작/멀티프로세스 간 공유, 재발급 rate-limit 회피)
_TOKEN_CACHE_PATH = Path.home() / ".cache" / "etf_rag" / "kis_token.json"

# 인메모리 토큰 상태: {"access_token": str, "expires_at": float}
_token_state: dict = {}
_token_lock = threading.Lock()

# 현재가 캐시: {ticker: {"data": dict, "fetched_at": float}}
_price_cache: dict = {}


def _kis_config() -> dict:
    """config.KIS 를 매 호출 시 읽어 테스트에서 patch 가능하게 한다."""
    from config import KIS
    return KIS


def is_enabled() -> bool:
    """KIS 연동 활성화 여부 (app_key/secret 모두 존재)."""
    return bool(_kis_config().get("enabled"))


# ── 토큰 발급/캐싱 ────────────────────────────────────────

def _load_cached_token() -> Optional[dict]:
    """디스크 캐시에서 유효한 토큰 로드 (없거나 만료 시 None)."""
    try:
        if not _TOKEN_CACHE_PATH.exists():
            return None
        with open(_TOKEN_CACHE_PATH, "r") as f:
            data = json.load(f)
        margin = _kis_config().get("token_margin", 600)
        if data.get("expires_at", 0) - margin > time.time():
            return data
    except Exception as e:
        logger.debug(f"KIS 토큰 캐시 로드 실패: {e}")
    return None


def _save_cached_token(state: dict) -> None:
    """토큰을 디스크 캐시에 저장 (best-effort)."""
    try:
        _TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_TOKEN_CACHE_PATH, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.debug(f"KIS 토큰 캐시 저장 실패: {e}")


def _get_access_token(force: bool = False) -> Optional[str]:
    """유효한 접근토큰 반환. 캐시 우선 → 만료/없음이면 신규 발급.

    KIS 토큰은 발급 후 24시간 유효, 재발급은 분당 1회 제한.
    margin(기본 10분) 안에 들면 선제 갱신한다.
    """
    cfg = _kis_config()
    if not cfg.get("enabled"):
        return None

    margin = cfg.get("token_margin", 600)
    now = time.time()

    with _token_lock:
        # 1) 인메모리 캐시
        if not force and _token_state.get("access_token") and \
                _token_state.get("expires_at", 0) - margin > now:
            return _token_state["access_token"]

        # 2) 디스크 캐시
        if not force:
            cached = _load_cached_token()
            if cached:
                _token_state.update(cached)
                return cached["access_token"]

        # 3) 신규 발급
        import requests
        url = f"{cfg['base_url']}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": cfg["app_key"],
            "appsecret": cfg["app_secret"],
        }
        try:
            resp = requests.post(url, json=body, timeout=cfg.get("timeout", 5))
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"KIS 토큰 발급 실패: {e}")
            return None

        token = data.get("access_token")
        if not token:
            logger.warning(f"KIS 토큰 응답에 access_token 없음: {data}")
            return None

        # expires_in(초) 우선, 없으면 24시간으로 가정
        expires_in = data.get("expires_in", 86400)
        try:
            expires_in = int(expires_in)
        except (TypeError, ValueError):
            expires_in = 86400

        state = {"access_token": token, "expires_at": now + expires_in}
        _token_state.update(state)
        _save_cached_token(state)
        return token


# ── 현재가 조회 ───────────────────────────────────────────

def _parse_price_output(output: dict) -> Optional[dict]:
    """KIS inquire-price output → realtime 표준 스키마 변환."""
    try:
        price = float(output.get("stck_prpr", 0))  # 주식 현재가
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None

    def _to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _to_int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    prev_close = _to_float(output.get("stck_sdpr"))     # 전일 종가
    change = _to_float(output.get("prdy_vrss"))         # 전일 대비
    change_pct = _to_float(output.get("prdy_ctrt"))     # 전일 대비율(%)
    sign = output.get("prdy_vrss_sign", "")             # 1상한 2상승 3보합 4하한 5하락
    if change is not None and sign in ("4", "5"):
        change = -abs(change)
    if change_pct is not None and sign in ("4", "5"):
        change_pct = -abs(change_pct)

    return {
        "price": round(price),
        "prev_close": round(prev_close) if prev_close else None,
        "change": round(change) if change is not None else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "volume": _to_int(output.get("acml_vol")),      # 누적 거래량
        "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "source": "kis",
    }


def get_current_price(ticker: str, cache_ttl: int = 300) -> Optional[dict]:
    """국내 주식/ETF 현재가 조회 (FHKST01010100).

    Args:
        ticker: KRX 6자리 종목코드 (ETF/주식 공통, 시장구분 'J' 사용)
        cache_ttl: 결과 캐시 TTL(초)

    Returns:
        성공 시 realtime 표준 dict (source="kis"), 실패/비활성 시 None.
    """
    if not is_enabled():
        return None

    now = time.time()
    cached = _price_cache.get(ticker)
    if cached and (now - cached["fetched_at"]) < cache_ttl:
        return cached["data"]

    token = _get_access_token()
    if not token:
        return None

    cfg = _kis_config()
    import requests
    url = f"{cfg['base_url']}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": cfg["app_key"],
        "appsecret": cfg["app_secret"],
        "tr_id": "FHKST01010100",
    }
    params = {
        "fid_cond_mrkt_div_code": "J",   # J: 주식/ETF/ETN
        "fid_input_iscd": ticker,
    }

    try:
        resp = requests.get(url, headers=headers, params=params,
                            timeout=cfg.get("timeout", 5))
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"KIS 현재가 조회 실패 ({ticker}): {e}")
        return None

    # rt_cd "0" = 정상
    if data.get("rt_cd") != "0":
        logger.warning(f"KIS 현재가 응답 오류 ({ticker}): "
                       f"{data.get('msg_cd')} {data.get('msg1')}")
        return None

    parsed = _parse_price_output(data.get("output", {}))
    if parsed is None:
        return None

    _price_cache[ticker] = {"data": parsed, "fetched_at": now}
    return parsed


def clear_cache() -> None:
    """현재가/토큰 인메모리 캐시 초기화 (디스크 캐시는 유지)."""
    _price_cache.clear()
    _token_state.clear()
