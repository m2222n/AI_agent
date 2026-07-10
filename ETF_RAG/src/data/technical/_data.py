"""
DB 접근 + TTL 캐시 — OHLCV / 종가 시계열 조회

싱글톤 DB 커넥션, yfinance fallback, 5분 TTL 캐시.
"""

import logging
import sqlite3
import threading
import time
from typing import Optional

from src.data.database import DB_PATH

logger = logging.getLogger(__name__)

# KOSPI 대표 ETF (베타 계산 시 시장 벤치마크)
MARKET_BENCHMARK = "069500"  # KODEX 200
BENCHMARK_TICKER = "069500"  # KODEX 200

# ── DB 커넥션 싱글톤 (매 호출마다 connect/close 방지) ──
_db_conn: Optional[sqlite3.Connection] = None
_db_lock = threading.Lock()


def _get_db_conn() -> sqlite3.Connection:
    """글로벌 DB 커넥션 반환 (싱글톤, 스레드 안전)."""
    global _db_conn
    with _db_lock:
        if _db_conn is None:
            _db_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            _db_conn.row_factory = sqlite3.Row
            _db_conn.execute("PRAGMA journal_mode=WAL")
        return _db_conn


def reset_db_connection() -> None:
    """DB 파일이 교체됐을 때 싱글톤 커넥션·TTL 캐시를 리셋한다.

    DB 파일을 unlink→재다운로드로 갈아끼우면, 이 싱글톤은 삭제된 옛 inode의
    열린 핸들을 계속 잡고 있어 새 데이터가 절대 반영되지 않는다. 커넥션을 닫고
    None으로 되돌려 다음 조회 시 새 파일로 재연결하게 하고, stale 값을 서빙하지
    않도록 OHLCV/종가 캐시도 비운다. (DB 새로고침 cron 엔드포인트에서 호출)
    """
    global _db_conn
    with _db_lock:
        if _db_conn is not None:
            try:
                _db_conn.close()
            except Exception:  # noqa: BLE001 — 닫기 실패해도 None 리셋은 진행
                logger.warning("기존 DB 커넥션 close 실패 — 무시하고 리셋", exc_info=True)
            _db_conn = None
        _ohlcv_cache.clear()
        _closes_cache.clear()
    logger.info("technical DB 커넥션·캐시 리셋 완료")


# ── 간단한 TTL 캐시 (동일 질문 내 중복 DB 쿼리 방지) ──
_CACHE_TTL = 300  # 5분
_ohlcv_cache: dict[tuple, tuple] = {}  # (ticker, days) → (timestamp, data)
_closes_cache: dict[tuple, tuple] = {}


def _ohlcv_cache_get(ticker: str, days: int) -> Optional[list]:
    key = (ticker, days)
    entry = _ohlcv_cache.get(key)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None


def _ohlcv_cache_put(ticker: str, days: int, data: list):
    _ohlcv_cache[(ticker, days)] = (time.time(), data)


def _closes_cache_get(ticker: str, days: int) -> Optional[list]:
    key = (ticker, days)
    entry = _closes_cache.get(key)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None


def _closes_cache_put(ticker: str, days: int, data: list):
    _closes_cache[(ticker, days)] = (time.time(), data)


def _yfinance_ohlcv(ticker: str, days: int = 250) -> list[dict]:
    """yfinance에서 과거 OHLCV 데이터 조회 (DB 없을 때 fallback).

    Returns:
        [{"date": "20260408", "open": ..., "high": ..., "low": ...,
          "close": ..., "volume": ...}, ...]
    """
    try:
        import yfinance as yf
        from src.data.realtime import krx_to_yfinance
        yf_ticker = krx_to_yfinance(ticker, "stock")
        # days+여유분 (영업일 변환)
        period_map = {250: "1y", 150: "9mo", 60: "3mo"}
        period = "1y"
        for threshold, p in sorted(period_map.items()):
            if days <= threshold:
                period = p
                break
        if days > 250:
            period = "2y"

        df = yf.download(yf_ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return []

        # MultiIndex 컬럼 처리 (yfinance >= 0.2.31)
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df = df.droplevel("Ticker", axis=1)

        result = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y%m%d")
            c = int(round(float(row["Close"])))
            h = int(round(float(row["High"])))
            l = int(round(float(row["Low"])))
            o = int(round(float(row["Open"])))
            v = int(float(row["Volume"]))
            if c > 0 and h > 0 and l > 0:
                result.append({
                    "date": date_str, "open": o, "high": h,
                    "low": l, "close": c, "volume": v,
                })
        return result[-days:] if len(result) > days else result
    except Exception as e:
        logger.warning(f"yfinance OHLCV 조회 실패 ({ticker}): {e}")
        return []


def _db_available() -> bool:
    """SQLite DB 파일이 존재하는지 확인."""
    return DB_PATH.exists()


def _get_closes(ticker: str, days: int = 250,
                conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """최근 N영업일 종가 조회 (날짜 오름차순).

    DB 미존재 시 yfinance fallback.

    Returns:
        [{"date": "20260408", "close": 210500}, ...]
    """
    if conn is None and not _db_available():
        ohlcv = _yfinance_ohlcv(ticker, days)
        return [{"date": d["date"], "close": d["close"]} for d in ohlcv]

    # 캐시된 결과가 있으면 반환
    cached = _closes_cache_get(ticker, days)
    if cached is not None:
        return cached

    use_conn = conn if conn is not None else _get_db_conn()

    rows = use_conn.execute("""
        SELECT date, close FROM daily_prices
        WHERE ticker = ? AND close > 0
        ORDER BY date DESC
        LIMIT ?
    """, (ticker, days)).fetchall()

    # 날짜 오름차순으로 뒤집기
    result = [{"date": r["date"], "close": r["close"]} for r in reversed(rows)]
    _closes_cache_put(ticker, days, result)
    return result


def _get_ohlcv(ticker: str, days: int = 250,
               conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """최근 N영업일 OHLCV 조회 (날짜 오름차순).

    DB 미존재 시 yfinance fallback.

    Returns:
        [{"date": "20260408", "open": 210000, "high": 212000,
          "low": 209000, "close": 210500, "volume": 1234567}, ...]
    """
    if conn is None and not _db_available():
        return _yfinance_ohlcv(ticker, days)

    # 캐시된 결과가 있으면 반환
    cached = _ohlcv_cache_get(ticker, days)
    if cached is not None:
        return cached

    use_conn = conn if conn is not None else _get_db_conn()

    # close>0만 필수. 과거 데이터(yfinance 백필 등)는 high/low가 0/null일 수 있어
    # high>0 AND low>0으로 거르면 과거가 통째로 잘려 기간 분석이 1년치로 제한됐음.
    # high/low가 없으면 close로 대체(종가만 있는 날의 자연스러운 OHLC 근사).
    rows = use_conn.execute("""
        SELECT date, open, high, low, close, volume FROM daily_prices
        WHERE ticker = ? AND close > 0
        ORDER BY date DESC
        LIMIT ?
    """, (ticker, days)).fetchall()

    result = []
    for r in reversed(rows):
        close = r["close"]
        high = r["high"] if r["high"] and r["high"] > 0 else close
        low = r["low"] if r["low"] and r["low"] > 0 else close
        opn = r["open"] if r["open"] and r["open"] > 0 else close
        result.append({
            "date": r["date"], "open": opn, "high": high,
            "low": low, "close": close, "volume": r["volume"] or 0,
        })
    _ohlcv_cache_put(ticker, days, result)
    return result
