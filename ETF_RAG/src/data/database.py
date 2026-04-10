"""
SQLite 데이터베이스 레이어 — ETF/주식 시계열 데이터 저장

기존 JSON 파일 기반 저장소를 대체하여 5년 보존 + 시계열 조회를 지원.
collector.py 출력을 그대로 받아 저장하고, loader.py 호환 포맷으로 반환.

사용법:
    from src.data.database import init_db, upsert_daily_data, get_latest_data

    conn = init_db()
    upsert_daily_data(conn, collected_data)  # collector.collect_all() 결과
    etfs = get_latest_data(conn)             # loader._normalize_collected() 호환
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "etf_rag.db"

_SCHEMA_SQL = """
-- 종목 마스터 (ETF + 주식)
CREATE TABLE IF NOT EXISTS instruments (
    ticker      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'etf',
    sector      TEXT DEFAULT '',
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 1
);

-- 일별 시세
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        INTEGER,
    high        INTEGER,
    low         INTEGER,
    close       INTEGER NOT NULL DEFAULT 0,
    volume      INTEGER,
    trade_value INTEGER,
    nav         REAL,
    base_index  REAL,
    change      INTEGER,
    change_pct  REAL,
    deviation   REAL,
    tracking_error REAL,
    PRIMARY KEY (ticker, date)
);

-- 기간별 수익률
CREATE TABLE IF NOT EXISTS returns (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    period      TEXT NOT NULL,
    return_pct  REAL,
    PRIMARY KEY (ticker, date, period)
);

-- 보유종목 (ETF PDF 구성종목)
CREATE TABLE IF NOT EXISTS holdings (
    ticker       TEXT NOT NULL,
    date         TEXT NOT NULL,
    stock_ticker TEXT NOT NULL,
    stock_name   TEXT,
    shares       REAL,
    amount       INTEGER,
    weight       REAL,
    PRIMARY KEY (ticker, date, stock_ticker)
);

-- 수집 로그
CREATE TABLE IF NOT EXISTS collection_log (
    date          TEXT PRIMARY KEY,
    collected_at  TEXT NOT NULL,
    total_count   INTEGER,
    holdings_count INTEGER,
    source        TEXT DEFAULT 'pykrx'
);

-- 주식 펀더멘털 (PER/PBR/EPS 등)
CREATE TABLE IF NOT EXISTS stock_fundamentals (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    market_cap  INTEGER,
    shares_outstanding INTEGER,
    bps         REAL,
    per         REAL,
    pbr         REAL,
    eps         REAL,
    div         REAL,
    dps         REAL,
    PRIMARY KEY (ticker, date)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);
CREATE INDEX IF NOT EXISTS idx_returns_date ON returns(date);
CREATE INDEX IF NOT EXISTS idx_holdings_date ON holdings(date);
CREATE INDEX IF NOT EXISTS idx_instruments_type ON instruments(type);
CREATE INDEX IF NOT EXISTS idx_stock_fundamentals_date ON stock_fundamentals(date);
"""


# ── 연결 / 초기화 ─────────────────────────────────────────────

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """SQLite 연결 (WAL 모드, Row 팩토리)"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """DB 초기화: 테이블 생성 + 마이그레이션 + 연결 반환"""
    conn = get_connection(db_path)
    conn.executescript(_SCHEMA_SQL)
    _migrate(conn)
    logger.info(f"DB 초기화: {db_path}")
    return conn


def _migrate(conn: sqlite3.Connection):
    """기존 DB 스키마 마이그레이션 (컬럼 추가 등)"""
    # instruments.sector 컬럼 추가 (v2)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(instruments)").fetchall()}
    if "sector" not in cols:
        conn.execute("ALTER TABLE instruments ADD COLUMN sector TEXT DEFAULT ''")
        logger.info("마이그레이션: instruments.sector 컬럼 추가")
    # sector 인덱스 (컬럼 존재 확인 후 생성)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_instruments_sector ON instruments(sector)")


# ── 쓰기: collector 출력 → DB ──────────────────────────────────

def upsert_daily_data(conn: sqlite3.Connection, data: dict) -> int:
    """
    수집 결과를 DB에 저장 (INSERT OR REPLACE).

    Args:
        data: collector.collect_all() 출력
              {"metadata": {...}, "etfs": [{ticker, name, date, ohlcv, ...}]}

    Returns:
        저장된 ETF 수
    """
    meta = data.get("metadata", {})
    etfs = data.get("etfs", [])
    if not etfs:
        return 0

    date = meta.get("collection_date", "")

    with conn:
        # 1) instruments
        conn.executemany("""
            INSERT INTO instruments (ticker, name, type, first_seen, last_seen)
            VALUES (?, ?, 'etf', ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                name = excluded.name,
                last_seen = excluded.last_seen,
                is_active = 1
        """, [(e["ticker"], e["name"], date, date) for e in etfs])

        # 2) daily_prices
        price_rows = []
        for e in etfs:
            ohlcv = e.get("ohlcv") or {}
            price_rows.append((
                e["ticker"], e.get("date", date),
                ohlcv.get("open"), ohlcv.get("high"),
                ohlcv.get("low"), ohlcv.get("close", 0),
                ohlcv.get("volume"), ohlcv.get("trade_value"),
                ohlcv.get("nav"), ohlcv.get("base_index"),
                ohlcv.get("change"), ohlcv.get("change_pct"),
                e.get("deviation"), e.get("tracking_error"),
            ))
        conn.executemany("""
            INSERT OR REPLACE INTO daily_prices
            (ticker, date, open, high, low, close, volume, trade_value,
             nav, base_index, change, change_pct, deviation, tracking_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, price_rows)

        # 3) returns
        return_rows = []
        for e in etfs:
            for period, pct in e.get("returns", {}).items():
                if pct is not None:
                    return_rows.append((
                        e["ticker"], e.get("date", date), period, pct
                    ))
        if return_rows:
            conn.executemany("""
                INSERT OR REPLACE INTO returns (ticker, date, period, return_pct)
                VALUES (?, ?, ?, ?)
            """, return_rows)

        # 4) holdings
        holding_rows = []
        max_sqlite_int = 2**63 - 1
        for e in etfs:
            for h in e.get("holdings", []):
                amount = h.get("amount")
                # pykrx가 음수 금액을 uint64로 반환하는 경우 → None 처리
                if amount is not None and abs(amount) > max_sqlite_int:
                    amount = None
                holding_rows.append((
                    e["ticker"], e.get("date", date),
                    h["stock_ticker"], h.get("stock_name", ""),
                    h.get("shares"), amount,
                    h.get("weight"),
                ))
        if holding_rows:
            conn.executemany("""
                INSERT OR REPLACE INTO holdings
                (ticker, date, stock_ticker, stock_name, shares, amount, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, holding_rows)

        # 5) collection_log
        conn.execute("""
            INSERT OR REPLACE INTO collection_log
            (date, collected_at, total_count, holdings_count)
            VALUES (?, ?, ?, ?)
        """, (
            date,
            meta.get("collected_at", datetime.now().isoformat()),
            meta.get("total_etfs", len(etfs)),
            meta.get("holdings_collected", 0),
        ))

    logger.info(f"DB 저장: {len(etfs)}개 ETF ({date})")
    return len(etfs)


def upsert_stock_data(conn: sqlite3.Connection, data: dict) -> int:
    """
    주식 수집 결과를 DB에 저장 (INSERT OR REPLACE).

    Args:
        data: stock_collector.collect_all() 출력
              {"metadata": {...}, "stocks": [{ticker, name, date, ohlcv, ...}]}

    Returns:
        저장된 종목 수
    """
    meta = data.get("metadata", {})
    stocks = data.get("stocks", [])
    if not stocks:
        return 0

    date = meta.get("collection_date", "")

    with conn:
        # 1) instruments (type='stock', sector 포함)
        conn.executemany("""
            INSERT INTO instruments (ticker, name, type, sector, first_seen, last_seen)
            VALUES (?, ?, 'stock', ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                name = excluded.name,
                sector = CASE WHEN excluded.sector != '' THEN excluded.sector
                              ELSE instruments.sector END,
                last_seen = excluded.last_seen,
                is_active = 1
        """, [(s["ticker"], s["name"], s.get("sector", ""), date, date)
              for s in stocks])

        # 2) daily_prices (주식은 nav/base_index/deviation/tracking_error 없음)
        price_rows = []
        for s in stocks:
            ohlcv = s.get("ohlcv") or {}
            price_rows.append((
                s["ticker"], s.get("date", date),
                ohlcv.get("open"), ohlcv.get("high"),
                ohlcv.get("low"), ohlcv.get("close", 0),
                ohlcv.get("volume"), ohlcv.get("trade_value"),
                None, None,  # nav, base_index
                None, ohlcv.get("change_pct"),  # change, change_pct
                None, None,  # deviation, tracking_error
            ))
        conn.executemany("""
            INSERT OR REPLACE INTO daily_prices
            (ticker, date, open, high, low, close, volume, trade_value,
             nav, base_index, change, change_pct, deviation, tracking_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, price_rows)

        # 3) returns
        return_rows = []
        for s in stocks:
            for period, pct in s.get("returns", {}).items():
                if pct is not None:
                    return_rows.append((
                        s["ticker"], s.get("date", date), period, pct
                    ))
        if return_rows:
            conn.executemany("""
                INSERT OR REPLACE INTO returns (ticker, date, period, return_pct)
                VALUES (?, ?, ?, ?)
            """, return_rows)

        # 4) stock_fundamentals
        fund_rows = []
        for s in stocks:
            fund = s.get("fundamental") or {}
            fund_rows.append((
                s["ticker"], s.get("date", date),
                s.get("market_cap", 0),
                s.get("shares_outstanding", 0),
                fund.get("bps"), fund.get("per"), fund.get("pbr"),
                fund.get("eps"), fund.get("div"), fund.get("dps"),
            ))
        conn.executemany("""
            INSERT OR REPLACE INTO stock_fundamentals
            (ticker, date, market_cap, shares_outstanding,
             bps, per, pbr, eps, div, dps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, fund_rows)

        # 5) collection_log
        conn.execute("""
            INSERT OR REPLACE INTO collection_log
            (date, collected_at, total_count, holdings_count, source)
            VALUES (?, ?, ?, 0, 'pykrx_stock')
        """, (
            date,
            meta.get("collected_at", datetime.now().isoformat()),
            meta.get("total_stocks", len(stocks)),
        ))

    logger.info(f"DB 저장: {len(stocks)}개 주식 ({date})")
    return len(stocks)


# ── 읽기: loader.py 호환 ──────────────────────────────────────

def get_latest_date(conn: sqlite3.Connection) -> Optional[str]:
    """최신 수집일 반환"""
    row = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
    return row[0] if row and row[0] else None


def get_latest_data(conn: sqlite3.Connection,
                    date: Optional[str] = None,
                    inst_type: str = "etf") -> List[dict]:
    """
    특정일의 ETF 데이터를 loader._normalize_collected() 호환 포맷으로 반환.

    date가 None이면 최신 수집일 사용.
    반환 포맷: [{ticker, name, date, open, high, low, close, volume,
                 trade_value, nav, base_index, change, change_pct,
                 deviation, tracking_error, returns, holdings}, ...]
    """
    if date is None:
        date = get_latest_date(conn)
    if not date:
        return []

    # 시세 + 종목명 조회
    rows = conn.execute("""
        SELECT p.*, i.name
        FROM daily_prices p
        JOIN instruments i ON p.ticker = i.ticker
        WHERE p.date = ? AND i.type = ?
        ORDER BY p.trade_value DESC
    """, (date, inst_type)).fetchall()

    if not rows:
        return []

    # 수익률 조회
    return_rows = conn.execute("""
        SELECT ticker, period, return_pct
        FROM returns WHERE date = ?
    """, (date,)).fetchall()

    returns_map = {}
    for r in return_rows:
        returns_map.setdefault(r["ticker"], {})[r["period"]] = r["return_pct"]

    # 보유종목 조회
    holding_rows = conn.execute("""
        SELECT ticker, stock_ticker, stock_name, shares, amount, weight
        FROM holdings WHERE date = ?
    """, (date,)).fetchall()

    holdings_map = {}
    for h in holding_rows:
        holdings_map.setdefault(h["ticker"], []).append({
            "stock_ticker": h["stock_ticker"],
            "stock_name": h["stock_name"] or "",
            "shares": h["shares"],
            "amount": h["amount"],
            "weight": h["weight"],
        })

    # 통일 포맷으로 변환
    result = []
    for row in rows:
        ticker = row["ticker"]
        result.append({
            "ticker": ticker,
            "name": row["name"],
            "date": row["date"],
            "open": row["open"] or 0,
            "high": row["high"] or 0,
            "low": row["low"] or 0,
            "close": row["close"] or 0,
            "volume": row["volume"] or 0,
            "trade_value": row["trade_value"] or 0,
            "nav": row["nav"] or 0,
            "base_index": row["base_index"] or 0,
            "change": row["change"] or 0,
            "change_pct": row["change_pct"] or 0.0,
            "deviation": row["deviation"],
            "tracking_error": row["tracking_error"],
            "returns": returns_map.get(ticker, {}),
            "holdings": holdings_map.get(ticker, []),
        })

    return result


def get_latest_stock_data(conn: sqlite3.Connection,
                          date: Optional[str] = None) -> List[dict]:
    """
    특정일의 주식 데이터를 반환.

    date가 None이면 주식 데이터의 최신 수집일 사용.
    반환 포맷: [{ticker, name, date, close, volume, trade_value, change_pct,
                 market_cap, shares_outstanding, per, pbr, eps, div, dps,
                 returns}, ...]
    """
    if date is None:
        # 주식 최신일 조회
        row = conn.execute("""
            SELECT MAX(p.date) FROM daily_prices p
            JOIN instruments i ON p.ticker = i.ticker
            WHERE i.type = 'stock'
        """).fetchone()
        date = row[0] if row and row[0] else None
    if not date:
        return []

    # 시세 + 종목명 + 업종
    rows = conn.execute("""
        SELECT p.*, i.name, i.sector
        FROM daily_prices p
        JOIN instruments i ON p.ticker = i.ticker
        WHERE p.date = ? AND i.type = 'stock'
        ORDER BY p.trade_value DESC
    """, (date,)).fetchall()

    if not rows:
        return []

    # 수익률
    return_rows = conn.execute("""
        SELECT r.ticker, r.period, r.return_pct
        FROM returns r
        JOIN instruments i ON r.ticker = i.ticker
        WHERE r.date = ? AND i.type = 'stock'
    """, (date,)).fetchall()

    returns_map = {}
    for r in return_rows:
        returns_map.setdefault(r["ticker"], {})[r["period"]] = r["return_pct"]

    # 펀더멘털
    fund_rows = conn.execute("""
        SELECT * FROM stock_fundamentals WHERE date = ?
    """, (date,)).fetchall()

    fund_map = {}
    for f in fund_rows:
        fund_map[f["ticker"]] = {
            "market_cap": f["market_cap"] or 0,
            "shares_outstanding": f["shares_outstanding"] or 0,
            "bps": f["bps"] or 0,
            "per": f["per"] or 0,
            "pbr": f["pbr"] or 0,
            "eps": f["eps"] or 0,
            "div": f["div"] or 0,
            "dps": f["dps"] or 0,
        }

    result = []
    for row in rows:
        ticker = row["ticker"]
        fund = fund_map.get(ticker, {})
        result.append({
            "ticker": ticker,
            "name": row["name"],
            "date": row["date"],
            "sector": row["sector"] or "",
            "open": row["open"] or 0,
            "high": row["high"] or 0,
            "low": row["low"] or 0,
            "close": row["close"] or 0,
            "volume": row["volume"] or 0,
            "trade_value": row["trade_value"] or 0,
            "change_pct": row["change_pct"] or 0.0,
            "market_cap": fund.get("market_cap", 0),
            "shares_outstanding": fund.get("shares_outstanding", 0),
            "per": fund.get("per", 0),
            "pbr": fund.get("pbr", 0),
            "eps": fund.get("eps", 0),
            "bps": fund.get("bps", 0),
            "div": fund.get("div", 0),
            "dps": fund.get("dps", 0),
            "returns": returns_map.get(ticker, {}),
        })

    return result


def get_historical_prices(conn: sqlite3.Connection,
                          ticker: str,
                          start_date: str,
                          end_date: str) -> List[dict]:
    """시계열 가격 조회 (날짜 오름차순)"""
    rows = conn.execute("""
        SELECT date, open, high, low, close, volume, trade_value,
               nav, change_pct
        FROM daily_prices
        WHERE ticker = ? AND date >= ? AND date <= ?
        ORDER BY date ASC
    """, (ticker, start_date, end_date)).fetchall()

    return [dict(r) for r in rows]


def search_instruments(conn: sqlite3.Connection,
                       keyword: str = "",
                       inst_type: str = "") -> List[dict]:
    """종목 검색 (이름/티커 키워드)"""
    query = "SELECT * FROM instruments WHERE is_active = 1"
    params = []

    if inst_type:
        query += " AND type = ?"
        params.append(inst_type)

    if keyword:
        query += " AND (name LIKE ? OR ticker LIKE ?)"
        like = f"%{keyword}%"
        params.extend([like, like])

    query += " ORDER BY last_seen DESC LIMIT 50"
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ── 유지보수 ──────────────────────────────────────────────────

def prune_old_data(conn: sqlite3.Connection, retention_days: int = 4380):
    """
    오래된 데이터 삭제 (기본 12년 = 4380일).
    holdings는 1년만 보존.
    """
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y%m%d")
    holdings_cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    with conn:
        for table in ["daily_prices", "returns", "stock_fundamentals"]:
            deleted = conn.execute(
                f"DELETE FROM {table} WHERE date < ?", (cutoff,)
            ).rowcount
            if deleted:
                logger.info(f"{table}에서 {deleted}행 삭제 ({cutoff} 이전)")

        deleted = conn.execute(
            "DELETE FROM holdings WHERE date < ?", (holdings_cutoff,)
        ).rowcount
        if deleted:
            logger.info(f"holdings에서 {deleted}행 삭제 ({holdings_cutoff} 이전)")

        # 90일 미확인 종목 비활성 처리
        inactive_cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        conn.execute(
            "UPDATE instruments SET is_active = 0 WHERE last_seen < ?",
            (inactive_cutoff,)
        )

    logger.info("데이터 정리 완료")


def import_json_file(conn: sqlite3.Connection, json_path: Path) -> int:
    """
    기존 JSON 수집 파일을 DB로 가져오기 (마이그레이션용).

    Args:
        json_path: etf_data_YYYYMMDD.json 경로

    Returns:
        가져온 ETF 수
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return upsert_daily_data(conn, data)


def get_db_stats(conn: sqlite3.Connection) -> dict:
    """DB 통계 (디버깅/모니터링용)"""
    stats = {}
    for table in ["instruments", "daily_prices", "returns", "holdings",
                   "stock_fundamentals", "collection_log"]:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        stats[table] = row[0]

    stats["latest_date"] = get_latest_date(conn)
    stats["date_range"] = {}

    row = conn.execute("SELECT MIN(date), MAX(date) FROM daily_prices").fetchone()
    if row and row[0]:
        stats["date_range"] = {"start": row[0], "end": row[1]}

    return stats
