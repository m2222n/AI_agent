"""
DB 스키마 + 초기화 — 테이블 정의, 연결, 마이그레이션
"""

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# DB는 영속 볼륨(ETF_DATA_DIR, 예: Railway /data)에 두어 재배포 시 재다운로드를
# 피한다. 미설정 시 기존 경로(src/data)와 동일 — config.PERSIST_DIR와 같은 규칙.
# (config 직접 import는 순환 위험이라 env를 직접 읽어 동일 로직 유지.)
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent  # src/data
DB_PATH = Path(os.getenv("ETF_DATA_DIR", str(_DEFAULT_DATA_DIR))) / "etf_rag.db"

_SCHEMA_SQL = """
-- 종목 마스터 (ETF + 주식)
CREATE TABLE IF NOT EXISTS instruments (
    ticker      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'etf',
    sector      TEXT DEFAULT '',
    market      TEXT DEFAULT '',   -- KOSPI | KOSDAQ (yfinance .KS/.KQ 정확 변환용)
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

-- DART 고유번호 ↔ 종목코드 매핑
CREATE TABLE IF NOT EXISTS dart_corp_codes (
    corp_code   TEXT PRIMARY KEY,
    ticker      TEXT NOT NULL,
    corp_name   TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- 분기 재무제표 (OpenDart)
CREATE TABLE IF NOT EXISTS stock_financials (
    ticker           TEXT NOT NULL,
    fiscal_year      TEXT NOT NULL,
    fiscal_quarter   INTEGER NOT NULL,
    report_code      TEXT NOT NULL,
    revenue          INTEGER,
    operating_profit INTEGER,
    net_income       INTEGER,
    operating_margin REAL,
    net_margin       REAL,
    revenue_growth_yoy REAL,
    op_growth_yoy    REAL,
    collected_at     TEXT NOT NULL,
    PRIMARY KEY (ticker, fiscal_year, fiscal_quarter)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);
CREATE INDEX IF NOT EXISTS idx_returns_date ON returns(date);
CREATE INDEX IF NOT EXISTS idx_holdings_date ON holdings(date);
CREATE INDEX IF NOT EXISTS idx_instruments_type ON instruments(type);
CREATE INDEX IF NOT EXISTS idx_stock_fundamentals_date ON stock_fundamentals(date);
CREATE INDEX IF NOT EXISTS idx_dart_corp_codes_ticker ON dart_corp_codes(ticker);
CREATE INDEX IF NOT EXISTS idx_stock_financials_ticker ON stock_financials(ticker);
CREATE INDEX IF NOT EXISTS idx_stock_financials_year ON stock_financials(fiscal_year);
"""


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
    # instruments.market 컬럼 추가 (v3) — yfinance .KS/.KQ 정확 변환용
    if "market" not in cols:
        conn.execute("ALTER TABLE instruments ADD COLUMN market TEXT DEFAULT ''")
        logger.info("마이그레이션: instruments.market 컬럼 추가")
    # sector 인덱스 (컬럼 존재 확인 후 생성)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_instruments_sector ON instruments(sector)")
