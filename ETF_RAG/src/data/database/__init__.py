"""
SQLite 데이터베이스 패키지 — ETF/주식 시계열 데이터 저장

서브모듈:
- _schema.py      : DB 경로, 스키마, 연결, 초기화, 마이그레이션
- _write.py       : ETF/주식 수집 결과 저장 (upsert)
- _read.py        : 데이터 조회 (loader.py 호환)
- _dart.py        : DART 재무제표 CRUD
- _maintenance.py : 정리, JSON 가져오기, 통계
"""

# ── 스키마 / 연결 ──
from src.data.database._schema import (
    DB_PATH,
    get_connection,
    init_db,
)

# ── 쓰기 ──
from src.data.database._write import (
    upsert_daily_data,
    upsert_stock_data,
)

# ── 읽기 ──
from src.data.database._read import (
    get_latest_date,
    get_latest_data,
    get_latest_stock_data,
    get_historical_prices,
    get_closes_batch,
    get_market_map,
    get_latest_dps,
    get_low_history_tickers,
    search_instruments,
)

# ── DART ──
from src.data.database._dart import (
    upsert_corp_codes,
    get_corp_code,
    get_all_corp_codes,
    upsert_financial_data,
    get_financial_data,
    get_latest_financial_summary,
)

# ── 유지보수 ──
from src.data.database._maintenance import (
    prune_old_data,
    import_json_file,
    get_db_stats,
)

__all__ = [
    "DB_PATH", "get_connection", "init_db",
    "upsert_daily_data", "upsert_stock_data",
    "get_latest_date", "get_latest_data", "get_latest_stock_data",
    "get_historical_prices", "get_closes_batch", "get_market_map", "get_latest_dps", "get_low_history_tickers",
    "search_instruments",
    "upsert_corp_codes", "get_corp_code", "get_all_corp_codes",
    "upsert_financial_data", "get_financial_data", "get_latest_financial_summary",
    "prune_old_data", "import_json_file", "get_db_stats",
]
