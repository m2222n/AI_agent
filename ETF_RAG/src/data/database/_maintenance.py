"""
DB 유지보수 — 정리, 마이그레이션, 통계
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from src.data.database._write import upsert_daily_data
from src.data.database._read import get_latest_date

logger = logging.getLogger(__name__)


def prune_old_data(conn: sqlite3.Connection, retention_days: int = 4380):
    """
    오래된 데이터 정리.
    - daily_prices, returns, stock_fundamentals: 영구 보존 (KRX 슬라이딩 윈도우로 재수집 불가)
    - holdings: 1년만 보존 (용량 관리)
    """
    holdings_cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    with conn:
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
                   "stock_fundamentals", "collection_log",
                   "dart_corp_codes", "stock_financials"]:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        stats[table] = row[0]

    stats["latest_date"] = get_latest_date(conn)
    stats["date_range"] = {}

    row = conn.execute("SELECT MIN(date), MAX(date) FROM daily_prices").fetchone()
    if row and row[0]:
        stats["date_range"] = {"start": row[0], "end": row[1]}

    return stats
