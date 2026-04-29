"""
DART 재무제표 CRUD — corp_code 매핑 + 분기 재무데이터
"""

import sqlite3
from datetime import datetime
from typing import List, Optional


def upsert_corp_codes(conn: sqlite3.Connection, codes: List[dict]) -> int:
    """
    DART corp_code ↔ ticker 매핑 저장.

    Args:
        codes: [{"corp_code": "00126380", "ticker": "005930", "corp_name": "삼성전자"}]

    Returns:
        저장된 건수
    """
    if not codes:
        return 0
    now = datetime.now().isoformat()
    with conn:
        conn.executemany("""
            INSERT OR REPLACE INTO dart_corp_codes
            (corp_code, ticker, corp_name, updated_at)
            VALUES (?, ?, ?, ?)
        """, [(c["corp_code"], c["ticker"], c["corp_name"], now) for c in codes])
    return len(codes)


def get_corp_code(conn: sqlite3.Connection, ticker: str) -> Optional[str]:
    """티커로 DART corp_code 조회"""
    row = conn.execute(
        "SELECT corp_code FROM dart_corp_codes WHERE ticker = ?", (ticker,)
    ).fetchone()
    return row[0] if row else None


def get_all_corp_codes(conn: sqlite3.Connection) -> dict:
    """전체 ticker → corp_code 매핑 반환"""
    rows = conn.execute("SELECT ticker, corp_code FROM dart_corp_codes").fetchall()
    return {r["ticker"]: r["corp_code"] for r in rows}


def upsert_financial_data(conn: sqlite3.Connection, rows: List[dict]) -> int:
    """
    분기 재무제표 데이터 저장.

    Args:
        rows: [{"ticker", "fiscal_year", "fiscal_quarter", "report_code",
                "revenue", "operating_profit", "net_income",
                "operating_margin", "net_margin",
                "revenue_growth_yoy", "op_growth_yoy"}]

    Returns:
        저장된 건수
    """
    if not rows:
        return 0
    now = datetime.now().isoformat()
    with conn:
        conn.executemany("""
            INSERT OR REPLACE INTO stock_financials
            (ticker, fiscal_year, fiscal_quarter, report_code,
             revenue, operating_profit, net_income,
             operating_margin, net_margin,
             revenue_growth_yoy, op_growth_yoy, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(
            r["ticker"], r["fiscal_year"], r["fiscal_quarter"], r["report_code"],
            r.get("revenue"), r.get("operating_profit"), r.get("net_income"),
            r.get("operating_margin"), r.get("net_margin"),
            r.get("revenue_growth_yoy"), r.get("op_growth_yoy"), now,
        ) for r in rows])
    return len(rows)


def get_financial_data(conn: sqlite3.Connection,
                       ticker: str,
                       quarters: int = 8) -> List[dict]:
    """
    특정 종목의 최근 N분기 재무제표 조회 (최신순).

    Returns:
        [{fiscal_year, fiscal_quarter, revenue, operating_profit, net_income,
          operating_margin, net_margin, revenue_growth_yoy, op_growth_yoy}]
    """
    rows = conn.execute("""
        SELECT * FROM stock_financials
        WHERE ticker = ?
        ORDER BY fiscal_year DESC, fiscal_quarter DESC
        LIMIT ?
    """, (ticker, quarters)).fetchall()
    return [dict(r) for r in rows]


def get_latest_financial_summary(conn: sqlite3.Connection,
                                 ticker: str) -> Optional[dict]:
    """최근 1분기 재무 요약 (enrichment용)"""
    row = conn.execute("""
        SELECT * FROM stock_financials
        WHERE ticker = ?
        ORDER BY fiscal_year DESC, fiscal_quarter DESC
        LIMIT 1
    """, (ticker,)).fetchone()
    return dict(row) if row else None
