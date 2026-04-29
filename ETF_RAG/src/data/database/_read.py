"""
DB 읽기 — loader.py 호환 데이터 조회
"""

import sqlite3
from typing import List, Optional


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
