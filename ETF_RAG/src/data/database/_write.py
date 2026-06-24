"""
DB 쓰기 — collector 출력을 DB에 저장 (ETF + 주식)
"""

import logging
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)


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
        # 1) instruments (ETF는 모두 KOSPI 상장 → market='KOSPI')
        conn.executemany("""
            INSERT INTO instruments (ticker, name, type, market, first_seen, last_seen)
            VALUES (?, ?, 'etf', 'KOSPI', ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                name = excluded.name,
                market = CASE WHEN instruments.market = '' THEN 'KOSPI'
                              ELSE instruments.market END,
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
        # 1) instruments (type='stock', sector·market 포함)
        conn.executemany("""
            INSERT INTO instruments (ticker, name, type, sector, market, first_seen, last_seen)
            VALUES (?, ?, 'stock', ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                name = excluded.name,
                sector = CASE WHEN excluded.sector != '' THEN excluded.sector
                              ELSE instruments.sector END,
                market = CASE WHEN excluded.market != '' THEN excluded.market
                              ELSE instruments.market END,
                last_seen = excluded.last_seen,
                is_active = 1
        """, [(s["ticker"], s["name"], s.get("sector", ""), s.get("market", ""), date, date)
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
