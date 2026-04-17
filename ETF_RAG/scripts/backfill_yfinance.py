"""
yfinance 기반 과거 데이터 백필 — KRX 슬라이딩 윈도우 누락분 수집

pykrx(KRX)는 현재 시점 기준 ~12년치만 제공하므로,
그 이전 데이터는 yfinance로 개별 종목 수집.
기본 대상: 2014-01-01 ~ 2014-04-17 (pykrx 미제공 구간)

사용법:
    python scripts/backfill_yfinance.py                          # 기본 (2014-01-01 ~ 2014-04-17)
    python scripts/backfill_yfinance.py --start 20140101 --end 20140301  # 기간 지정
    python scripts/backfill_yfinance.py --type stock             # 주식만
    python scripts/backfill_yfinance.py --type etf               # ETF만

제한사항:
    - trade_value(거래대금), nav, base_index는 yfinance 미제공 → NULL
    - 수정주가(Adj Close) 기준이라 pykrx 데이터와 미세 차이 가능
    - 개별 종목 조회라 전종목 수집 시 시간 소요 (~1-2시간)
"""

import sys
import time
import logging
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "backfill_yfinance.log"),
    ],
)
logger = logging.getLogger(__name__)


def get_tickers_from_db(conn: sqlite3.Connection, inst_type: str) -> list:
    """DB에 등록된 종목 코드+이름 반환."""
    rows = conn.execute(
        "SELECT ticker, name FROM instruments WHERE type = ? ORDER BY ticker",
        (inst_type,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_already_collected_dates(conn: sqlite3.Connection, ticker: str) -> set:
    """특정 종목의 이미 수집된 날짜 셋."""
    rows = conn.execute(
        "SELECT date FROM daily_prices WHERE ticker = ?", (ticker,)
    ).fetchall()
    return {r[0] for r in rows}


def collect_ticker_yfinance(
    conn: sqlite3.Connection,
    ticker: str,
    name: str,
    inst_type: str,
    start: str,
    end: str,
) -> int:
    """yfinance로 단일 종목 OHLCV 수집 → DB 저장. 반환: 저장 건수."""
    import yfinance as yf

    import pandas as pd

    # KRX → yfinance 티커 변환
    start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:]}"

    if inst_type == "etf":
        yf_ticker = f"{ticker}.KS"
    else:
        # 주식: KS 먼저, KQ fallback
        yf_ticker = None
        for suffix in ("KS", "KQ"):
            candidate = f"{ticker}.{suffix}"
            try:
                df_test = yf.download(
                    candidate, start=start_fmt, end=end_fmt,
                    progress=False, timeout=10, auto_adjust=False,
                )
                if not df_test.empty:
                    yf_ticker = candidate
                    break
            except Exception:
                continue
        if yf_ticker is None:
            return 0

    # 다운로드 (auto_adjust=False → 원시 가격, pykrx와 호환)
    try:
        df = yf.download(
            yf_ticker, start=start_fmt, end=end_fmt,
            progress=False, timeout=30, auto_adjust=False,
        )
    except Exception as e:
        logger.warning(f"{ticker} ({yf_ticker}) 다운로드 실패: {e}")
        return 0

    if df.empty:
        return 0

    # MultiIndex 처리 (yfinance 0.2.x에서 단일 종목도 MultiIndex 반환)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 이미 수집된 날짜 제외
    existing_dates = get_already_collected_dates(conn, ticker)

    saved = 0
    for date_idx, row in df.iterrows():
        date_str = date_idx.strftime("%Y%m%d")
        if date_str in existing_dates:
            continue

        close_val = int(round(float(row.get("Close", 0))))
        if close_val <= 0:
            continue

        conn.execute(
            """INSERT OR IGNORE INTO daily_prices
               (ticker, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker,
                date_str,
                int(round(float(row.get("Open", 0)))),
                int(round(float(row.get("High", 0)))),
                int(round(float(row.get("Low", 0)))),
                close_val,
                int(float(row.get("Volume", 0))),
            ),
        )
        saved += 1

    if saved > 0:
        conn.commit()

    return saved


def main():
    parser = argparse.ArgumentParser(description="yfinance 과거 데이터 백필 (KRX 미제공 구간)")
    parser.add_argument("--start", default="20140101", help="시작일 YYYYMMDD (기본: 20140101)")
    parser.add_argument("--end", default="20140418", help="종료일 YYYYMMDD (기본: 20140418, 미포함)")
    parser.add_argument("--type", choices=["etf", "stock", "all"], default="all",
                        help="수집 대상 (기본: all)")
    args = parser.parse_args()

    logger.info(f"=== yfinance 백필 시작: {args.start} ~ {args.end}, 대상: {args.type} ===")

    from src.data.database import init_db
    conn = init_db()

    total_saved = 0
    total_failed = 0
    total_skipped = 0

    for inst_type in (["etf", "stock"] if args.type == "all" else [args.type]):
        tickers = get_tickers_from_db(conn, inst_type)
        logger.info(f"{inst_type.upper()} {len(tickers)}종목 수집 시작")

        for i, (ticker, name) in enumerate(tickers, 1):
            try:
                saved = collect_ticker_yfinance(
                    conn, ticker, name, inst_type, args.start, args.end
                )
                if saved > 0:
                    total_saved += saved
                    if i % 100 == 0 or i <= 10:
                        logger.info(f"  [{i}/{len(tickers)}] {ticker} {name}: {saved}일 저장")
                else:
                    total_skipped += 1

                # rate limit 방지
                if i % 50 == 0:
                    time.sleep(1)

            except Exception as e:
                total_failed += 1
                logger.warning(f"  [{i}/{len(tickers)}] {ticker} 실패: {e}")

            if i % 200 == 0:
                logger.info(
                    f"  --- 진행: {i}/{len(tickers)} "
                    f"(저장 {total_saved}, 스킵 {total_skipped}, 실패 {total_failed}) ---"
                )

    conn.close()
    logger.info("=" * 60)
    logger.info(f"yfinance 백필 완료!")
    logger.info(f"  기간: {args.start} ~ {args.end}")
    logger.info(f"  저장: {total_saved}건")
    logger.info(f"  스킵: {total_skipped}종목 (데이터 없음)")
    logger.info(f"  실패: {total_failed}종목")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
