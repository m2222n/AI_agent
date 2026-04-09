"""
과거 데이터 백필 스크립트 — ETF + 주식 전종목 5년치

pykrx 일괄 API로 영업일마다 전종목 시세를 SQLite에 저장.
보유종목/괴리율/추적오차는 과거 데이터라 의미 없으므로 시세+수익률만 수집.
주식은 시세+시가총액+펀더멘털(PER/PBR)도 포함.

사용법:
    # 5년치 전체 백필 (기본)
    python scripts/backfill_historical.py

    # 기간 지정
    python scripts/backfill_historical.py --start 20210401 --end 20260409

    # ETF만
    python scripts/backfill_historical.py --type etf

    # 주식만
    python scripts/backfill_historical.py --type stock

    # 진행 상황 확인 (이어서 수집)
    python scripts/backfill_historical.py --resume

예상 시간: ETF 5년 ~70분, 주식 5년 ~100분 (네트워크 상태에 따라 다름)
"""

import os
import sys
import time
import logging
import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from pykrx import stock
from src.data.collector import ensure_krx_login, REQUEST_DELAY
from src.data.database import init_db, DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "backfill.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── 영업일 목록 생성 ─────────────────────────────────────────

def get_business_days(start: str, end: str) -> list:
    """pykrx에서 영업일 목록을 가져옴"""
    logger.info(f"영업일 목록 조회: {start} ~ {end}")
    try:
        days = stock.get_previous_business_days(fromdate=start, todate=end)
        result = [d.strftime("%Y%m%d") for d in days]
        logger.info(f"영업일 {len(result)}일")
        return result
    except Exception as e:
        logger.error(f"영업일 조회 실패: {e}")
        return []


def get_already_collected(conn: sqlite3.Connection, inst_type: str) -> set:
    """이미 수집된 날짜 목록 (resume용)"""
    if inst_type == "etf":
        rows = conn.execute("""
            SELECT DISTINCT dp.date FROM daily_prices dp
            JOIN instruments i ON dp.ticker = i.ticker
            WHERE i.type = 'etf'
        """).fetchall()
    else:
        rows = conn.execute("""
            SELECT DISTINCT dp.date FROM daily_prices dp
            JOIN instruments i ON dp.ticker = i.ticker
            WHERE i.type = 'stock'
        """).fetchall()
    return {r[0] for r in rows}


# ── ETF 일별 수집 (시세 + 등락률만, 보유종목/괴리율 제외) ────

def collect_etf_day(conn: sqlite3.Connection, date: str) -> int:
    """ETF 전종목 하루치 시세를 DB에 저장"""
    try:
        # 1) 종목 목록
        tickers = stock.get_etf_ticker_list(date)
        if not tickers:
            return 0

        # 2) 시세/NAV 일괄
        df_ohlcv = stock.get_etf_ohlcv_by_ticker(date)
        time.sleep(REQUEST_DELAY)

        # 3) 등락률 일괄
        try:
            df_change = stock.get_etf_price_change_by_ticker(date, date)
        except Exception:
            df_change = None
        time.sleep(REQUEST_DELAY)

        # DB 저장
        now = datetime.now().isoformat()
        etfs = []
        for ticker in tickers:
            name = stock.get_etf_ticker_name(ticker) or ""
            ohlcv_row = df_ohlcv.loc[ticker] if ticker in df_ohlcv.index else None
            change_row = df_change.loc[ticker] if (
                df_change is not None and ticker in df_change.index
            ) else None

            ohlcv = {}
            if ohlcv_row is not None:
                ohlcv = {
                    "open": int(ohlcv_row.get("시가", 0)),
                    "high": int(ohlcv_row.get("고가", 0)),
                    "low": int(ohlcv_row.get("저가", 0)),
                    "close": int(ohlcv_row.get("종가", 0)),
                    "volume": int(ohlcv_row.get("거래량", 0)),
                    "trade_value": int(ohlcv_row.get("거래대금", 0)),
                    "nav": round(float(ohlcv_row.get("NAV", 0)), 2),
                    "base_index": round(float(ohlcv_row.get("기초지수", 0)), 2),
                }

            if change_row is not None:
                ohlcv["change"] = int(change_row.get("변동폭", 0))
                ohlcv["change_pct"] = round(float(change_row.get("등락률", 0)), 2)

            etfs.append({
                "ticker": ticker, "name": name, "date": date, "ohlcv": ohlcv,
            })

        # upsert_daily_data 호환 포맷으로 저장
        data = {
            "metadata": {
                "collection_date": date,
                "collected_at": now,
                "total_etfs": len(etfs),
                "holdings_collected": 0,
            },
            "etfs": etfs,
        }

        from src.data.database import upsert_daily_data
        count = upsert_daily_data(conn, data)
        return count

    except Exception as e:
        logger.warning(f"ETF {date} 수집 실패: {e}")
        return 0


# ── 주식 일별 수집 (시세 + 시가총액 + 펀더멘털) ──────────────

def collect_stock_day(conn: sqlite3.Connection, date: str) -> int:
    """주식 전종목 하루치 시세를 DB에 저장"""
    try:
        # 1) 시세 일괄 (KOSPI + KOSDAQ)
        df_ohlcv = stock.get_market_ohlcv_by_ticker(date, market="ALL")
        if df_ohlcv.empty:
            return 0
        time.sleep(REQUEST_DELAY)

        # 2) 시가총액 일괄
        df_cap = stock.get_market_cap_by_ticker(date, market="ALL")
        time.sleep(REQUEST_DELAY)

        # 3) 펀더멘털 (KOSPI + KOSDAQ 개별)
        fund = {}
        for mkt in ["KOSPI", "KOSDAQ"]:
            try:
                df_fund = stock.get_market_fundamental_by_ticker(date, market=mkt)
                for ticker, row in df_fund.iterrows():
                    fund[ticker] = {
                        "bps": round(float(row.get("BPS", 0)), 2),
                        "per": round(float(row.get("PER", 0)), 2),
                        "pbr": round(float(row.get("PBR", 0)), 2),
                        "eps": round(float(row.get("EPS", 0)), 2),
                        "div": round(float(row.get("DIV", 0)), 2),
                        "dps": round(float(row.get("DPS", 0)), 2),
                    }
            except Exception:
                pass
        time.sleep(REQUEST_DELAY)

        # DB 저장
        now = datetime.now().isoformat()
        stocks = []
        for ticker, row in df_ohlcv.iterrows():
            name = stock.get_market_ticker_name(ticker) or ""
            cap_row = df_cap.loc[ticker] if ticker in df_cap.index else None

            s = {
                "ticker": ticker,
                "name": name,
                "date": date,
                "ohlcv": {
                    "open": int(row.get("시가", 0)),
                    "high": int(row.get("고가", 0)),
                    "low": int(row.get("저가", 0)),
                    "close": int(row.get("종가", 0)),
                    "volume": int(row.get("거래량", 0)),
                    "trade_value": int(row.get("거래대금", 0)),
                    "change_pct": round(float(row.get("등락률", 0)), 2),
                },
                "market_cap": int(cap_row.get("시가총액", 0)) if cap_row is not None else 0,
                "shares_outstanding": int(cap_row.get("상장주식수", 0)) if cap_row is not None else 0,
                "fundamental": fund.get(ticker, {}),
                "returns": {},  # 백필에서는 수익률 스킵 (일별 시세로 나중에 계산 가능)
            }
            stocks.append(s)

        data = {
            "metadata": {
                "collection_date": date,
                "collected_at": now,
                "total_stocks": len(stocks),
            },
            "stocks": stocks,
        }

        from src.data.database import upsert_stock_data
        count = upsert_stock_data(conn, data)
        return count

    except Exception as e:
        logger.warning(f"주식 {date} 수집 실패: {e}")
        return 0


# ── 메인 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="과거 데이터 백필 (ETF + 주식 전종목)")
    parser.add_argument("--start", default=None, help="시작일 (YYYYMMDD, 기본: 5년 전)")
    parser.add_argument("--end", default=None, help="종료일 (YYYYMMDD, 기본: 어제)")
    parser.add_argument("--type", choices=["etf", "stock", "all"], default="all",
                        help="수집 대상 (기본: all)")
    parser.add_argument("--resume", action="store_true",
                        help="이미 수집된 날짜 건너뛰기 (이어서 수집)")
    args = parser.parse_args()

    # 날짜 범위
    if args.end:
        end_date = args.end
    else:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    if args.start:
        start_date = args.start
    else:
        start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y%m%d")

    logger.info(f"=== 백필 시작: {start_date} ~ {end_date}, 대상: {args.type} ===")

    # KRX 로그인
    ensure_krx_login()

    # 영업일 목록
    business_days = get_business_days(start_date, end_date)
    if not business_days:
        logger.error("영업일을 찾을 수 없습니다.")
        return

    # DB 초기화
    conn = init_db()

    # resume 모드: 이미 수집된 날짜 제외
    if args.resume:
        if args.type in ("etf", "all"):
            etf_done = get_already_collected(conn, "etf")
            logger.info(f"ETF 이미 수집된 날짜: {len(etf_done)}일")
        if args.type in ("stock", "all"):
            stock_done = get_already_collected(conn, "stock")
            logger.info(f"주식 이미 수집된 날짜: {len(stock_done)}일")
    else:
        etf_done = set()
        stock_done = set()

    total_days = len(business_days)
    etf_total = 0
    stock_total = 0
    failed_days = []

    for i, date in enumerate(business_days, 1):
        progress = f"[{i}/{total_days}] {date}"

        # ETF
        if args.type in ("etf", "all"):
            if date in etf_done:
                logger.debug(f"{progress} ETF 스킵 (이미 수집)")
            else:
                count = collect_etf_day(conn, date)
                if count > 0:
                    etf_total += count
                    logger.info(f"{progress} ETF {count}종목 저장")
                else:
                    failed_days.append(("etf", date))
                    logger.warning(f"{progress} ETF 수집 실패")

        # 주식
        if args.type in ("stock", "all"):
            if date in stock_done:
                logger.debug(f"{progress} 주식 스킵 (이미 수집)")
            else:
                count = collect_stock_day(conn, date)
                if count > 0:
                    stock_total += count
                    logger.info(f"{progress} 주식 {count}종목 저장")
                else:
                    failed_days.append(("stock", date))
                    logger.warning(f"{progress} 주식 수집 실패")

        # 10일마다 진행 상황 요약
        if i % 10 == 0:
            logger.info(f"--- 진행: {i}/{total_days}일 완료 "
                        f"(ETF 누적 {etf_total:,}, 주식 누적 {stock_total:,}) ---")

    # 완료 통계
    conn.close()
    logger.info("=" * 60)
    logger.info(f"백필 완료!")
    logger.info(f"  기간: {start_date} ~ {end_date} ({total_days}영업일)")
    logger.info(f"  ETF: 총 {etf_total:,}건 저장")
    logger.info(f"  주식: 총 {stock_total:,}건 저장")
    if failed_days:
        logger.warning(f"  실패: {len(failed_days)}건")
        for ftype, fdate in failed_days[:10]:
            logger.warning(f"    - {ftype} {fdate}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
