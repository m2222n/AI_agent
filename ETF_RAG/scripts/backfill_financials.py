"""
재무제표 전체 백필 스크립트 — 2015~2025 전 분기, 전종목

DART API 일일 한도(10,000건)를 감안하여 request_delay를 최소화하고,
이미 수집된 건은 자동 스킵합니다.

사용법:
    python scripts/backfill_financials.py
    python scripts/backfill_financials.py --start 2020 --end 2025
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "backfill_financials.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="재무제표 전체 백필 (2015~2025)")
    parser.add_argument("--start", type=int, default=2015, help="시작 연도")
    parser.add_argument("--end", type=int, default=2025, help="종료 연도")
    parser.add_argument("--max", type=int, default=3000, help="분기당 최대 종목 수")
    parser.add_argument("--delay", type=float, default=0.2, help="API 요청 간격 (초)")
    args = parser.parse_args()

    from src.data.dart_collector import (
        _init_dart, collect_single_financial, _calc_yoy_growth,
        REPORT_CODES, _get_latest_quarter,
    )
    from src.data.database import (
        init_db, get_all_corp_codes, upsert_financial_data,
        get_financial_data,
    )

    _init_dart()
    conn = init_db()

    # corp_code 매핑
    corp_codes = get_all_corp_codes(conn)
    if not corp_codes:
        logger.error("corp_code 매핑이 없습니다. --refresh-codes를 먼저 실행하세요.")
        return

    # 대상 종목: 거래대금 상위
    rows = conn.execute("""
        SELECT p.ticker FROM daily_prices p
        JOIN instruments i ON p.ticker = i.ticker
        WHERE i.type = 'stock' AND p.trade_value >= 1000000000
        AND p.date = (SELECT MAX(date) FROM daily_prices
                      WHERE ticker = p.ticker)
        GROUP BY p.ticker
        ORDER BY MAX(p.trade_value) DESC
        LIMIT ?
    """, (args.max,)).fetchall()
    tickers = [r["ticker"] for r in rows]
    logger.info(f"대상 종목: {len(tickers)}개")

    # 최근 발표 분기
    latest_year, latest_q = _get_latest_quarter()
    logger.info(f"최근 발표 분기: {latest_year}Q{latest_q}")

    total_collected = 0
    total_skipped = 0
    total_failed = 0
    api_calls = 0
    start_time = time.time()

    for year in range(args.start, args.end + 1):
        for quarter in [1, 2, 3, 4]:
            # 미래 분기 스킵
            if year > int(latest_year):
                continue
            if year == int(latest_year) and quarter > latest_q:
                continue

            logger.info(f"=== {year}Q{quarter} 시작 ({len(tickers)}종목) ===")
            collected = 0
            skipped = 0
            failed = 0

            for i, ticker in enumerate(tickers):
                corp_code = corp_codes.get(ticker)
                if not corp_code:
                    skipped += 1
                    continue

                # 이미 수집된 데이터 스킵
                existing = get_financial_data(conn, ticker, quarters=50)
                already_exists = any(
                    d["fiscal_year"] == str(year) and d["fiscal_quarter"] == quarter
                    for d in existing
                )
                if already_exists:
                    skipped += 1
                    continue

                # API 일일 한도 체크 (안전 마진 포함)
                if api_calls >= 9500:
                    elapsed = time.time() - start_time
                    logger.warning(f"API 일일 한도 근접 ({api_calls}건, {elapsed/3600:.1f}시간 경과). 중단.")
                    logger.info(f"여기까지 수집: {total_collected}건, 스킵: {total_skipped}건, 실패: {total_failed}건")
                    conn.close()
                    return

                result = collect_single_financial(corp_code, str(year), quarter, args.delay)
                api_calls += 1

                if result is None:
                    failed += 1
                    continue

                # YoY 성장률
                growth = _calc_yoy_growth(conn, ticker, str(year), quarter, result)

                row = {
                    "ticker": ticker,
                    "fiscal_year": str(year),
                    "fiscal_quarter": quarter,
                    "report_code": REPORT_CODES[quarter],
                    **result,
                    **growth,
                }
                upsert_financial_data(conn, [row])
                collected += 1

                if (i + 1) % 100 == 0:
                    logger.info(f"  {year}Q{quarter} 진행: {i+1}/{len(tickers)} "
                               f"(수집 {collected}, 스킵 {skipped}, 실패 {failed}, "
                               f"API {api_calls}건)")

            total_collected += collected
            total_skipped += skipped
            total_failed += failed

            logger.info(f"=== {year}Q{quarter} 완료: 수집 {collected}, 스킵 {skipped}, "
                       f"실패 {failed} (누적 API {api_calls}건) ===")

    elapsed = time.time() - start_time
    conn.close()
    logger.info("=" * 60)
    logger.info(f"백필 완료!")
    logger.info(f"  기간: {args.start}~{args.end}")
    logger.info(f"  수집: {total_collected}건, 스킵: {total_skipped}건, 실패: {total_failed}건")
    logger.info(f"  API 호출: {api_calls}건")
    logger.info(f"  소요시간: {elapsed/60:.1f}분")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
