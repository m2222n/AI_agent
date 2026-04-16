"""
DART 재무제표 전종목 점진적 백필 — 매일 9500건씩

전종목(거래대금 무관)을 2015년부터 현재까지 수집.
이미 수집된 건은 자동 스킵(resume). 하루 한도 도달 시 중단.
다음 날 이어서 수집.

사용법:
    python -m scripts.backfill_financials_runner           # 매일 9500건
    python -m scripts.backfill_financials_runner --limit 100  # 테스트
    python -m scripts.backfill_financials_runner --status   # 진행 상황
"""

import sys
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_all_stock_tickers(conn) -> list:
    """DB의 모든 주식 종목 코드 반환 (거래대금 무관)."""
    rows = conn.execute("""
        SELECT DISTINCT i.ticker
        FROM instruments i
        WHERE i.type = 'stock'
        ORDER BY i.ticker
    """).fetchall()
    return [r["ticker"] for r in rows]


def get_all_quarters(start_year: int = 2015, end_year: int = None) -> list:
    """수집 대상 (year, quarter) 목록 생성."""
    from src.data.dart_collector import _get_latest_quarter

    latest_year, latest_q = _get_latest_quarter()
    latest_year = int(latest_year)

    if end_year is not None:
        latest_year = min(latest_year, end_year)

    quarters = []
    for year in range(start_year, latest_year + 1):
        for q in [1, 2, 3, 4]:
            if year == latest_year and q > latest_q:
                break
            quarters.append((str(year), q))
    # 최신 분기부터 역순 — 최신 데이터가 성공률 높고 가치도 높음
    quarters.reverse()
    return quarters


def show_status(conn):
    """현재 백필 진행 상황 출력."""
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM stock_financials")
    total_rows = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT ticker) FROM stock_financials")
    total_tickers = cur.fetchone()[0]

    total_stocks = len(get_all_stock_tickers(conn))

    from src.data.database import get_all_corp_codes
    corp_codes = get_all_corp_codes(conn)

    quarters = get_all_quarters()
    # 매핑 가능한 종목만 카운트
    mappable = sum(1 for t in get_all_stock_tickers(conn) if t in corp_codes)
    total_possible = mappable * len(quarters)

    print(f"=== DART 재무제표 백필 현황 ===")
    print(f"수집 완료: {total_rows}행, {total_tickers}종목")
    print(f"전체 주식: {total_stocks}종목")
    print(f"DART 매핑 가능: {mappable}종목")
    print(f"수집 대상 (종목×분기): {total_possible}건")
    remaining = total_possible - total_rows
    print(f"예상 남은 건수: ~{remaining}건")
    print(f"예상 남은 일수 (9500건/일): ~{remaining // 9500}일")


def run_daily_backfill(conn, daily_limit: int = 9500,
                       start_year: int = 2015, end_year: int = None):
    """하루치 백필 실행 — daily_limit 건 수집 후 중단."""
    from src.data.database import (
        get_all_corp_codes, upsert_financial_data, get_financial_data,
    )
    from src.data.dart_collector import (
        collect_single_financial, _calc_yoy_growth, REPORT_CODES,
    )
    from config import DART_COLLECTION

    request_delay = DART_COLLECTION.get("request_delay", 0.5)
    corp_codes = get_all_corp_codes(conn)
    if not corp_codes:
        logger.error("corp_code 매핑이 없습니다. dart_collector --refresh-codes를 먼저 실행하세요.")
        return 0

    tickers = get_all_stock_tickers(conn)
    quarters = get_all_quarters(start_year=start_year, end_year=end_year)

    # 이미 수집된 (ticker, year, quarter) 셋을 미리 로드 — DB 반복 조회 방지
    existing_set = set()
    for row in conn.execute(
        "SELECT ticker, fiscal_year, fiscal_quarter FROM stock_financials"
    ).fetchall():
        existing_set.add((row[0], row[1], row[2]))
    logger.info(f"이미 수집된 데이터: {len(existing_set)}건")

    logger.info(f"전종목 백필: {len(tickers)}종목 × {len(quarters)}분기, 일일 한도 {daily_limit}건")

    collected = 0
    skipped = 0
    failed = 0
    api_calls = 0  # API 호출 횟수 (스킵 제외)
    consecutive_failures = 0  # 연속 실패 카운터
    MAX_CONSECUTIVE_FAILURES = 200  # 200건 연속 실패 시 API 한도 소진 간주

    for year, quarter in quarters:
        for ticker in tickers:
            if api_calls >= daily_limit:
                logger.info(
                    f"일일 한도 도달 ({daily_limit}건) — "
                    f"수집 {collected}, 스킵 {skipped}, 실패 {failed}"
                )
                return collected

            corp_code = corp_codes.get(ticker)
            if not corp_code:
                skipped += 1
                continue

            # 이미 수집된 데이터 스킵 (메모리 셋 조회 — O(1))
            if (ticker, year, quarter) in existing_set:
                skipped += 1
                continue

            # API 호출
            if api_calls == 0:
                logger.info(f"첫 API 호출 시작: {ticker} {year} Q{quarter}")
            result = collect_single_financial(corp_code, year, quarter, request_delay)
            api_calls += 1

            if result is None:
                failed += 1
                consecutive_failures += 1
                # 연속 실패가 임계치 초과 시 API 한도 소진으로 간주하고 조기 종료
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.warning(
                        f"연속 {MAX_CONSECUTIVE_FAILURES}건 실패 — API 한도 소진 추정, 조기 종료. "
                        f"API {api_calls}, 수집 {collected}, 스킵 {skipped}, 실패 {failed}"
                    )
                    return collected
                continue

            # 성공 시 연속 실패 카운터 리셋
            consecutive_failures = 0

            # YoY 성장률
            growth = _calc_yoy_growth(conn, ticker, year, quarter, result)

            row = {
                "ticker": ticker,
                "fiscal_year": year,
                "fiscal_quarter": quarter,
                "report_code": REPORT_CODES[quarter],
                **result,
                **growth,
            }
            upsert_financial_data(conn, [row])
            existing_set.add((ticker, year, quarter))
            collected += 1

            if api_calls % 50 == 0 or (api_calls <= 50 and api_calls % 10 == 0):
                logger.info(
                    f"  진행: API {api_calls}/{daily_limit} "
                    f"(수집 {collected}, 스킵 {skipped}, 실패 {failed})"
                )

    logger.info(
        f"전종목 백필 완료! 더 이상 수집할 데이터 없음 — "
        f"수집 {collected}, 스킵 {skipped}, 실패 {failed}"
    )
    return collected


def main():
    parser = argparse.ArgumentParser(description="DART 전종목 점진적 백필")
    parser.add_argument("--limit", type=int, default=9500,
                        help="일일 수집 한도 (기본 9500)")
    parser.add_argument("--start-year", type=int, default=2015,
                        help="수집 시작 연도 (기본 2015)")
    parser.add_argument("--end-year", type=int, default=None,
                        help="수집 종료 연도 (기본: 최신)")
    parser.add_argument("--status", action="store_true",
                        help="진행 상황만 출력")
    args = parser.parse_args()

    from src.data.database import init_db
    from src.data.dart_collector import _init_dart

    _init_dart()
    conn = init_db()

    if args.status:
        show_status(conn)
        return

    start = time.time()
    collected = run_daily_backfill(
        conn, daily_limit=args.limit,
        start_year=args.start_year, end_year=args.end_year,
    )
    elapsed = time.time() - start

    logger.info(f"오늘 수집: {collected}건, 소요시간: {elapsed:.0f}초")

    # 진행 상황 출력
    show_status(conn)


if __name__ == "__main__":
    main()
