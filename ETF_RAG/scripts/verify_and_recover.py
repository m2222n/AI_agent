"""
수집 데이터 검증 + 누락 자동 보충 스크립트

최근 N영업일 중 DB에 누락된 날짜를 감지하고 자동으로 재수집합니다.
daily_collect.sh 끝에서 호출되어 수집 실패 시 자동 복구합니다.

사용법:
    python scripts/verify_and_recover.py          # 최근 5영업일 검증+보충
    python scripts/verify_and_recover.py --days 10 # 최근 10영업일
    python scripts/verify_and_recover.py --check   # 검증만 (보충 안 함)
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5


def get_expected_business_days(n_days: int = 10) -> list[str]:
    """오늘로부터 과거 N영업일 목록 생성.

    주말은 건너뛰지만 공휴일은 하드코딩하지 않음.
    공휴일은 find_missing_dates()에서 DB 데이터 부재 + 수집 시도 실패로 자연 감지.
    """
    result = []
    dt = datetime.now()

    # 오늘이 주말이면 금요일부터 시작
    while _is_weekend(dt):
        dt -= timedelta(days=1)

    # 오늘 장마감 전(16:00)이면 어제부터
    if datetime.now().hour < 16:
        dt -= timedelta(days=1)

    while len(result) < n_days:
        if _is_weekend(dt):
            dt -= timedelta(days=1)
            continue
        result.append(dt.strftime("%Y%m%d"))
        dt -= timedelta(days=1)

    return sorted(result)


def find_missing_dates(conn, expected_dates: list[str]) -> list[str]:
    """DB에서 누락된 영업일 찾기."""
    missing = []
    for date in expected_dates:
        cur = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM daily_prices WHERE date = ?",
            (date,),
        )
        count = cur.fetchone()[0]
        if count < 500:  # 최소 500종목은 있어야 정상
            missing.append((date, count))
            logger.warning(f"누락 감지: {date} — {count}종목 (최소 500 필요)")
        else:
            logger.info(f"정상: {date} — {count}종목")
    return missing


def recover_missing(conn, missing_dates: list[tuple]) -> dict:
    """누락된 날짜 데이터를 재수집.

    수집 시도 후 0건이면 공휴일로 판단 (실패가 아님).
    """
    from scripts.backfill_historical import collect_etf_day, collect_stock_day

    results = {"recovered": [], "failed": [], "holidays": []}

    for date, existing_count in missing_dates:
        logger.info(f"보충 수집 시작: {date} (기존 {existing_count}종목)")

        etf_count = collect_etf_day(conn, date)
        stock_count = collect_stock_day(conn, date)

        if etf_count + stock_count > 0:
            logger.info(f"보충 완료: {date} — ETF {etf_count} + 주식 {stock_count}")
            results["recovered"].append(date)
        elif existing_count == 0:
            # 기존 0건 + 수집 0건 = 공휴일 (KRX 데이터 자체가 없음)
            logger.info(f"공휴일 감지: {date} — KRX 데이터 없음 (정상)")
            results["holidays"].append(date)
        else:
            logger.error(f"보충 실패: {date}")
            results["failed"].append(date)

    return results


def main():
    parser = argparse.ArgumentParser(description="수집 데이터 검증 + 누락 보충")
    parser.add_argument("--days", type=int, default=10, help="검증할 최근 영업일 수")
    parser.add_argument("--check", action="store_true", help="검증만 (보충 안 함)")
    args = parser.parse_args()

    from src.data.database import init_db
    conn = init_db()

    # 1. 기대 영업일 생성
    expected = get_expected_business_days(args.days)
    logger.info(f"검증 대상: {expected}")

    # 2. 누락 감지
    missing = find_missing_dates(conn, expected)

    if not missing:
        logger.info(f"최근 {args.days}영업일 데이터 정상 ✅")
        conn.close()
        return

    logger.warning(f"누락 {len(missing)}일 감지: {[m[0] for m in missing]}")

    if args.check:
        logger.info("--check 모드: 보충 생략")
        conn.close()
        sys.exit(1)  # 누락 있으면 exit 1 (알림용)

    # 3. KRX 로그인 + 자동 보충
    from src.data.collector import ensure_krx_login
    try:
        ensure_krx_login()
    except RuntimeError:
        logger.warning("KRX 로그인 정보 없음 — 로그인 없이 시도")

    results = recover_missing(conn, missing)
    conn.close()

    # 4. 결과 리포트
    if results.get("holidays"):
        logger.info(f"공휴일 (스킵): {results['holidays']}")
    if results["recovered"]:
        logger.info(f"보충 성공: {results['recovered']}")
    if results["failed"]:
        logger.error(f"보충 실패: {results['failed']}")
        # macOS 알림
        import subprocess
        failed_str = ", ".join(results["failed"])
        subprocess.run([
            "osascript", "-e",
            f'display notification "데이터 보충 실패: {failed_str}" '
            f'with title "ETF RAG 수집 오류"',
        ], capture_output=True)
        sys.exit(1)

    logger.info("검증+보충 완료 ✅")


if __name__ == "__main__":
    main()
