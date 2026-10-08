"""
수집 데이터 검증 + 누락 자동 보충 스크립트

최근 N영업일 중 DB에 누락된 날짜를 감지하고 자동으로 재수집합니다.
daily_collect.sh 끝에서 호출되어 수집 실패 시 자동 복구합니다.

사용법:
    python scripts/verify_and_recover.py          # 최근 5영업일 검증+보충
    python scripts/verify_and_recover.py --days 10 # 최근 10영업일
    python scripts/verify_and_recover.py --check   # 검증만 (보충 안 함)
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

# 영업일 계산은 KST 기준 — GitHub Actions 러너(UTC)에서도 한국 날짜로 판단해야 한다.
KST = timezone(timedelta(hours=9))

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
    now = datetime.now(KST)
    dt = now

    # 오늘이 주말이면 금요일부터 시작
    while _is_weekend(dt):
        dt -= timedelta(days=1)

    # 오늘 장마감 전(16:00 KST)이면 어제부터
    if now.hour < 16:
        dt -= timedelta(days=1)

    while len(result) < n_days:
        if _is_weekend(dt):
            dt -= timedelta(days=1)
            continue
        result.append(dt.strftime("%Y%m%d"))
        dt -= timedelta(days=1)

    return sorted(result)


# 정상 영업일 하한 — ETF/주식을 각각 검사(한쪽만 빠진 날 감지). 휴장일은 둘 다 0.
_MIN_ETF = 500
_MIN_STOCK = 1500

# 유령 행 정리 대상 — 날짜 컬럼을 가진 수집 테이블. (returns는 백필이 쓰지 않지만
# 같은 날짜가 남아 있으면 함께 지운다.)
_DATED_TABLES = ("daily_prices", "stock_fundamentals", "returns", "collection_log")


def purge_date(conn, date: str) -> int:
    """특정 날짜의 수집 행을 전 테이블에서 삭제. 삭제된 총 행 수 반환.

    휴장일에 KRX가 돌려준 전-0 프레임이 저장된 '유령 행'을 제거할 때 쓴다.
    테이블이 없거나 date 컬럼이 없으면 그 테이블은 건너뛴다(테스트 인메모리 DB 호환).
    """
    total = 0
    for table in _DATED_TABLES:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        if "date" not in cols:
            continue
        cur = conn.execute(f"DELETE FROM {table} WHERE date = ?", (date,))
        total += cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    return total


def find_missing_dates(conn, expected_dates: list[str]) -> list[str]:
    """DB에서 누락된 영업일 찾기.

    ETF/주식을 분리 검사 — 한쪽만 빠진 날(예: 주식 수집 실패로 ETF만 존재)도
    감지한다. 둘 다 0이면 휴장일로 보고 정상 처리(recover에서 0건=휴장 확인).

    유령 행(행은 있는데 종가가 전부 0 — 휴장일 응답이 저장된 것)은 그 자리에서
    삭제하고 '데이터 없음'으로 재분류한다. 그래야 recover가 다시 수집을 시도하고,
    진짜 휴장일이면 0건으로 끝나 깨끗한 상태가 유지된다.
    """
    missing = []
    for date in expected_dates:
        row = conn.execute(
            """
            SELECT
              SUM(CASE WHEN i.type='etf' THEN 1 ELSE 0 END) etf_n,
              SUM(CASE WHEN i.type='stock' THEN 1 ELSE 0 END) stock_n,
              SUM(CASE WHEN p.close > 0 THEN 1 ELSE 0 END) priced_n
            FROM daily_prices p JOIN instruments i ON p.ticker = i.ticker
            WHERE p.date = ?
            """,
            (date,),
        ).fetchone()
        etf_n = row[0] or 0
        stock_n = row[1] or 0
        priced_n = row[2] or 0

        if (etf_n + stock_n) > 0 and priced_n == 0:
            # 행은 있으나 종가가 전부 0 → 휴장일 유령 행. 삭제 후 재수집 대상으로.
            removed = purge_date(conn, date)
            logger.warning(
                f"유령 행 감지: {date} — {etf_n + stock_n}종목 전부 종가 0 "
                f"(휴장일 응답 저장분) → {removed}행 삭제 후 재검사 대상"
            )
            etf_n = stock_n = 0

        if etf_n == 0 and stock_n == 0:
            # 둘 다 없음 → 휴장일 후보. recover의 수집 시도(0건=휴장)로 자연 확인.
            missing.append((date, 0))
            logger.warning(f"누락 감지: {date} — 데이터 없음(휴장일 가능)")
        elif etf_n < _MIN_ETF or stock_n < _MIN_STOCK:
            # 한쪽만 부족 → 부분 누락(수집 중단 등). 보충 대상.
            missing.append((date, etf_n + stock_n))
            logger.warning(
                f"부분 누락 감지: {date} — ETF {etf_n}/주식 {stock_n} "
                f"(하한 ETF {_MIN_ETF}/주식 {_MIN_STOCK})"
            )
        else:
            logger.info(f"정상: {date} — ETF {etf_n}/주식 {stock_n}")
    return missing


def recover_missing(conn, missing_dates: list[tuple]) -> dict:
    """누락된 날짜 데이터를 재수집.

    collect_*_day 반환 규약(backfill_historical 참조):
      >0 저장 / 0 휴장일(KRX 데이터 없음) / None 수집 실패(예외).
    - 한쪽이라도 None이면 **실패**로 분류한다. 예전엔 실패도 0을 돌려줘 로그인
      실패·API 오류가 "공휴일"로 둔갑해 ✅로 끝났다(2026-10-08 실제 발생).
    - 둘 다 0이고 기존 행도 0이면 진짜 휴장일.
    """
    from scripts.backfill_historical import collect_etf_day, collect_stock_day

    results = {"recovered": [], "failed": [], "holidays": []}

    for date, existing_count in missing_dates:
        logger.info(f"보충 수집 시작: {date} (기존 {existing_count}종목)")

        etf_count = collect_etf_day(conn, date)
        stock_count = collect_stock_day(conn, date)

        if etf_count is None or stock_count is None:
            logger.error(
                f"보충 실패: {date} — 수집 예외 "
                f"(ETF {'실패' if etf_count is None else etf_count} / "
                f"주식 {'실패' if stock_count is None else stock_count})"
            )
            results["failed"].append(date)
        elif etf_count + stock_count > 0:
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
    # 자격증명이 아예 없을 때만 로그인 없이 진행한다. 자격증명이 있는데 로그인이
    # 실패하면 중단 — 예전엔 둘 다 RuntimeError로 뭉뚱그려 "정보 없음"으로 넘겼고,
    # 그 결과 미로그인 상태의 빈 응답이 전부 "공휴일"로 처리돼 ✅로 끝났다.
    from src.data.collector import ensure_krx_login
    if not os.environ.get("KRX_ID") or not os.environ.get("KRX_PW"):
        logger.warning("KRX 로그인 정보 없음 — 로그인 없이 시도")
    else:
        try:
            ensure_krx_login()
        except RuntimeError as e:
            logger.error(f"KRX 로그인 실패 — 보충 중단: {e}")
            conn.close()
            sys.exit(1)

    results = recover_missing(conn, missing)
    conn.close()

    # 4. 결과 리포트
    if results.get("holidays"):
        logger.info(f"공휴일 (스킵): {results['holidays']}")
    if results["recovered"]:
        logger.info(f"보충 성공: {results['recovered']}")
    if results["failed"]:
        logger.error(f"보충 실패: {results['failed']}")
        # macOS 알림 — osascript가 없는 환경(GitHub Actions/Linux)에선 건너뛴다.
        import shutil
        import subprocess
        if shutil.which("osascript"):
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
