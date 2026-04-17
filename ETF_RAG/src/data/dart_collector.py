"""
OpenDart 재무제표 수집기 — dart-fss 기반

분기별 재무제표(매출액, 영업이익, 당기순이익)를 OpenDart API로 수집하여
SQLite DB에 저장. 거래대금 상위 종목만 수집 (rate limit 고려).

사용법:
    python -m src.data.dart_collector                    # 최근 분기
    python -m src.data.dart_collector --refresh-codes    # corp_code 목록 갱신
    python -m src.data.dart_collector --backfill         # 3년 백필
    python -m src.data.dart_collector --test             # 10종목 테스트

환경변수 (.env):
    DART_API_KEY: OpenDart API 키 (https://opendart.fss.or.kr)
"""

import logging
import argparse
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# 보고서 코드 → 분기 매핑
REPORT_CODES = {
    1: "11013",  # Q1 (1분기보고서)
    2: "11012",  # H1 (반기보고서) — 누적치, Q1 빼야 Q2 단독
    3: "11014",  # Q3 (3분기보고서)
    4: "11011",  # Annual (사업보고서)
}

# 재무제표에서 추출할 계정명
ACCOUNT_NAMES = {
    "revenue": ["매출액", "수익(매출액)", "영업수익"],
    "operating_profit": ["영업이익", "영업이익(손실)"],
    "net_income": ["당기순이익", "당기순이익(손실)", "분기순이익"],
}


def _init_dart():
    """dart-fss 초기화 (API 키 설정 + import)"""
    import dart_fss as dart
    from config import DART_API_KEY

    if not DART_API_KEY:
        raise ValueError("DART_API_KEY 환경변수가 필요합니다.")

    dart.set_api_key(DART_API_KEY)
    return dart


def refresh_corp_codes(conn) -> int:
    """
    DART corp_code 전체 목록 다운로드 → DB 저장.
    KOSPI/KOSDAQ 상장 종목만 필터링.
    """
    from src.data.database import upsert_corp_codes

    dart = _init_dart()
    logger.info("DART corp_code 목록 다운로드 중...")

    corp_list = dart.get_corp_list()
    codes = []

    for corp in corp_list.corps:
        stock_code = getattr(corp, "stock_code", "") or ""
        if len(stock_code) == 6 and stock_code.isdigit():
            codes.append({
                "corp_code": corp.corp_code,
                "ticker": stock_code,
                "corp_name": corp.corp_name,
            })

    count = upsert_corp_codes(conn, codes)
    logger.info(f"corp_code {count}건 저장 완료")
    return count


def _extract_account_value(accounts: list, target_names: list) -> Optional[int]:
    """계정 목록에서 특정 계정의 금액 추출 (CFS 우선)"""
    # CFS(연결재무제표) 항목에서 먼저 검색
    for acc in accounts:
        acc_nm = acc.get("account_nm", "") or acc.get("acc_nm", "")
        fs_div = acc.get("fs_div", "")
        if acc_nm in target_names and fs_div == "CFS":
            amount_str = acc.get("thstrm_amount", "")
            if amount_str and amount_str != "-":
                try:
                    return int(amount_str.replace(",", ""))
                except (ValueError, TypeError):
                    pass
    # OFS(별도재무제표) fallback
    for acc in accounts:
        acc_nm = acc.get("account_nm", "") or acc.get("acc_nm", "")
        if acc_nm in target_names:
            amount_str = acc.get("thstrm_amount", "")
            if amount_str and amount_str != "-":
                try:
                    return int(amount_str.replace(",", ""))
                except (ValueError, TypeError):
                    pass
    return None


# collect_single_financial 반환값 구분용 센티넬
NO_DATA = "NO_DATA"  # 정상적으로 데이터가 없음 (해당 분기 공시 없음)


def collect_single_financial(corp_code: str, year: str, quarter: int,
                             request_delay: float = 0.5):
    """
    단일 기업의 특정 분기 재무제표 수집.

    Returns:
        dict: 성공 시 {revenue, operating_profit, ...}
        "NO_DATA": 정상적으로 데이터 없음 (해당 분기 공시 없음)
        None: API 오류 (rate limit, 네트워크 등)
    """
    from dart_fss.api.finance import fnltt_singl_acnt

    report_code = REPORT_CODES.get(quarter)
    if not report_code:
        return NO_DATA

    time.sleep(request_delay)

    try:
        # fnltt_singl_acnt는 CFS+OFS 모두 반환, fs_div 필드로 구분
        result = fnltt_singl_acnt(
            corp_code=corp_code,
            bsns_year=year,
            reprt_code=report_code,
        )
    except Exception as e:
        error_str = str(e).lower()
        # rate limit / 인증 오류는 진짜 API 에러
        if "limited" in error_str or "429" in error_str or "unauthorized" in error_str:
            logger.warning(f"API 오류: {corp_code} {year}Q{quarter} — {e}")
            return None
        # 그 외(데이터 없음 등)는 정상 — 013 에러코드 = "조회된 데이터가 없습니다"
        logger.debug(f"재무제표 없음: {corp_code} {year}Q{quarter} — {e}")
        return NO_DATA

    accounts = result.get("list", [])
    if not accounts:
        return NO_DATA

    revenue = _extract_account_value(accounts, ACCOUNT_NAMES["revenue"])
    op_profit = _extract_account_value(accounts, ACCOUNT_NAMES["operating_profit"])
    net_income = _extract_account_value(accounts, ACCOUNT_NAMES["net_income"])

    # 마진율 계산
    op_margin = None
    net_margin = None
    if revenue and revenue != 0:
        if op_profit is not None:
            op_margin = round(op_profit / revenue * 100, 2)
        if net_income is not None:
            net_margin = round(net_income / revenue * 100, 2)

    return {
        "revenue": revenue,
        "operating_profit": op_profit,
        "net_income": net_income,
        "operating_margin": op_margin,
        "net_margin": net_margin,
    }


def _calc_yoy_growth(conn, ticker: str, year: str, quarter: int,
                     current: dict) -> dict:
    """전년동기 대비 성장률 계산"""
    from src.data.database import get_financial_data

    prev_year = str(int(year) - 1)

    # DB에서 전년동기 조회
    all_data = get_financial_data(conn, ticker, quarters=20)
    prev = None
    for d in all_data:
        if d["fiscal_year"] == prev_year and d["fiscal_quarter"] == quarter:
            prev = d
            break

    growth = {}
    if prev:
        prev_rev = prev.get("revenue")
        prev_op = prev.get("operating_profit")
        cur_rev = current.get("revenue")
        cur_op = current.get("operating_profit")

        if prev_rev and prev_rev != 0 and cur_rev is not None:
            growth["revenue_growth_yoy"] = round(
                (cur_rev - prev_rev) / abs(prev_rev) * 100, 2
            )
        if prev_op and prev_op != 0 and cur_op is not None:
            growth["op_growth_yoy"] = round(
                (cur_op - prev_op) / abs(prev_op) * 100, 2
            )

    return growth


def _get_latest_quarter() -> tuple:
    """현재 시점에서 가장 최근 발표된 분기 추정 (DART 데이터 지연 고려)"""
    now = datetime.now()
    year = now.year
    month = now.month

    # DART 데이터 지연: 분기 종료 후 ~45일
    # Q1 (3월 말) → 5월 중순 이후 조회 가능
    # Q2 (6월 말) → 8월 중순 이후
    # Q3 (9월 말) → 11월 중순 이후
    # Q4 (12월 말) → 3월 중순 이후
    if month >= 5:
        # Q1 이상 가능
        if month >= 8:
            # Q2 이상
            if month >= 11:
                return (str(year), 3)
            return (str(year), 2)
        return (str(year), 1)
    elif month >= 3:
        return (str(year - 1), 4)
    else:
        return (str(year - 1), 3)


def collect_batch_financials(conn, year: str, quarter: int,
                             tickers: list = None,
                             max_count: int = 500) -> int:
    """
    배치 수집: 거래대금 상위 종목의 특정 분기 재무제표 수집.

    Args:
        conn: SQLite 연결
        year: 사업연도 (예: "2025")
        quarter: 분기 (1~4)
        tickers: 수집 대상 종목 (None이면 거래대금 상위 자동 선택)
        max_count: 최대 수집 종목 수

    Returns:
        수집 성공 건수
    """
    from src.data.database import (
        get_all_corp_codes, upsert_financial_data, get_financial_data,
    )
    from config import DART_COLLECTION

    request_delay = DART_COLLECTION.get("request_delay", 0.5)
    min_trade_value = DART_COLLECTION.get("min_trade_value", 1_000_000_000)

    # corp_code 매핑
    corp_codes = get_all_corp_codes(conn)
    if not corp_codes:
        logger.error("corp_code 매핑이 없습니다. --refresh-codes를 먼저 실행하세요.")
        return 0

    # 대상 종목 결정
    if tickers is None:
        rows = conn.execute("""
            SELECT p.ticker FROM daily_prices p
            JOIN instruments i ON p.ticker = i.ticker
            WHERE i.type = 'stock' AND p.trade_value >= ?
            AND p.date = (SELECT MAX(date) FROM daily_prices
                          WHERE ticker = p.ticker)
            GROUP BY p.ticker
            ORDER BY MAX(p.trade_value) DESC
            LIMIT ?
        """, (min_trade_value, max_count)).fetchall()
        tickers = [r["ticker"] for r in rows]

    logger.info(f"수집 대상: {len(tickers)}종목, {year}Q{quarter}")

    collected = 0
    skipped = 0
    failed = 0

    for i, ticker in enumerate(tickers):
        corp_code = corp_codes.get(ticker)
        if not corp_code:
            skipped += 1
            continue

        # 이미 수집된 데이터 스킵
        existing = get_financial_data(conn, ticker, quarters=20)
        already_exists = any(
            d["fiscal_year"] == year and d["fiscal_quarter"] == quarter
            for d in existing
        )
        if already_exists:
            skipped += 1
            continue

        result = collect_single_financial(corp_code, year, quarter, request_delay)
        if result is None or result == NO_DATA:
            failed += 1
            continue

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
        collected += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  진행: {i + 1}/{len(tickers)} (수집 {collected}, 스킵 {skipped}, 실패 {failed})")

    logger.info(
        f"배치 수집 완료: {year}Q{quarter} — "
        f"수집 {collected}, 스킵 {skipped}, 실패 {failed}"
    )
    return collected


def backfill_financials(conn, start_year: int = None, end_year: int = None) -> int:
    """
    과거 재무제표 백필 (기본 3년).

    Returns:
        총 수집 건수
    """
    from config import DART_COLLECTION

    now = datetime.now()
    if end_year is None:
        end_year = now.year
    if start_year is None:
        start_year = end_year - DART_COLLECTION.get("backfill_years", 3)

    total = 0
    for year in range(start_year, end_year + 1):
        for quarter in [1, 2, 3, 4]:
            # 미래 분기 스킵
            latest_year, latest_q = _get_latest_quarter()
            if year > int(latest_year):
                continue
            if year == int(latest_year) and quarter > latest_q:
                continue

            logger.info(f"백필: {year}Q{quarter}")
            count = collect_batch_financials(conn, str(year), quarter)
            total += count

    logger.info(f"백필 완료: 총 {total}건")
    return total


def main():
    import sys
    from pathlib import Path

    # 프로젝트 루트를 sys.path에 추가
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="OpenDart 재무제표 수집")
    parser.add_argument("--refresh-codes", action="store_true",
                        help="DART corp_code 목록 갱신")
    parser.add_argument("--backfill", action="store_true",
                        help="3년 백필")
    parser.add_argument("--year", type=str, help="사업연도 (예: 2025)")
    parser.add_argument("--quarter", type=int, choices=[1, 2, 3, 4],
                        help="분기 (1~4)")
    parser.add_argument("--test", action="store_true",
                        help="테스트 (10종목만)")
    parser.add_argument("--max", type=int, default=500,
                        help="최대 수집 종목 수 (기본 500)")
    args = parser.parse_args()

    from src.data.database import init_db

    _init_dart()
    conn = init_db()

    if args.refresh_codes:
        refresh_corp_codes(conn)
        return

    if args.backfill:
        backfill_financials(conn)
        return

    # 기본: 최근 분기 수집
    if args.year and args.quarter:
        year, quarter = args.year, args.quarter
    else:
        year, quarter = _get_latest_quarter()

    max_count = 10 if args.test else args.max
    logger.info(f"수집 시작: {year}Q{quarter} (최대 {max_count}종목)")

    collect_batch_financials(conn, year, quarter, max_count=max_count)
    logger.info("완료!")


if __name__ == "__main__":
    main()
