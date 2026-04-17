"""
GitHub Actions용 전체 수집 스크립트 — deploy/ JSON + SQLite DB 동시 갱신

기존 collect_for_deploy.py의 JSON 수집 + daily_collect.sh의 DB 수집을 통합.
GitHub Actions에서 Release asset으로 DB를 관리하여 Mac 없이도 동작.

사용법:
    python scripts/collect_full.py                    # 최근 영업일 자동 감지
    python scripts/collect_full.py --date 20260417   # 특정일 수집
    python scripts/collect_full.py --db-path /tmp/etf_rag.db  # DB 경로 지정
    python scripts/collect_full.py --skip-db          # deploy JSON만
"""

import json
import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# 프로젝트 경로 설정
PROJECT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
ETF_RAG_DIR = PROJECT_DIR / "ETF_RAG"

# scripts/ 와 ETF_RAG/ 둘 다 import 가능하게
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(ETF_RAG_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# collect_for_deploy.py에서 함수 import
from collect_for_deploy import (
    login_krx,
    find_latest_business_day,
    collect_etf_for_deploy,
    collect_stock_for_deploy,
    collect_financial_summary,
    DEPLOY_DIR,
)


def collect_to_db(db_path: Path, date: str, stock_data: dict = None) -> dict:
    """pykrx 수집 결과를 SQLite DB에 저장.

    ETF/주식 데이터는 이미 collect_*_for_deploy()로 수집된 것을 재사용.
    """
    # DB_PATH 오버라이드
    import config
    config.DB_PATH = db_path

    from src.data.database import init_db, upsert_daily_data, upsert_stock_data, prune_old_data

    conn = init_db()

    # ETF 수집 → DB
    logger.info(f"[DB] ETF 수집 ({date})")
    etf_data = collect_etf_for_deploy(date)
    etf_count = upsert_daily_data(conn, {
        "metadata": etf_data["metadata"],
        "etfs": etf_data["etfs"],
    })
    logger.info(f"[DB] ETF {etf_count}종목 저장")

    # 주식은 이미 수집된 데이터 재사용 (또는 새로 수집)
    if stock_data is None:
        stock_data = collect_stock_for_deploy(date)

    stock_count = upsert_stock_data(conn, {
        "metadata": stock_data["metadata"],
        "stocks": stock_data["stocks"],
    })
    logger.info(f"[DB] 주식 {stock_count}종목 저장")

    prune_old_data(conn)
    conn.close()

    return {
        "etf_count": etf_count,
        "stock_count": stock_count,
        "etf_data": etf_data,
        "stock_data": stock_data,
    }


def main():
    parser = argparse.ArgumentParser(description="전체 수집 (deploy JSON + SQLite DB)")
    parser.add_argument("--date", type=str, help="수집 기준일 (YYYYMMDD)")
    parser.add_argument("--db-path", type=str, default=None,
                        help="SQLite DB 경로 (기본: ETF_RAG/src/data/etf_rag.db)")
    parser.add_argument("--skip-db", action="store_true",
                        help="DB 수집 건너뛰기 (deploy JSON만)")
    args = parser.parse_args()

    # KRX 로그인
    if not login_krx():
        logger.error("KRX 로그인 실패")
        sys.exit(1)
    logger.info("KRX 로그인 성공")

    # 기준일
    date = args.date if args.date else find_latest_business_day()
    logger.info(f"수집 기준일: {date}")

    # DB 경로
    db_path = Path(args.db_path) if args.db_path else ETF_RAG_DIR / "src" / "data" / "etf_rag.db"

    etf_data = None
    stock_data = None

    # --- DB 수집 ---
    if not args.skip_db and db_path.exists():
        logger.info(f"[DB] 경로: {db_path} ({db_path.stat().st_size / 1024 / 1024:.0f}MB)")

        # 주식 데이터 먼저 수집 (DB + deploy 둘 다 사용)
        stock_data = collect_stock_for_deploy(date)

        # 재무제표 (월요일만)
        if datetime.now().weekday() == 0:
            collect_financial_summary(stock_data, max_count=50)

        result = collect_to_db(db_path, date, stock_data=stock_data)
        etf_data = result["etf_data"]
        logger.info(f"[DB] 완료: ETF {result['etf_count']}, 주식 {result['stock_count']}")
    else:
        if not args.skip_db:
            logger.warning(f"[DB] 파일 없음 ({db_path}) — DB 수집 건너뜀")

    # --- deploy/ JSON 수집 (DB에서 안 했으면 별도 수집) ---
    if etf_data is None:
        etf_data = collect_etf_for_deploy(date)
    if stock_data is None:
        stock_data = collect_stock_for_deploy(date)
        if datetime.now().weekday() == 0:
            collect_financial_summary(stock_data, max_count=50)

    # --- deploy/ JSON 저장 ---
    etf_path = DEPLOY_DIR / "etf_data.json"
    with open(etf_path, "w", encoding="utf-8") as f:
        json.dump(etf_data, f, ensure_ascii=False, indent=2)

    stock_path = DEPLOY_DIR / "stock_data.json"
    with open(stock_path, "w", encoding="utf-8") as f:
        json.dump(stock_data, f, ensure_ascii=False, indent=2)

    # 검증
    etf_count = len(etf_data["etfs"])
    stock_count = len(stock_data["stocks"])
    logger.info(f"[Deploy] ETF {etf_count}종목, 주식 {stock_count}종목 저장")

    if etf_count < 500 or stock_count < 1000:
        logger.error(f"수집 결과가 너무 적습니다! ETF={etf_count}, 주식={stock_count}")
        sys.exit(1)

    logger.info(f"=== 전체 수집 완료: {date} ===")


if __name__ == "__main__":
    main()
