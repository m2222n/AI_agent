"""
기존 JSON 수집 파일을 SQLite DB로 마이그레이션

사용법:
    python scripts/migrate_json_to_db.py
    python scripts/migrate_json_to_db.py --dry-run  # 파일 목록만 확인
"""

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import COLLECTED_DIR
from src.data.database import init_db, import_json_file, get_db_stats


def main():
    parser = argparse.ArgumentParser(description="JSON → SQLite 마이그레이션")
    parser.add_argument("--dry-run", action="store_true", help="파일 목록만 확인")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # JSON 파일 탐색
    if not COLLECTED_DIR.exists():
        print(f"수집 디렉토리 없음: {COLLECTED_DIR}")
        return

    json_files = sorted(COLLECTED_DIR.glob("etf_data_*.json"))
    if not json_files:
        print("마이그레이션할 JSON 파일 없음")
        return

    print(f"발견된 JSON 파일: {len(json_files)}개")
    for f in json_files:
        print(f"  - {f.name}")

    if args.dry_run:
        return

    # DB 초기화 + 마이그레이션
    conn = init_db()

    total = 0
    for json_path in json_files:
        try:
            count = import_json_file(conn, json_path)
            total += count
            print(f"  {json_path.name}: {count}개 ETF")
        except Exception as e:
            print(f"  {json_path.name}: 실패 - {e}")

    conn.close()

    print(f"\n마이그레이션 완료: {total}개 ETF (총 {len(json_files)}개 파일)")

    # 통계 확인
    conn = init_db()
    stats = get_db_stats(conn)
    conn.close()
    print(f"\nDB 통계:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
