#!/bin/bash
# SQLite DB를 GitHub Release asset으로 업로드 (초기 설정 또는 수동 동기화용)
#
# 사용법:
#   ./scripts/upload_db_to_release.sh
#
# 사전 조건:
#   - gh CLI 인증 완료 (gh auth login)
#   - zstd 설치 (brew install zstd)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DB_FILE="$PROJECT_DIR/ETF_RAG/src/data/etf_rag.db"
COMPRESSED="/tmp/etf_rag.db.zst"

if [ ! -f "$DB_FILE" ]; then
    echo "DB 파일 없음: $DB_FILE"
    exit 1
fi

echo "DB 크기: $(du -h "$DB_FILE" | cut -f1)"

# zstd 압축 (레벨 19 = 고압축)
echo "압축 중..."
zstd -19 "$DB_FILE" -o "$COMPRESSED" --force
echo "압축 크기: $(du -h "$COMPRESSED" | cut -f1)"

cd "$PROJECT_DIR"

# release 생성 (없으면)
if ! gh release view db-latest > /dev/null 2>&1; then
    echo "Release 'db-latest' 생성..."
    gh release create db-latest \
        --title "SQLite DB (자동 업데이트)" \
        --notes "매일 GitHub Actions에서 자동 업데이트되는 SQLite DB. 로컬에서 수동 업로드도 가능." \
        --latest=false
fi

# 업로드
echo "업로드 중..."
gh release upload db-latest "$COMPRESSED" --clobber
echo "완료! DB가 Release 'db-latest'에 업로드되었습니다."
