#!/bin/bash
# OpenDart 재무제표 전종목 백필 스크립트
# 매일 19:00에 launchd로 실행, 하루 39,000건씩 점진적 수집 (DART 한도 40,000)
# 전종목 완료 시 자동 중단 (수집할 게 없으면 0건으로 끝남)
# 50건 연속 API 오류 시 한도 소진으로 간주, 조기 종료
# '데이터 없음'은 정상 스킵 (financials_no_data 테이블에 기록)
#
# 수집 범위: 2015~현재, 전종목 (거래대금 무관)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/Users/m2222n/Work/.venv/bin/python3"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

DATE_LABEL=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/dart_backfill_${DATE_LABEL}.log"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

echo "=== DART 재무제표 백필 시작: $(date) ===" >> "$LOG_FILE"

# 네트워크 확인
if ! host opendart.fss.or.kr > /dev/null 2>&1; then
    echo "=== 네트워크 불가 — 백필 중단: $(date) ===" >> "$LOG_FILE"
    exit 1
fi

# 전종목 백필 (하루 39,000건 제한, resume 자동)
# 기본값: --limit 39000 (backfill_financials_runner.py 참조)
if $PYTHON -m scripts.backfill_financials_runner >> "$LOG_FILE" 2>&1; then
    echo "=== DART 백필 완료: $(date) ===" >> "$LOG_FILE"
else
    EXIT_CODE=$?
    echo "=== DART 백필 실패 (exit code: $EXIT_CODE): $(date) ===" >> "$LOG_FILE"
    osascript -e "display notification \"DART 재무제표 백필 실패. 로그: $LOG_FILE\" with title \"ETF RAG 백필 오류\"" 2>/dev/null || true
    exit 1
fi

# 오래된 로그 정리 (30일)
find "$LOG_DIR" -name "dart_backfill_*.log" -mtime +30 -delete 2>/dev/null || true
