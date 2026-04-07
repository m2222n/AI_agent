#!/bin/bash
# ETF 일배치 수집 스크립트
# 매일 장마감 후(18:00) 실행 — launchd 또는 cron으로 등록
#
# 사용법:
#   ./scripts/daily_collect.sh              # 최근 영업일 기준
#   ./scripts/daily_collect.sh 20260407     # 특정일 기준

set -euo pipefail

# ── 경로 설정 ─────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/Users/m2222n/Work/.venv/bin/python3"
COLLECTOR="$PROJECT_DIR/src/data/collector.py"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# 로그 파일 (날짜별)
DATE_LABEL=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/collect_${DATE_LABEL}.log"

# ── 수집 실행 ─────────────────────────────────────────────────
echo "=== ETF 일배치 수집 시작: $(date) ===" >> "$LOG_FILE"

# 기준일 인자 (없으면 자동 감지)
DATE_ARG=""
if [ -n "${1:-}" ]; then
    DATE_ARG="--date $1"
fi

cd "$PROJECT_DIR"

if $PYTHON "$COLLECTOR" $DATE_ARG --holdings 100 >> "$LOG_FILE" 2>&1; then
    echo "=== 수집 완료: $(date) ===" >> "$LOG_FILE"

    # 수집 결과 요약 (마지막 줄에 종목 수 표시)
    SUMMARY=$(tail -3 "$LOG_FILE" | grep "완료" || echo "완료 메시지 없음")
    echo "$SUMMARY"
else
    EXIT_CODE=$?
    echo "=== 수집 실패 (exit code: $EXIT_CODE): $(date) ===" >> "$LOG_FILE"

    # 실패 알림 (macOS 알림)
    osascript -e "display notification \"ETF 수집 실패 (exit $EXIT_CODE). 로그: $LOG_FILE\" with title \"ETF RAG 수집 오류\"" 2>/dev/null || true

    exit $EXIT_CODE
fi

# ── 오래된 수집 파일 정리 (30일 이상) ─────────────────────────
COLLECTED_DIR="$PROJECT_DIR/src/data/collected"
if [ -d "$COLLECTED_DIR" ]; then
    find "$COLLECTED_DIR" -name "etf_data_*.json" -mtime +30 -delete 2>/dev/null || true
fi

# ── 오래된 로그 정리 (30일 이상) ──────────────────────────────
find "$LOG_DIR" -name "collect_*.log" -mtime +30 -delete 2>/dev/null || true
