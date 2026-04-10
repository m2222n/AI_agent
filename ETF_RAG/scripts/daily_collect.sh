#!/bin/bash
# ETF + 주식 일배치 수집 스크립트
# 매일 장마감 후(18:30) 실행 — launchd 또는 cron으로 등록
#
# 사용법:
#   ./scripts/daily_collect.sh              # 최근 영업일 기준
#   ./scripts/daily_collect.sh 20260407     # 특정일 기준

set -euo pipefail

# ── 경로 설정 ─────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/Users/m2222n/Work/.venv/bin/python3"
# -m 모듈 모드로 실행 (PYTHONPATH 기반 import 안정성 확보)
ETF_MODULE="src.data.collector"
STOCK_MODULE="src.data.stock_collector"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# 로그 파일 (날짜별)
DATE_LABEL=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/collect_${DATE_LABEL}.log"

# ── ETF 수집 ─────────────────────────────────────────────────
echo "=== ETF 일배치 수집 시작: $(date) ===" >> "$LOG_FILE"

# 기준일 인자 (없으면 자동 감지)
DATE_ARG=""
if [ -n "${1:-}" ]; then
    DATE_ARG="--date $1"
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

# ── 네트워크 확인 (KRX DNS 해석 가능한지) ──────────────────
# Mac 절전 복귀 직후 DNS 미준비 대응: 최대 6회 × 20초 = 2분 대기
MAX_RETRY=6
NETWORK_OK=false
for i in $(seq 1 $MAX_RETRY); do
    if host data.krx.co.kr > /dev/null 2>&1; then
        NETWORK_OK=true
        break
    fi
    echo "네트워크 대기 중... ($i/$MAX_RETRY)" >> "$LOG_FILE"
    sleep 20
done

if ! $NETWORK_OK; then
    echo "=== 네트워크 불가 — 수집 중단: $(date) ===" >> "$LOG_FILE"
    osascript -e 'display notification "네트워크 연결 실패. 수집 중단." with title "ETF RAG 수집 오류"' 2>/dev/null || true
    exit 1
fi

ETF_OK=true
if $PYTHON -m $ETF_MODULE $DATE_ARG --holdings 100 >> "$LOG_FILE" 2>&1; then
    echo "=== ETF 수집 완료: $(date) ===" >> "$LOG_FILE"
else
    EXIT_CODE=$?
    echo "=== ETF 수집 실패 (exit code: $EXIT_CODE): $(date) ===" >> "$LOG_FILE"
    ETF_OK=false
fi

# ── 주식 수집 ────────────────────────────────────────────────
echo "=== 주식 일배치 수집 시작: $(date) ===" >> "$LOG_FILE"

STOCK_OK=true
if $PYTHON -m $STOCK_MODULE $DATE_ARG >> "$LOG_FILE" 2>&1; then
    echo "=== 주식 수집 완료: $(date) ===" >> "$LOG_FILE"
else
    EXIT_CODE=$?
    echo "=== 주식 수집 실패 (exit code: $EXIT_CODE): $(date) ===" >> "$LOG_FILE"
    STOCK_OK=false
fi

# ── 결과 요약 ────────────────────────────────────────────────
if $ETF_OK && $STOCK_OK; then
    SUMMARY=$(tail -5 "$LOG_FILE" | grep "완료" || echo "수집 완료")
    echo "$SUMMARY"
elif ! $ETF_OK || ! $STOCK_OK; then
    # 하나라도 실패 시 알림
    FAIL_MSG=""
    $ETF_OK || FAIL_MSG="ETF "
    $STOCK_OK || FAIL_MSG="${FAIL_MSG}주식 "
    osascript -e "display notification \"${FAIL_MSG}수집 실패. 로그: $LOG_FILE\" with title \"ETF RAG 수집 오류\"" 2>/dev/null || true

    # 둘 다 실패한 경우만 exit 1
    if ! $ETF_OK && ! $STOCK_OK; then
        exit 1
    fi
fi

# ── 오래된 수집 파일 정리 (30일 이상) ─────────────────────────
COLLECTED_DIR="$PROJECT_DIR/src/data/collected"
if [ -d "$COLLECTED_DIR" ]; then
    find "$COLLECTED_DIR" -name "etf_data_*.json" -mtime +30 -delete 2>/dev/null || true
    find "$COLLECTED_DIR" -name "stock_data_*.json" -mtime +30 -delete 2>/dev/null || true
fi

# ── 오래된 로그 정리 (30일 이상) ──────────────────────────────
find "$LOG_DIR" -name "collect_*.log" -mtime +30 -delete 2>/dev/null || true
