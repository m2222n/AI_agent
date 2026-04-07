# ETF 일배치 수집 자동화

## 수동 실행
```bash
./scripts/daily_collect.sh              # 최근 영업일 기준
./scripts/daily_collect.sh 20260407     # 특정일 기준
```

## macOS launchd 등록 (권장)
```bash
# 1. plist를 LaunchAgents에 심볼릭 링크
ln -sf "$(pwd)/scripts/com.etfrag.daily-collect.plist" ~/Library/LaunchAgents/

# 2. 등록 (즉시 활성화)
launchctl load ~/Library/LaunchAgents/com.etfrag.daily-collect.plist

# 확인
launchctl list | grep etfrag
```

### 해제
```bash
launchctl unload ~/Library/LaunchAgents/com.etfrag.daily-collect.plist
rm ~/Library/LaunchAgents/com.etfrag.daily-collect.plist
```

### 즉시 실행 (테스트)
```bash
launchctl start com.etfrag.daily-collect
```

## 스케줄
- **매일 18:00** (장마감 후)
- Mac이 꺼져있으면 다음 부팅 시 실행됨

## 로그
- 수집 로그: `logs/collect_YYYYMMDD.log`
- launchd 로그: `logs/launchd_stdout.log`, `logs/launchd_stderr.log`

## 자동 정리
- 30일 이상 된 수집 파일(`collected/etf_data_*.json`) 자동 삭제
- 30일 이상 된 로그 파일 자동 삭제
