# 데이터 수집 자동화

## 현재 구성 (2026-10-08~): GitHub Actions 단일 파이프라인

수집·검증·누락 보충·Release DB 업로드·프로덕션 반영·실패 알림이 전부
`.github/workflows/daily-collect.yml` 한 곳에서 돈다. **로컬 launchd는 해제됨**(아래 참조).

```
18:30 KST  Daily KRX Data Collection
  ├─ Release DB 다운로드(db-latest)
  ├─ collect_full.py        당일 수집 (deploy JSON + DB)
  ├─ verify_and_recover.py  최근 10영업일 검증 → 빈 날 재수집, 유령 행(종가 전부 0) 삭제
  ├─ Release DB 업로드
  ├─ deploy JSON 커밋
  └─ /admin/refresh-db → 알림 → 가상투자 스냅샷
20:30 KST  Watchdog — 미실행 시 재트리거 + Issue
09:00 KST  Keep-Alive — 백엔드 /health 4회 검사, 실패 시 Issue
```

수집 실패·보충 실패는 `notify-failure`가 GitHub Issue를 자동 생성한다.

## ⚠️ 로컬 launchd — 해제됨 (2026-10-08)

**이유**: 로컬 venv 2개가 Python 3.9.2인데 pykrx ≥1.2.9(2026-09-19)는 Python ≥3.10 필수.
KRX API 변경을 못 따라가 2026-09-29부터 매일 "로그인 성공 → 데이터 요청만 실패"로
조용히 죽어 있었고(알림은 macOS 팝업뿐), Actions는 매 실행 최신 pykrx를 설치해 무사했다.
감시되지 않는 두 번째 수집 경로를 유지하는 것보다 **파이프라인을 하나로** 두는 게 낫다.

- `launchctl unload` 완료. plist·`daily_collect.sh`는 **수동 실행용으로 보존**.
- 다시 켜려면 Python 3.11 venv에 `pykrx>=1.2.9 requests python-dotenv dart-fss`를 설치하고
  `daily_collect.sh`의 `PYTHON=` 경로를 바꾼 뒤 `launchctl load`. 권장하지 않음.

## 수동 실행 (필요 시)

```bash
./scripts/daily_collect.sh              # 최근 영업일 기준 (PYTHON 경로가 3.9 venv라 현재 실패함)
./scripts/daily_collect.sh 20260407     # 특정일 기준
```

## 수동 복구 — Release DB에 구멍이 났을 때 (2026-10-08 실제 절차)

Actions가 며칠 실패했는데(예: KRX 비번 만료) 그 뒤 보충 스텝도 실패했다면 로컬에서 직접:

```bash
# 0) Python 3.11 임시 venv (로컬 .venv는 3.9라 pykrx 1.2.9 설치 불가)
python3.11 -m venv /tmp/collect_venv
/tmp/collect_venv/bin/pip install "pykrx>=1.2.9" python-dotenv requests pandas numpy zstandard

# 1) Release DB 내려받기·해제
mkdir -p /tmp/reldb && cd /tmp/reldb
curl -sL -o etf_rag.db.zst https://github.com/m2222n/AI_agent/releases/download/db-latest/etf_rag.db.zst
zstd -d etf_rag.db.zst -o etf_rag.db

# 2) 검증(--check) → 보충. ETF_DATA_DIR로 DB 경로를 지정한다.
cd /path/to/ETF_RAG
PYTHONPATH=. ETF_DATA_DIR=/tmp/reldb /tmp/collect_venv/bin/python scripts/verify_and_recover.py --days 10 --check
PYTHONPATH=. ETF_DATA_DIR=/tmp/reldb /tmp/collect_venv/bin/python scripts/verify_and_recover.py --days 10

# 3) 유령 행 전수 검사 (휴장일에 종가 전부 0으로 저장된 날) — 있으면 삭제
sqlite3 /tmp/reldb/etf_rag.db "SELECT date FROM daily_prices GROUP BY date HAVING SUM(close=0)=COUNT(*);"

# 4) 무결성 → 압축 → 업로드 → 바이트 비교
sqlite3 /tmp/reldb/etf_rag.db "PRAGMA quick_check; VACUUM;"
zstd -19 -T0 /tmp/reldb/etf_rag.db -o /tmp/reldb/etf_rag.db.zst --force
gh release upload db-latest /tmp/reldb/etf_rag.db.zst --clobber --repo m2222n/AI_agent
```

프로덕션 반영은 다음 Actions 실행의 `/admin/refresh-db`가 한다(별도 작업 불필요).

**함정**: `verify_and_recover`가 몇 초 만에 "공휴일 N일 ✅"로 끝나면 보충된 게 아니라
**데이터 요청이 실패한 것**이다(구버전 pykrx·미로그인). 2026-10-08 수정으로 예외는 실패로
분류되지만, 로그에 `보충 완료: ... ETF 1171 + 주식 2873` 같은 실제 건수가 찍혔는지 확인할 것.

## 로그
- Actions: GitHub → Actions → Daily KRX Data Collection
- 로컬 수동 실행: `logs/collect_YYYYMMDD.log`
