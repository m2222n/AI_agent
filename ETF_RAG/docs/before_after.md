# ETF RAG 챗봇 — Before / After 비교

> 부트캠프 과제 수준 → 실서비스 수준으로 전환하는 과정 기록

---

## Phase 0 → Phase 1 (데이터 계층)

| 항목 | Before (Phase 0) | After (Phase 1) |
|------|-------------------|------------------|
| **데이터 소스** | etf_data.json 하드코딩 (8개 ETF) | KRX 실시간 수집 via pykrx (1,084종목) |
| **데이터 갱신** | 수동 편집 (사실상 불가) | 일배치 자동 수집 (장마감 후) |
| **시세 정보** | NAV, AUM 텍스트 (고정값) | OHLCV + NAV + 등락률 + 괴리율 + 추적오차 (매일 갱신) |
| **보유종목** | top_holdings 텍스트 배열 | PDF(Portfolio Deposit File) 실제 데이터, 금액/비중 포함 |
| **ETF 커버리지** | 8종목 (KODEX 200, TIGER S&P500 등) | 국내 ETF 전종목 1,084개 |
| **수집 속도** | N/A | 시세/NAV: 전종목 1초 (일괄 API), 보유종목: 상위 100개 ~2.5분 |
| **데이터 정합성** | 없음 | validate_result() — 누락/이상치 자동 검증 |
| **인증** | 불필요 | KRX 로그인 워크어라운드 (pykrx 세션 패치) |

### 코드 변화

**loader.py**
```
Before: JSON 파일 읽기 → 고정 필드 매핑 → Document 생성
After:  수집 데이터 우선 로드 → 정규화 → 두 포맷 모두 Document 변환
        (수집 없으면 하드코딩 fallback)
```

**collector.py (신규)**
```
Before: 없음
After:  KRX 로그인 → 일괄 시세 수집 → 괴리율/추적오차 → 보유종목(상위 N개)
        CLI: --test, --date, --max, --holdings
```

**config.py**
```
Before: ETF_DATA_PATH 하드코딩
After:  COLLECTED_DIR + get_latest_collected_path() 자동 탐색
```

### Document 내용 비교

**Before (하드코딩)**
```
ETF ID: ETF-001
상품명: KODEX 200 (069500)
카테고리: 국내 주식형
추종지수: KOSPI 200
운용사: 삼성자산운용
총보수: 0.15%
순자산가치(NAV): 35,420원     ← 고정값, 언제 기준인지 불명
```

**After (수집 데이터)**
```
상품명: KODEX 200 (069500)
기준일: 20260406               ← 정확한 기준일
종가: 80,800원                 ← 실시간 가격
NAV: 80,647.71원
등락률: +2.91%
거래량: 14,703,488주
거래대금: 1,184,866,376,189원
기초지수: 798.32
괴리율: -0.17%
추적오차율: 0.05%
주요 보유종목: 삼성전자 (31.77%), SK하이닉스 (6.8%), ...
```

### 테스트 변화

| | Before | After |
|---|--------|-------|
| 테스트 수 | 22개 | 26개 |
| 데이터 로더 테스트 | 5개 (하드코딩만) | 9개 (수집+fallback+config) |
| mock 활용 | 없음 | unittest.mock.patch 도입 |

---

## 다음 Phase에서 기대되는 변화

### Phase 2 (RAG 파이프라인)
| | Before | After (예정) |
|---|--------|-------------|
| Vector DB | FAISS (인메모리, 매번 재생성) | Pinecone (영속, 하이브리드 검색) |
| 검색 | similarity_search | BM25 + Dense + Cohere Rerank |
| 한국어 | 기본 토크나이저 | Kiwi 형태소 분석 |
| 평가 | 수동 17건 | RAGAS 자동 평가 50건+ |

### Phase 3 (에이전트)
| | Before | After (예정) |
|---|--------|-------------|
| 질문 분류 | 키워드 매칭 5가지 | LangGraph + Function Calling |
| 도구 | 없음 | RAG 검색, 실시간 시세, ETF 비교, 보유종목 조회 |
| 모델 | GPT-4o only | GPT-4o-mini (기본) + GPT-4o (복잡) 라우팅 |

---

_Last Updated: 2026-04-07_
