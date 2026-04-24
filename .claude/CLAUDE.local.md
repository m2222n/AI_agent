# CLAUDE.local.md - AI_Agent 프로젝트 로컬 메모

## SKKU AIEX 부트캠프 (K_HIGHTECH PLATFORM) 학습 내용 메모
> MS Azure OpenAI 기반 RAG 서비스 구현과정 (2026.03.12-03.13, 16시간)
> 강사: 김영욱 (Hello AI 대표 / Microsoft RD / AI MVP)

### Day 1 (2026.03.12) - 기초/이론 + Streamlit 챗봇

**프로젝트에 적용할 내용:**

#### 1. Azure OpenAI 백엔드 지원 추가 (교안 p.52-53)
- 현재: OpenAI 직접 API만 지원
- 추가: Azure OpenAI endpoint 지원 (기업 환경 대응)
- Azure 연결 패턴:
  ```python
  openai.azure_endpoint = ENDPOINT
  openai.api_key = API_KEY
  openai.api_type = "azure"
  openai.api_version = "2024-12-01-preview"
  # model명은 Azure 배포명 사용 (예: "gpt-4.1-mini")
  ```
- config.py에 LLM_BACKEND 설정 추가 ("openai" | "azure")
- .env에 AZURE_ENDPOINT, AZURE_API_KEY 등 추가

#### 2. Structured Output / Function Calling (교안 p.62)
- JSON 스키마로 LLM 응답 형식 강제 → 파싱 안정성 + 할루시네이션 방어
- Pydantic/Zod로 실시간 검증
- 현재 프로젝트: LLM 응답을 텍스트로만 받음 → 구조화된 응답으로 전환 고려
- Phase 3 (LLM 응답 품질) 단계에서 적용

#### 3. 할루시네이션 방어 전략 3단계 (교안 p.41)
- RAG (검색 증강): 외부 데이터 참조 ← 이미 구현됨
- CoV (검증의 사슬, Chain of Verification): 자기 검증 ← Phase 3에서 추가
- Self-Consistency: 다수결 채택 ← 비용 대비 효과 검토 필요

#### 4. 프롬프트 엔지니어링 참고 (교안 p.34-42)
- 4대 구성요소: Role + Context + Task + Format ← 이미 prompts.py에 반영됨
- 부정 제약 (Negative Constraints): "~하지 마" 형태 제약 ← prompts.py 보강 시 참고
- 멀티 페르소나: 서로 다른 관점의 전문가 토론 시뮬레이션 ← ETF 비교 질문에 활용 가능
- Temperature 가이드: 0.0-0.2 (결정론적/코딩), 0.7-1.0 (창의성) ← 현재 0.3 적절

**이미 구현 완료된 내용 (스킵):**
- Streamlit 기본 + Chat Interface (p.67-80) → 이미 완성
- System Message + Temperature 활용 (p.56-57) → prompts.py에 반영됨
- dotenv 환경변수 관리 (p.54-55) → config.py에 반영됨
- 세션 상태 기반 채팅 기록 (p.78-80) → chat.py에 반영됨

### Day 2 (2026.03.13) - RAG 파이프라인 구축

**프로젝트에 적용할 내용:**

#### 5. Vector DB: FAISS → ChromaDB 마이그레이션 참고 코드 (교안 p.33, 38)
- 현재: FAISS 인메모리 → Phase 2에서 Chroma로 교체 예정
- 부트캠프 실습 코드 그대로 활용 가능:
  ```python
  from langchain_chroma import Chroma
  db = Chroma.from_documents(
      texts, embedding=embedding_model,
      persist_directory="./chroma_db"
  )
  # 재로드 시:
  db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
  ```
- persist_directory로 영속성 확보 → 현재 FAISS의 매번 재생성 문제 해결
- **주의:** 이전에 Chroma SQLite 버전 문제로 FAISS 전환한 이력 있음 → Streamlit Cloud 환경에서 재확인 필요

#### 6. Chunking 전략: RecursiveCharacterTextSplitter 기본값 (교안 p.28-29)
- Phase 2 Chunking 설계의 **시작점**으로 활용:
  ```python
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=100,
      length_function=tiktoken_len)
  texts = text_splitter.split_documents(pages)
  ```
- chunk_overlap=100: 문맥이 끊기지 않도록 앞뒤 내용 중첩
- tiktoken 기반 길이 측정 (cl100k_base) → 토큰 단위로 정확한 분할
- ETF 투자설명서 PDF 파싱 시 이 설정을 기본으로 시작, 이후 튜닝

#### 7. 임베딩 모델 선택 가이드 (교안 p.6, 31)
- 2026년 주요 모델 비교:
  - **OpenAI text-embedding-3-large** (3072차원): 대규모 검색 표준
  - **OpenAI text-embedding-3-small** (1536차원): 비용 효율적
  - **BGE-M3**: 한국어 포함 다국어 성능 우수 ← Phase 2 한국어 특화 실험 후보
  - **Jina Embeddings v4**: 이미지까지 이해하는 멀티모달
  - **Qwen3-Embedding**: 코딩/추론 능력 우수
- 현재 프로젝트: OpenAI 임베딩 사용 중 → Phase 2에서 BGE-M3와 비교 실험
- **코사인 유사도** 측정 함수 (교안 p.31):
  ```python
  from numpy import dot
  from numpy.linalg import norm
  def cos_sim(A, B):
      return dot(A, B) / (norm(A) * norm(B))
  ```

#### 8. RetrievalQA + MMR 검색 (교안 p.36, 39)
- 현재: 단순 similarity_search (중복 문서 반환 가능)
- 개선: **MMR (Maximal Marginal Relevance)** → 유사하면서도 다양한 문서 선택
  ```python
  qa = RetrievalQA.from_chain_type(
      llm=llm, chain_type='stuff',
      retriever=db.as_retriever(
          search_type='mmr',
          search_kwargs={'k': 3, 'fetch_k': 10}),
      return_source_documents=True)
  ```
- `fetch_k=10`개 후보에서 MMR로 `k=3`개 최종 선택 → 다양성 확보
- Phase 2의 retriever.py 개선 시 즉시 적용 가능

#### 9. Hybrid Search + Re-ranking 개념 (교안 p.9)
- **Chunk Overlap**: 문맥 끊김 방지 (이미 위 6번에서 적용)
- **Hybrid Search**: 키워드 검색(정확성) + 벡터 검색(의미) 결합
  - Phase 2 로드맵의 "BM25 + Dense Vector"와 동일 개념
- **Re-ranking (재정렬)**: 검색 결과 중 가장 적합한 순서로 다시 나열하여 LLM에 전달
  - Cross-encoder 기반 → Phase 2에서 적용 예정

#### 10. LangChain → LangGraph 전환 고려 (교안 p.12-19)
- 현재: LangChain 기반 선형 파이프라인 (질문→검색→응답)
- LangGraph 장점:
  - **순환 구조 (Cycles)**: 에이전트가 스스로 검토/재시도 가능
  - **상태 관리**: 단계별 정보 유실 방지
  - **Conditional Edge**: 조건부 분기 (예: 검색 결과 부족 시 재검색)
  - **Human-in-the-loop**: 민감한 작업 전 사용자 승인
  - **Checkpoints**: 서버 재시작 후에도 상태 복원
- LangChain은 '사슬'(선형), LangGraph는 '지도'(분기+순환)
- **적용 시점**: Phase 3 이후, 복잡한 워크플로우 필요 시 검토
  - 예: 질문 분류 → 검색 부족 시 재검색 → 검증 → 응답 의 순환 구조
  - ReAct 패턴 (Thought → Action → Observation → Final Answer)

#### 11. PDF 문서 로딩 파이프라인 (교안 p.26)
- Phase 1의 "ETF 투자설명서 PDF 파싱"에 직접 활용:
  ```python
  from langchain_community.document_loaders import PyPDFLoader
  loader = PyPDFLoader('path/to/etf_prospectus.pdf')
  pages = loader.load_and_split()
  ```
- PDF → 페이지별 Document 객체 → Chunking → 임베딩 → Vector DB 저장

**이미 구현 완료된 내용 (스킵):**
- AzureChatOpenAI 기본 invoke/temperature 설정 (p.23) → Day 1에서 이미 정리
- SystemMessage/HumanMessage 활용 (p.24) → prompts.py에 반영됨
- .env 환경 구성 (p.21) → config.py에 반영됨

---

## TODO: 부트캠프 내용 → 프로젝트 적용

### Phase 1에서 적용 (Day 2 - PDF/데이터)
- [x] PyPDFLoader로 ETF 투자설명서 PDF 로딩 파이프라인 구축 (→ #11) — `pdf_loader.py` 구현 완료
- [x] tiktoken 기반 토큰 카운트 함수 추가 (→ #6) — `_trim_history()` + PDF 청킹에 적용

### Phase 2에서 적용 (Day 2 - RAG 고도화, 핵심)
- [x] FAISS persist 구현 (→ #5) — `save_local/load_local` + 해시 기반 캐시 무효화 (ChromaDB 대신 FAISS 유지)
- [x] RecursiveCharacterTextSplitter 적용 (chunk_size=1000, overlap=100) (→ #6) — `pdf_loader.py`에 적용
- [ ] 임베딩 모델 비교 실험: OpenAI vs BGE-M3 (한국어 특화) (→ #7)
- [x] retriever.py에 MMR 검색 적용 (→ #8) — `HybridRetriever`에 Jaccard 기반 MMR 구현
- [x] Hybrid Search 구현 (BM25 + Dense Vector) (→ #9) — `FAISS + Kiwi BM25 + RRF` 구현
- [ ] Re-ranking 적용 (Cross-encoder / Cohere Rerank) (→ #9)

### Phase 3에서 적용 (Day 1+2 종합)
- [x] Structured Output 적용 - LLM 분류에 Pydantic 스키마 강제 (Day 1 → #2) — `classify_with_llm()` 적용
- [x] CoV (Chain of Verification) 할루시네이션 방어 (Day 1 → #3) — LangGraph verify 노드 추가 (compare/recommend/risk)
- [x] 부정 제약(Negative Constraints) prompts.py에 보강 (Day 1 → #4) — 수치 근거 필수, 허위 생성 금지 등 적용
- [x] LangGraph 전환 (Day 2 → #10) — `agent.py` 에이전트 그래프 + 도구 13개 + CoV 검증

### 검토 후 결정
- [ ] Self-Consistency (다수결) - 비용 대비 효과 검토 (Day 1) → 보류 (CoV로 대체)
- [ ] 멀티 페르소나 - ETF 비교 질문에 다관점 분석 적용 (Day 1) → 보류
- [x] LangGraph 본격 도입 — 완료 (에이전트 + CoV 검증 그래프)
- [ ] Azure OpenAI 백엔드 지원 - 현재 불필요, 기업 환경 대응 필요 시 재검토 (Day 1)

---

---

## Semiconductor AI 과정에서 추가 적용 내용 (2026.03.26-03.27)
> MS Azure 기반 Semiconductor AI Special 실무 과정 (SKKU AIEX CAMPUS)
> 소스코드: `/Users/m2222n/Work/Personal/Semiconductor_LLM/`

### Phase 1에서 적용 (외부 API 연동)
- [x] 실시간 ETF 데이터 API 호출 시 에러 핸들링 패턴 적용 (→ #14) — timeout/retry/rate limit 구현 완료

### Phase 3에서 적용 (핵심 - Function Calling + Multi-Tool Agent)
- [x] Function Calling으로 질문 분류 전환 (→ #12, #13) — LangGraph + 도구 13개 + Structured Output 분류
- [x] Multi-Tool Agent 구조 (→ #13) — RAG 검색 + yfinance + pykrx + DART + 기술적 지표 + 포트폴리오 + 예측 통합
- [x] 구조화 데이터 + 비구조화 데이터 통합 응답 (→ #13) — `_enrich_with_structured_data()` 구현

### 검토 후 결정
- [ ] Async 병렬 API 호출 (`asyncio.gather()`) - 복수 API 동시 호출로 응답 속도 개선 (→ #15)
- [x] Microsoft Agent Framework → LangGraph 채택 완료 (Azure 종속성 없음)

#### 12. Function Calling / Tool Use 패턴 (Semiconductor_LLM: 01, 04)
- OpenAI Function Calling의 2-call agent loop:
  1. 첫 번째 호출: LLM이 질문을 보고 어떤 도구를 쓸지 결정 (tool_choice="auto")
  2. 도구 실행 후 결과를 messages에 추가
  3. 두 번째 호출: 도구 결과를 바탕으로 최종 응답 생성
- JSON Schema로 도구 메타데이터 정의 → LLM이 파라미터 자동 추출
- 현재 classifier.py의 키워드 기반 5가지 분류보다 훨씬 유연
- **적용 시:** classifier.py를 대체하지 않고, tool 선택 레이어를 상위에 추가
- 참고 코드: `Semiconductor_LLM/01.openai_sample.py` (기본), `04.MAP_tools.py` (고급)

#### 13. Multi-Tool Agent 구조 (Semiconductor_LLM: 04)
- 하나의 에이전트에 여러 도구를 등록하고 LLM이 상황에 맞게 선택:
  ```python
  tools = [search_etf_documents, get_etf_price, search_holdings, calculate_metrics]
  ```
- ETF 챗봇에 적용할 도구 후보:
  - `search_etf_documents`: ChromaDB에서 투자설명서/운용보고서 검색 (RAG)
  - `get_etf_price`: 실시간 NAV/수익률/거래량 조회 (외부 API)
  - `compare_etfs`: ETF 비교 분석 + 표/차트 생성 (로컬 계산)
  - `search_holdings`: 보유종목/섹터 비중 조회 (외부 API)
- 현재 선형 파이프라인(질문→검색→응답)을 tool-based로 전환
- ChromaDB RAG를 도구 중 하나로 편입 (`@tool` 데코레이터 패턴)
- 참고 코드: `Semiconductor_LLM/04.MAP_tools.py` (ChromaDB + 외부 API + 도구 통합)

#### 14. 외부 API 에러 핸들링 패턴 (Semiconductor_LLM: 04)
- Phase 1 크롤링/API 호출 시 적용할 패턴:
  ```python
  try:
      response = requests.get(url, timeout=5)
      response.raise_for_status()
  except requests.exceptions.Timeout:
      return "서버 응답 시간 초과. 잠시 후 다시 시도해주세요."
  except requests.exceptions.RequestException as e:
      return f"API 오류: {str(e)}"
  ```
- timeout 설정 필수, raise_for_status()로 HTTP 에러 감지
- 참고 코드: `Semiconductor_LLM/04.MAP_tools.py` (get_weather, get_exchange_rate 함수)

#### 15. Async 병렬 API 호출 (Semiconductor_LLM: 05)
- `asyncio.gather()`로 복수 API를 동시에 호출:
  ```python
  price, holdings, performance = await asyncio.gather(
      get_etf_price(ticker),
      get_holdings(ticker),
      get_performance(ticker)
  )
  ```
- ETF 가격 + 보유종목 + 수익률을 동시에 가져와서 응답 속도 개선
- Streamlit과의 호환성 확인 필요 (Streamlit은 기본적으로 동기 실행)
- 참고 코드: `Semiconductor_LLM/05.IoTHub.py` (asyncio.gather 패턴)

---

## Phase 1-1 개발 기록 (2026-04-06)

### collector.py 주요 구현 사항

#### KRX 로그인 워크어라운드
- **원인:** KRX가 2026-02-27부터 로그인 필수 정책으로 변경 (pykrx#276)
- **해결:** `login_krx()` + `_patch_pykrx_session()` 구현 — pykrx 내부 HTTP를 로그인된 세션으로 교체
- **KRX 계정:** .env에 KRX_ID, KRX_PW 저장 (data.krx.co.kr 무료 회원가입)
- pykrx 공식 PR #282는 아직 미머지 → 자체 워크어라운드

#### 수집 전략 — 일괄 API 우선
- **시세/NAV/등락률:** `get_etf_ohlcv_by_ticker(date)`, `get_etf_price_change_by_ticker(date, date)` → 전종목 1초
- **괴리율/추적오차:** `get_etf_price_deviation()`, `get_etf_tracking_error()` → 개별 호출 필요 (1084개 × 1.5초)
- **보유종목:** `get_etf_portfolio_deposit_file(ticker, date)` → 개별 호출, 거래대금 상위 N개만

#### 발견한 pykrx 주의사항
- `get_etf_portfolio_deposit_file(ticker, date)`: **ticker가 첫 번째** 인자 (공식 문서 시그니처 확인)
  - 내부 krx 모듈은 `(date, ticker)` 순서 → 외부 API와 내부 API 순서가 다름!
- 비중(weight) 값이 float32로 반환 → `round()` 필수
- 채권 코드(10202G 등)는 `get_market_ticker_name()`으로 조회 불가 → 빈 문자열 처리

#### 수집 데이터 구조 (etf_data_YYYYMMDD.json)
```json
{
  "metadata": {"collection_date": "20260406", "total_etfs": 1084, "holdings_collected": 100},
  "etfs": [{
    "ticker": "069500", "name": "KODEX 200", "date": "20260406",
    "ohlcv": {"open": 80210, "high": 81200, "low": 80100, "close": 80800,
              "volume": 14703488, "trade_value": 1184866376189,
              "nav": 80647.71, "base_index": 798.32,
              "change": 735, "change_pct": 2.91},
    "deviation": -0.17, "tracking_error": 0.05,
    "holdings": [{"stock_ticker": "005930", "stock_name": "삼성전자",
                   "shares": 8140.0, "amount": 667480000, "weight": 31.77}]
  }]
}
```

#### 다음 작업 (Phase 1 남은 것)
1. ~~loader.py 리팩토링~~ → ✅ Phase 1-2에서 완료
2. ~~config.py 자동 연동~~ → ✅ Phase 1-2에서 완료
3. ~~수익률 계산~~ → ✅ Phase 1-1 수익률 수집 추가 완료
4. ~~주요 ETF 선별 기준~~ → ✅ ETF_SELECTION 필터링 구현 완료
5. KIS OpenAPI 연동 (실시간 시세)
6. 일배치 자동화 (cron 또는 GitHub Actions)

---

## Phase 1-2 개발 기록 (2026-04-07)

### loader.py 리팩토링 — 수집 데이터 우선 로드
- **`load_etf_data()`**: 수집 데이터(collected/) 우선 → 없으면 하드코딩(etf_data.json) fallback
- **`_normalize_collected()`**: 수집 JSON → 통일된 dict 구조로 정규화 (ohlcv 중첩 풀기)
- **`_create_doc_from_collected()`**: 수집 데이터용 LangChain Document 생성 (종가/NAV/등락률/보유종목 포함)
- **`_create_doc_from_hardcoded()`**: 기존 하드코딩용 Document 생성 (분리)
- **`create_documents()`**: 데이터 포맷 자동 감지 → 적절한 변환 함수 호출

### config.py 수정
- `COLLECTED_DIR = DATA_DIR / "collected"` 추가
- `get_latest_collected_path()`: collected 폴더에서 최신 JSON 자동 탐색 (glob + reverse sort)
- Python 3.9 호환: `Optional[Path]` 사용 (`Path | None` 문법 불가)

### retriever.py 수정
- metadata 접근 시 `doc.metadata.get("id") or doc.metadata.get("ticker", "")` — 수집 데이터는 id 필드 없음

### sidebar.py 수정
- ETF 목록: 수집 데이터일 때 거래대금 상위 20개만 표시 (전체 1084개 표시 불가)
- 수집 데이터 표시 필드: 종가, 등락률, 거래대금
- 하드코딩 데이터 표시 필드: 카테고리, 위험등급, 총보수

### 테스트 업데이트 (26개 전체 통과)
- `test_data_loader.py`: 9개 신규 (수집 로드, Document 변환, holdings 없는 경우, 필수 필드, fallback, 하드코딩 docs, metadata, config 경로, 빈 디렉토리)
- `conftest.py`: `etf_data` fixture에 `get_latest_collected_path` mock 추가 — 다른 테스트가 하드코딩 데이터로 동작하도록 보장

### Before/After 문서 작성
- `docs/before_after.md`: Phase 0 → Phase 1 변화 비교 (데이터 소스, 코드, Document 내용, 테스트)

---

## Phase 1-1 수익률 + ETF 선별 기록 (2026-04-07)

### collector.py — 수익률 수집 추가
- **`collect_bulk_returns(date)`**: `get_etf_price_change_by_ticker(fromdate, todate)` × 5기간
- 기간: 1일(1d), 1주(1w), 1개월(1m), 3개월(3m), 1년(1y)
- 전종목 일괄 조회 × 5회 = 약 8초 (매우 효율적)
- 수집 JSON에 `"returns": {"1d": 2.91, "1w": 5.12, ...}` 필드 추가
- `collect_all()` 수집 순서: 시세/NAV → 등락률 → **수익률** → 괴리율/추적오차 → 보유종목

### loader.py — 수익률 표시 + ETF 필터링
- Document에 "수익률: 1일: +2.91%, 1주: +5.12%, ..." 텍스트 추가
- `_filter_etfs()`: 거래대금 1억 미만, 종가 0원 제외
- 로드 시 자동 필터링 (수집 데이터에만 적용, 하드코딩에는 미적용)

### config.py — ETF 선별 기준
- `ETF_SELECTION`: `min_trade_value=100_000_000`, `exclude_zero_close=True`
- 전종목 1084개 → 유동성 있는 종목만 RAG 대상으로 필터링

### 테스트 (29개 전체 통과)
- 신규 3개: `test_collected_doc_has_returns`, `test_filter_excludes_zero_close`, `test_filter_excludes_low_trade_value`

### 다음 작업 (Phase 1 남은 것)
1. KIS OpenAPI 연동 (Phase 1-3)
2. ~~일배치 자동화 (Phase 1-4)~~ → ✅ 완료
3. 메타데이터(정적) vs 시세데이터(동적) 분리 설계

---

## Phase 1-4 일배치 자동화 기록 (2026-04-07)

### scripts/daily_collect.sh
- 수집 실행 + 날짜별 로그(`logs/collect_YYYYMMDD.log`) + 30일 자동 정리
- 실패 시 macOS 알림(`osascript display notification`)
- 수동 실행: `./scripts/daily_collect.sh [YYYYMMDD]`

### scripts/com.etfrag.daily-collect.plist
- macOS launchd 스케줄: 매일 18:00 (장마감 후)
- 등록: `ln -sf ... ~/Library/LaunchAgents/ && launchctl load ...`
- Mac 꺼져있으면 다음 부팅 시 실행

### 주의사항
- pykrx가 `/Users/m2222n/Work/.venv/`에만 설치되어 있음 — 스크립트에서 해당 Python 경로를 하드코딩
- ETF_RAG/.venv에는 pykrx 없음 (langchain, streamlit 등만 있음)
- 추후 requirements 통합 또는 별도 수집 전용 venv 검토 필요

---

## Phase 2-1 하이브리드 검색 구현 기록 (2026-04-07)

### retriever.py — FAISS + Kiwi BM25 하이브리드 검색
- **`HybridRetriever`** 클래스: FAISS(dense) + BM25(sparse)를 RRF로 결합
- **`tokenize_korean()`**: Kiwi 형태소 분석기로 NNG/NNP/VV/VA/SL 태그만 추출
  - "수익률" → "수익" + "률"로 분리됨 (Kiwi 특성, BM25에서는 문제없음)
- **RRF (Reciprocal Rank Fusion)**: `1/(k+rank+1)` 공식, k=60 (표준값)
  - dense_weight=0.7, sparse_weight=0.3 (config.py `HYBRID_SEARCH`에서 조정 가능)
- **하위 호환**: `retrieve_relevant_docs()`가 FAISS 직접 전달도 지원 (isinstance 분기)
- Kiwi는 싱글턴 패턴 (`_get_kiwi()`) — 초기화 비용 1회만

### vectorstore.py — 임베딩 모델 명시
- `get_embeddings()` 함수 추가: `OpenAIEmbeddings(model="text-embedding-3-small")`
- config.py에 `EMBEDDING_MODEL` 상수 추가

### config.py — 하이브리드 검색 설정
- `HYBRID_SEARCH`: dense_weight, sparse_weight, bm25_k, dense_k, final_k
- `EMBEDDING_MODEL = "text-embedding-3-small"`

### app.py / chat.py — 통합
- `init_vector_db()` → `init_retriever()` (HybridRetriever 반환)
- `process_question(question, client, vectorstore)` → `process_question(question, client, retriever)`

### requirements.txt 추가
- `kiwipiepy>=0.18.0`, `rank_bm25>=0.2.2`

### .venv 재생성
- 디렉토리 rename(2week_etf_chatbot→ETF_RAG) 이후 venv 인터프리터 경로 깨짐 → 재생성

### 테스트 (44개 전체 통과)
- 신규 15개 (`test_retriever.py`):
  - 토큰화 4개: 명사/영문/빈문자열/ETF질문
  - 하이브리드 6개: 초기화/결과반환/반도체쿼리/채권쿼리/final_k/점수양수
  - 통합 3개: HybridRetriever사용/FAISS fallback/threshold초과
  - doc_key 2개: ticker우선/id fallback

### MMR 추가 (2026-04-07)
- `HybridRetriever.search(use_mmr=True)` — RRF 후보 × 3 → MMR로 다양성 확보
- `_apply_mmr()`: Jaccard 유사도 기반, λ=0.7 (config.py `HYBRID_SEARCH["mmr_lambda"]`)
- `_jaccard_similarity()`: 토큰 집합 기반 문서 간 유사도

### PDF 문서 처리 파이프라인 (2026-04-07)
- `src/data/pdf_loader.py` — PyPDFLoader + RecursiveCharacterTextSplitter
- chunk_size=1000, chunk_overlap=100
- 파일명 규칙: `{ticker}_{name}_{doc_type}.pdf` → 메타데이터 자동 추출
- `create_documents(include_pdfs=True)` — ETF 메타 + PDF 통합
- PDF 디렉토리: `src/data/pdfs/` (파일 추가 시 자동 인식)

### 테스트 (51개 전체 통과)
- 신규 7개: MMR 3개 (다양성/RRF순서/Jaccard) + PDF 4개 (no_dir/empty_dir/메타추출/부분매칭)

### 다음 작업 (Phase 2 남은 것)
1. ETF 투자설명서 PDF 수집 및 적용
2. 보류: Pinecone 마이그레이션, Cohere Rerank, RAGAS 평가
3. 보류: KIS OpenAPI (Phase 1-3)

---

## 품질 안정화 스프린트 기록 (2026-04-07)

### Task 1: 프롬프트 모순 수정 (할루시네이션 방지)
- **문제:** `prompts.py`는 "문서 정보만 사용" ↔ `client.py`는 context=None일 때 "일반적인 ETF 지식으로 답변" → 모순
- **수정 (client.py):** no-context 분기를 "추측하지 마세요 + 데이터 없다고 솔직히 안내"로 변경
- **수정 (prompts.py):** base_constraints에 "검색 결과 없으면 절대 추측 금지", "구체적 수치는 문서 값만 사용" 추가

### Task 2: 검색 신뢰도 임계값 추가
- **문제:** 하이브리드 검색이 점수가 극히 낮아도 결과를 반환 → 무관한 ETF가 context에 포함
- **수정 (config.py):** `HYBRID_SEARCH["min_rrf_score"] = 0.002` 추가
- **수정 (retriever.py):** `retrieve_relevant_docs()`에서 RRF 최소 점수 필터링 → 미달 시 context=None

### Task 3: 토큰 관리 (tiktoken)
- **문제:** `chat_history[-10:]` 단순 슬라이스 → 긴 대화 시 context window 초과 가능
- **수정 (client.py):** `_trim_history(messages, max_tokens=6000)` 함수 추가
  - tiktoken으로 토큰 수 카운팅, 최신 메시지 우선 유지, 초과 시 오래된 메시지 제거
  - `_get_encoder()` 싱글턴, `_count_tokens()` 유틸

### Task 4: 검색 결과 캐싱
- **문제:** 동일 질문 재질문 시 FAISS+BM25 전부 재실행
- **수정 (chat.py):** `@st.cache_data(ttl=3600)` 적용한 `_cached_search()` 함수 추가
  - 1시간 TTL, LLM 호출은 대화 히스토리 의존이므로 캐싱하지 않음

### Task 5: requirements 정리
- `tiktoken>=0.5.0`, `langchain-text-splitters>=0.0.1`, `numpy>=1.24.0` 추가

### Task 6: 분류기 강화
- **문제:** ETF 브랜드가 KODEX/TIGER만 → ACE, ARIRANG, KBSTAR 등 누락
- **수정 (classifier.py):**
  - ETF 브랜드 18개로 확장 (ACE, ARIRANG, KBSTAR, HANARO, KOSEF, SOLS, PLUS 등)
  - 6자리 티커 패턴 매칭 (`\b\d{6}\b`)
  - 비교 키워드 확장 ("중에서", "어떤것", "셋 중", "뭐가 더", "이랑", "하고", "랑")
  - 정보 키워드 확장 ("수익률", "종가", "거래량", "보유종목")
  - 추천 키워드 확장 ("적합한", "알맞은", "찾아줘")
  - 위험 키워드 확장 ("하락", "폭락")
  - `_normalize()` 함수: 공백 정규화 + 소문자 변환

### 테스트 결과: 61개 전체 통과
- classifier 10개 + data_loader 12개 + prompts 7개 + retriever 22개 + PDF 관련 4개 = 총 55 + 6(기존 누락분) = 61개

---

## Phase 2 참고 메모

- **Azure AI Search를 Vector DB 대안으로 검토** (2026-03-23)
  - Chroma는 Streamlit Cloud에서 SQLite 버전 문제로 실패 이력 있음
  - Azure AI Search는 하이브리드 검색(BM25+벡터) + Semantic Ranker 내장 → Phase 2 목표를 코드 구현 없이 서비스로 해결 가능
  - 단, 개인 Azure 구독 필요 (부트캠프 실습 계정은 임시)

---

## 검색 정확도 개선 (2026-04-08)

### RAGAS 평가 첫 실행 + ETF 이름 매칭 도입
- **문제**: 하이브리드 검색(FAISS+BM25)만으로는 714개 ETF 중 정확한 문서를 못 찾음 (Hit Rate 45%)
  - "KODEX 200 수익률" 질문에 다른 ETF가 상위에 올라옴
  - 원인: ETF 문서가 구조적으로 유사하여 임베딩이 구분 못함
- **해결**: ETF 이름/티커 직접 매칭 (pre-filter)
  - `_name_index` / `_ticker_index`: 문서 이름→인덱스 매핑 (HybridRetriever 초기화 시 구축)
  - `_match_etf_by_name()`: 질문 텍스트에 실제 ETF 이름이 포함되어 있으면 직접 매칭
  - 긴 이름부터 greedy matching (예: "KODEX 200선물인버스2X" > "KODEX 200")
  - 매칭 결과를 하이브리드 검색보다 우선 배치 (score=1.0)
- **결과**: Hit Rate 45% → **75%** (+30%p), simple 유형 30% → **80%**
- **남은 한계**: 브랜드명 없는 질문 ("반도체 ETF", "나스닥이랑 반도체") → 하이브리드 검색 의존
  - LangGraph 에이전트의 재검색 로직으로 추가 개선 예정

### 평가 결과 파일
- `eval/results/eval_20260408_101035.json` — 개선 전 (Hit Rate 45%)
- `eval/results/eval_20260408_101709.json` — 개선 후 (Hit Rate 75%)

---

## Phase 3-1 LangGraph 에이전트 전환 (2026-04-08)

### 아키텍처 변경
- 기존: `classifier.py`(키워드) → `retriever.py`(검색) → `client.py`(OpenAI 직접 호출) → 스트리밍 응답
- 신규: `agent.py`(LangGraph 그래프) → LLM이 도구 자동 선택 → 검색 → 최종 답변

### 신규 파일
- **`src/llm/agent.py`** — LangGraph 에이전트 코어
  - `AgentState`: messages + question_type + tool_call_count
  - `classify_with_llm()`: LLM 기반 질문 분류 (키워드 classifier fallback)
  - `call_model()` / `call_tools()`: 그래프 노드
  - `should_call_tools()`: 조건부 엣지 (tool_calls 있으면 도구 실행, 최대 2회)
  - `build_graph()`: StateGraph 컴파일
  - `run_agent()`: 비스트리밍 실행
  - `stream_agent()`: 스트리밍 실행 (stream_mode="updates")
- **`src/llm/tools.py`** — Function Calling 도구 3개
  - `search_etf(query)`: 하이브리드 검색 (retrieve_relevant_docs 호출)
  - `compare_etfs(etf_name_1, etf_name_2)`: 두 ETF 개별 검색 비교
  - `get_etf_list(category)`: 카테고리별 ETF 목록 (k=5)
  - `set_retriever()`: 앱 초기화 시 retriever 주입 (모듈 레벨 글로벌)
- **`tests/test_agent.py`** — 에이전트 테스트 11개
  - 도구 5개: search 결과/빈결과, compare, list 결과/빈결과
  - 라우팅 3개: tool_calls 있을때/없을때/횟수초과
  - 모델 라우팅 1개: COMPLEX_TYPES 정의 확인
  - 그래프 1개: build_graph 컴파일

### 모델 라우팅
- **단순 질문** (simple, general) → `GPT-4o-mini` (비용 절감)
- **복잡 질문** (compare, recommend, risk) → `GPT-4o`
- `_get_model()`: 캐싱된 ChatOpenAI 인스턴스 반환

### 수정된 기존 파일
- **`app.py`**: `create_client()` 제거, `set_retriever()` 호출 추가, `process_question(question)` 시그니처 변경
- **`src/ui/chat.py`**: 전면 리팩토링 — `stream_agent()` 사용, 기존 `_cached_search()` + `call_llm_streaming()` 제거
  - 이벤트 기반 UI 업데이트: question_type → tool_call → token → done
  - 모델명 표시 추가
- **`requirements.txt`**: `langgraph>=0.6.0` 추가

### 기존 코드 유지 (하위 호환)
- `classifier.py`: LLM 분류 실패 시 fallback으로 사용
- `client.py`: 삭제하지 않음 (import 의존성 + API 키 조회 함수 여전히 사용)
- `retriever.py`: 변경 없음 — tools.py에서 `retrieve_relevant_docs()` 호출

### 테스트: 68개 전체 통과
- 기존 57개 + 신규 11개 (test_agent.py)

---

## 토큰 단위 스트리밍 구현 (2026-04-08)

### 문제
- `stream_agent()`가 `stream_mode="updates"`를 사용 → 노드 완료 후 전체 답변을 한꺼번에 반환
- Streamlit UI에서 타이핑 효과 없이 답변이 한 번에 나타남

### 해결
- `stream_mode=["messages", "updates"]` 듀얼 모드로 전환
  - `messages` 모드: `AIMessageChunk` 토큰을 개별적으로 yield → 타이핑 효과
  - `updates` 모드: 도구 호출/결과 이벤트 감지 (tool_call, tool_result)
- 이벤트 포맷: `(mode, data)` 튜플로 반환됨
  - `("messages", (AIMessageChunk, metadata))` — 토큰
  - `("updates", {"node_name": {...}})` — 노드 상태
- `tool_call_chunks`가 있는 AIMessageChunk는 건너뜀 (도구 호출은 updates에서 처리)
- `AIMessageChunk.content`를 누적하여 `{"event": "token", "data": 누적텍스트}` yield

### 수정 파일
- **`agent.py`**: `stream_agent()` 전면 수정, `AIMessageChunk` import 추가
- **`test_agent.py`**: 스트리밍 테스트 5개 추가
  - 토큰 누적, 도구 호출/결과, 모델 라우팅, tool_call_chunks 필터링

### chat.py 변경 불필요
- 기존 `{"event": "token", "data": full_response}` 인터페이스 유지 → UI 코드 수정 없음

### 테스트: 73개 전체 통과
- 기존 68개 + 스트리밍 5개

### RAGAS 재평가 결과 (에이전트 전환 후)
- **결론: 검색 품질 변화 없음** — Hit Rate 88%, Precision 0.567, Recall 0.880
- 에이전트 전환 전(eval_20260408_102500) vs 후(eval_20260408_112127): 완전 동일
- 이유: retriever 코드 변경 없음 — 에이전트는 도구 선택 레이어만 추가, 검색 로직은 동일
- 유형별: simple 90.5%, compare 87.5%, recommend 90.9%, risk 80%, general 80%
- 실패 6건: 모호한 질문(Q21), 보유종목 매칭(Q29, Q48), 범위 외(Q44, Q45), 개념(Q35)

---

## LangSmith 비용 모니터링 연동 (2026-04-08)

### 구현 방식
- LangChain/LangGraph는 환경변수만 설정하면 **자동 트레이싱** (코드 변경 최소화)
- `langsmith` 패키지 이미 설치됨 (`plugins: langsmith-0.4.37`)
- 무료 tier: 5,000 traces/월

### 수정 파일
- **`.env.example`**: `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` 추가
- **`config.py`**: `is_langsmith_enabled()` 함수 추가 (환경변수 검증)
- **`app.py`**: `from config import is_langsmith_enabled` import 추가
- **`src/ui/sidebar.py`**: LangSmith 활성화 시 사이드바에 상태 표시
- **`tests/test_config.py`**: 4개 테스트 (활성/비활성 조건)

### 사용법
1. https://smith.langchain.com 가입 (GitHub 로그인)
2. API Key 발급 → `.env`에 설정:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=lsv2_pt_xxxxx
   LANGCHAIN_PROJECT=etf-rag-chatbot
   ```
3. 앱 실행 시 자동으로 모든 LLM 호출/도구 실행이 트레이싱됨
4. LangSmith 대시보드에서 비용/레이턴시/에러율 확인

### Streamlit Cloud 배포
- Streamlit Cloud: Settings → Secrets에 동일 환경변수 추가
- 미설정 시 트레이싱 비활성화 (앱 동작에 영향 없음)

### 테스트: 77개 전체 통과
- 기존 73개 + config 4개

### 다음 작업
- ~~데이터 저장소 설계 (SQLite 전환, 과거 3년 보존)~~ → ✅ 완료
- 주식 데이터 수집기 추가

---

## SQLite 데이터 저장소 구현 (2026-04-08)

### 신규 파일
- **`src/data/database.py`** — SQLite CRUD 모듈
  - 5 테이블: instruments, daily_prices, returns, holdings, collection_log
  - WAL 모드 (Streamlit 읽기 + 수집 쓰기 동시 가능)
  - Composite PK (ticker, date) → INSERT OR REPLACE로 자연 upsert
  - `init_db()`, `upsert_daily_data()`, `get_latest_data()`, `get_historical_prices()`
  - `search_instruments()`, `prune_old_data(days=1095)`, `import_json_file()`, `get_db_stats()`
  - holdings.amount 오버플로 방어: pykrx가 uint64 값(~1.8×10^19) 반환 → `abs(amount) > 2^63-1` 시 None
- **`scripts/migrate_json_to_db.py`** — JSON → SQLite 일회성 마이그레이션
  - collected/*.json 전체 import → 1088 instruments, 1098 daily_prices, 5203 returns, 1600 holdings
- **`tests/test_database.py`** — 22개 테스트
  - init 2개, write 8개 (upsert/replace/empty), read 7개 (latest/historical/search), maintenance 3개, import 1개, stats 1개

### 수정된 기존 파일
- **`loader.py`**: 3-tier 우선순위 — SQLite DB → collected/ JSON → 하드코딩 fallback
- **`collector.py`**: 듀얼 라이트 — JSON 저장 후 SQLite에도 upsert (실패해도 JSON은 정상)
- **`config.py`**: `DB_PATH = DATA_DIR / "etf_rag.db"` 추가
- **`.gitignore`**: `*.db`, `*.db-wal`, `*.db-shm` 패턴 추가
- **`tests/conftest.py`**: `etf_data` fixture에 `DB_PATH` mock 추가
- **`tests/test_data_loader.py`**: 전체 12개 테스트에 `DB_PATH` mock 추가

### 테스트: 99개 전체 통과
- 기존 77개 + database 22개

### 다음 작업
- 주식 데이터 수집기 추가 (pykrx stock API)

---

## 주식 데이터 확장 (2026-04-08)

### 신규 파일
- **`src/data/stock_collector.py`** — pykrx 기반 주식 일배치 수집
  - KOSPI + KOSDAQ 전종목 (market="ALL" → 두 시장 반복)
  - 4-step 수집: OHLCV → 시가총액/발행주식수 → 펀더멘털(PER/PBR/EPS/BPS/DIV/DPS) → 수익률
  - 듀얼 라이트: `stock_data_YYYYMMDD.json` + SQLite `upsert_stock_data()`
  - CLI: `python -m src.data.stock_collector [--date YYYYMMDD] [--market KOSPI|KOSDAQ|ALL] [--max N] [--test]`
  - `collect_bulk_ohlcv()`, `collect_bulk_market_cap()`, `collect_bulk_fundamental()`, `collect_bulk_returns()`
- **`tests/test_stock_collector.py`** — 22개 테스트
  - DB write 8개, DB read 6개, ETF/stock 분리 1개, stats 1개, collector mock 3개, validation 2개, save 1개

### 수정된 기존 파일
- **`src/data/database.py`**:
  - `stock_fundamentals` 테이블 추가 (ticker, date, market_cap, shares_outstanding, bps, per, pbr, eps, div, dps)
  - `upsert_stock_data(conn, data)` — instruments(type='stock') + daily_prices + returns + stock_fundamentals + collection_log
  - `get_latest_stock_data(conn, date)` — stock + fundamentals JOIN, instruments.type='stock' 필터
  - `prune_old_data()` / `get_db_stats()` — stock_fundamentals 포함
- **`src/data/loader.py`**:
  - `load_stock_data()` — SQLite DB에서만 로드 (fallback 없음)
  - `_filter_stocks()` — ETF와 동일 기준 (거래대금 1억+, 종가 0 제외)
  - `create_stock_documents()` / `_create_doc_from_stock()` — 주식 Document 변환 (PER/PBR/EPS/시가총액/배당)
  - `_format_market_cap()` — 조원/억원 단위 변환
  - ETF/주식 모두 `asset_type` 메타데이터 추가 ("etf" / "stock")
- **`src/llm/tools.py`**:
  - `search_stock(query)` 도구 추가 — 주식 검색용 Function Calling 도구
  - `set_retriever()` — stock_retriever 파라미터 추가
  - `ALL_TOOLS` 4개 (search_etf, compare_etfs, get_etf_list, search_stock)
- **`src/llm/prompts.py`**: 역할을 "투자 전문 어드바이저" (ETF+주식)로 확장
- **`scripts/daily_collect.sh`**: ETF + 주식 순차 수집, 개별 성공/실패 추적, stock_data_*.json 정리 추가
- **`tests/test_data_loader.py`**: 7개 주식 테스트 추가 (19개 total)
- **`tests/test_agent.py`**: ALL_TOOLS 4개 assertion 업데이트
- **`tests/test_prompts.py`**: "투자 전문 어드바이저" assertion 업데이트

### 테스트: 128개 전체 통과

### 평가 데이터셋 확장 (2026-04-08)
- **eval_dataset.json**: 50개 → 65개 (주식 13개 + 혼합 2개 추가)
  - 주식 simple 7개 (삼성전자, SK하이닉스, 현대차, 현대건설, KB금융, 한화에어로스페이스)
  - 주식 compare 2개 (삼성전자 vs SK하이닉스, 현대차 vs 삼성SDI)
  - 주식 recommend 2개 (거래대금, 배당수익률)
  - 주식 general 1개 (PBR 개념)
  - 혼합 2개 (삼성전자 ETF 편입, 반도체 관련 주식+ETF)
  - `asset_type` 필드 추가 ("etf" / "stock" / "mixed")
- **run_eval.py**: ETF + 주식 듀얼 retriever 초기화, asset_type별 적절한 retriever 선택
- **stock_collector.py**: `--max` 옵션이 거래대금 상위 기준으로 정렬 후 자르도록 수정

### 평가 결과 (주식 확장 후)
- 전체 Hit Rate: **90.8%** (59/65)
- ETF: 88.0% (44/50) — 기존과 동일
- 주식: **100%** (13/13) — 이름 매칭 완벽 동작
- 혼합: **100%** (2/2)
- 유형별: simple 93.3%, compare 90.0%, recommend 92.9%, risk 80.0%, general 83.3%
- 결과 파일: `eval/results/eval_20260408_122613.json`

### 다음 작업
1. UI 확장 (사이드바 ETF/주식 탭 분리)
2. Phase 4: UI/UX 개편

---

## Phase 4-1 README 리라이트 + 비교 차트 구현 (2026-04-08)

### README.md 전면 리라이트
- 기존 "2주차 프로토타입" 수준 → 포트폴리오/면접용 완전 새 작성
- Mermaid 아키텍처 플로차트 추가
- ChatGPT 대비 차별점 비교 테이블
- RAGAS 평가 결과 (Hit Rate 90.8%), 비용 분석 ($5~17/월)RAG
- 프로젝트 구조, 기술 스택, 설치/실행 안내 포함

### 개인 GitHub 프로필 (m2222n/m2222n) 업데이트
- AI_agent 섹션: 제목/통계/기술 상세/Demo URL 업데이트
- Streamlit Cloud URL 변경 반영 (구 URL → 신 URL)

### 비교 차트 자동 생성 (structured_data 파이프라인)
- **`src/llm/tools.py`**:
  - 구조화 데이터 인덱스 (`_etf_data_index`, `_stock_data_index`) — `_build_data_index()`로 구축
  - `_find_structured_data(name_or_ticker)`: 정확 매칭 → 부분 매칭
  - `_extract_comparison_fields(data)`: ETF(nav, deviation, holdings) / 주식(per, pbr, market_cap) 분기
  - `compare_etfs`: 구조화 JSON 반환 (`{"__type__": "comparison_table", "items": [...]}`) + 텍스트 fallback
  - `set_retriever(etf_data=, stock_data=)`: `if etf_data is not None:` 조건으로 인덱스 클리어 가능
- **`src/llm/agent.py`**: `structured_data` 이벤트 yield (`'"__type__"' in msg.content`)
- **`src/ui/charts.py`** (신규):
  - `try_parse_comparison()`: comparison_table JSON 추출
  - `render_comparison()`: 마크다운 테이블 + `st.bar_chart` 수익률 비교
  - ETF 전용 행 (NAV, 괴리율) / 주식 전용 행 (PER, PBR, 시가총액, 배당) 자동 분기
- **`src/ui/chat.py`**: `structured_data` 이벤트 캡처, `comparison_data` 히스토리 저장, `chart_placeholder` 렌더링
- **`app.py`**: `set_retriever(etf_data=etf_data, stock_data=stock_data)` 호출

### 테스트: 148개 전체 통과
- 신규 20개: test_agent.py 8개 (구조화 데이터 조회/필드추출/비교/스트리밍) + test_charts.py 12개 (파싱/포맷)
- 주요 수정: `set_retriever(etf_data=[], stock_data=[])` + `if etf_data is not None:` 조건으로 테스트 격리 문제 해결

### 다음 작업
1. ~~UI/UX 전면 개편~~ → ✅ Phase 4-2에서 완료
2. ~~에러 핸들링 완성~~ → ✅ Phase 4-2에서 완료
3. ~~사용자 피드백 루프~~ → ✅ Phase 4-2에서 완료

---

## Phase 4-2 UI/UX + 에러 핸들링 + 피드백 + 통합 응답 (2026-04-08)

### 에러 핸들링 완성
- **`src/llm/agent.py`**:
  - `_make_error_message(e)`: 예외 유형별 사용자 친화적 한국어 메시지 (Rate Limit/Timeout/Connection/Auth/Generic)
  - `call_model()`: LLM 호출 실패 시 try-except → 에러 AIMessage 반환
  - `call_tools()`: 도구 실행 실패 시 try-except → 에러 ToolMessage 반환
  - `stream_agent()`: 전체 스트리밍 try-except → `error` 이벤트 yield
- **`src/ui/chat.py`**:
  - `_get_user_error_message(e)`: UI용 에러 분류 (agent.py와 동일 패턴)
  - `error` 이벤트 핸들링 (st.warning)
  - Exception 시 `return` 대신 에러 메시지를 응답으로 표시
  - `last_question` 세션 저장 누락 수정
- **`app.py`**:
  - `load_stock_data()`: try-except로 주식 로드 실패 시 빈 리스트 반환 (ETF만 동작)
  - `init_retriever()`: try-except + 에러 상세 표시 + st.stop()
- **`src/utils/logging.py`**: `log_interaction()`, `log_feedback()`에 OSError try-except 추가

### 사용자 피드백 루프 개선
- **`src/ui/components.py`**:
  - 부정 피드백 시 사유 선택 (radio: 4가지 사유 + 기타 텍스트 입력)
  - 피드백 중복 방지 (`feedback_submitted` 세션 상태)
  - `st.toast`로 피드백 확인 (기존 st.success 대체)
  - 초기화 시 피드백 상태도 리셋
- **`src/utils/logging.py`**: `get_feedback_stats()` 함수 추가
  - 긍정/부정 카운트, 만족도%, 부정 사유별 집계
- **`src/ui/sidebar.py`**: 성능 모니터링에 만족도 메트릭 추가

### UI/UX 전면 개편
- **`src/ui/styles.py`** (신규): 커스텀 CSS
  - 채팅 메시지 배경색 (사용자: 파란 계열, 어시스턴트: 회색 계열)
  - 둥근 모서리, hover 애니메이션, 메트릭 카드 스타일
  - 비교 테이블 헤더 강조, 일관된 border-radius
  - 모바일 반응형 (`@media max-width: 768px`)
  - max-width: 900px로 가독성 개선
- **`src/ui/sidebar.py`** 개편:
  - 데이터 현황 메트릭 (ETF/주식 종목수 + 기준일)
  - 등락률 색상 표시 (🔴 상승 / 🔵 하락 / ⚪ 보합)
  - 거래대금 읽기 쉬운 포맷 (조/억/만 단위)
  - 2컬럼 레이아웃 (종가 + 거래대금)
  - 서비스 안내 텍스트 정리

### 구조화+비구조화 통합 응답
- **`src/llm/tools.py`**: `_enrich_with_structured_data(sources, index)` 함수 추가
  - 텍스트 검색 결과(RAG)에 구조화 데이터(실시간 가격/수익률) 자동 보강
  - `search_etf`: ETF 인덱스에서 종가/NAV/수익률 보강
  - `search_stock`: 주식 인덱스에서 종가/PER/수익률 보강
  - LLM이 정확한 수치를 참조할 수 있도록 `[실시간 데이터 요약]` 섹션 추가

### 테스트: 168개 전체 통과
- 신규 20개:
  - test_agent.py +10개: 에러 메시지 5개, call_model/call_tools 예외 2개, 스트리밍 에러 1개, 구조화 보강 1개, 도구 예외 1개
  - test_ui_features.py 12개 (신규): 에러 메시지 5개, 피드백 통계 4개, CSS 2개, 피드백 사유 1개

### 다음 작업
1. ~~Phase 4-3: 실시간 시세~~ → ✅ yfinance로 완료
2. Phase 4-4: Multi-Agent, Pinecone, 한국어 임베딩

---

## Phase 4-3 yfinance 장중 시세 연동 (2026-04-08)

### 설계 배경
- KIS OpenAPI는 증권 계좌 개설 필요 (수일 소요) → 개인 포트폴리오 프로젝트에 과도
- yfinance: 계좌 불필요, pip install만으로 즉시 사용, 한국 종목 지원 (.KS/.KQ)
- 듀얼 소스 전략: pykrx(메인, 장마감 후 확정) + yfinance(보조, 장중 15분 지연)

### 신규 파일
- **`src/data/realtime.py`** — yfinance 장중 시세 조회 모듈
  - `is_market_open(now)`: KST 기준 평일 09:00~15:30 판단
  - `krx_to_yfinance(ticker, asset_type)`: 6자리 KRX 코드 → `.KS`/`.KQ` 변환
    - ETF는 항상 `.KS` (KOSPI), 주식은 `.KS` 시도 후 `.KQ` fallback
    - 결과 캐시 (`_market_suffix_cache`) — 티커당 1회만 resolution
  - `get_realtime_price(ticker, asset_type, cache_ttl)`: 현재가 조회
    - 장 외 시간 → None 반환
    - 5분 인메모리 캐시 (yfinance rate limit 방어)
    - `yf.Ticker.fast_info` 사용 (최소 API 호출)
    - 반환: price, prev_close, change, change_pct, volume, timestamp, source
  - `clear_cache()`: 양쪽 캐시 초기화
- **`tests/test_realtime.py`** — 22개 테스트
  - 장 운영 시간 7개 (장중/장전/장후/주말/경계값)
  - 티커 변환 4개 (ETF/KOSPI/KOSDAQ/캐시)
  - 실시간 조회 6개 (성공/장외/캐시히트/캐시만료/에러/no_last_price)
  - 캐시 초기화 1개
  - 도구 통합 4개 (ALL_TOOLS 카운트/not_found/fallback/realtime_data)

### 수정된 기존 파일
- **`src/llm/tools.py`**:
  - `get_realtime_price(name_or_ticker)` 도구 추가 (5번째 LangGraph 도구)
  - 장중: yfinance 실시간 데이터 반환 (현재가, 전일대비, 거래량)
  - 장 외/실패: pykrx 구조화 데이터 fallback (종가, 수익률, NAV/PER)
  - `ALL_TOOLS` 5개로 확장
- **`src/llm/prompts.py`**: 실시간 가격 도구 사용 안내 추가
- **`config.py`**: `REALTIME_PRICE` 설정 (cache_ttl, market_open/close, enabled)
- **`requirements.txt`**: `yfinance>=0.2.0` 추가
- **`tests/test_agent.py`**: ALL_TOOLS 5개 assertion 업데이트

### 테스트: 190개 전체 통과
- 기존 168개 + 신규 22개 (test_realtime.py)

### 다음 작업
1. ~~섹터 분석~~ → ✅ 완료
2. Phase 4-4: Multi-Agent, Pinecone, 한국어 임베딩
3. KIS OpenAPI는 계좌 개설 후 추후 추가

---

## Phase 4-3 섹터 분석 도구 (2026-04-08)

### 설계
- ETF 보유종목 데이터를 역활용: 종목→ETF 역인덱스로 cross-reference
- "삼성전자 담고 있는 ETF", "반도체 관련 ETF 보유종목" 등의 질문에 답변 가능
- 기존 구조화 데이터 인덱스 위에 역인덱스만 추가 → 최소 변경

### 구현
- **`src/llm/tools.py`**:
  - `_holdings_reverse_index`: {stock_ticker → [{etf_name, etf_ticker, weight}]} 역인덱스
  - `_build_holdings_reverse_index(etf_data)`: ETF 보유종목 → 종목별 편입 ETF 목록 구축
    - 종목 티커 + 종목명(소문자) 양쪽으로 조회 가능
  - `set_retriever()`: ETF 데이터 주입 시 역인덱스도 함께 구축
  - `analyze_sector(query)` 도구 추가 (6번째 LangGraph 도구):
    1. 정확 매칭: 티커/종목명으로 해당 종목을 보유한 ETF 목록 (비중 높은 순)
    2. 부분 매칭: 키워드로 관련 종목 검색 후 편입 ETF 목록
    3. 통계: 평균 비중, 최대 비중 ETF 표시
  - `ALL_TOOLS` 6개로 확장
- **`src/llm/prompts.py`**: 섹터 분석 도구 사용 안내 추가

### 테스트: 204개 전체 통과
- 신규 14개 (test_sector.py):
  - 역인덱스 4개 (기본/종목명/빈데이터/비중값)
  - 도구 8개 (정확매칭/종목명/정렬/SK하이닉스/현대차/없는종목/부분매칭/통계)
  - 엣지케이스 1개 (보유종목 없을 때)
  - ALL_TOOLS 1개 (6개 확인)

---

## RAGAS Full 평가 + 프롬프트 개선 (2026-04-08)

### 평가 파이프라인 리라이트
- **`eval/run_eval.py`**: 직접 OpenAI → `run_agent()` 에이전트 기반
- RAGAS 0.4.3: `SingleTurnSample` → `EvaluationDataset` → `evaluate()`
- `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` (RAGAS 네이티브 embed_query 버그 회피)
- `--sample N` CLI 옵션, `_json_safe()` numpy 직렬화

### 프롬프트 개선 3가지
1. **면책 + 데이터 기반 의견** (`prompts.py`): risk/recommend에 적극적 의견 제시 + 면책 footer
2. **데이터 없는 항목 출력 방지**: simple/compare에서 수수료/위험등급/배당정책 삭제
3. **보유종목 enrichment** (`tools.py`): `_enrich_with_structured_data()`에 holdings 상위 10개

### 평가 결과 비교 (4차까지)
| 지표 | Baseline | 2차 | 3차 | 4차(최종) |
|------|----------|-----|-----|-----------|
| Faithfulness | 0.500 | 0.521 | 0.529 | **0.549** |
| F (RAG only) | - | - | - | **0.578** |
| Answer Relevancy | 0.423 | 0.301 | 0.341 | **0.340** |
| Context Recall | 0.336 | 0.400 | 0.492 | **0.469** |

### 3차: RAGAS context에 구조화 데이터 포함
- `run_eval.py`: `_enrich_with_structured_data()` 결과를 RAGAS context에 포함
- 에이전트가 LLM에 전달하는 것과 동일한 context로 평가 → CR 대폭 개선

### 4차: recommend/risk 데이터 근거 강화
- `prompts.py` recommend: "#중요" 섹션 추가 — 추천 이유를 반드시 검색 데이터 수치로만 설명
- `prompts.py` risk: "#중요" 섹션 추가 — 위험 분석에 실제 수익률/등락률 인용 강제
- `run_eval.py`: `faithfulness_rag_only` 지표 추가 (general 제외)

### Faithfulness 유형별 분석 (4차)
- simple: 0.720 / compare: 0.698 / risk: 0.441 / recommend: 0.251 / general: 0.014
- general (F=0.014): 구조적 한계 — LLM 지식 질문이므로 F 측정 부적합 → rag_only로 분리
- recommend (F=0.251): 추천 논리가 context 외 지식 사용 → 데이터 근거 강제했으나 완전 해결 불가

### AR 하락 원인
- 면책 문구 + 한국어 역질문 생성 실패 (AR=0이 30/65개)
- 추가 최적화 ROI 낮음

### 결과 파일
- `eval_20260408_162135.json` (baseline), `eval_20260408_163838.json` (2차)
- `eval_20260408_173420.json` (3차), `eval_20260408_175504.json` (4차)

### 상세 RAGAS 기록
- 메모리 파일 참조: `memory/project_ai_agent_ragas.md`

---

## 주식 도구 확장 + 3년 백필 (2026-04-08)

### 3년 과거 데이터 백필 완료
- **`scripts/backfill_historical.py`** — 728 영업일 (2023-04-10 ~ 2026-04-08)
  - ETF: 608,525 레코드, Stock: 2,057,558 레코드 → SQLite DB
  - `--resume` 모드: 이미 수집된 날짜 스킵 (안전 재시작)
  - ETF: 티커 목록 + OHLCV + 등락률 (보유종목/괴리율은 개별 API라 제외)
  - Stock: OHLCV + 시가총액 + 펀더멘털 (PER/PBR/EPS/BPS/DIV/DPS)

### 주식 도구 확장 (6개 → 8개)
- **`compare_stocks(stock_name_1, stock_name_2)`**: 주식 비교 분석
  - 구조화 JSON (`comparison_table`, `asset_type: "stock"`) → charts.py 자동 렌더링 (PER/PBR/시가총액/배당)
  - 구조화 데이터 없으면 RAG 텍스트 fallback
- **`get_stock_list(category)`**: 키워드 기반 주식 목록 (반도체, 자동차, 바이오 등)
  - `_stock_retriever` or `_retriever` → `retrieve_relevant_docs(k=5)` + `_enrich_with_structured_data()`
- **`prompts.py`**: 주식 비교/목록 도구 + 밸류에이션 분석 안내 추가

### 테스트: 206개 전체 통과
- charts 주식 비교 테스트 2개 추가
- 3개 파일 ALL_TOOLS assertion 6→8 업데이트 (test_agent, test_sector, test_realtime)

### 평가 데이터셋: 75개 (ETF 50 + 주식 22 + 혼합 3)
- 주식 10개 추가: compare 2 (NAVER/카카오, LG에너지솔루션/삼성SDI), recommend 4 (자동차/바이오/은행/시가총액), simple 2 (기아 PER, POSCO홀딩스), mixed 1 (삼성전자 vs KODEX 반도체)

---

## Streamlit Cloud 배포 데이터 (2026-04-09)

### 문제
- Streamlit Cloud에 SQLite DB/collected/ 없음 → 하드코딩 8개 ETF 샘플만 로드
- ETF 검색, 비교, 현재가 조회 모두 실패 (시세 데이터 없음)
- 주식 기능 전체 불가 (fallback 데이터 없음)

### 해결: deploy/ 배포용 데이터 폴더
- `src/data/deploy/etf_data.json` (922K) + `stock_data.json` (134K)
- loader.py 4-tier fallback: DB → collected/ → **deploy/** → hardcoded
- `_normalize_stock_collected()` 추가: 주식 JSON 정규화 (collected/deploy 공통)
- config.py: `get_deploy_etf_path()`, `get_deploy_stock_path()` + stock collected 헬퍼

### 테스트: 208개 통과 (+2: deploy ETF/주식 fallback)

---

## GitHub Actions 자동수집 (2026-04-10)

### 배경
- Mac 꺼져 있어도 Streamlit Cloud 앱에 최신 데이터 반영 필요
- 선택지: 1) 외부 서버 cron, 2) GitHub Actions, 3) Streamlit Cloud scheduled job
- **GitHub Actions 선택** — 개인 프로젝트에 가장 적합 (무료, 설정 간단, Git 연동 자연스러움)

### 구현
- **`.github/workflows/daily-collect.yml`**:
  - 스케줄: 18:30 KST (09:30 UTC), 월~금 (`cron: '30 9 * * 1-5'`)
  - `workflow_dispatch`로 수동 실행도 가능
  - `permissions: contents: write` — push 권한 (초기 누락으로 403 에러, 즉시 수정)
  - 흐름: checkout → Python 3.11 → `pip install pykrx requests` → 수집 → 변경 감지 → commit+push
- **`scripts/collect_for_deploy.py`** — GitHub Actions용 경량 수집:
  - SQLite DB 없이 deploy/ JSON만 생성
  - ETF: 시세/NAV/등락률/수익률 (괴리율/추적오차/보유종목 생략 — deploy용 불필요)
  - 주식: 시세/시가총액/펀더멘털/업종/수익률 전부 수집
  - 검증: ETF < 500 또는 주식 < 1000이면 exit 1
- **launchd**: 18:00 → 18:30으로 변경 (KRX 결산 ~17:30 + 안전 마진 1시간)
- **GitHub Secrets**: `KRX_ID`, `KRX_PW` 설정 완료

### 동작 흐름
```
18:30 KST → GitHub Actions 실행
  → pykrx KRX 로그인
  → ETF ~1,088종목 + 주식 ~3,100종목 수집 (~1분 30초)
  → deploy/etf_data.json, deploy/stock_data.json 업데이트
  → git commit + push (github-actions[bot])
  → Streamlit Cloud가 GitHub 변경 감지 → 자동 재배포
```

### 듀얼 자동수집 체계
| | GitHub Actions | launchd (로컬) |
|---|---|---|
| 대상 | deploy/ JSON (Streamlit Cloud용) | SQLite DB + collected/ JSON |
| Mac 필요 | 불필요 | 필요 |
| 시간 | 18:30 KST | 18:30 KST |
| 데이터 | 시세+펀더멘털 (경량) | 시세+펀더멘털+괴리율+보유종목 (전체) |

### 첫 실행 결과 (수동 트리거)
- 수집 성공: ETF 1,088 + 주식 3,102종목
- 첫 시도 push 실패 (403 Permission denied) → `permissions: contents: write` 추가로 해결
- 재실행 성공: 1분 25초, deploy/ JSON 업데이트 + push → Streamlit Cloud 재배포 확인

---

---

## C-4 OpenDart 재무제표 데이터 구현 (2026-04-14)

### 설계 배경
- pykrx에서 수집하는 PER/PBR/EPS/DIV만으로는 밸류에이션 분석 깊이 부족
- OpenDart API (무료 10,000건/일)로 분기별 매출/영업이익/순이익/마진율/성장률 추가
- dart-fss v0.4.15 라이브러리 사용 (fnltt_singl_acnt() 저수준 API)

### DB 스키마 (database.py)
- **`dart_corp_codes`** 테이블: DART 8자리 corp_code ↔ 주식 6자리 ticker 매핑
- **`stock_financials`** 테이블: 분기 재무제표 (PK: ticker + fiscal_year + fiscal_quarter)
  - revenue, operating_profit, net_income, operating_margin, net_margin, revenue_growth_yoy, op_growth_yoy
- CRUD 6함수: upsert_corp_codes, get_corp_code, get_all_corp_codes, upsert_financial_data, get_financial_data, get_latest_financial_summary
- 3개 인덱스 추가

### dart_collector.py (신규)
- `refresh_corp_codes(conn)`: dart-fss corp_code 목록 → DB (KOSPI/KOSDAQ만)
- `collect_single_financial(corp_code, year, quarter)`: 단일 기업 분기 재무제표
  - 연결(CFS) 우선 → 별도(OFS) fallback
  - 매출액/영업이익/당기순이익 추출 (여러 계정명 변형 대응)
  - 반기보고서(11012)는 누적치 → Q1 빼서 Q2 단독 계산
- `collect_batch_financials()`: 거래대금 상위 종목 배치 수집 (0.5초 딜레이)
- `backfill_financials()`: 3년 백필 (resume 지원)
- CLI: `--refresh-codes`, `--backfill`, `--year`, `--quarter`, `--test`, `--max`

### 도구 추가 (tools.py)
- **`get_financial_statements(name_or_ticker, quarters=4)`**: 12번째 LangGraph 도구
  - DB 조회 → 분기별 마크다운 표 (매출/영업이익/순이익/마진/YoY)
  - DB 데이터 없으면 안내 메시지 반환
- `_enrich_with_structured_data()`: 주식 검색 시 최근 분기 실적 한 줄 요약 자동 추가
- 프롬프트(prompts.py): 재무제표 키워드 + 해석 기준 (영업이익률 10%+ 양호, 20%+ 우수)

### 테스트: 302개 전체 통과 (기존 279 + 신규 23)
- test_dart_collector.py 23개:
  - TestCorpCodes 5개, TestFinancialData 10개, TestDbStatsIncludesNewTables 1개
  - TestDartCollectorHelpers 5개, TestToolsRegistration 2개
- 기존 4개 파일 ALL_TOOLS assertion 11→12 업데이트

### API 키 발급 + 실제 수집 (2026-04-14)
- DART API 키 발급 완료: `68cc9bc4...` (.env에 저장, GitHub Secrets 등록)
- `dart-fss v0.4.15` 사용: `fnltt_singl_acnt()` (30개 요약 항목, CFS+OFS 혼합)
  - `fs_div` 파라미터 불가 (inspect.signature로 확인) → 응답의 `fs_div` 필드로 CFS/OFS 분류
  - `_extract_account_value()`: CFS 우선 검색 → OFS fallback
- 테스트 수집 결과: 10종목 중 7성공, 3실패 (2025 보고서 미공시)
  - 금융회사(KB금융, 신한지주)는 "매출액" 없음 → revenue=N/A, operating_profit 정상
- 10년 백필 진행: 2015~2025, ~101종목 × 44분기, 283행 수집 중 (백그라운드)
  - 2014년 데이터는 OpenDart에 없음 ("조회된 데이타가 없습니다")

### deploy 연동 (2026-04-14)
- `collect_for_deploy.py`: `collect_financial_summary()` 함수 추가 (~130줄)
  - 거래대금 상위 50종목에 `financial_summary` 필드 추가
  - DART corp_code 다운로드 → 매칭 → CFS 우선 재무데이터 추출
  - **월요일만 실행** (`weekday() == 0`) — 분기 데이터라 매일 불필요
- `loader.py`: `_normalize_stock_collected()`에 `financial_summary` 필드 전달
- `tools.py`: DB 데이터 없을 때 deploy JSON의 `financial_summary` fallback 표시
- GitHub Actions: `dart-fss` pip install + `DART_API_KEY` secret 추가

### eval 데이터셋 (2026-04-14)
- 124개 → 134개 (재무제표 10개 추가)
  - simple 6: 삼성전자 매출, SK하이닉스 영업이익률, 현대차 실적, NAVER 매출, LG화학 재무, 카카오 순이익
  - compare 2: 삼성vs하이닉스 영업이익률, 반도체 실적비교
  - recommend 1: 영업이익률 높은 주식
  - general 1: 영업이익률이란

---

## Mac 독립 자동화 + 데이터 영구 보존 (2026-04-17)

### 데이터 영구 보존
- **문제**: `prune_old_data()`가 12년(4380일) 넘는 데이터 삭제 → 2026-04-18에 2014-04-18 데이터 삭제 위기
- **원인**: KRX 슬라이딩 윈도우 (~12년) — 한번 삭제하면 재수집 불가
- **해결**: daily_prices/returns/stock_fundamentals 영구 보존 (holdings만 1년 정리)

### GitHub Actions DB Release 관리
- **문제**: Mac이 고장/교체되면 SQLite DB 수집 불가 → Streamlit Cloud에 영향
- **해결**: GitHub Release asset으로 DB 관리
  - `collect_full.py`: deploy JSON + SQLite DB 통합 수집
  - `daily-collect.yml`: Release에서 DB 다운로드 → 수집 → 업로드 사이클
  - `upload_db_to_release.sh`: 로컬 DB 초기 업로드 (zstd 압축)

### DART 백필 개선
- `dart_collector.py`: NO_DATA 센티넬 — '데이터 없음'과 API 오류 구분
- `backfill_financials_runner.py`:
  - `financials_no_data` 테이블로 빈 분기 기록 (다음 실행에서 스킵)
  - 일일 한도 9,500→39,000 (DART 실제 40,000)
  - 연속 API 오류 200→50으로 조기 종료 기준 변경

### yfinance 백필 스크립트
- `backfill_yfinance.py`: KRX 슬라이딩 윈도우 밖 구간(2014-01-01~04-17) 보충
- `auto_adjust=False`로 원시 가격 수집 (액면분할 미조정 = pykrx 데이터와 일관)

### 테스트 수정
- `test_prune_old_data` → `test_prune_old_data_preserves_prices` (영구 보존 검증)
- `test_prune_preserves_recent_data` → `test_prune_deletes_old_holdings` (holdings 삭제 검증)

---

## 분석 지표 개선 + KST 자동화 (2026-04-17)

### predictor.py 오버홀
- **SMA→EMA 전환**: MACD 피처에서 SMA 근사치 → 진짜 EMA 계산 (`_calc_ema_at()` 헬퍼)
  - EMA 시드: 최초 period개 SMA → 이후 `k = 2/(period+1)` 반복
- **Bootstrap CI**: 기존 ±1σ → 잔차 리샘플링 500회 Bootstrap 백분위 (90% CI)
  - `_bootstrap_ci()`: residuals < 10개일 때 RMSE fallback
- **6m/1y 기간 추가**: `HORIZON_MAP`에 `"6m": 120, "1y": 240` 추가
  - `_calc_statistical_prediction()`에서 `data_days = max(500, horizon_days * 5 + 60)` 동적 조정
- **시나리오 확률 개선**: `_calc_scenarios()` — sigmoid slope 4→3, historical win_rate 블렌딩
  - sample_count >= 10일 때: 70% sigmoid + 30% win_rate
  - sample_count < 10: 기존 sigmoid만 (저표본 보호)

### 포트폴리오 벤치마크 비교
- **`technical.py`**: `BENCHMARK_TICKER = "069500"` (KODEX 200)
  - `_calc_benchmark()` 함수 추가 (~60줄): total_return, annualized, volatility, sharpe, max_drawdown, alpha, tracking_error
- **`tools.py`**: `simulate_portfolio` 도구 출력에 벤치마크 비교 섹션 추가
  - "KODEX 200 대비: 초과수익률 +X.XX%p, 알파 X.XX%, 추적오차 X.XX%"

### 비교 시계열 차트 (상대 수익률 추이)
- **`chart_generator.py`**: `generate_comparison_chart()` 함수 추가 (~80줄)
  - 2~4개 종목의 base=100 정규화 가격 성능 차트 (matplotlib)
  - `_COMPARE_COLORS = ["#1A73E8", "#E8453C", "#34A853", "#FBBC04"]`
  - base64 PNG 반환 → `comparison_chart_b64` 키로 comparison dict에 포함
- **`tools.py`**: `compare_etfs`와 `compare_stocks` 모두 시계열 차트 생성
- **`charts.py`**: "기간별 상대 수익률 추이" 섹션 렌더링 (`comparison_chart_b64`)

### KST 타임존 자동화
- **`scripts/collect_full.py`**: `datetime.now()` → `datetime.now(KST)` 전면 교체
  - `KST = timezone(timedelta(hours=9))` — GitHub Actions UTC 환경 대응
  - 월요일 판별: `now_kst.weekday() == 0` (KST 기준)
  - 월요일에 재무제표 전종목 갱신: `run_daily_backfill()` (최근 1년만 스캔)
- **`.github/workflows/daily-collect.yml`**: KST 월요일 감지 스텝 추가
  - `TZ=Asia/Seoul date +%u` → `is_monday` output
  - timeout 45→60분, commit 날짜 UTC→KST 변경
- **`com.etfrag.dart-backfill.plist`**: `Weekday=1` 추가 (매일→월요일만)

### RAGAS 재평가
- Hit Rate **100%** (162/162) 유지 확인
- Full RAGAS 평가 실행 (--sample 24, stratified)

### 테스트: 404→419개 (+15)
- `test_predictor.py`: +14개 (EMA 4 + Bootstrap CI 4 + 장기 기간 2 + win_rate 시나리오 2 + 기존 수정 2)
- `test_technical.py`: +3개 (벤치마크 포함/없음/양수알파)
- 기존 `test_valid_horizons`, `test_different_horizons` 6m/1y 추가

### Streamlit Cloud 빈 응답 수정 (2026-04-17)
- **증상**: "삼성전자 기술적 분석" 등 질문 시 답변 영역 완전히 빈 상태
- **원인 1**: `should_call_tools`에서 도구 호출 2회 초과 시 `"end"` 반환 → AIMessage에 content 없이 tool_calls만 존재 → `final_answer` 빈 문자열
- **수정 1**: `force_answer` 노드 추가 — 도구 호출 제한 도달 시 ToolMessage로 "현재 정보로 답변하라" 주입 → agent 노드로 재순환
- **원인 2**: Streamlit Cloud (Python 3.13) + LangGraph `stream_mode=["messages", "updates"]` 조합에서 `messages` 모드의 AIMessageChunk 토큰 스트리밍 미발생
- **수정 2**: `updates` 모드 fallback — agent 노드의 최종 AIMessage.content를 `final_answer`로 사용 + chat.py `done` 이벤트에서 answer 길이 비교 fallback
- **파일**: `agent.py` (force_answer 노드, updates fallback), `chat.py` (done answer fallback, 상세 로깅)
- **테스트**: 419→421개 (+2: `test_force_answer_single_tool_call`, `test_force_answer_multiple_tool_calls`)

---

## 답변 품질 강화 (2026-04-21)

### CoV 전체 도구 확대
- **기존**: compare/recommend/risk 유형만 CoV 검증
- **변경**: `COV_TYPES = {"simple", "compare", "recommend", "risk", "technical", "correlation", "portfolio"}` (general만 제외)
- agent.py `should_verify()`: question_type in COV_TYPES이면 verify 노드로 분기

### force_answer 개선
- 도구 호출 제한 도달 시 기존에는 빈 ToolMessage만 주입
- **변경**: 이전 도구 결과 요약을 포함 ("수집된 정보: ..." + "반드시 위 정보를 바탕으로 답변하세요")
- agent.py `force_answer()`: messages에서 ToolMessage 내용 수집 → 요약 주입

### R² 신뢰도 엄격화
- predictor.py `_calc_statistical_prediction()`:
  - "높음" R²>0.3 (was 0.1), "보통" R²>0.1 (was 0.05)
  - 등급 A는 score≥5 (was 4)
  - 3단계 리스크 메시지 (R²>0.3 / R²>0.1 / R²≤0.1)

### 증거 절단 한도 확대
- agent.py: ToolMessage 2000자 (was 1000), CoV 프롬프트 5000자 (was 3000)

### 차트 해석 캡션
- chat.py: 기술적 분석 차트 렌더링 후 해석 가이드 캡션 자동 표시
  - "상단: 종가 + MA(5/20/60) + 볼린저 밴드 | 중단: RSI(14) | 하단: 거래량 + MACD"

### 에러 재시도 UI
- chat.py: 예외 발생 시 "🔄 다시 시도" 버튼 표시
  - `on_click` 대신 `st.button` + `st.session_state["_retry_question"]` + `st.rerun()` 패턴
  - 실패한 user 메시지 pop 후 재실행

### 테스트: 421→431개 (+10)
- test_agent.py +6: CoV 확대 2, force_answer 증거 포함 2, ToolMessage 길이 2
- test_predictor.py +2: R² 엄격화 등급 2
- test_ui_features.py +2: 에러 재시도 UI 2

---

## 탭 UI 분리 + 후속 질문 + 데이터 범위 (2026-04-21~22)

### 탭 분리 UI
- **`src/ui/tabs.py`** (신규): 탭별 전용 렌더러
  - `render_technical_tab()`: 종목 입력 → 11개 지표 + 차트 (chart_generator.py 직접 호출)
  - `render_financial_tab()`: 종목 입력 → 분기별 재무 테이블 + 추이 바차트
  - `render_comparison_tab()`: 2종목 입력 → 비교 테이블 + 수익률 바차트 + 상대 수익률 추이 차트
  - `render_outlook_tab()`: 종목 입력 → 3축 예측 (기술적/펀더멘털/통계) + 시나리오
  - `_resolve_ticker()`: 종목명/티커 → 구조화 데이터 조회 + 유사 종목 제안
  - `_ticker_input()`: st.selectbox with search (자동완성, ~4,200종목)
- **`app.py`**: `st.tabs(["💬 종합 채팅", "📊 기술적 분석", "📑 재무제표", "⚖️ 비교 분석", "🔮 가격 전망"])`
- 탭은 에이전트 없이 직접 데이터 함수 호출 → 빠른 응답

### 후속 질문 버튼
- **구현 이력**: `st.button` 반환값 + `st.rerun()` 방식에서 `on_click` 콜백 방식으로 최종 전환
  - 문제: st.button → True on rerun → st.rerun() → 2번째 rerun에서 True=False → 실행 안됨 (2클릭 필요)
  - 해결: `on_click=_set_followup` 콜백 (rerun 전 실행) + guard (`_retry_question` 있으면 버튼 렌더링 스킵)
- **`src/ui/chat.py`**:
  - `_set_followup(fq)`: on_click 콜백 — `st.session_state["_retry_question"] = fq`
  - `_render_followup_buttons()`: 후속 질문 버튼 렌더링 (guard + on_click)
  - `_get_followup_suggestions()`: 도구 사용 기반 후속 질문 2~3개 제안
  - 히스토리 렌더링 시 마지막 assistant 메시지의 followups도 표시

### 프롬프트 데이터 범위 안내
- **문제**: "데이터 언제부터?" 질문에 GPT가 학습 데이터 기준 "2023년 10월"로 답변
- **수정**: `prompts.py` base_constraints에 데이터 범위 섹션 추가
  - 시세(OHLCV): 2014년 4월부터
  - 재무제표: 2015년부터 (DART 기반)
  - "절대로 LLM 학습 데이터 기준으로 답하지 마세요"

### 차트 제목/범례 겹침 수정
- **`chart_generator.py`**: `top=0.93→0.88`, 제목 y=0.98/0.955, 범례 upper left→upper right

### 종목 입력 자동완성 (2026-04-22)
- **`src/ui/tabs.py`**: `_ticker_input()` — `st.text_input` → `st.selectbox` with search로 전환
  - `_build_autocomplete_options()`: `_etf_data_index` + `_stock_data_index`에서 "종목명 (티커)" 형식 옵션 생성
  - session_state 캐싱 (빈 리스트는 캐시하지 않음 — 인덱스 미초기화 방어)
  - "삼" 입력 → 삼성전자, 삼성SDI 등 자동 필터 / "005930" 입력 → "삼성전자 (005930)" 표시
  - 4개 탭 모두 적용 (tech_ticker, fin_ticker, cmp_ticker1, cmp_ticker2, outlook_ticker)

### Streamlit Cloud 재무제표 미동작 이슈
- **원인**: `@st.cache_resource`가 DB 다운로드 실패(404)를 캐시 → 이후 성공적 다운로드 시도 차단
  - db-latest Release 생성 전에 앱이 먼저 부팅되어 404 캐시됨
- **해결**: Streamlit Cloud 대시보드에서 앱 **Reboot** (캐시 초기화)

### GitHub Actions dotenv 버그 수정 (2026-04-22)
- **원인**: `collect_full.py` → `config.py` → `from dotenv import load_dotenv` → `ModuleNotFoundError`
- **수정**: `.github/workflows/daily-collect.yml`에 `python-dotenv` pip install 추가
- 4/22 스케줄 실행 실패 → 수정 후 수동 트리거로 성공, 데이터 정상 수집

### 탭 종목 검색 부분 매칭 전환 (2026-04-22)
- **문제**: `st.selectbox`의 네이티브 검색이 ~4,200개 옵션에서 "삼성" 부분 매칭 불가 (No results)
- **수정**: `_ticker_input()`을 `st.text_input` + 수동 부분 매칭 + `st.selectbox`로 변경
  - 텍스트 입력 → `q in opt.lower()` 필터링 → 매칭 1개면 자동선택, 복수면 selectbox 표시
  - "삼성" → 33건 매칭 (ETF+주식), 종목명/티커 모두 검색 가능

### 재무제표 탭 분기 수 동적 조정 (2026-04-22)
- **변경**: 고정 selectbox [4,6,8,12] → `st.slider` (1 ~ 최대분기)
  - 전체 데이터를 `get_financial_data(quarters=200)`으로 조회 후 max_q 계산
  - 기본값: min(8, 전체 분기 수)
- 설명 문구: "최근 8분기" → "2015년부터 재무제표를 조회합니다"
- 분기 수 selectbox 열 제거 → 2열 레이아웃으로 간소화

### yfinance 백필 실행 (2026-04-22)
- `backfill_yfinance.py` 실행 (2014-01-01 ~ 2014-04-17, ETF+주식 전종목)
- KRX 슬라이딩 윈도우 밖 구간 보충 (로컬 DB에만 영향)

### 커밋 이력 (04-21~22)
- `37591d5`: feat: 탭 분리 UI + 후속질문 on_click 리팩토링
- `d910cfb`: fix: 데이터 범위 질문 시 LLM 학습 데이터 대신 실제 DB 기준 안내
- `35c7df3`: fix: 후속질문 버튼 on_click 콜백 → st.button 반환값 방식으로 변경 (중간 시도)
- `c9a57c4`: fix: 프롬프트 데이터 범위 — 재무제표 2015년부터로 수정
- `d6ab94f`: fix: 후속질문 on_click 재적용 + 차트 제목/범례 겹침 수정
- `e6a87d6`: feat: 탭 종목 입력에 자동완성 검색 추가 (selectbox)
- `9424364`: docs: CLAUDE.md, CLAUDE.local.md, README 전체 업데이트
- `fba311c`: fix: GitHub Actions에 python-dotenv 의존성 추가
- `5566c4c`: fix: 탭 종목 검색을 부분 매칭으로 변경 + 재무제표 분기 수 동적 조정

### 커밋 이력 (04-23)
- `a748203`~`7b619d2`: st_searchbox→text_input+selectbox→text_input+버튼 자동완성 (Streamlit Cloud 호환)
- `862c119`: fix: 재무제표 fiscal_year 타입 에러 수정 (str→int 변환)
- `2f0e906`: feat: 홈 버튼 왼쪽 상단 이동 + 기능 카드 탭 네비게이션 (JS 주입)
- `11f4b71`: perf: 탭 UI 성능 최적화 (캐싱 + selectbox 전환)

### UI/UX 변경 (04-23)
- **홈 버튼**: 왼쪽 상단 이동, "🏠 홈으로 돌아가기" 텍스트, 대화 히스토리 초기화
- **기능 카드 4개**: 클릭 시 해당 탭으로 자동 전환 (JS `[role="tab"]` 클릭 주입)
- **자동완성**: st_searchbox 제거 → text_input + selectbox 방식 (Streamlit Cloud 호환)
- **재무제표**: 연도 범위 select_slider (2015~현재), caption 변경

### 성능 최적화 (04-23)
- `_build_autocomplete_options()` → `@st.cache_resource` (rerun마다 4,200종목 순회 제거)
- `_ticker_input()` 버튼 20개 → `st.selectbox` 1개 (위젯 렌더링 20→1)
- 기술적 지표/차트/재무제표 DB → `@st.cache_data` (ttl=1h/10m)
- 비교 차트/가격 전망도 캐싱 적용

### 커밋 이력 (04-23 오후)
- `7171b7d`: fix: 탭 순서 변경 + 재무제표 카드 추가 + 카드 네비게이션 버그 수정
- `2b078de`: feat: 단계별 로딩 spinner + 실시간 시세 사이드바 연동 + 홈 버튼 탭 리셋
- `0dd5cb2`: fix: X축 연도 변경점 라벨 개선 (연도 시작 시 강제 표시)
- `d9ef610`: feat: 기술적 분석 차트 기간 파라미터 추가 (days)

### UI/UX 2차 개선 (04-23 오후)
- **탭 순서**: 종합채팅/기술적분석/재무제표/가격전망/비교분석 (전망↔비교 교환)
- **기능 카드 5개**: 실시간 시세(→사이드바)/기술적분석/재무제표/가격전망/비교분석
- **단계별 로딩**: `@st.cache_resource(show_spinner=...)` 3단계 분리 (DB→데이터→인덱스)
- **사이드바 연동**: 실시간 시세 카드 → 주식 탭 전환 + 검색 input 포커스 (JS `setTimeout`)
- **홈 버튼**: 세션 초기화 + `_goto_tab=0` → JS 종합채팅 탭 전환
- **카드 네비게이션 버그**: JS가 사이드바+메인 탭을 모두 선택 → `.stMainBlockContainer` 스코핑
- **X축 연도 라벨**: `_build_xlabels()` 재작성 — 연도 변경점 강제 삽입, 근접 라벨 `min_gap` 제거
- **기술 분석 days 파이프라인**:
  - `get_technical_summary(ticker, days=250)` — days 파라미터 추가, fetch_days = max(days, 250)
  - `first_date`/`last_date` 반환 → 도구 출력에 "데이터 범위" 표시
  - `get_technical_indicators(days)` → `get_technical_summary(days)` + `generate_technical_chart(days)` 전달
  - 프롬프트: days 사용법 + 데이터 범위 안내 지침
- **차트 기준일 연도**: `_fmt_date_full()` (YYYY/MM/DD) 추가
- **사이드바 개선**: st.metric→st.markdown (종목수 잘림 수정), "🔄 매일 18:30 자동 업데이트" 안내

---

## 프롬프트 6차 개선: 종목 추천 품질 강화 + 면책 문구 순화 (2026-04-23)

### 문제 발견 (친구 피드백)
- "투자 기간별 추천 종목" 질문에 대해 답변 품질 3가지 문제:
  1. **과거 수익률만으로 판단**: 급등 종목을 무조건 추천 (고점 리스크 무시)
  2. **동일 종목 중복**: 장기/단기 등 여러 카테고리에 같은 종목 반복 등장
  3. **PER 200+ 종목 추천**: 고평가 경고 없이 추천

### 수정 내용 (src/llm/prompts.py)

#### A. base_constraints에 "종목 추천/선정 원칙" 섹션 추가
- 과거 수익률만으로 판단 금지 (급등 = 고점 리스크)
- 다각도 분석 필수: 밸류에이션(PER/PBR) + 기술적 지표(RSI/MACD) + 재무제표(매출/이익) + 유동성
- PER 100배 이상 "고PER 주의" 필수 언급
- 카테고리 간 종목 중복 금지
- 투자 기간별 선정 기준 구분 (장기/중장기/단기)
- 근거 부족 시 억지 추천 금지 ("확신 있는 추천이 어렵습니다")
- 막연한 표현 대신 구체적 수치/지표 근거 제시

#### B. recommend 타입 프롬프트 전면 개편
- 4단계 분석 프로세스 (파악→필터→다각도검증→추천)
- "추천 품질 기준" 7개 규칙
- "추천 종목별 필수 체크 항목" (밸류에이션/모멘텀/펀더멘털/리스크)
- Few-shot 예시 변경: KODEX 고배당 → 삼성전자 다각도 분석
- 출력 포맷에 "종목 간 비교 요약" 추가

#### C. 면책 문구 순화/통일 (5곳)
- 기존: "※ 본 정보는 투자 참고용이며, 투자 판단은 본인의 책임입니다."
- 변경: "📌 위 내용은 데이터 기반 참고 정보입니다. 실제 투자 시에는 추가적인 조사와 전문가 상담을 권장합니다."
- base_constraints 면책 안내도 동일 톤으로 변경

### 테스트 수정 (tests/test_prompts.py)
- `test_recommend_has_few_shot`: "KODEX 고배당" → "삼성전자" assertion 변경

### 테스트: 431개 전체 통과 (변경 없음)

---

## 수집 자동화 안정성 강화 + 모바일 가독성 개선 (2026-04-23 야간)

### 수집 자동화 안정성
- **문제**: GitHub Actions scheduled workflow가 간헐적으로 실행 안 됨 (4/23 수집 누락 발견)
- **Watchdog 워크플로우** (`.github/workflows/watchdog-collect.yml`):
  - 20:30 KST (수집 2시간 후) 실행
  - `gh run list` + jq로 최근 14시간 내 daily-collect 성공 여부 확인
  - 미실행 시 자동 재트리거 → 15분 대기 → 재확인
  - 재트리거도 실패 시 GitHub Issue 자동 생성 (중복 방지)
- **실패 알림** (`daily-collect.yml` 확장):
  - `notify-failure` job 추가 (`needs: collect`, `if: failure()`)
  - 실패 시 GitHub Issue 자동 생성 (날짜별 중복 방지)
  - permissions에 `actions: write`, `issues: write` 추가

### 모바일 가독성 개선 (`src/ui/styles.py`)
- **line-height 1.7**: `.stMarkdown, .stChatMessage` 본문 텍스트 가독성 향상
- **테이블 가로 스크롤**: `overflow-x: auto` + `-webkit-overflow-scrolling: touch` (넓은 비교표 모바일 대응)
- **테이블 헤더**: `white-space: nowrap` (헤더 줄바꿈 방지)
- **캡션 크기**: 0.78rem → 0.82rem
- **768px 태블릿 breakpoint 추가**:
  - 탭 라벨 축소 (0.82rem, 패딩 축소)
  - 멀티컬럼 2열 축소 (`flex: 1 1 45%`, `flex-wrap: wrap`)
- **480px 소형 폰 breakpoint 신규**:
  - 멀티컬럼 세로 스택 (`flex: 1 1 100%`, `min-width: 100%`)
  - h1 1.25rem, 채팅 메시지 패딩 축소
  - 테이블 0.8rem + 셀 패딩 축소
  - 메트릭 카드 컴팩트 (패딩 0.5rem, 값 1.1rem)
  - 탭 라벨 0.75rem
  - 사이드바 260px

### 테스트: 431개 전체 통과

---

## 코드 품질 리팩토링 + 멀티 도구 병렬 호출 (2026-04-24)

### 코드 품질 개선
- **에러 처리 통합**: `chat.py`의 `_get_user_error_message()` → `agent.py`의 `_make_error_message()`로 위임 (중복 제거)
- **모듈 캡슐화**: `tabs.py`가 `_tools_module._etf_data_index` 직접 접근 → `tools.py`에 공개 API 추가
  - `get_available_tickers()`: 자동완성용 종목 옵션 반환
  - `get_data_indices()`: ETF/주식 데이터 인덱스 반환 (읽기 전용)
- **N+1 DB 쿼리 제거**: `_enrich_with_structured_data()` — 종목별 DB 연결 → 1회 배치 조회로 개선

### 멀티 도구 병렬 호출 (E-1 #1 완료)
- `call_tools()` — 2개+ 도구 호출 시 `ThreadPoolExecutor` 병렬 실행 (max_workers=4)
- 단일 도구는 순차 실행 (스레드풀 오버헤드 방지)
- tool_call 순서 보장 (dict 매핑 → 원본 순서로 ToolMessage 생성)
- **효과**: 예측 질문에서 `get_technical_indicators` + `get_financial_statements` 동시 실행 → ~10초 → ~5초
- 모든 도구가 read-only (초기화 후), SQLite WAL 모드 → 스레드 안전
- 스트리밍/UI 변경 없음 (병렬화가 call_tools 내부에서 완결)

### 테스트: 434개 전체 통과 (+3)

### 대화 맥락 유지 프롬프트 (E-1 #2 완료)
- `prompts.py` base_constraints에 `#대화 맥락 유지` 섹션 추가
- 대명사/지시어 참조 ("그 종목", "아까 그거") → 이전 대화에서 종목 식별
- 후속 질문에서 종목명 생략 시 최근 언급 종목으로 간주
- chat_history[-10:]는 이미 에이전트에 전달됨 → 프롬프트만 추가 (코드 변경 없음)

### 동적 예시 질문 생성 (E-1 #4 완료)
- `generate_dynamic_examples(etf_data, stock_data)` — 당일 수집 데이터 기반
- 4개 카테고리: 오늘의 급등주, 급락주, 거래대금 TOP, 비교 분석
- 급등: change_pct 상위 2개 ("XX 오늘 +3.5% 왜 올랐어?")
- 급락: change_pct 하위 2개 ("XX -2.1% 하락, 기술적 분석해줘")
- 거래대금: trade_value 상위 2개 ("XX 앞으로 어떨까?")
- 비교: 급등 1위 vs 거래대금 1위 (겹치지 않을 때만)
- 기존 하드코딩 예시는 하단에 유지 (fallback)
- 테스트 13개 추가 (447개 전체 통과)

### 응답 포맷 개선 (E-1 #3 완료)
- `split_into_sections()`: ##/### 헤더 기준 마크다운 섹션 분리
- `render_sectioned_answer()`: 첫 섹션 펼침, 나머지 `st.expander` 접기
- 조건: 500자+ & 3섹션+ 일 때만 (짧은 답변은 기존 마크다운)
- 스트리밍 중 마크다운 → 완료 후 섹션별 재렌더링
- 히스토리 재렌더링에도 동일 적용
- 테스트 14개 추가 (461개 전체 통과)

### Cohere Rerank v3.5 (E-2 #5 완료)
- `retriever.py`: `_rerank()` 메서드 추가 (RRF 후, MMR 전)
- Cohere `rerank-v3.5` cross-encoder로 후보 재정렬
- COHERE_API_KEY 없으면 자동 비활성화 (graceful fallback)
- API 오류/패키지 미설치 시에도 원래 순서로 fallback
- `config.py`: RERANK 설정 dict (enabled, model, top_n)
- `requirements.txt`: `cohere>=5.0` 추가
- `.env.example`: COHERE_API_KEY 안내 추가
- 파이프라인: Stage 0(이름매칭) → Stage 1(FAISS) → Stage 2(BM25) → Stage 3(RRF) → **Stage 4(Rerank)** → Stage 5(MMR)
- 테스트 12개 추가 (473개 전체 통과)

### 테스트: 473개 전체 통과 (+39 today)

---

_Last Updated: 2026-04-24_
_Phase 0~4 + C-1~C-6 + D-1~D-3 완료 + 프롬프트 6차 + 수집 Watchdog + 모바일 반응형 + E-1(병렬도구/대화맥락/동적예시/응답포맷) + E-2(Cohere Rerank) + 도구 13개 + 테스트 473개 + eval 162개 + Hit Rate 100%_
