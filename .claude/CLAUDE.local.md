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
- [ ] PyPDFLoader로 ETF 투자설명서 PDF 로딩 파이프라인 구축 (→ #11)
- [ ] tiktoken 기반 토큰 카운트 함수 추가 (→ #6)

### Phase 2에서 적용 (Day 2 - RAG 고도화, 핵심)
- [ ] FAISS → ChromaDB 마이그레이션 (→ #5)
  - persist_directory로 영속성 확보
  - Streamlit Cloud에서 SQLite 버전 호환 재확인
- [ ] RecursiveCharacterTextSplitter 적용 (chunk_size=1000, overlap=100) (→ #6)
- [ ] 임베딩 모델 비교 실험: OpenAI vs BGE-M3 (한국어 특화) (→ #7)
- [ ] retriever.py에 MMR 검색 적용 (similarity_search → mmr, fetch_k=10, k=3) (→ #8)
- [ ] Hybrid Search 구현 (BM25 + Dense Vector) (→ #9)
- [ ] Re-ranking 적용 (Cross-encoder) (→ #9)

### Phase 3에서 적용 (Day 1+2 종합)
- [ ] Structured Output 적용 - LLM 응답을 JSON 스키마로 강제 (Day 1 → #2)
- [ ] CoV (Chain of Verification) 할루시네이션 방어 로직 추가 (Day 1 → #3)
- [ ] 부정 제약(Negative Constraints) prompts.py에 보강 (Day 1 → #4)
- [ ] LangGraph 전환 검토 - 검색 부족 시 재검색 순환 구조 (Day 2 → #10)

### 검토 후 결정
- [ ] Self-Consistency (다수결) - 비용 대비 효과 검토 (Day 1)
- [ ] 멀티 페르소나 - ETF 비교 질문에 다관점 분석 적용 (Day 1)
- [ ] LangGraph 본격 도입 시점 - Phase 3 이후 복잡 워크플로우 필요 시 (Day 2)
- [ ] Azure OpenAI 백엔드 지원 - 현재 불필요, 기업 환경 대응 필요 시 재검토 (Day 1)

---

---

## Semiconductor AI 과정에서 추가 적용 내용 (2026.03.26-03.27)
> MS Azure 기반 Semiconductor AI Special 실무 과정 (SKKU AIEX CAMPUS)
> 소스코드: `/Users/m2222n/Work/Personal/Semiconductor_LLM/`

### Phase 1에서 적용 (외부 API 연동)
- [ ] 실시간 ETF 데이터 API 호출 시 에러 핸들링 패턴 적용 (→ 아래 #14)

### Phase 3에서 적용 (핵심 - Function Calling + Multi-Tool Agent)
- [ ] Function Calling으로 질문 분류 전환: classifier.py 키워드 매칭 → LLM이 도구 자동 선택 (→ #12, #13)
- [ ] Multi-Tool Agent 구조: RAG 검색 + 실시간 API + 로컬 계산을 하나의 에이전트에서 통합 (→ #13)
- [ ] 구조화 데이터(가격/수익률)와 비구조화 데이터(투자설명서) 통합 응답 구현 (→ #13)

### 검토 후 결정
- [ ] Async 병렬 API 호출 (`asyncio.gather()`) - 복수 API 동시 호출로 응답 속도 개선 (→ #15)
- [ ] Microsoft Agent Framework → LangGraph가 더 적합, Azure 종속성 없이 동일 패턴 구현 가능

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

## Phase 2 참고 메모

- **Azure AI Search를 Vector DB 대안으로 검토** (2026-03-23)
  - Chroma는 Streamlit Cloud에서 SQLite 버전 문제로 실패 이력 있음
  - Azure AI Search는 하이브리드 검색(BM25+벡터) + Semantic Ranker 내장 → Phase 2 목표를 코드 구현 없이 서비스로 해결 가능
  - 단, 개인 Azure 구독 필요 (부트캠프 실습 계정은 임시)

---

_Last Updated: 2026-04-07_
_Phase 1 + Phase 2-1(하이브리드+MMR) + Phase 2-2(PDF 파이프라인) + 부트캠프/Semiconductor AI 교안_
