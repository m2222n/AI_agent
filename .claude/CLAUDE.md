# ETF RAG 챗봇 - 실서비스 프로젝트

## 프로젝트 개요

**목표:** 부트캠프 과제 수준의 ETF 챗봇을 **실제 사용 가능한 서비스**로 발전시키면서 RAG + AI Agent를 깊이 학습한다.

**핵심 차별점 (ChatGPT와 다른 이유):**
- 실시간 ETF 데이터 (오늘의 NAV, 수익률, 거래량) — ChatGPT는 학습 데이터 기준, 우리는 오늘 기준
- 공식 투자설명서/운용보고서 기반 정확한 답변 + 출처 보장
- ETF 비교 분석 특화 (표/차트 자동 생성)
- Function Calling 기반 Multi-Tool Agent — 질문에 따라 도구를 자동 선택

**GitHub:** https://github.com/m2222n/AI_agent.git
**배포:** https://aiagent-5ejryv4fsnjvhrevzwn3ct.streamlit.app/

---

## 기술 스택

| 구분 | 현재 (Phase 2) | 목표 (서비스) |
|------|----------------|--------------|
| LLM | GPT-4o only | GPT-4o-mini (기본) + GPT-4o (복잡 질문) — 라우팅 |
| Vector DB | **FAISS** (인메모리) | **Pinecone** (free tier, 서버리스) |
| 데이터 | **pykrx** (ETF 1084 + 주식 KOSPI/KOSDAQ 전종목) + **yfinance** (장중 15분 지연) | + **한국투자증권 OpenAPI** (실시간) |
| 검색 | **Hybrid Search** (FAISS + Kiwi BM25, RRF 결합) | + **Cohere Rerank v3** |
| 임베딩 | **OpenAI text-embedding-3-small** | (→ 추후 BGE-M3 비교) |
| 문서 | 없음 | ETF 투자설명서 PDF 파싱 (PyPDFLoader + RecursiveCharacterTextSplitter) |
| 에이전트 | **LangGraph** + Function Calling + 모델 라우팅 | ✅ 구현 완료 |
| 분류 | ~~키워드 매칭 classifier.py~~ → **LLM 분류** | **LangGraph** 기반 LLM 라우팅 + Function Calling |
| 평가 | 수동 17건 + pytest 51개 | **RAGAS** 자동 평가 (Faithfulness, Relevancy, Context Recall) |
| 한국어 | **Kiwi** 형태소 분석 (BM25 토크나이저) | ✅ 적용 완료 |
| 모니터링 | **LangSmith** (free tier, 환경변수로 활성화) | **LangSmith** (free tier, 파이프라인 트레이싱) |
| 배포 | Streamlit Cloud | Streamlit Cloud (→ 추후 Railway + 커스텀 도메인) |

**예상 월 비용:** $5~17 (개인 프로젝트)

---

## 로드맵

### Phase 0: 프로젝트 리셋 ✅ 완료
> 단일 파일 741줄 → 모듈 분리. 뼈대 잡기.

- [x] 프로젝트 구조 재설계 (모듈 분리)
- [x] 설정 관리 체계 (config.py, .env)
- [x] 테스트 프레임워크 세팅 (pytest, 22개 테스트 통과)
- [x] 기존 app.py 로직을 새 구조로 마이그레이션
- [x] 디렉토리 rename: `2week_etf_chatbot` → `ETF_RAG`

---

### Phase 1: 진짜 데이터 확보 ✅ 완료 (1-3 보류)
> 가짜 데이터로는 서비스가 아님. 이게 없으면 나머지는 전부 의미 없음.

**1-1. pykrx 기반 일배치 데이터 수집**
- [x] pykrx로 국내 ETF 전종목 목록 수집 (종목코드, 이름) — 1084종목 확인
- [x] KRX 로그인 워크어라운드 구현 (2026-02 정책변경 대응, pykrx#276)
- [x] 일별 데이터 수집: 시세(OHLCV) + NAV + 기초지수 + 등락률 — 일괄 API로 전종목 1초
- [x] ETF 보유종목(PDF 구성종목) 수집 — 거래대금 상위 100개
- [x] 추적오차율, 괴리율 수집
- [x] 수익률 계산 (1일/1주/1개월/3개월/1년) — `collect_bulk_returns()` 일괄 API × 5기간
- [x] 주요 ETF 선별 기준 정의 — 거래대금 1억+, 종가 0 제외 (`ETF_SELECTION` in config.py)

**1-2. 데이터 저장 구조** ✅ 완료
- [x] etf_data.json 하드코딩 → 자동 갱신 구조로 전환 (loader.py 리팩토링 완료)
- [x] config.py: `get_latest_collected_path()` — 최신 수집 파일 자동 탐색
- [x] retriever.py: 수집 데이터 metadata 호환성 수정 (id/ticker fallback)
- [x] sidebar.py: 수집 데이터 포맷 대응 (종가/등락률/거래대금 표시, 상위 20개)
- [x] 테스트 29개 전체 통과 (loader 12개: 수익률+필터링 추가)
- [x] **SQLite 데이터베이스** — 3년 보존, WAL 모드, 5테이블 (instruments, daily_prices, returns, holdings, collection_log)
- [x] loader.py 4-tier 우선순위: SQLite DB → collected/ → deploy/ → 하드코딩 fallback
- [x] deploy/ 배포용 데이터 (Streamlit Cloud용, Git 추적, ~1MB)
- [x] collector.py 듀얼 라이트: JSON + SQLite 동시 저장
- [x] 데이터 정합성 검증 로직 — validate_result() 구현 완료

**1-3. 한국투자증권 OpenAPI 연동**
- [ ] KIS Developers 계좌 개설 + API 키 발급
- [ ] 실시간 시세 조회 연동 (REST, 추후 WebSocket)
- [ ] 에러 핸들링 패턴 적용 (timeout, retry, rate limit)

**1-4. 수집 자동화** ✅ 완료
- [x] 일배치 셸 스크립트 (`scripts/daily_collect.sh`) — 수집 + 로깅 + 정리
- [x] macOS launchd plist (`scripts/com.etfrag.daily-collect.plist`) — 매일 18:00 자동 실행
- [x] 수집 결과 로깅 (`logs/collect_YYYYMMDD.log`) + 실패 시 macOS 알림
- [x] 30일 이상 된 수집 파일/로그 자동 삭제

**자기 검증:** "내일 실제 ETF 가격이 반영되나?" → No면 실패

---

### Phase 2: RAG 파이프라인 재구축 ✅ 핵심 완료 (2-3~2-4 보류)
> RAG를 "제대로" 하는 단계. 면접에서 "왜 이 구조인가?" 에 답할 수 있어야 한다.

**2-1. 하이브리드 검색 (FAISS + Kiwi BM25)** ✅ 완료
- [x] **Kiwi** 형태소 분석기 도입 (한국어 BM25 토크나이저, `kiwipiepy`)
- [x] **FAISS + BM25 하이브리드 검색** 구현 (`HybridRetriever` 클래스)
- [x] **RRF (Reciprocal Rank Fusion)** 결합 — dense 70% + sparse 30%
- [x] **MMR (Maximal Marginal Relevance)** — Jaccard 유사도 기반 다양성 확보 (λ=0.7)
- [x] 임베딩: **OpenAI text-embedding-3-small** 명시 적용
- [x] FAISS 단독 검색 하위 호환 유지 (retriever.py)
- [x] **ETF 이름/티커 직접 매칭** — 질문에서 ETF 이름을 찾아 문서 직접 매핑 (Hit Rate 45%→75%)

**2-2. PDF 문서 처리 파이프라인** ✅ 완료 (파이프라인 구축, PDF 미적용)
- [x] `pdf_loader.py` — PyPDFLoader + RecursiveCharacterTextSplitter (chunk_size=1000, overlap=100)
- [x] 파일명 기반 메타데이터 추출 ({ticker}_{name}_{doc_type}.pdf)
- [x] `create_documents(include_pdfs=True)`로 ETF 데이터 + PDF 통합
- [ ] ETF 투자설명서 PDF 수집 및 적용 (pdfs/ 디렉토리에 파일 추가 시 자동 인식)

**2-3. Vector DB 교체 (추후)**
- [ ] FAISS → **Pinecone** 마이그레이션 (free tier, 서버리스)
- [ ] Pinecone sparse-dense 하이브리드 검색으로 전환

**2-4. Re-ranking (추후)**
- [ ] **Cohere Rerank v3** 적용 (1차 검색 → 재정렬)

**2-5. 평가 체계** ✅ 기본 구축 완료
- [x] RAGAS 평가 파이프라인 구축 (`eval/run_eval.py` — retrieval-only + full RAGAS 모드)
- [x] 평가 데이터셋 구축 (`eval/eval_dataset.json` — 50개 질문)
- [x] 변경 전후 정량 비교 기록 (`eval/results/` — JSON, 5회 평가)
- [x] 에이전트 전환 후 재평가: Hit Rate 88% 유지 (검색 품질 변화 없음)
- [x] 주식 질문 25개 추가 (총 75개), 주식 검색 평가 파이프라인 확장
- [x] 주식 확장 후 재평가: 전체 90.8%, ETF 88%, 주식 100%, 혼합 100%
- [x] 주식 도구 확장 + 75개 데이터셋 재평가: 전체 **91.9%**, ETF 88%, 주식 100%, 혼합 100%
- [x] RAGAS Full 평가 (에이전트 기반): Baseline F=0.500, AR=0.423, CR=0.336
- [x] 프롬프트 개선 후 재평가: F=0.521(+0.021), AR=0.301(-0.122), CR=0.400(+0.064)
- [x] RAGAS 평가 context에 구조화 데이터 포함: F=0.529, AR=0.341, CR=0.492(+0.156)
- [x] recommend/risk 프롬프트 데이터 근거 강화: F=0.549, F(RAG)=0.578, CR=0.469

**자기 검증:** "100개 문서에서 정확한 답을 찾는가?" → 정량 평가 없으면 실패

---

### Phase 3: 에이전트 + LLM 응답 품질 ← 현재 단계
> "ChatGPT보다 나은 점이 있나?" 에 답할 수 있어야 한다.

**3-1. LangGraph 기반 에이전트** ✅ 구현 완료
- [x] LangGraph 도입 — 키워드 classifier.py → LLM 라우팅 그래프 (`agent.py`)
- [x] LLM 기반 질문 분류 (`classify_with_llm()`, 키워드 fallback 유지)
- [x] Function Calling 도구 정의 (`tools.py`) — 8개:
  - `search_etf`: 하이브리드 RAG 검색
  - `compare_etfs`: ETF 비교 분석 (개별 검색 후 병합)
  - `get_etf_list`: 카테고리별 ETF 목록 검색
  - `search_stock`: 주식 RAG 검색
  - `compare_stocks`: 주식 비교 분석 (PER/PBR/시가총액/배당)
  - `get_stock_list`: 주식 카테고리별 목록 검색
  - `get_realtime_price`: 장중 실시간 시세 (yfinance, 15분 지연) + 장 외 종가 fallback
  - `analyze_sector`: 종목→ETF 역인덱스 기반 보유종목/섹터 분석
- [x] 검색 결과 부족 시 재검색 순환 구조 (Conditional Edge, 최대 2회)
- [x] 스트리밍 에이전트 (`stream_agent()` — 이벤트 기반 UI 업데이트)
- [x] 토큰 단위 스트리밍 (`stream_mode=["messages","updates"]` — AIMessageChunk 누적)

**3-2. 모델 라우팅** ✅ 구현 완료
- [x] 단순 질문 (simple/general) → GPT-4o-mini, 복잡한 비교/분석 (compare/recommend/risk) → GPT-4o
- [x] 라우팅 기준: LLM 분류 결과 기반 자동 선택
- [x] 비용 모니터링: LangSmith 트레이싱 연동 (환경변수 설정 시 자동 활성화)

**3-3. 응답 품질**
- [x] 구조화 데이터(가격/수익률) + 비구조화 데이터(투자설명서) 통합 응답 (_enrich_with_structured_data)
- [x] Hallucination 방어: 검색 결과 없으면 "모른다" (프롬프트 모순 수정 + min_rrf_score 필터)
- [x] 프롬프트 개선: 데이터 기반 의견 제공 + 면책 문구 (투자 권유 아닌 참고용 의견)
- [x] 프롬프트 개선: 데이터에 없는 항목(수수료, 위험등급, 배당정책) 출력 방지
- [x] 프롬프트 개선: general 질문에 금융 지식 활용 허용
- [x] 보유종목(상위 10개) 구조화 데이터 enrichment 추가
- [ ] Hallucination 방어: CoV 검증 (추후)
- [x] 대화 히스토리 토큰 관리 (tiktoken 카운팅, _trim_history)
- [x] 비교 질문 시 표/차트 자동 생성 (structured_data 이벤트 + charts.py)
- [x] 검색 결과 캐싱 (@st.cache_data, ttl=1h)

**자기 검증:** "ChatGPT보다 나은 점이 있나?" → 없으면 실패

---

### Phase 4: 서비스 마감 + 포트폴리오
> "친구한테 URL 보내서 쓰라고 할 수 있나?" 에 부끄럽지 않아야 한다.

**4-1. 즉시 (커밋/배포/문서화)** ✅ 완료
- [x] Git 커밋 + 푸시 (주식 확장 전체)
- [x] README + 아키텍처 다이어그램 (Mermaid 플로차트, 포트폴리오용)
- [x] 비용 분석 문서 (README에 월 $5~17 비용 분석 포함)

**4-2. UI/UX 개편**
- [x] 비교 질문 시 표/차트 자동 생성 (st.bar_chart + 마크다운 테이블, ETF+주식 모두 지원)
- [x] UI/UX 전면 개편 (커스텀 CSS, 반응형, styles.py)
- [x] 에러 핸들링 완성 (API 타임아웃/인증/네트워크/Rate Limit 분류, graceful degradation)
- [x] 사용자 피드백 루프 (부정 피드백 사유 수집, 만족도 통계, sidebar 표시)
- [x] LangSmith 모니터링 연동 (환경변수 설정 시 자동 트레이싱)

**4-3. 데이터/분석 확장**
- [x] yfinance 장중 시세 연동 (15분 지연, 계좌 불필요, get_realtime_price 도구)
- [x] 종목→ETF 역인덱스 + 섹터 분석 (analyze_sector 도구, 보유종목 cross-reference)
- [ ] KIS OpenAPI 실시간 시세 연동 (장중 실시간 데이터, 계좌 개설 필요, 추후)
- [ ] 포트폴리오 시뮬레이션 (과거 3년 데이터 기반 백테스트)

**4-4. 아키텍처 고도화 (추후)**
- [ ] Multi-Agent 구조 (리서치 → 분석 → 답변 에이전트 분리)
- [ ] Pinecone + Cohere Rerank (문서 수 증가 시)
- [ ] 한국어 임베딩 모델 비교 (BGE-M3 vs text-embedding-3-small A/B 테스트)
- [ ] KRX 시세정보 재배포 라이선스 검토 (상용화 시 필수)

**자기 검증:** "친구한테 URL 보내서 쓰라고 할 수 있나?" → 부끄러우면 실패

---

## 프로젝트 구조

```
ETF_RAG/
├── app.py                  # Streamlit 진입점 (HybridRetriever 사용)
├── config.py               # 설정/경로/상수 관리 (HYBRID_SEARCH, EMBEDDING_MODEL 등)
├── requirements.txt
├── .env.example
├── src/
│   ├── data/
│   │   ├── loader.py       # load_etf_data(), create_documents(include_pdfs), _filter_etfs()
│   │   ├── database.py     # SQLite CRUD (init_db, upsert_daily_data, get_latest_data, prune_old_data)
│   │   ├── pdf_loader.py   # load_pdf_documents() — PDF 파싱 + 청킹 파이프라인
│   │   ├── realtime.py     # yfinance 장중 시세 조회 (15분 지연, 5분 캐시, KRX→yfinance 티커 변환)
│   │   ├── collector.py    # pykrx 기반 ETF 일배치 수집 (일괄 API + 개별 PDF + SQLite 듀얼라이트)
│   │   ├── stock_collector.py # pykrx 기반 주식 일배치 수집 (KOSPI+KOSDAQ, 시세+시총+펀더멘털)
│   │   ├── etf_data.json   # 하드코딩 샘플 (8개 ETF, fallback용)
│   │   ├── etf_rag.db      # SQLite DB (WAL 모드, .gitignore)
│   │   ├── collected/      # 수집 결과 JSON (.gitignore, 로컬 전용)
│   │   ├── deploy/         # 배포용 데이터 (Git 추적, Streamlit Cloud용)
│   │   └── pdfs/           # ETF 투자설명서 PDF (파일 추가 시 자동 인식)
│   ├── rag/
│   │   ├── vectorstore.py  # create_vectorstore(), get_embeddings() — text-embedding-3-small
│   │   └── retriever.py    # HybridRetriever (FAISS+Kiwi BM25+RRF+MMR), retrieve_relevant_docs()
│   ├── llm/
│   │   ├── agent.py        # LangGraph 에이전트 (라우팅 + 도구 + 재검색)
│   │   ├── tools.py        # Function Calling 도구 8개 (search_etf, compare_etfs, get_etf_list, search_stock, compare_stocks, get_stock_list, get_realtime_price, analyze_sector) + 구조화/역인덱스
│   │   ├── client.py       # get_api_key(), create_client(), call_llm_streaming()
│   │   ├── prompts.py      # build_system_prompt()
│   │   └── classifier.py   # classify_question_type() (LLM 분류 fallback)
│   ├── ui/
│   │   ├── sidebar.py      # render_sidebar()
│   │   ├── chat.py         # process_question() (structured_data 이벤트 처리 포함)
│   │   ├── charts.py       # 비교 차트/테이블 렌더링 (try_parse_comparison, render_comparison)
│   │   ├── styles.py       # 커스텀 CSS (반응형, 테이블 스타일, 모바일 대응)
│   │   └── components.py   # render_example_questions(), render_feedback_buttons(부정사유 수집)
│   └── utils/
│       └── logging.py      # log_interaction(), log_feedback()
├── eval/
│   ├── eval_dataset.json          # RAGAS 평가 데이터셋 (75개 질문: ETF 50 + 주식 22 + 혼합 3)
│   ├── run_eval.py                # 평가 실행 스크립트 (--no-llm / full RAGAS)
│   └── results/                   # 평가 결과 JSON (eval_YYYYMMDD_HHMMSS.json)
├── tests/                  # pytest 208개 (agent 34 + charts 15 + classifier 10 + config 4 + database 22 + loader 21 + prompts 7 + realtime 22 + retriever 28 + sector 14 + stock 22 + ui_features 12)
├── scripts/
│   ├── daily_collect.sh               # 일배치 수집 셸 스크립트
│   ├── backfill_historical.py         # 3년 과거 데이터 백필 (ETF+주식 전종목, --resume 지원)
│   ├── migrate_json_to_db.py          # JSON → SQLite 일회성 마이그레이션
│   ├── com.etfrag.daily-collect.plist  # macOS launchd 스케줄
│   └── README_cron.md                 # 자동화 설정 안내
└── docs/
    └── TODO_deferred.md               # 보류된 작업 목록 (Pinecone, Cohere, KIS, RAGAS)
```

---

## 부트캠프 기록 (아카이브)

<details>
<summary>1~3주차 과제 이력 (클릭하여 펼치기)</summary>

### 1주차: 고객 요구사항 분석 (완료)
- RFP 분석 → 고객 니즈 도출
- 시스템 아키텍처 설계
- 멀티 에이전트 구조 기획

### 2주차: 프로토타입 개발 (완료)
- 기술 스택: GPT-4o + FAISS + LangChain + Streamlit
- 세션 대화, 스트리밍 응답, 출처 표시, 피드백 수집

### 3주차: 고도화 (완료)
- 프롬프트 엔지니어링 (역할지정/CoT/Few-shot)
- 질문 유형 자동 분류 (5가지)
- 시나리오 테스트 17건, 100% 성공
- 모니터링/로깅, UX 개선

### 기술 이슈
- Chroma SQLite 버전 문제 → FAISS로 변경 → Phase 2에서 Pinecone으로 최종 전환

</details>

---

## 개발 규칙

- 각 Phase 완료 시 반드시 자기 검증 질문에 답하고 결과를 기록
- 새 기능은 반드시 테스트 코드와 함께 작성
- RAG 관련 변경은 반드시 정량 평가(RAGAS) 전후 비교 기록
- 커밋은 Phase 단위가 아니라 기능 단위로 잘게 나누기
- 법적 이슈: 네이버 크롤링 금지 (ToS 위반), KRX 실시간 시세 재배포 시 라이선스 필요

---

_Last Updated: 2026-04-09 (deploy/ 배포 데이터 추가, 테스트 208개 통과)_
