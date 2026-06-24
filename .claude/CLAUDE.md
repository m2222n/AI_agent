# ETF RAG 챗봇 - 실서비스 프로젝트

## 프로젝트 개요

**목표:** 부트캠프 과제 수준의 ETF 챗봇을 **실제 사용 가능한 서비스**로 발전시키면서 RAG + AI Agent를 깊이 학습한다.

**핵심 차별점 (ChatGPT와 다른 이유):**
- 실시간 ETF 데이터 (오늘의 NAV, 수익률, 거래량) — ChatGPT는 학습 데이터 기준, 우리는 오늘 기준
- 공식 투자설명서/운용보고서 기반 정확한 답변 + 출처 보장
- ETF 비교 분석 특화 (표/차트 자동 생성)
- Function Calling 기반 Multi-Tool Agent — 질문에 따라 도구를 자동 선택

**GitHub:** https://github.com/m2222n/AI_agent.git
**배포 (Streamlit, 프로토타입):** https://aiagent-5ejryv4fsnjvhrevzwn3ct.streamlit.app/
**배포 (SaaS, Railway — 2026-06-09 실제 배포 성공):**
- 프론트(Next.js): https://radiant-abundance-production-bdf0.up.railway.app
- 백엔드(FastAPI): https://aiagent-production-75ca.up.railway.app

---

## 기술 스택

| 구분 | 현재 (Phase 2) | 목표 (서비스) |
|------|----------------|--------------|
| LLM | GPT-4o only | GPT-4o-mini (기본) + GPT-4o (복잡 질문) — 라우팅 |
| Vector DB | **FAISS** (인메모리) | **Pinecone** (free tier, 서버리스) |
| 데이터 | **pykrx** (ETF ~1,088 + 주식 ~3,100 전종목, 12년 보존) + **yfinance** (장중 15분 지연) + **한국투자증권 OpenAPI** (REST 현재가/호가 + WebSocket 실시간 체결, PR #50~#54) | ✅ 완료 |
| 검색 | **Hybrid Search** (FAISS + Kiwi BM25, RRF + **Cohere Rerank v3.5** + MMR) | ✅ 완료 |
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
- [x] **SQLite 데이터베이스** — 영구 보존 (daily_prices/returns/stock_fundamentals), WAL 모드, 8테이블 (instruments, daily_prices, returns, holdings, collection_log, stock_fundamentals, dart_corp_codes, financials_no_data)
- [x] loader.py 4-tier 우선순위: SQLite DB → collected/ → deploy/ → 하드코딩 fallback
- [x] deploy/ 배포용 데이터 (Streamlit Cloud용, Git 추적, ~1MB)
- [x] collector.py 듀얼 라이트: JSON + SQLite 동시 저장
- [x] 데이터 정합성 검증 로직 — validate_result() 구현 완료

**1-3. 한국투자증권 OpenAPI 연동**
- [x] 한국투자증권 계좌 개설 (2026-06-10)
- [x] KIS Developers 앱 등록 + API 키(appkey/appsecret) 발급
- [x] 실시간 시세 조회 연동 (REST, PR #50) — `src/data/kis_client.py` 현재가(FHKST01010100), realtime.py가 KIS 우선→yfinance fallback. 추후 WebSocket
- [x] 에러 핸들링 패턴 적용 (timeout, OAuth 토큰 디스크 캐시/재발급 분당 1회 제한 회피, rt_cd/HTTP 오류 시 None→fallback)

**1-4. 수집 자동화** ✅ 완료
- [x] 일배치 셸 스크립트 (`scripts/daily_collect.sh`) — 수집 + 로깅 + 정리
- [x] macOS launchd plist (`scripts/com.etfrag.daily-collect.plist`) — 매일 18:30 자동 실행 (로컬 SQLite DB 업데이트)
- [x] **GitHub Actions** (`.github/workflows/daily-collect.yml`) — 매일 18:30 KST, deploy/ JSON + SQLite DB 갱신 → auto-commit/push → Streamlit Cloud 자동 재배포
- [x] `scripts/collect_for_deploy.py` — GitHub Actions용 경량 수집 (SQLite 없이 JSON만, 재무제표는 월요일만)
- [x] `scripts/collect_full.py` — GitHub Actions용 통합 수집 (deploy JSON + SQLite DB 동시 갱신, KST 타임존, 월요일 재무제표 갱신)
- [x] `scripts/upload_db_to_release.sh` — 로컬 DB → GitHub Release asset 초기 업로드 (zstd 압축)
- [x] GitHub Release `db-latest` — SQLite DB를 Release asset으로 관리 (Mac 없이도 DB 갱신 가능)
- [x] 수집 결과 로깅 (`logs/collect_YYYYMMDD.log`) + 실패 시 macOS 알림
- [x] 30일 이상 된 수집 파일/로그 자동 삭제
- [x] 수집 검증 + 누락 자동 보충 (`scripts/verify_and_recover.py` — 최근 5영업일 DB 검증, 자동 재수집)
- [x] pykrx 로깅 오류 방어 (`_PykrxFilter` — `record.args`가 dict인 malformed 로그 필터링)
- [x] 12년 백필 완료 (2014-01-01 ~ 2026-04-10, ETF+주식 전종목, 800만 행, 1.5GB)
- [x] yfinance 백필 스크립트 — KRX 슬라이딩 윈도우 밖 구간(2014-01-01~04-17) 보충 (auto_adjust=False 원시 가격)
- [x] 데이터 영구 보존 정책 — prune_old_data에서 daily_prices/returns/stock_fundamentals 삭제 중단 (KRX 재수집 불가)
- [x] 수집 Watchdog 워크플로우 (`.github/workflows/watchdog-collect.yml`) — 20:30 KST 검증, 미실행 시 자동 재트리거 + GitHub Issue 알림
- [x] 수집 실패 알림 (`daily-collect.yml` notify-failure job) — 실패 시 GitHub Issue 자동 생성

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
- [x] **부분 키워드 매칭** — Kiwi 토큰화 후 per-keyword best match (최단 이름 우선, 중복 방지)
- [x] **접두어 매칭** — "TIGER 차이나전기차" → "TIGER 차이나전기차SOLACTIVE" (공백 제거 비교)
- [x] **한글 별칭 매핑** — 영문 포함 종목명의 한글 등가 별칭 등록 (POSCO→포스코, LG→엘지 등)

**2-2. PDF 문서 처리 파이프라인** ✅ 완료 (파이프라인 구축, PDF 미적용)
- [x] `pdf_loader.py` — PyPDFLoader + RecursiveCharacterTextSplitter (chunk_size=1000, overlap=100)
- [x] 파일명 기반 메타데이터 추출 ({ticker}_{name}_{doc_type}.pdf)
- [x] `create_documents(include_pdfs=True)`로 ETF 데이터 + PDF 통합
- [ ] ETF 투자설명서 PDF 수집 및 적용 (pdfs/ 디렉토리에 파일 추가 시 자동 인식)

**2-3. Vector DB 듀얼 백엔드** ✅ 완료
- [x] Pinecone 서버리스 백엔드 추가 (free tier, aws us-east-1, 자동 인덱스 생성)
- [x] FAISS/Pinecone 자동 선택 + Pinecone 실패 시 FAISS 자동 fallback
- [ ] Pinecone sparse-dense 하이브리드 검색으로 전환

**2-4. Re-ranking** ✅ 완료
- [x] **Cohere Rerank v3.5** 적용 — RRF 결합 후 cross-encoder 재정렬, MMR 전 삽입
- [x] Graceful fallback: API 키 없으면 자동 비활성화, API 오류 시 원래 순서 유지

**2-5. 평가 체계** ✅ 기본 구축 완료
- [x] RAGAS 평가 파이프라인 구축 (`eval/run_eval.py` — retrieval-only + full RAGAS 모드)
- [x] 평가 데이터셋 구축 (`eval/eval_dataset.json` — 50개 질문)
- [x] 변경 전후 정량 비교 기록 (`eval/results/` — JSON, 5회 평가)
- [x] 에이전트 전환 후 재평가: Hit Rate 88% 유지 (검색 품질 변화 없음)
- [x] 주식 질문 25개 추가 (총 75개), 주식 검색 평가 파이프라인 확장
- [x] 주식 확장 후 재평가: 전체 90.8%, ETF 88%, 주식 100%, 혼합 100%
- [x] 주식 도구 확장 + 75개 데이터셋 재평가: 전체 91.9%, ETF 88%, 주식 100%, 혼합 100%
- [x] 5차 프롬프트 개선 + min_rrf_score 필터: 전체 **95.2%**
- [x] RAGAS Full 평가 (에이전트 기반): Baseline F=0.500, AR=0.423, CR=0.336
- [x] 프롬프트 개선 후 재평가: F=0.521(+0.021), AR=0.301(-0.122), CR=0.400(+0.064)
- [x] RAGAS 평가 context에 구조화 데이터 포함: F=0.529, AR=0.341, CR=0.492(+0.156)
- [x] recommend/risk 프롬프트 데이터 근거 강화: F=0.549, F(RAG)=0.578, CR=0.469
- [x] 5차: 구조화 데이터 활용 원칙 프롬프트 추가 + enrichment 헤더 강화 + min_rrf_score 0.002→0.01 → **Hit Rate 91.9%→95.2%**
- [x] 6차: ground_truth 날짜 독립 개선 + 도구 결과 context 포함 + stratified sampling (8유형 균등) → **Hit Rate 95.5%, CR(RAG)=0.371**
- [x] 7차: eval 데이터셋 보정 (4개) + retriever 검색 개선 (부분 키워드 매칭, 접두어 매칭, 한글 별칭) → **Hit Rate 100.0% (162/162)**
- [x] 8차: RAGAS 답변 품질 재개선 — 컨텍스트 조립 강화 + 프롬프트 수치 인용 강제 + AR 한국어 역질문 프롬프트 + ground_truth 44개 보정 → **F=0.688(+0.277), AR=0.709(+0.601), CR=0.854(+0.521)**

**자기 검증:** "100개 문서에서 정확한 답을 찾는가?" → 정량 평가 없으면 실패

---

### Phase 3: 에이전트 + LLM 응답 품질 ✅ 완료
> "ChatGPT보다 나은 점이 있나?" 에 답할 수 있어야 한다.

**3-1. LangGraph 기반 에이전트** ✅ 구현 완료
- [x] LangGraph 도입 — 키워드 classifier.py → LLM 라우팅 그래프 (`agent.py`)
- [x] LLM 기반 질문 분류 (`classify_with_llm()`, 키워드 fallback 유지)
- [x] Function Calling 도구 정의 (`tools.py`) — 14개:
  - `search_etf`: 하이브리드 RAG 검색
  - `compare_etfs`: ETF 비교 분석 (개별 검색 후 병합)
  - `get_etf_list`: 카테고리별 ETF 목록 검색
  - `search_stock`: 주식 RAG 검색
  - `compare_stocks`: 주식 비교 분석 (PER/PBR/시가총액/배당)
  - `get_stock_list`: 주식 카테고리별 목록 검색
  - `get_realtime_price`: 장중 실시간 시세 (yfinance, 15분 지연) + 장 외 종가 fallback
  - `analyze_sector`: 종목→ETF 역인덱스 기반 보유종목/섹터 분석 + 밸류에이션 위치
  - `get_technical_indicators`: 기술적 지표 분석 (MA/RSI/MACD/볼린저/골든크로스)
  - `get_stock_correlation`: 종목 간 상관관계 + 베타 계수 분석
  - `simulate_portfolio`: 포트폴리오 백테스트 (수익률/MDD/샤프/변동성) + KODEX 200 벤치마크 비교 (알파/추적오차)
  - `get_financial_statements`: 분기별 재무제표 (매출/영업이익/순이익/마진/성장률, OpenDart)
  - `predict_price_outlook`: 4축 가격 전망 (기술적+펀더멘털+Ridge회귀+Prophet, EMA 피처, Bootstrap CI, 6m/1y 지원)
  - `get_stock_news`: 종목 뉴스 수집 + GPT 감성 분석 (Google News RSS, 긍정/부정/중립/혼재)
- [x] 검색 결과 부족 시 재검색 순환 구조 (Conditional Edge, 최대 2회)
- [x] 스트리밍 에이전트 (`stream_agent()` — 이벤트 기반 UI 업데이트)
- [x] 토큰 단위 스트리밍 (`stream_mode=["messages","updates"]` — AIMessageChunk 누적)
- [x] 멀티 도구 병렬 호출 (`ThreadPoolExecutor`, 2개+ 도구 동시 실행, 순서 보장)

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
- [x] 프롬프트 6차: 종목 추천/선정 원칙 추가 (다각도 분석 필수, PER 100+ 경고, 카테고리 중복 금지, 기간별 기준)
- [x] 프롬프트 6차: recommend 타입 전면 개편 (4단계 프로세스, 품질 기준 7개, 다각도 검증 체크리스트)
- [x] 프롬프트 6차: 면책 문구 순화/통일 (5곳, "투자 판단은 본인의 책임" → "추가 조사와 전문가 상담 권장")
- [x] 보유종목(상위 10개) 구조화 데이터 enrichment 추가
- [x] Hallucination 방어: CoV 검증 — LangGraph verify 노드 (도구 사용 전체 질문 대상, general만 제외)
- [x] Structured Output 적용 — Pydantic `QuestionClassification` + `with_structured_output()` (LLM 분류 JSON 강제)
- [x] FAISS 디스크 캐싱 — `save_local/load_local` + MD5 해시 기반 캐시 무효화 (냉부팅 시 임베딩 재호출 방지)
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
- [x] 탭 분리 UI (종합 채팅 / 기술적 분석 / 재무제표 / 비교 분석 / 가격 전망 / 섹터 분석) — tabs.py
- [x] 후속 질문 버튼 (on_click 콜백 + session_state 패턴)
- [x] 탭 종목 입력 자동완성 (text_input + 부분 매칭 selectbox, ~4,200종목 검색, 이름/티커 매칭)
- [x] 재무제표 탭 분기 수 동적 조정 (slider 1~최대분기, 2015년부터 전체 데이터)
- [x] 프롬프트 데이터 범위 안내 (시세 2014년~, 재무제표 2015년~, LLM 학습 데이터 기준 답변 방지)
- [x] 차트 제목/범례 겹침 수정 (subplots_adjust + fig.text 위치 조정)
- [x] 홈 버튼 왼쪽 상단 배치 ("🏠 홈으로 돌아가기", 대화 히스토리 초기화)
- [x] 기능 카드 4개 탭 네비게이션 (클릭 시 해당 탭 자동 전환, JS 주입)
- [x] 탭 UI 성능 최적화 (@st.cache_resource/cache_data, selectbox 전환, 차트/DB 캐싱)
- [x] 탭 순서 변경 (전망↔비교) + 재무제표 카드 추가 + 카드 네비게이션 JS 스코핑 수정
- [x] 단계별 로딩 spinner (3단계: DB 다운로드→데이터 로드→인덱스 구축)
- [x] 실시간 시세 카드 → 사이드바 주식 탭 전환 + 검색 포커스 (JS)
- [x] 홈 버튼 탭 리셋 (세션 초기화 + 종합채팅 탭 전환)
- [x] 차트 X축 연도 변경점 강제 표시 + 차트 제목 기준일 연도 표시
- [x] 기술 분석 days 파라미터 파이프라인 (사용자 기간 지정, 데이터 범위 안내)
- [x] 사이드바: st.metric→st.markdown (잘림 수정) + 업데이트 시간 안내
- [x] 모바일 반응형 CSS 강화 (line-height, 테이블 가로 스크롤, 768px 태블릿 + 480px 소형 폰 breakpoint, 멀티컬럼 스택)
- [x] 섹터(업종) 분석 탭 추가 (업종별 등락률/시총 차트, 업종 상세 종목 차트, 밸류에이션 요약)
- [x] 관심종목(watchlist) 기능 (사이드바 ⭐ 토글, 홈 리셋 시 보존)
- [x] 비교 탭 밸류에이션 차트 (PER/PBR/배당 side-by-side), 기술 탭 장중 시세 차트 (yfinance 15분봉)
- [x] 포트폴리오 시뮬레이션 차트 (wealth curve + drawdown), 재무제표 실적 추이 차트 (매출/이익 바 + 마진 라인)

**4-3. 데이터/분석 확장**
- [x] yfinance 장중 시세 연동 (15분 지연, 계좌 불필요, get_realtime_price 도구)
- [x] 종목→ETF 역인덱스 + 섹터 분석 (analyze_sector 도구, 보유종목 cross-reference)
- [x] KIS OpenAPI 실시간 시세 연동 (PR #50·#52·#53·#54) — REST 현재가(FHKST01010100)/호가(FHKST01010200) + WebSocket 체결(H0STCNT0 온디맨드 구독). 기술탭 PriceCard(WS→REST fallback)+OrderbookCard. 상세: CLAUDE.local.md "Phase F-2"
- [x] 포트폴리오 시뮬레이션 (Phase C-5 `simulate_portfolio` 도구 — 12년 데이터 백테스트, 수익률/MDD/샤프 + KODEX 200 벤치마크)

**4-4. 아키텍처 고도화 (→ Phase F/G로 통합)**
- [x] Pinecone 듀얼 백엔드 (FAISS+Pinecone, 자동 fallback)
- [ ] **Phase F: SaaS 전환** — FastAPI + React/Next.js + KIS 실시간 시세 + KoELECTRA 감성 분석
  - [x] **F-1 백엔드 골격** (2026-06-08): `api/` 패키지 — FastAPI `/health`·`/chat`·`/stream`(SSE). 기존 agent(`run_agent`/`stream_agent`)를 Streamlit 없이 래핑, 동기 호출은 threadpool 경유. Streamlit 앱과 병행.
  - [x] **F-4a 프론트 골격** (2026-06-08): `frontend/` Next.js 16(App Router/TS/Tailwind v4) — health 게이트 + 비스트리밍 `/chat` 채팅 UI.
  - [x] **F-4b SSE 스트리밍** (2026-06-08): `/stream` SSE 실시간 토큰 타이핑(@microsoft/fetch-event-source) + react-markdown/remark-gfm + 도구 상태줄.
  - [x] **F-4c 차트/비교표** (2026-06-08): structured_data 인라인 렌더 — technical/portfolio_chart base64 PNG, comparison_table 항목별 전치 표(+상대수익률 차트).
  - [x] **F-4d 멀티턴/마감** (2026-06-08): chat_history 멀티턴(대명사 참조) + 후속질문 칩 + localStorage 영속(텍스트만) + 대화 초기화 + 모바일 반응형. **F-4 프론트 채팅 완성.**
  - [x] **탭 REST API** (2026-06-08): `api/tabs.py` — `/tabs/{technical,financial,comparison,outlook,sector,tickers}` 엔드포인트. Streamlit 탭 함수를 Streamlit 없이 래핑(기존 함수 재사용). 테스트 670개.
  - [x] **프론트 탭(공통+기술분석)** (2026-06-08): NavBar(6탭 URL 라우트) + TickerSearch(디바운스 자동완성) + `/technical` 페이지(지표+차트).
  - [x] **프론트 탭 완성(재무/비교/전망/섹터)** (2026-06-08): 나머지 4탭 같은 패턴. **F-4 프론트 6탭(채팅+5데이터) 완성 — Streamlit 기능 패리티 달성.**
  - [x] **F-5 도커화/배포 설정** (2026-06-08): 백엔드/프론트 Dockerfile + docker-compose + DEPLOY.md(Railway) + CORS_ORIGINS env화. 실제 배포는 직접(계정/비용). 영속 볼륨(ETF_DATA_DIR)은 권장 후속으로 문서화.
  - [x] **F-1잔여 (A) JWT 인증 백엔드** (2026-06-08): 동기 SQLAlchemy 사용자 DB(stock DB와 분리, Postgres prod/sqlite dev) + bcrypt + PyJWT + `/auth/signup,login,me` + get_current_user.
  - [x] **F-1잔여 (B) 유저별 저장 CRUD** (2026-06-08): Watchlist/ChatHistory 모델 + `/me/watchlist`·`/me/history` CRUD(get_current_user 뒤, 유저 격리). 테스트 684개.
  - [x] **F-1잔여 (C) 프론트 인증 UI** (2026-06-09): lib/auth(토큰)+AuthContext + `/login`(로그인/회원가입 토글) + NavBar 로그인/로그아웃 + chat/stream Bearer 스레딩 + **로그인 시 서버 대화이력 로드/append**(비로그인은 localStorage). **로그인 선택제 — 비로그인도 전 기능 사용.** **Phase F-1 완료(인증+유저저장+프론트).** 후속: 실제 배포, PWA(저비용), F-2 KIS.
  - [x] **관심종목(watchlist) 프론트 UI** (2026-06-09): `/me/watchlist` 연결 — useWatchlist hook(낙관적 토글) + ⭐ 토글(기술탭) + 홈 관심종목 칩(→/technical?ticker=). 로그인 시에만.
  - [x] **PWA(설치형, 비용 0)** (2026-06-09): app/manifest.ts(standalone) + 아이콘(192/512) + public/sw.js(same-origin GET network-first, API 미캐시) + SW 등록(prod). 휴대폰 "홈 화면에 추가"→앱처럼 전체화면. 앱스토어 불필요. 푸시는 후속(VAPID+서버).
  - [x] **🚀 실제 Railway 배포 성공** (2026-06-09): 2서비스(백엔드 ETF_RAG/ + 프론트 ETF_RAG/frontend/) Dockerfile 배포. env: 백엔드 OPENAI_API_KEY/JWT_SECRET/CORS_ORIGINS, 프론트 NEXT_PUBLIC_API_BASE(빌드ARG). **실제 동작 확인(채팅·탭).** 배포 함정: ①fc-cache not found→fontconfig 추가(#37) ②프론트 Railway가 $PORT=8080 주입→Next standalone이 8080 listen→도메인 포트도 8080으로 맞춤(3000 아님). 무료 trial $5/30일(이후 Hobby $5/월~). 후속: F-2 KIS(신분증), 푸시, 도메인.
  - [x] **채팅 패리티 보강** (2026-06-10): Streamlit 대비 누락분 일부 — 동적 추천질문(`/tabs/movers` 급등/급락/거래대금) + 피드백(`/feedback` 익명/로그인) + 에러 재시도 버튼.
  - [x] **사이드바 신설** (2026-06-10): `/tabs/overview`(데이터현황+ETF/주식 거래대금TOP+섹터) + Sidebar(데스크톱 좌측, 종목검색, 클릭→기술분석).
  - [x] **기술 분석 탭 보강** (2026-06-10): `/tabs/intraday`(장중 15분봉) + 11개 지표 전부(스토캐스틱/일목/CCI/ADX/OBV/ATR) + 골든/데드크로스.
  - [x] **🎯 Streamlit ↔ SaaS 기능 패리티 완료** (2026-06-10): PR #42~#48 (7개). 비교탭(returns 버그수정/5기간 수익률/분기실적/기간선택) + 재무탭(1~5년 기간) + 전망탭(Prophet·통계 축 렌더 버그수정) + 5탭 데이터범위 안내문 + 기술탭 10년 + **방문자 카운터**(`/stats/visit` Supabase, Railway 백엔드 env 등록·라이브 검증 완료 누적 148) + 사이드바 업종 필터. 2회 Explore 대조로 누락분 색출. **상세는 CLAUDE.local.md "SaaS ↔ Streamlit 격차 체크리스트".**
  - [x] **F-2 KIS 실시간 시세** (2026-06-12~15, PR #50·#52·#53·#54): `kis_client`(REST 현재가 FHKST01010100/호가 FHKST01010200, OAuth 토큰 디스크캐시) + `kis_ws`(WebSocket 체결 H0STCNT0 온디맨드 구독/refcount). `/tabs/price`·`/tabs/orderbook`·`/tabs/price/stream`(SSE). 기술탭 PriceCard(WS 우선→REST 폴링 fallback)+OrderbookCard. 장중 라이브 검증. yfinance/종가 fallback 유지. 상세: CLAUDE.local.md "Phase F-2".
  - [x] **모바일 사이드바 드로워 + 반응형** (2026-06-12, PR #51): 사이드바가 lg:block이라 모바일에서 통째로 숨겨지던 격차 → ☰ 드로워(ChromeShell) + 데이터탭 그리드/줄바꿈 반응형. 상세: "Phase F-mobile".
  - [x] **웹 푸시 알림** (2026-06-15, PR #55·#56): VAPID + `PushSubscription` + SW push 핸들러 + PushToggle(구독) / `run_watchlist_alerts`(관심종목 ±5% 일일 자동발송, `/push/run-watchlist-alerts` X-Cron-Token, daily-collect 트리거). transformers 불필요. 상세: "Phase F: 웹 푸시".
  - [x] **코드 점검 라운드** (2026-06-15, PR #57): PR #50~#56 점검 → kis_ws 재연결 실버그 1건 수정(+회귀테스트). 나머지 클린.
  - [x] **F-3 로컬 감성 분석** (2026-06-16, PR #58): 뉴스 감성 분류를 로컬 `snunlp/KR-FinBert-SC`(사전학습, 선택적 설치)로 — 분류 로컬→GPT fallback, 요약은 GPT 하이브리드. torch 미설치 시 기존 GPT. 상세: "Phase F-3".
  - [x] **유저 DB 영구화 (Postgres)** (2026-06-18): Railway에 PostgreSQL 플러그인 + `DATABASE_URL=${{Postgres.DATABASE_URL}}` 연결 → 재배포해도 회원/관심종목 보존(이전엔 컨테이너 SQLite라 소실). 코드는 이미 지원, 설정만. curl 라이브 검증(가입201→탈퇴204→재로그인401).
  - [x] **계정 관리** (2026-06-18, PR #63): 비밀번호 변경(`PUT /auth/password`) + 닉네임(표시용, `PUT /auth/profile` + User.nickname) + 회원 탈퇴(`DELETE /auth/me`, 연관 watchlist/chat/push 정리). 프론트 `/account` 페이지 + NavBar ⚙️닉네임. ID찾기=이메일이 곧 ID, 비번찾기=메일인프라 후속. 테스트 +13.
  - [x] **UI 개선 4건** (2026-06-18~23): ①'티커'→'종목코드' 화면 문구(#64) ②재무제표 주식만 검색(ETF 제외)+10년+직접설정(#65) ③비교분석 기간 직접설정(#66) ④섹터 기간 추이 차트(업종 선택 후 1주~10년, 시총 상위20 가중지수, `/tabs/sector?period=`, #68). 상세: CLAUDE.local.md "UI 개선 요청 4건".
  - [x] **주가 DB malformed 자동복구** (2026-06-19, PR #67): Railway에서 `database disk image is malformed`로 주가 SQLite 로드 실패 → `ensure_db`가 `exists()`만 보고 손상 파일 재사용한 게 원인(Release 파일 자체는 정상 확인). `_is_valid_sqlite`(PRAGMA quick_check) + 손상 시 삭제·재다운로드 + 다운로드 크기 검증. 재배포로 복구 확인. 테스트 +8.
  - [x] **가상투자(모의투자) 탭** (2026-06-23, PR #69): 로그인 유저 가상 현금 1억으로 실제 시세 기반 모의투자. `api/paper.py`(`/me/paper/*`) — 4모델(계좌/보유/내역/스냅샷) + 매수(평단가 가중평균)/매도(실현손익) + 체결가=현재가(`_price_blocking` 재사용) + 포트폴리오 평가손익 + 거래내역 + 계좌 초기화 + 유저간 수익률 랭킹. 프론트 `/invest` 탭. 테스트 +15.
  - [x] **가상투자 수익률 추이 차트** (2026-06-23, PR #70): 일별 평가액 스냅샷(PaperSnapshot) 누적 → 수익률 추이 라인. 거래 직후 + `/me/paper/snapshot-all`(X-Cron-Token, daily-collect 트리거) 2경로 기록. `/me/paper/history`(차트). 테스트 +5(전체 821).
  - [x] **가상투자 계좌 초기화 = 라운드 결산** (2026-06-24, PR #71): 초기화 시 직전 라운드 성과(기간·종목별 실현+미실현 손익)를 `PaperRound`에 결산 보존 후 1억 새출발. 확인 모달("초기화" 입력)+`confirm` 서버검증. `/me/paper/rounds`(지난 성적). 프론트 "📚 지난 성적" 섹션. 테스트 +6(전체 825).
- [ ] **Phase G: 모바일 앱** — React Native (웹 70% 재사용), 푸시 알림, 오프라인 캐시
- [ ] 한국어 임베딩 모델 비교 (BGE-M3 vs text-embedding-3-small, 검색 품질 불만 시)
- [ ] KRX 시세정보 재배포 라이선스 검토 (상용화 시 필수)

**자기 검증:** "친구한테 URL 보내서 쓰라고 할 수 있나?" → 부끄러우면 실패

---

### Phase D: 기술적 지표 확장 + 차트 이미지 + 예측 응답 ✅ 완료
> "기술적 분석 차트까지 보여주는 서비스인가?" 에 답할 수 있어야 한다.

**D-1. 기술적 지표 6개 추가** ✅ 구현 완료
- [x] `_get_ohlcv()` 헬퍼 추가 (OHLCV 전체 조회, _get_closes()와 병행)
- [x] `calc_stochastic()` — %K/%D, 과매수/과매도 신호
- [x] `calc_ichimoku()` — 전환선/기준선/선행스팬A·B/치코우/구름대 상태
- [x] `calc_cci()` — CCI 지표, ±100 기준 신호
- [x] `calc_adx()` — ADX/+DI/-DI, 추세 강도 판정
- [x] `calc_obv()` — OBV + 20일 MA, 매집/분산 판정
- [x] `calc_atr()` — ATR + ATR%, 변동성 수준 판정
- [x] `get_technical_summary()` 확장 — 6개 지표 추가 반환
- [x] `get_technical_indicators` 도구 포맷 확장 (6개 지표 출력 섹션)
- [x] 프롬프트에 6개 지표 키워드 + 해석 기준 추가
- [x] 테스트 46개 추가 (6개 지표 × ~6-8 + 통합)

**D-2. 기술적 분석 차트 이미지** ✅ 구현 완료
- [x] `src/data/chart_generator.py` — matplotlib 3단 차트 (가격+MA+볼린저 / RSI / 거래량+MACD)
- [x] base64 PNG 반환 → structured_data 이벤트 → `st.image()` 렌더링
- [x] 깔끔한 디자인 (커스텀 컬러, 스파인 제거, 그리드, 주석 라벨)
- [x] 한글 폰트 fallback (AppleGothic → NanumGothic → sans-serif)
- [x] `packages.txt` — Streamlit Cloud 한글 폰트 (fonts-nanum)
- [x] `src/ui/charts.py` — `try_parse_structured_data()` 일반화 (comparison_table + technical_chart)

**D-3. 예측 질문 종합 응답** ✅ 구현 완료
- [x] 예측/전망 질문 시 `get_technical_indicators` + `get_financial_statements` 함께 사용하도록 프롬프트 안내
- [x] 응답 구조: 기술적 분석 요약 → 재무제표 요약 → 종합 판단 → 리스크 → 면책
- [x] 확정적 표현 금지, 데이터 기반 가능성 표현

---

### Phase C: 정량 분석 ✅ 완료
> "기술적 분석까지 되는 서비스인가?" 에 답할 수 있어야 한다.

**C-1. 기술적 지표** ✅ 구현 완료
- [x] `src/data/technical.py` — MA(5/20/60/120), EMA, RSI(14), MACD(12,26,9), 볼린저 밴드(20,2) 계산
- [x] 골든크로스/데드크로스 판정 (5/20, 20/60, 60/120 MA 교차 감지)
- [x] 추세 판정 (MA 정배열/역배열 기반)
- [x] `get_technical_indicators` 도구 추가 (9번째 LangGraph 도구)
- [x] 프롬프트에 기술적 지표 사용 안내 + 해석 기준 추가
- [x] 테스트 29개 (MA 6 + RSI 5 + MACD 4 + 볼린저 5 + 크로스 4 + 통합 3 + 도구 2)

**C-2. 업종별 밸류에이션 분포** ✅ 구현 완료
- [x] 업종 내 상대적 위치 (PER/PBR 백분위) — `_calc_percentile()`, `_format_valuation_position()`
- [x] analyze_sector 도구에 밸류에이션 분포 통계 추가 (PER 분포 구간, 중간값, PBR<1 저평가 수, 고배당 수)
- [x] 종목 검색 시 업종 내 밸류에이션 상대 위치 자동 표시

**C-3. 종목 간 상관관계/베타** ✅ 구현 완료
- [x] `calc_correlation()` — 일봉 수익률 기반 상관계수 (공통 날짜 매칭)
- [x] `calc_beta()` — 시장 대비 민감도 (Cov/Var, 벤치마크: KODEX 200)
- [x] `get_stock_correlation` 도구 추가 (10번째 LangGraph 도구)
- [x] 프롬프트에 상관관계/베타 해석 기준 추가
- [x] 테스트 17개 (일간수익률 3 + 상관계수 3 + 베타 3 + 밸류에이션 6 + 도구 2)

**C-4. 재무제표 데이터** ✅ 구현 완료
- [x] DB 스키마: `dart_corp_codes` + `stock_financials` 2개 테이블 + CRUD 6함수
- [x] `src/data/dart_collector.py` — OpenDart 수집 모듈 (dart-fss, CFS→OFS fallback, CLI)
- [x] `get_financial_statements` 도구 추가 (12번째 LangGraph 도구)
- [x] `_enrich_with_structured_data()` 최근 분기 실적 요약 추가
- [x] 프롬프트 재무제표 키워드 + 해석 기준 추가
- [x] 테스트 23개 (test_dart_collector.py)
- [x] DART API 키 발급 + 실제 수집 테스트 성공 (7/10 종목, 3개는 2025 미공시)
- [x] deploy 연동: `collect_for_deploy.py`에 `collect_financial_summary()` 추가 (월요일만 실행)
- [x] tools.py deploy fallback: DB 없을 때 stock_data.json의 `financial_summary` 사용
- [x] GitHub Actions: `dart-fss` + `DART_API_KEY` secret 추가
- [x] 평가 데이터셋 재무제표 질문 10개 추가 (124 → 134개)
- [x] 전종목 백필 완료 (147,048건, DART corp_code 있는 3,342종목)
- [x] 매주 월요일 DART 최신 분기 자동 수집 (`daily_collect.sh` 내 `dart_collector --max 3500`)

**C-5. 포트폴리오 시뮬레이션** ✅ 구현 완료
- [x] `simulate_portfolio()` — 백테스트 (총수익률, 연환산, MDD, 샤프, 변동성)
- [x] `simulate_portfolio` 도구 추가 (11번째 LangGraph 도구)
- [x] 자연어 파싱 ("삼성전자 50%, SK하이닉스 50%"), 기간 선택 (6m~5y)
- [x] 테스트 10개 (시뮬레이션 8 + 도구 2)

**C-6. 평가 데이터셋 확장** ✅ 완료 (2026-04-10, 04-14 추가)
- [x] 75개 → 124개 (49개 추가: technical 18, correlation 12, portfolio 12, general 7)
- [x] 124개 → 134개 (재무제표 10개 추가: simple 6, compare 2, recommend 1, general 1)
- [x] 134개 → 146개 (D-1 지표 12개 추가: 스토캐스틱/일목균형표/CCI/ADX/OBV/ATR/예측)
- [x] 146개 → 154개 (가격 전망 모델 8개 추가)
- [x] 154개 → 162개 (예측/전망 ETF+주식 8개 추가, ground_truth 안정화)
- [x] 8개 질문 유형: simple, compare, recommend, risk, general, technical, correlation, portfolio

---

## 프로젝트 구조

```
ETF_RAG/
├── app.py                  # Streamlit 진입점 (HybridRetriever 사용)
├── config.py               # 설정/경로/상수 관리 (HYBRID_SEARCH, EMBEDDING_MODEL 등)
├── requirements.txt
├── packages.txt            # Streamlit Cloud 시스템 패키지 (fonts-nanum 한글 폰트)
├── .env.example
├── src/
│   ├── data/
│   │   ├── loader.py       # load_etf_data(), create_documents(include_pdfs), _filter_etfs()
│   │   ├── database/       # SQLite CRUD 패키지 (5 서브모듈)
│   │   │   ├── __init__.py     # 18개 공개 API re-export
│   │   │   ├── _schema.py      # DB_PATH, 스키마, get_connection(), init_db(), _migrate()
│   │   │   ├── _write.py       # upsert_daily_data(), upsert_stock_data()
│   │   │   ├── _read.py        # get_latest_date/data/stock_data(), get_historical_prices(), search_instruments()
│   │   │   ├── _dart.py        # DART corp_code 매핑 + 분기 재무 CRUD (6함수)
│   │   │   └── _maintenance.py # prune_old_data(), import_json_file(), get_db_stats()
│   │   ├── pdf_loader.py   # load_pdf_documents() — PDF 파싱 + 청킹 파이프라인
│   │   ├── realtime.py     # yfinance 장중 시세 조회 (15분 지연, 5분 캐시, KRX→yfinance 티커 변환)
│   │   ├── technical/      # 기술적 지표 패키지 (5 서브모듈)
│   │   │   ├── __init__.py     # 전체 공개 API re-export
│   │   │   ├── _data.py        # DB 연결 싱글턴, TTL 캐시, _get_closes(), _get_ohlcv()
│   │   │   ├── _indicators.py  # MA/EMA/RSI/MACD/볼린저/크로스 계산
│   │   │   ├── _advanced.py    # 스토캐스틱/일목균형표/CCI/ADX/OBV/ATR
│   │   │   ├── _portfolio.py   # 상관계수/베타/포트폴리오 시뮬레이션/벤치마크
│   │   │   └── _summary.py     # get_technical_summary() 통합 지표
│   │   ├── chart_generator/    # matplotlib 차트 패키지 (5 서브모듈)
│   │   │   ├── __init__.py     # 8개 generate_* 함수 re-export
│   │   │   ├── _style.py       # 한글 폰트, 컬러 팔레트, 공통 스타일
│   │   │   ├── _series.py      # 시계열 데이터 조회 + X축 라벨 헬퍼
│   │   │   ├── technical.py    # 기술적 분석/비교/장중 차트
│   │   │   ├── financial.py    # 재무제표/밸류에이션/포트폴리오 차트
│   │   │   └── sector.py       # 섹터 개요/상세 차트
│   │   ├── collector.py    # pykrx 기반 ETF 일배치 수집 (일괄 API + 개별 PDF + SQLite 듀얼라이트)
│   │   ├── stock_collector.py # pykrx 기반 주식 일배치 수집 (KOSPI+KOSDAQ, 시세+시총+펀더멘털)
│   │   ├── dart_collector.py  # OpenDart 분기 재무제표 수집 (dart-fss, CFS→OFS fallback, CLI)
│   │   ├── predictor.py    # 4축 가격 전망 모델 (기술적+펀더멘털+Ridge회귀+Prophet+EMA피처+Bootstrap CI+6m/1y)
│   │   ├── news.py         # Google News RSS + 감성 분석 (로컬 KR-FinBert-SC→GPT fallback, 요약은 GPT)
│   │   ├── sentiment.py    # 로컬 금융 감성 분류 (KR-FinBert-SC 선택적, transformers 미설치 시 None→GPT)
│   │   ├── etf_data.json   # 하드코딩 샘플 (8개 ETF, fallback용)
│   │   ├── etf_rag.db      # SQLite DB (WAL 모드, .gitignore)
│   │   ├── collected/      # 수집 결과 JSON (.gitignore, 로컬 전용)
│   │   ├── deploy/         # 배포용 데이터 (Git 추적, Streamlit Cloud용)
│   │   └── pdfs/           # ETF 투자설명서 PDF (파일 추가 시 자동 인식)
│   ├── rag/
│   │   ├── vectorstore.py  # create_vectorstore() — FAISS/Pinecone 듀얼 백엔드 (MD5 해시 캐시, 자동 fallback)
│   │   ├── utils.py        # 공유 유틸리티 (compute_docs_hash — FAISS/BM25 캐시 무효화용)
│   │   └── retriever.py    # HybridRetriever (FAISS+Kiwi BM25+RRF+Cohere Rerank+MMR, BM25 pickle 캐시), retrieve_relevant_docs()
│   ├── llm/
│   │   ├── agent.py        # LangGraph 에이전트 (라우팅 + 도구 + 재검색 + CoV 검증 + Structured Output + force_answer + 병렬 도구 호출)
│   │   ├── tools/          # Function Calling 도구 패키지 (7 서브모듈)
│   │   │   ├── __init__.py     # 공개 API re-export + __getattr__/__setattr__ 위임
│   │   │   ├── _state.py       # 모듈 레벨 상태 (retriever, 데이터 인덱스, 역인덱스)
│   │   │   ├── _helpers.py     # 종목 검색, 필드 추출, enrichment 헬퍼
│   │   │   ├── _search.py      # search_etf/stock, compare_etfs/stocks, get_etf/stock_list
│   │   │   ├── _analysis.py    # get_realtime_price, analyze_sector, get_technical_indicators
│   │   │   ├── _quantitative.py # get_stock_correlation, simulate_portfolio, get_financial_statements
│   │   │   └── _forecast.py    # predict_price_outlook, get_stock_news
│   │   ├── client.py       # get_api_key(), create_client(), call_llm_streaming()
│   │   ├── prompts.py      # build_system_prompt()
│   │   └── classifier.py   # classify_question_type() (LLM 분류 fallback)
│   ├── ui/
│   │   ├── sidebar.py      # render_sidebar()
│   │   ├── chat.py         # process_question() (structured_data 이벤트 처리, 후속질문 on_click 콜백)
│   │   ├── charts.py       # 구조화 데이터 렌더링 (비교 테이블 + 시계열 차트 + 기술적 분석 차트)
│   │   ├── tabs.py         # 탭별 전용 UI (기술적 분석/재무제표/비교 분석/가격 전망/섹터 분석, text_input 부분매칭 자동완성)
│   │   ├── styles.py       # 커스텀 CSS (반응형, 테이블 스타일, 모바일 대응: 768px 태블릿 + 480px 소형 폰)
│   │   └── components.py   # render_example_questions(동적+기본), generate_dynamic_examples(급등/급락/거래대금), render_feedback_buttons(부정사유 수집)
├── eval/
│   ├── eval_dataset.json          # RAGAS 평가 데이터셋 (192개 질문, 11개 유형 — forecast/news/sector 포함)
│   ├── run_eval.py                # 평가 실행 스크립트 (--no-llm / full RAGAS)
│   └── results/                   # 평가 결과 JSON (eval_YYYYMMDD_HHMMSS.json)
│   └── utils/
│       ├── formatters.py   # 공통 포맷터 (format_market_cap, format_trade_value, format_number)
│       └── logging.py      # log_interaction(), log_feedback()
├── .gitignore              # Python/SQLite/IDE/OS 파일 제외 (.env, *.db, collected/, logs/ 등)
├── tests/                  # pytest 771개
├── .github/
│   └── workflows/
│       ├── daily-collect.yml          # GitHub Actions 자동 수집 (18:30 KST, deploy/ JSON + Release DB 갱신 + 실패 시 Issue)
│       ├── ci.yml                     # CI 파이프라인 (PR/push 시 pytest + coverage 자동 실행)
│       └── watchdog-collect.yml       # 수집 검증 Watchdog (20:30 KST, 미실행 시 재트리거 + Issue 알림)
├── scripts/
│   ├── daily_collect.sh               # 일배치 수집 (ETF+주식+월요일DART+검증/복구, 로컬 Mac용)
│   ├── collect_for_deploy.py          # GitHub Actions용 경량 수집 (deploy/ JSON 전용)
│   ├── backfill_historical.py         # 12년 과거 데이터 백필 (ETF+주식 전종목, --resume 지원)
│   ├── backfill_financials_runner.py  # DART 재무제표 전종목 백필 (39,000건/일, NO_DATA 구분, resume)
│   ├── verify_and_recover.py          # 수집 검증 + 누락 자동 보충 (최근 5영업일)
│   ├── backfill_financials.sh         # DART 백필 셸 스크립트 (launchd용)
│   ├── backfill_yfinance.py           # yfinance KRX 슬라이딩 윈도우 밖 백필 (2014-01-01~04-17)
│   ├── migrate_json_to_db.py          # JSON → SQLite 일회성 마이그레이션
│   ├── com.etfrag.daily-collect.plist  # macOS launchd 스케줄 (18:30)
│   └── README_cron.md                 # 자동화 설정 안내
├── scripts/                           # 프로젝트 루트 스크립트
│   ├── collect_full.py                # GitHub Actions용 통합 수집 (deploy JSON + SQLite DB)
│   └── upload_db_to_release.sh        # 로컬 DB → GitHub Release asset 업로드
└── docs/
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

## 운영 회복력 (Operational Resilience)

무료 호스팅 서비스들은 유휴 시 자동 정지되므로 다음 메커니즘으로 가동을 유지한다.

| 서비스 | 정지 정책 | 방어 메커니즘 |
|--------|----------|--------------|
| Streamlit Cloud | 7일 무활동 → 비활성화 | `keep-alive.yml` 매일 09:00 KST ping |
| Supabase (free) | 7일 무활동 → 일시정지 (90일까지 보존, 그 후 삭제) | `keep-alive.yml` REST GET ping (SUPABASE_URL/KEY secret 등록 시) |
| GitHub Actions | 60일 push 없음 → schedule cron 비활성화 | `daily-collect.yml`이 매일 push하므로 자동 유지 |
| Pinecone (선택) | 무료 인덱스 무활동 → 삭제 | FAISS fallback 코드 경로로 자동 복구 |

**pykrx 회복 패턴 (2026-06-01 학습):**
- pykrx 내부 DataFrame에 KRX ticker 중복이 발생하면 `get_etf_ticker_name(t)`이 string 대신 pandas Series 반환 → SQLite 바인딩 실패
- 모든 종목명 조회는 `_safe_get_etf_name` / `_safe_get_ticker_name`을 거치고, 내부의 `_coerce_name()`이 Series면 `iloc[0]`로 첫 값만 추출
- 새 pykrx 호출 추가 시 raw 함수 직접 호출 금지 — 반드시 safe 래퍼 경유

---

## 외부 공개 자료

- **Tistory 블로그 시리즈 8편 작성 완료** (2026-06-01): `Project/투자 AI 챗봇` 카테고리. Phase 0~E 개발 기록 + 의사결정 + 수치 변화. 향후 Phase F/G 진행 시 9편 이후 추가 예정.

---

_Last Updated: 2026-06-24 (가상투자 완성 #69~71 + 시세부족 신규종목 기술분석 자동완성 제외 #72(0192L0 류, min_days 필터). 테스트 827. 직전: 유저DB Postgres 영구화 + 계정관리 #63 + UI개선 4건 #64~66·#68 + 주가DB malformed #67. 후원=BMC+토스 보류. git push HTTP/1.1 우회 필요. 다음: 후원 / 블로그9편 / Railway 유료전환. 상세: CLAUDE.local.md)_

> ✅ **CI 액션 Node 24 대응 완료** (2026-06-08): `checkout@v4→v6`, `setup-python@v5→v6` (ci.yml + daily-collect.yml 4곳). CI green 확인. 2026-06-16 Node 24 강제 전환 대비.

> 🔬 **임베딩 A/B 실험 (small 유지 결정, 2026-06-08)**: `eval/exp_embedding_ab.py` — `text-embedding-3-small`(현행) vs `-3-large` 격리 비교. **full 파이프라인은 둘 다 100%**(직접매칭+BM25+Rerank가 천장), 순수 dense-only로 격리하면 large가 Hit@1 0.45→0.79(+0.33)·MRR 0.47→0.80로 압도하나 인덱싱 5.6배 느림(19s→106s)·비용 2배. **결론: small 유지** — 하이브리드 검색 덕에 실서비스 체감 개선 0. PDF 투자설명서 등 비정형 문서 확대로 dense 의존도가 커지면 large 재검토.

> 🚀 **Phase F 착수 — F-1 백엔드 골격 (2026-06-08)**: `api/` 패키지 (FastAPI). `/health`·`/chat`·`/stream`(SSE)으로 기존 agent를 Streamlit 없이 노출. 동기 `run_agent`/`stream_agent`는 threadpool 경유, init은 `app.py:init_all()`을 데코레이터 없이 복제(`api/deps.py`). 테스트 **655개**(+6). 단일 워커 전용. Streamlit 앱과 병행. 상세: CLAUDE.local.md.
