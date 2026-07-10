<div align="center">

# 📈 투자 질의응답 챗봇

**한국 주식·ETF를 AI에게 물어보면, 오늘 데이터로 답하는 투자 분석 서비스**

"삼성전자 요즘 어때?", "2차전지 ETF 비교해줘", "이 종목 6개월 전망은?" 같은 질문을 하면
AI 에이전트가 알아서 필요한 데이터(실시간 시세·12년 차트·재무제표·뉴스)를 찾아 분석해 답합니다.
**ChatGPT와 달리 학습 시점이 아닌 "오늘 종가" 기준으로, 실제 KRX 전종목 데이터에 근거해서** 답하는 것이 핵심입니다.

### [👉 웹앱 바로 써보기](https://radiant-abundance-production-bdf0.up.railway.app)

별도 설치 없이 브라우저에서 바로 사용 (모바일 설치형 PWA 지원)

[![CI](https://github.com/m2222n/AI_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/m2222n/AI_agent/actions/workflows/ci.yml)
&nbsp;·&nbsp; 테스트 924개(백엔드 907 + 프론트 17) &nbsp;·&nbsp; RAG 검색 정확도 100% &nbsp;·&nbsp; AI 도구 14종

</div>

---

## 어떤 서비스인가요?

KRX(한국거래소)에 상장된 **ETF + KOSPI/KOSDAQ 주식 전종목**의 데이터를 매일 자동으로 모으고,
그 위에 **AI 에이전트(LangGraph)** 와 **검색 엔진(RAG)** 을 얹어 자연어 질문에 답하는 서비스입니다.

세 가지로 요약하면:

1. **물어보면 답한다** — 질문을 던지면 AI가 14개 분석 도구 중 알맞은 것을 스스로 골라 실행하고, 근거와 함께 답합니다.
2. **데이터가 최신이고 진짜다** — 매일 18:30 KRX 종가를 자동 수집하고, 장중에는 한국투자증권 실시간 시세를 씁니다. 추정이 아니라 실제 숫자로 답합니다.
3. **혼자서도 굴러간다** — 수집·검증·배포가 전부 자동화돼 있어, 내 컴퓨터를 꺼둬도 매일 데이터가 갱신되고 누구나 URL로 접속해 씁니다.

### ChatGPT로는 안 되는 것

| | 이 서비스 | ChatGPT |
|---|---|---|
| **데이터 기준일** | 오늘 종가 (매일 자동 수집) | 학습 시점 (보통 수개월 전) |
| **종목 범위** | ETF + KOSPI/KOSDAQ 주식 전종목 | 일부 유명 종목만 |
| **답변 근거** | 실제 종가·NAV·PER·수익률 숫자 | 추정 → 할루시네이션 위험 |
| **과거 데이터** | 12년 일별 (2014~, 880만 행) | 없음 |
| **장중 시세** | 한국투자증권 실시간 (현재가/호가/체결) | 불가 |
| **기술적 분석** | 11개 지표 + 차트, 기간별 수익률 | 불가 |
| **재무제표** | 분기 실적 (매출/영업이익/마진/YoY) | 불가 |
| **가격 전망** | 4개 모델 종합 (통계·머신러닝·시계열) | 불가 |
| **가상투자** | 1억 가상자금으로 모의 매매 + 랭킹 | 불가 |

---

## 주요 기능

### 🤖 질문하면 알아서 분석하는 AI 에이전트

질문을 LLM이 분류해 모델(GPT-4o / GPT-4o-mini)을 자동 선택하고,
아래 14개 도구 중 필요한 것을 골라(여러 개 동시에도) 실행한 뒤,
CoV(검증 단계)로 답이 데이터와 맞는지 한 번 더 확인하고 답합니다.

| 도구 | 하는 일 |
|------|------|
| `search_etf` / `search_stock` | 종목 검색 (하이브리드 RAG: 벡터 + 키워드 + 재정렬) |
| `compare_etfs` / `compare_stocks` | 2종목 비교 (표·차트 자동 생성, PER/PBR/시총/배당/재무) |
| `get_etf_list` / `get_stock_list` | 카테고리·키워드별 종목 목록 |
| `get_realtime_price` | 장중 실시간 시세 (한국투자증권 → yfinance → 종가 순) |
| `analyze_sector` | "삼성전자 담은 ETF 찾기" 같은 역추적 + 업종 밸류에이션 |
| `get_technical_indicators` | 기술적 지표 11종 (RSI·MACD·볼린저·일목 등) + 차트 |
| `get_stock_correlation` | 종목 간 상관관계 + 베타 |
| `simulate_portfolio` | 포트폴리오 백테스트 (수익률/최대낙폭/샤프) + 벤치마크 비교 |
| `get_financial_statements` | 분기별 재무제표 (매출/영업이익/마진/YoY, OpenDart) |
| `predict_price_outlook` | 4축 가격 전망 (기술적 + 펀더멘털 + Ridge 회귀 + Prophet) |
| `get_stock_news` | 종목 뉴스 + 감성 분석 (Google News + 로컬 금융 BERT → GPT) |

### 🔍 정확한 검색을 위한 하이브리드 RAG

질문에서 종목을 찾을 때, 의미 기반 검색과 키워드 검색을 결합해 정확도를 높입니다.

```
질문 → ① 종목명·코드 직접 매칭 → ② 벡터 검색(FAISS) + ③ 키워드 검색(Kiwi BM25)
     → ④ 결과 결합(RRF) → ⑤ 재정렬(Cohere Rerank) → ⑥ 다양성 확보(MMR) → 답변
```

종목이 매칭되면 해당 종목의 **투자설명서 PDF 청크도 함께 검색**돼(총보수·위험등급·환헤지 등 정형 데이터에 없는 정보), 시세·재무 같은 구조화 데이터와 결합해 답합니다.

검색 정확도(Hit Rate)는 평가셋 192개 질문(11개 유형)에서 **100%**.
답변 품질(RAGAS)은 Faithfulness 0.69~0.99 · Answer Relevancy 0.75 · Context Recall 0.85~1.0 수준입니다.

**측정 기반으로 개발합니다** — 감이 아니라 숫자로:
- **임베딩 모델 정량 비교** ([리포트](ETF_RAG/eval/EMBEDDING_COMPARISON.md)): OpenAI small/large vs BGE-M3(한국어 특화)를 MRR·Hit@k로 비교. dense-only에선 large가 압도(MRR 0.49→0.87)하나 **full 파이프라인은 셋 다 천장(1.0)** → 하이브리드 검색이 임베딩 약점을 메워, 현행 small 유지가 합리적이라는 결론.
- **검색 품질 CI 게이트**: `run_eval.py --min-hit-rate` + GitHub Actions `rag-eval` job — PR마다 검색 Hit Rate가 임계 미만이면 자동 실패시켜 회귀를 막습니다.

### 🖥️ 8개 탭으로 나뉜 웹앱

채팅뿐 아니라, 분석 종류별로 전용 탭을 제공합니다.

| 탭 | 내용 |
|----|------|
| **채팅** | 자유 질문 + 실시간 스트리밍 답변 + 후속 질문 추천 |
| **기술적 분석** | 11개 지표 + 차트, 기간별(6개월~10년) 수익률 |
| **재무제표** | 분기 매출/영업이익/마진/성장률 (1~10년) |
| **비교 분석** | 2종목 상대 수익률 + 밸류에이션 (1주~10년) |
| **가격 전망** | 4개 모델 종합 전망 + 시나리오·확률·리스크 |
| **섹터 분석** | 업종별 등락률·밸류에이션 + 기간 추이 차트 |
| **뉴스** | 종목 뉴스 + 감성 분석 + 일별 감성 시계열 차트 |
| **가상투자** | 1억 가상자금 모의 매매 + 평가손익·보유기간 + 비중 차트 + 거래 통계(승률·손익비) + 배당 정산 + 유저 랭킹 + 수익률 추이 + 거래내역 CSV |

- **계정(선택)**: 이메일 로그인 시 관심종목·대화이력·가상투자가 영구 저장 — 비로그인도 전 기능 사용 가능 (가입 시 나이대·성별을 분류정보로만 수집, 이름 등 식별정보는 받지 않음). 비밀번호 재설정(이메일 링크) 지원.
- **실시간 시세**: 기술 탭에서 한국투자증권 WebSocket 체결가 + 호가 10단계 표시
- **모바일**: 설치형 PWA + 반응형 + 관심종목 급등/급락 웹 푸시 알림

### 📈 매일 자동으로 갱신되는 데이터

- **자동 수집**: GitHub Actions가 매일 18:30 KST에 종가 수집 → 검증(Watchdog) → 배포까지 자동 (내 PC 불필요)
- **12년 데이터**: 2014년~현재, 전종목 880만 행 (KOSPI/KOSDAQ 시장 구분 포함)
- **재무제표**: OpenDart 전종목 백필 + 주간 자동 갱신

---

## 아키텍처

```
┌─ 프론트 (Next.js 16 + Tailwind) ──────── Railway
│   채팅(SSE 스트리밍) · 8탭 · 로그인 · PWA · 푸시
│        │ REST / SSE
┌─ 백엔드 (FastAPI) ────────────────────── Railway
│   /chat·/stream(SSE) · /tabs/* · /auth · /me/* · /push · /admin
│   └─ LangGraph 에이전트 (14 도구) + 하이브리드 RAG
│        │
├─ 유저 DB: PostgreSQL (계정·관심종목·이력·가상투자, 영구)
├─ 주가 DB: SQLite 880만 행 (영속 볼륨에 보존, GitHub Release로 배포)
└─ 외부 API: OpenAI · Cohere · 한국투자증권 KIS · OpenDart · Google News
```

| 구분 | 기술 |
|------|------|
| **AI 에이전트 / LLM** | LangGraph + Function Calling(14 도구) + 모델 라우팅(GPT-4o / 4o-mini) + CoV 검증 |
| **검색 / 임베딩** | Kiwi BM25 + FAISS + RRF + Cohere Rerank v3.5 + MMR / OpenAI text-embedding-3-small |
| **벡터 DB** | FAISS(디스크 캐싱) + Pinecone(서버리스, 자동 fallback) |
| **데이터 수집** | pykrx · 한국투자증권 KIS OpenAPI(REST+WebSocket) · yfinance(fallback) · dart-fss |
| **분석 / 예측** | 기술 지표 11종 + 상관/베타 + 포트폴리오 백테스트 + Ridge 회귀 + Prophet |
| **백엔드** | FastAPI(SSE) + JWT 인증(+비밀번호 재설정 이메일) + SQLAlchemy(PostgreSQL) + Alembic 마이그레이션 + pywebpush(VAPID) |
| **프론트엔드** | Next.js 16 + Tailwind v4 (SSE 스트리밍, PWA, 모바일 드로워) + Vitest 단위 테스트 |
| **품질 / 평가** | RAGAS 평가 · 임베딩 모델 정량 벤치(MRR/Hit@k) · 검색 품질 CI 게이트 · pytest 907개 + Vitest 17개 · GitHub Actions CI |
| **배포 / 운영** | Railway 2서비스(백엔드+프론트) + 영속 볼륨 · GitHub Actions 일일 자동 수집 · 무중단 DB 새로고침(cron) |

**운영 비용**: OpenAI API $5~17/월 + Railway Hobby $5/월~ (개인 프로젝트 규모). 한국투자증권 실시간·로컬 감성 분석은 추가 비용 0.

> 참고: 초기 프로토타입은 Streamlit으로 만들었고, 현재는 FastAPI + Next.js 기반 웹앱으로 전환했습니다.
> Streamlit 버전도 [여전히 동작](https://aiagent-5ejryv4fsnjvhrevzwn3ct.streamlit.app/)합니다.

---

## 프로젝트 구조

```
ETF_RAG/
├── api/                       # FastAPI 백엔드
│   ├── main.py                # /chat·/stream(SSE)·/health·/feedback·/stats
│   ├── tabs.py                # 데이터 탭 REST(뉴스 포함) + /tabs/price·orderbook·price/stream(SSE)
│   ├── auth.py                # JWT 인증 (가입/로그인/비번변경/닉네임·성별/탈퇴/비번재설정)
│   ├── admin.py               # 운영 cron (무중단 DB 새로고침) + 가입자/방문 통계
│   ├── email.py               # 비밀번호 재설정 메일 발송 (Resend)
│   ├── user_data.py           # 관심종목·대화이력 CRUD
│   ├── paper.py               # 가상투자 (매수/매도/포트폴리오/랭킹/스냅샷/라운드)
│   ├── push.py                # 웹 푸시 (VAPID 구독 + 관심종목 일일 알림)
│   └── db.py / models_db.py   # SQLAlchemy (User/Watchlist/ChatHistory/Push/Paper*)
├── alembic/                   # DB 스키마 마이그레이션 (Alembic, 버전 관리)
├── frontend/                  # Next.js 16 + Tailwind v4 (채팅·8탭·로그인·PWA)
├── src/
│   ├── data/
│   │   ├── database/          # SQLite CRUD 패키지 (WAL, 시장 구분 포함)
│   │   ├── collector.py / stock_collector.py / dart_collector.py   # 일배치 수집
│   │   ├── realtime.py · kis_client.py · kis_ws.py   # 시세 (KIS REST/WS → yfinance)
│   │   ├── technical/ · chart_generator/             # 지표 11종 · matplotlib 차트
│   │   ├── predictor.py · news.py · sentiment.py     # 가격 전망 · 뉴스/감성
│   │   └── db_downloader.py                          # Release DB 다운로드(무결성 검증)
│   ├── rag/                   # retriever(하이브리드) · vectorstore(FAISS/Pinecone)
│   ├── llm/                   # agent(LangGraph) · tools(14개) · prompts · classifier
│   └── ui/                    # Streamlit 프로토타입 UI
├── app.py                     # Streamlit 진입점 (프로토타입)
├── eval/                      # RAGAS 평가(192문항, 11유형) + 임베딩 비교(exp_embedding_ab.py, EMBEDDING_COMPARISON.md)
├── tests/                     # pytest 907개 (프론트는 frontend/ Vitest 17개)
├── scripts/                   # 수집·백필·배포 스크립트 + launchd
└── .github/workflows/         # daily-collect · watchdog-collect · ci
```

---

## 직접 실행하기

```bash
git clone https://github.com/m2222n/AI_agent.git
cd AI_agent/ETF_RAG
pip install -r requirements.txt
cp .env.example .env          # OPENAI_API_KEY 등 설정

# 프로토타입 (Streamlit)
streamlit run app.py

# 또는 SaaS 구성 — 백엔드 (FastAPI)
uvicorn api.main:app --port 8000
# 프론트엔드 (Next.js, 별도 터미널)
cd frontend && npm install && npm run dev   # NEXT_PUBLIC_API_BASE=http://localhost:8000

# 테스트
pytest tests/
```

**선택 설정**
- `transformers`/`torch` 설치 시 뉴스 감성을 로컬 모델(KR-FinBert-SC)로 분류(비용 0), 미설치 시 GPT로 fallback
- 한국투자증권 실시간(`KIS_*`)·웹 푸시(`VAPID_*`)는 `.env`에 키 설정 시 자동 활성화
- 배포 시 `ETF_DATA_DIR`를 영속 볼륨 경로로 지정하면 콜드스타트(DB 재다운로드)가 사라집니다

---

## 개발 로드맵

- [x] **기반 (Phase 0~4)** — 모듈 구조 → pykrx 수집·SQLite·12년 백필·자동화 → 하이브리드 검색·RAGAS → LangGraph 에이전트·모델 라우팅 → 탭 UI·실시간 시세·섹터 분석
- [x] **정량 분석 (Phase C~D)** — 기술 지표 11종·상관/베타·포트폴리오 백테스트·재무제표 + 4축 가격 전망(Ridge + Prophet) + 차트
- [x] **검색 고도화 (Phase E)** — Cohere Rerank · E2E 테스트 · RAGAS 재개선 · 뉴스 감성 · Prophet · Pinecone · 패키지 리팩토링
- [x] **SaaS 전환 (Phase F)** — FastAPI + Next.js 16 + Railway 실배포 · JWT 인증·유저별 저장 · 한국투자증권 실시간 시세(REST+WebSocket) · 웹 푸시(PWA) · 로컬 감성 분석
- [x] **가상투자** — 1억 가상자금 매매 · 평가손익 · 유저 랭킹 · 수익률 추이 · 라운드 결산
- [x] **운영 안정화** — PostgreSQL 유저 DB 영구화 · 영속 볼륨(콜드스타트 제거) · 시장 구분 기반 시세 정확화 · DB 신선도 검사 + ETF/주식 분리 수집 검증(부분 누락 자동 복구)
- [x] **평가 자동화** — 임베딩 모델 정량 벤치(BGE-M3 vs OpenAI) · 검색 품질 CI 게이트(Hit Rate 회귀 방어)
- [x] **콘텐츠 확장** — 투자설명서 PDF RAG(정형 데이터에 없는 총보수·위험등급 검색) · 뉴스 감성 시계열 전용 탭
- [x] **운영 고도화** — Alembic 스키마 마이그레이션 도입 · 무중단 DB 새로고침(cron, stale 방지) · 가입자/방문 통계 · 성별 수집 · 비밀번호 재설정(이메일) · 코드 리팩터링(중복 제거·N+1 개선, 테스트 907개)
- [ ] **모바일 앱 (Phase G)** — React Native (웹 코드 재사용), 네이티브 푸시, 오프라인 캐시

---

<div align="center">

**개인 프로젝트** · 정태민 ([@m2222n](https://github.com/m2222n))

</div>
