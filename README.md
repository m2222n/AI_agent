<div align="center">

# 📈 투자 질의응답 챗봇

**LangGraph Agent + Hybrid RAG 기반 ETF/주식 투자 정보 시스템**

KRX 전종목 ETF + 주식 데이터를 기반으로, AI 에이전트가 질문에 맞는 도구를 자동 선택하여 정확한 답변을 제공합니다.

[![Streamlit](https://img.shields.io/badge/Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://aiagent-5ejryv4fsnjvhrevzwn3ct.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/Tests-421_Passed-2ea44f?style=for-the-badge)](#)
[![RAGAS](https://img.shields.io/badge/Hit_Rate-100%25-blue?style=for-the-badge)](#)

</div>

---

## ChatGPT와 뭐가 다른가?

| | 이 챗봇 | ChatGPT |
|---|---|---|
| **데이터** | KRX 오늘 기준 (매일 18:30 자동 수집) | 학습 데이터 기준 (수개월 전) |
| **종목 커버리지** | ETF 1,088 + 주식 KOSPI/KOSDAQ 전종목 | 일부 유명 종목만 |
| **정확도** | 실제 종가/NAV/PER/수익률 기반 답변 | 추정 + 할루시네이션 위험 |
| **비교 분석** | 구조화 테이블 + 차트 자동 생성 | 텍스트만 |
| **장중 시세** | yfinance 15분 지연 실시간 (장중) | 불가 |
| **보유종목 분석** | 역인덱스로 "삼성전자 담은 ETF" 즉시 답변 | 검색 불가 |
| **과거 데이터** | 12년 일별 데이터 (2014~2026, 800만 행) | 없음 |
| **기술적 분석** | MA/RSI/MACD/볼린저/골든크로스 자동 분석 | 불가 |
| **재무제표** | OpenDart 분기 실적 (매출/영업이익/마진/YoY) | 불가 |
| **포트폴리오** | 백테스트 시뮬레이션 (수익률/MDD/샤프) | 불가 |

---

## 핵심 기능

### 🤖 LangGraph 에이전트 (13개 도구)

```mermaid
graph LR
    Q[질문] --> C{LLM 분류}
    C -->|단순/일반| Mini[GPT-4o-mini]
    C -->|비교/추천/위험| Pro[GPT-4o]
    Mini --> T{도구 선택}
    Pro --> T
    T --> S1[search_etf]
    T --> S2[compare_etfs]
    T --> S3[get_etf_list]
    T --> S4[search_stock]
    T --> S5[compare_stocks]
    T --> S6[get_stock_list]
    T --> S7[get_realtime_price]
    T --> S8[analyze_sector]
    T --> S9[get_technical_indicators]
    T --> S10[get_stock_correlation]
    T --> S11[simulate_portfolio]
    T --> S12[get_financial_statements]
    T --> S13[predict_price_outlook]
    S1 & S2 & S3 & S4 & S5 & S6 & S7 & S8 & S9 & S10 & S11 & S12 & S13 --> R[답변 생성]
```

| 도구 | 기능 |
|------|------|
| `search_etf` | 하이브리드 RAG 검색 (FAISS + Kiwi BM25 + RRF + MMR) |
| `compare_etfs` | ETF 비교 분석 (표/차트 자동 생성) |
| `get_etf_list` | 카테고리별 ETF 목록 검색 |
| `search_stock` | 주식 RAG 검색 |
| `compare_stocks` | 주식 비교 분석 (PER/PBR/시가총액/배당/재무제표) |
| `get_stock_list` | 키워드 기반 주식 목록 검색 |
| `get_realtime_price` | 장중 실시간 시세 (yfinance) + 장 외 종가 fallback |
| `analyze_sector` | 종목→ETF 역인덱스 보유종목/섹터 분석 + 업종 밸류에이션 |
| `get_technical_indicators` | 기술적 지표 (MA/RSI/MACD/볼린저/골든크로스/추세) |
| `get_stock_correlation` | 종목 간 상관관계 + 베타 계수 분석 |
| `simulate_portfolio` | 포트폴리오 백테스트 (수익률/MDD/샤프/변동성) |
| `get_financial_statements` | 분기별 재무제표 (매출/영업이익/마진/YoY, OpenDart) |
| `predict_price_outlook` | 3축 가격 전망 (기술적+펀더멘털+Ridge회귀, Bootstrap CI) |

### 🔍 하이브리드 검색 파이프라인

```
질문 입력
  │
  ├─ Step 0: ETF/주식 이름·티커 직접 매칭 (score=1.0)
  ├─ Step 1: FAISS Dense 검색 (벡터 유사도, k=20)
  ├─ Step 2: Kiwi BM25 Sparse 검색 (한국어 형태소, k=20)
  ├─ Step 3: RRF 결합 (dense 70% + sparse 30%)
  ├─ Step 4: MMR 다양성 확보 (λ=0.7)
  └─ Step 5: 구조화 데이터 enrichment → 최종 답변
```

### 📊 데이터 파이프라인

- **자동 수집 (이중화)**:
  - **GitHub Actions** — 매일 18:30 KST, deploy/ JSON 갱신 → auto-commit → Streamlit Cloud 자동 재배포
  - **macOS launchd** — 매일 18:30 로컬 SQLite DB 업데이트 + 19:00 DART 재무제표 백필
- **12년 과거 데이터**: 2014~2026, ETF+주식 전종목, 800만 행 (SQLite 1.5GB)
- **재무제표**: OpenDart API 분기별 실적 (전종목 점진적 백필 중)
- **장중 시세**: yfinance 15분 지연, 5분 캐시

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| **에이전트** | LangGraph + Function Calling (13개 도구) + 모델 라우팅 |
| **LLM** | GPT-4o (복잡 질문) + GPT-4o-mini (단순 질문) |
| **임베딩** | OpenAI text-embedding-3-small |
| **Vector DB** | FAISS (인메모리) |
| **검색** | Kiwi BM25 + FAISS Dense + RRF + MMR + 이름 매칭 |
| **데이터** | pykrx (ETF 1,088 + 주식 전종목) + yfinance (장중) + dart-fss (재무제표) |
| **저장** | SQLite WAL (12년 보존, 1.5GB) + JSON fallback |
| **평가** | RAGAS (Hit Rate 100%, 162개 데이터셋) |
| **테스트** | pytest 421개 |
| **자동 수집** | GitHub Actions (deploy/ 갱신) + macOS launchd (로컬 DB + DART 백필) |
| **모니터링** | LangSmith (free tier) |
| **UI** | Streamlit (커스텀 CSS, 반응형) |
| **배포** | Streamlit Cloud |

**월 비용**: $5~17 (OpenAI API 기준, 개인 프로젝트)

---

## 평가 결과

### 검색 품질 (Hit Rate)
```
평가 데이터셋: 162개 질문 (8개 유형)
├── simple, compare, recommend, risk, general,
│   technical, correlation, portfolio
├── ETF / 주식 / 혼합 전 유형
└── 전체 Hit Rate:     100.0% (162/162)
```

**주요 성과:** force_answer 노드로 빈 응답 방지, 부분 키워드 매칭 + 접두어 매칭 + 한글 별칭으로 검색 품질 100% 달성

### RAGAS 답변 품질 (7차)
| 지표 | 값 |
|------|-----|
| Hit Rate | 100% (162/162) |
| Faithfulness | 0.411 |
| Answer Relevancy | 0.108 |
| Context Recall | 0.333 |

---

## 프로젝트 구조

```
ETF_RAG/
├── app.py                     # Streamlit 진입점
├── config.py                  # 설정/상수 관리
├── src/
│   ├── data/
│   │   ├── loader.py          # 데이터 로딩 (SQLite→JSON→fallback)
│   │   ├── database.py        # SQLite CRUD (WAL, 8 테이블)
│   │   ├── collector.py       # ETF 일배치 수집 (pykrx)
│   │   ├── stock_collector.py # 주식 일배치 수집
│   │   ├── dart_collector.py  # OpenDart 재무제표 수집 (dart-fss)
│   │   ├── realtime.py        # 장중 시세 (yfinance)
│   │   └── technical.py       # 기술적 지표 (MA/RSI/MACD/볼린저/상관계수/베타)
│   ├── rag/
│   │   ├── retriever.py       # HybridRetriever (FAISS+BM25+RRF+MMR)
│   │   └── vectorstore.py     # FAISS 벡터스토어
│   ├── llm/
│   │   ├── agent.py           # LangGraph 에이전트 (라우팅+도구+재검색)
│   │   ├── tools.py           # Function Calling 도구 12개
│   │   ├── prompts.py         # 유형별 시스템 프롬프트
│   │   └── classifier.py      # 질문 분류 (LLM + 키워드 fallback)
│   └── ui/
│       ├── chat.py            # 스트리밍 채팅 처리
│       ├── charts.py          # 비교 테이블/차트 렌더링
│       ├── sidebar.py         # 사이드바 (데이터 현황)
│       └── styles.py          # 커스텀 CSS
├── eval/                      # RAGAS 평가 (162개 질문)
├── tests/                     # pytest 421개
├── scripts/
│   ├── daily_collect.sh                # ETF+주식 일배치 수집
│   ├── collect_for_deploy.py           # GitHub Actions용 경량 수집
│   ├── backfill_historical.py          # 12년 과거 데이터 백필
│   ├── backfill_financials_runner.py   # DART 전종목 재무제표 백필
│   ├── backfill_financials.sh          # 백필 셸 래퍼 (launchd용)
│   ├── com.etfrag.daily-collect.plist  # launchd 스케줄 (18:30)
│   └── com.etfrag.dart-backfill.plist  # launchd 스케줄 (19:00, DART)
└── .github/workflows/
    └── daily-collect.yml               # GitHub Actions (18:30 KST)
```

---

## 실행 방법

### 1. 설치

```bash
git clone https://github.com/m2222n/AI_agent.git
cd AI_agent/ETF_RAG
pip install -r requirements.txt
```

### 2. 환경 변수

```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY, DART_API_KEY 설정
```

### 3. 데이터 수집 (선택)

```bash
# ETF 수집
python -m src.data.collector

# 주식 수집
python -m src.data.stock_collector

# 재무제표 수집 (OpenDart)
python -m src.data.dart_collector

# 12년 과거 데이터 백필
python scripts/backfill_historical.py --resume
```

### 4. 실행

```bash
streamlit run app.py
```

### 5. 테스트

```bash
pytest tests/ -v
```

---

## 로드맵

- [x] **Phase 0**: 프로젝트 구조 리셋 (모듈 분리)
- [x] **Phase 1**: pykrx 데이터 수집 + SQLite DB + 주식 확장 + 12년 백필 + 자동화 (launchd + GitHub Actions)
- [x] **Phase 2**: 하이브리드 검색 (FAISS + BM25 + RRF + MMR) + RAGAS 평가
- [x] **Phase 3**: LangGraph 에이전트 + 도구 + 모델 라우팅 + 프롬프트 개선
- [x] **Phase 4**: UI/UX 개편 + 에러 핸들링 + 실시간 시세 + 섹터 분석
- [x] **Phase C**: 정량 분석 — 기술적 지표 + 상관관계/베타 + 포트폴리오 시뮬레이션 + 재무제표 (OpenDart)
- [x] **Phase D**: 기술적 지표 확장 (6개 추가) + 차트 이미지 + 가격 전망 모델 (3축 Ridge회귀)
- [ ] **추후**: Pinecone + Cohere Rerank + KIS OpenAPI

---

<div align="center">

**개인 프로젝트** | 정태민 ([@m2222n](https://github.com/m2222n))

</div>
