<div align="center">

# 📈 ETF RAG 챗봇

**LLM + Hybrid Search 기반 ETF 투자 정보 질의응답 시스템**

KRX 전종목 데이터를 기반으로, 질문에 맞는 ETF 정보를 정확하게 검색하고 답변합니다.

[![Streamlit](https://img.shields.io/badge/Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://aiagent-5ejryv4fsnjvhrevzwn3ct.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/Tests-57_Passed-2ea44f?style=for-the-badge)](#)

</div>

---

## 핵심 기능

| 기능 | 설명 |
|------|------|
| **하이브리드 검색** | FAISS(벡터) + Kiwi BM25(키워드) + RRF 결합으로 정확한 검색 |
| **ETF 이름 직접 매칭** | 714개 ETF 이름/티커를 질문에서 직접 인식 (Hit Rate 88%) |
| **MMR 다양성 확보** | Jaccard 유사도 기반 MMR로 중복 없는 다양한 결과 |
| **질문 유형 자동 분류** | 단순정보 / 비교 / 추천 / 위험분석 / 일반 — 5가지 유형별 최적 프롬프트 |
| **할루시네이션 방지** | 검색 결과 없으면 "모른다" 응답 + RRF 최소 점수 필터링 |
| **일배치 데이터 수집** | pykrx 기반 KRX 전종목 자동 수집 (시세/NAV/수익률/보유종목/괴리율) |

## 기술 스택

| 구분 | 기술 |
|------|------|
| **LLM** | OpenAI GPT-4o |
| **임베딩** | OpenAI text-embedding-3-small |
| **Vector DB** | FAISS (인메모리) |
| **검색** | Kiwi BM25 + FAISS Dense + RRF + MMR |
| **에이전트** | LangGraph (Phase 3 진행 중) |
| **데이터** | pykrx (KRX 일배치), 714개 ETF |
| **평가** | RAGAS (Hit Rate 88%, 50개 데이터셋) |
| **UI** | Streamlit |
| **배포** | Streamlit Cloud |

## 검색 파이프라인

```
질문 입력
  │
  ├─ Step 0: ETF 이름/티커 직접 매칭 (score=1.0)
  │          "KODEX 200" → 069500 즉시 매핑
  │
  ├─ Step 1: FAISS Dense 검색 (벡터 유사도, k=20)
  ├─ Step 2: Kiwi BM25 Sparse 검색 (키워드 매칭, k=20)
  ├─ Step 3: RRF 결합 (dense 70% + sparse 30%)
  ├─ Step 4: MMR 다양성 확보 (λ=0.7)
  │
  └─ Step 5: 이름 매칭 + 하이브리드 결과 병합 → top-k 반환
```

## 프로젝트 구조

```
ETF_RAG/
├── app.py                  # Streamlit 진입점
├── config.py               # 설정/상수 관리
├── src/
│   ├── data/
│   │   ├── loader.py       # 데이터 로딩 + ETF 필터링
│   │   ├── collector.py    # pykrx 일배치 수집
│   │   └── pdf_loader.py   # PDF 파싱 파이프라인
│   ├── rag/
│   │   ├── retriever.py    # HybridRetriever (FAISS+BM25+RRF+MMR)
│   │   └── vectorstore.py  # FAISS 벡터스토어
│   ├── llm/
│   │   ├── client.py       # OpenAI API 클라이언트
│   │   ├── prompts.py      # 유형별 시스템 프롬프트
│   │   └── classifier.py   # 질문 유형 분류
│   └── ui/
│       ├── chat.py         # 채팅 처리
│       ├── sidebar.py      # 사이드바
│       └── components.py   # UI 컴포넌트
├── eval/
│   ├── eval_dataset.json   # RAGAS 평가 데이터셋 (50개)
│   └── run_eval.py         # 평가 실행 스크립트
├── tests/                  # pytest 57개
└── scripts/                # 일배치 자동화 (launchd)
```

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
# .env 파일에 OPENAI_API_KEY 설정
```

### 3. 데이터 수집 (선택)

```bash
python -m src.data.collector
# collected/ 디렉토리에 etf_data_YYYYMMDD.json 생성
```

### 4. 실행

```bash
streamlit run app.py
```

## 평가 결과

```
평가 데이터셋: 50개 질문
├── simple (18개):   Hit Rate 88%
├── compare (8개):   Hit Rate 88%
├── recommend (10개): Hit Rate 90%
├── risk (5개):      Hit Rate 80%
├── general (4개):   N/A (검색 불필요)
└── 전체 Hit Rate:   88%
```

## 로드맵

- [x] **Phase 0**: 프로젝트 구조 리셋
- [x] **Phase 1**: pykrx 데이터 수집 + 일배치 자동화
- [x] **Phase 2**: 하이브리드 검색 (FAISS + BM25 + RRF + MMR) + RAGAS 평가
- [ ] **Phase 3**: LangGraph 에이전트 + Function Calling + 모델 라우팅 ← 진행 중
- [ ] **Phase 4**: 서비스 마감 + UI/UX + 모니터링

---

<div align="center">

**개인 프로젝트** | 정태민 ([@m2222n](https://github.com/m2222n))

</div>
