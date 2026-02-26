# ETF RAG 챗봇 - 실서비스 프로젝트

## 프로젝트 개요

**목표:** 부트캠프 과제 수준의 ETF 챗봇을 **실제 사용 가능한 서비스**로 발전시키면서 RAG를 깊이 학습한다.

**핵심 차별점 (ChatGPT와 다른 이유):**
- 실시간 ETF 데이터 (오늘의 NAV, 수익률, 거래량)
- 공식 투자설명서/운용보고서 기반 정확한 답변 + 출처 보장
- ETF 비교 분석 특화 (표/차트 자동 생성)

**GitHub:** https://github.com/m2222n/AI_agent.git
**배포:** https://aiagent-mrfkacatrcjjpzrmjzsdfc.streamlit.app/

---

## 기술 스택

| 구분 | 현재 (부트캠프) | 목표 (서비스) |
|------|----------------|--------------|
| LLM | GPT-4o only | GPT-4o + GPT-4o-mini (비용 최적화) |
| Vector DB | FAISS (인메모리) | Chroma 또는 Qdrant |
| 데이터 | 하드코딩 JSON 8개 | 실시간 크롤링 100개+ ETF |
| 검색 | 단순 similarity_search | Hybrid Search (BM25 + Dense) + Re-ranking |
| 문서 | 없음 | ETF 투자설명서 PDF 파싱 |
| 평가 | 수동 17건 테스트 | RAGAS 자동 평가 |
| 구조 | 단일 파일 741줄 | 모듈별 분리 |
| UI | Streamlit 기본 | 커스텀 테마 + 차트/표 |

---

## 로드맵

### Phase 0: 프로젝트 리셋 [현재 단계]
> 지금의 단일 파일 구조로는 어떤 개선도 고통. 먼저 뼈대를 잡는다.

**완료:**
- [x] 프로젝트 구조 재설계 (모듈 분리)
- [x] 설정 관리 체계 (config.py, .env)
- [x] 테스트 프레임워크 세팅 (pytest, 22개 테스트 통과)
- [x] 기존 app.py 로직을 새 구조로 마이그레이션
- [x] 디렉토리 rename: `2week_etf_chatbot` → `ETF_RAG`

**현재 구조:**
```
ETF_RAG/
├── app.py                  # Streamlit 진입점 (~75줄, 오케스트레이션만)
├── config.py               # 설정/경로/상수 관리
├── requirements.txt
├── .env.example
├── src/
│   ├── data/
│   │   ├── loader.py       # load_etf_data(), create_documents()
│   │   └── etf_data.json   # (Phase 1에서 크롤러로 대체)
│   ├── rag/
│   │   ├── vectorstore.py  # create_vectorstore()
│   │   └── retriever.py    # retrieve_relevant_docs()
│   ├── llm/
│   │   ├── client.py       # get_api_key(), create_client(), call_llm_streaming()
│   │   ├── prompts.py      # build_system_prompt()
│   │   └── classifier.py   # classify_question_type()
│   ├── ui/
│   │   ├── sidebar.py      # render_sidebar()
│   │   ├── chat.py         # init_session_state(), render_chat_history(), process_question()
│   │   └── components.py   # render_example_questions(), render_feedback_buttons()
│   └── utils/
│       └── logging.py      # log_interaction(), log_feedback(), get_performance_stats()
├── tests/
│   ├── conftest.py         # 공유 fixture
│   ├── test_classifier.py  # 10개 테스트
│   ├── test_prompts.py     # 7개 테스트
│   └── test_data_loader.py # 5개 테스트
└── docs/
    ├── report_2week.md     # 부트캠프 과제 보고서 (아카이브)
    ├── report_3week.md
    ├── test_scenarios.py   # 기존 시나리오 테스트 (아카이브)
    └── test_report.json
```

**자기 검증:** "새 기능 추가할 때 기존 코드를 건드려야 하나?" → No. 각 모듈이 독립적.

---

### Phase 1: 진짜 데이터 확보
> 가짜 데이터로는 서비스가 아님. 이게 없으면 나머지는 전부 의미 없음.

**할 일:**
- [ ] KRX/네이버금융 ETF 데이터 크롤링 파이프라인 구축
- [ ] 국내 상장 ETF 100개+ 기본 정보 수집
- [ ] 일별 NAV/수익률/거래량 업데이트 스크립트
- [ ] ETF 투자설명서(PDF) 수집 및 파싱
- [ ] 데이터 정합성 검증 로직
- [ ] 크롤링 스케줄러 (수동 → 자동)

**자기 검증:** "내일 실제 ETF 가격이 반영되나?" → No면 실패

---

### Phase 2: RAG 파이프라인 고도화
> RAG 공부의 핵심. 여기서 배우는 게 이 프로젝트의 가장 큰 가치.

**할 일:**
- [ ] Chunking 전략 설계 (고정 길이 vs 의미 단위 vs Recursive)
- [ ] 메타데이터 태깅 (ETF ID, 카테고리, 문서 유형 등)
- [ ] Embedding 모델 비교 실험 (OpenAI vs 한국어 특화 모델)
- [ ] Vector DB 교체 (FAISS → Chroma/Qdrant)
- [ ] Hybrid Search 구현 (BM25 + Dense Vector)
- [ ] Re-ranking 적용 (Cross-encoder)
- [ ] Multi-query Retrieval (질문 변형으로 검색 품질 향상)
- [ ] 평가 체계 구축 (RAGAS: Faithfulness, Relevancy, Context Recall)
- [ ] 평가 결과 정량 기록 (변경 전후 비교)

**학습 포인트:**
| 주제 | 배울 것 |
|------|---------|
| Chunking | 문서를 어떻게 나누느냐에 따라 검색 품질이 결정됨 |
| Embedding | 한국어 금융 도메인에 어떤 모델이 최적인지 |
| Hybrid Search | Dense만으로는 키워드 매칭이 약함, BM25 보완 |
| Re-ranking | 1차 검색 후 정밀 정렬로 정확도 향상 |
| RAGAS | RAG 품질을 숫자로 증명하는 방법 |

**자기 검증:** "100개 문서에서 정확한 답을 찾는가?" → 정량 평가 없으면 실패

---

### Phase 3: LLM 응답 품질
> "ChatGPT보다 나은 점이 있나?" 에 답할 수 있어야 한다.

**할 일:**
- [ ] 질문 분류를 LLM 기반으로 전환 (키워드 → 의미 기반)
- [ ] 구조화 데이터(가격/수익률)와 비구조화 데이터(문서) 통합 응답
- [ ] Hallucination 감지/방지 로직
- [ ] 비교 질문 시 차트/표 자동 생성
- [ ] GPT-4o-mini fallback (단순 질문은 저비용 모델)
- [ ] 응답 캐싱 (동일 질문 재활용)

**자기 검증:** "ChatGPT보다 나은 점이 있나?" → 없으면 실패

---

### Phase 4: 서비스 마감
> "친구한테 URL 보내서 쓰라고 할 수 있나?" 에 부끄럽지 않아야 한다.

**할 일:**
- [ ] UI/UX 전면 개편 (커스텀 테마, 반응형)
- [ ] 에러 핸들링 완성
- [ ] 배포 인프라 정비 (환경 분리, 시크릿 관리)
- [ ] 사용자 피드백 기반 개선 루프
- [ ] README 및 프로젝트 문서화 (포트폴리오 용)

**자기 검증:** "친구한테 URL 보내서 쓰라고 할 수 있나?" → 부끄러우면 실패

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
- Chroma SQLite 버전 문제 → FAISS로 변경

</details>

---

## 개발 규칙

- 각 Phase 완료 시 반드시 자기 검증 질문에 답하고 결과를 기록
- 새 기능은 반드시 테스트 코드와 함께 작성
- RAG 관련 변경은 반드시 정량 평가(RAGAS) 전후 비교 기록
- 커밋은 Phase 단위가 아니라 기능 단위로 잘게 나누기

---

_Last Updated: 2026-02-26_
