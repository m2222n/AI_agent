# 2주차 과제 보고서
## ETF 질의응답 챗봇 프로토타입

---

## 1. 프로젝트 개요

### 1.1 목적
ETF(상장지수펀드) 투자 정보를 자연어로 검색하고 답변받을 수 있는 AI 챗봇 프로토타입 개발

### 1.2 배경
- 1주차 과제에서 도출한 고객 요구사항 기반
- ETF 문의 증가에 따른 24시간 자동 응답 시스템 필요성
- RAG(Retrieval-Augmented Generation) 기술을 활용한 정확한 정보 제공

### 1.3 타겟 사용자
- **일반 투자자**: ETF 상품 정보, 투자 전략 문의
- **내부 상담 직원**: 고객 응대 시 빠른 정보 검색

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      사용자 인터페이스                        │
│                    (Streamlit Web UI)                       │
└───────────────────────────┬─────────────────────────────────┘
                            │ 질문 입력
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     RAG 파이프라인                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Retriever  │ → │   Context   │ → │  LLM Agent  │      │
│  │  (Chroma)   │    │   Builder   │    │  (GPT-4o)   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         ↑                                    │              │
│         │                                    ▼              │
│  ┌─────────────┐                      ┌─────────────┐      │
│  │  Embedding  │                      │  Streaming  │      │
│  │  (OpenAI)   │                      │  Response   │      │
│  └─────────────┘                      └─────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      데이터 레이어                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  ETF Data   │    │  Chat Log   │    │  Feedback   │      │
│  │   (JSON)    │    │  (JSONL)    │    │   (JSONL)   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 기술 스택 선정 근거

| 기술 | 선정 이유 |
|------|-----------|
| **GPT-4o** | 한국어 이해도 우수, 금융 도메인 지식, 스트리밍 지원 |
| **Chroma** | 로컬 환경에서 무료 사용, LangChain과 높은 호환성, 빠른 프로토타이핑 |
| **LangChain** | RAG 파이프라인 구축 표준, 풍부한 생태계, 확장성 |
| **Streamlit** | 빠른 UI 개발, Python 친화적, 무료 배포 지원 |
| **OpenAI Embeddings** | 한국어 임베딩 품질 우수, API 안정성 |

---

## 3. 핵심 기능 구현

### 3.1 RAG 파이프라인

#### 3.1.1 문서 임베딩 및 저장
```python
# ETF 데이터를 Document 객체로 변환 후 벡터 DB에 저장
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="etf_collection"
)
```

#### 3.1.2 의미 기반 검색
```python
def retrieve_relevant_docs(vectorstore, query: str, k: int = 3):
    results = vectorstore.similarity_search_with_score(query, k=k)
    # 유사도 점수 기반 필터링 (threshold: 1.5)
    filtered_results = [(doc, score) for doc, score in results if score < 1.5]
    ...
```

### 3.2 프롬프트 엔지니어링

R-C-G-C (Role-Context-Goal-Constraint) 프레임워크 적용:

```python
system_prompt = """너는 ETF 투자 전문 상담사야. 다음 규칙을 반드시 지켜:

1. 역할(Role): ETF 투자 정보를 정확하게 제공하는 전문가
2. 맥락(Context): 제공된 ETF 문서 정보를 기반으로 답변
3. 목표(Goal): 투자자가 ETF 상품을 이해하고 적절한 투자 결정을 내릴 수 있도록 도움
4. 제약조건(Constraint):
   - 문서에 없는 내용은 추측하지 말고 "해당 정보는 제공된 문서에 없습니다"라고 답해
   - 답변 중 특정 ETF 정보를 인용할 때는 반드시 [ETF-001] 형식으로 출처를 표시해
   ...
"""
```

**Temperature 설정**: 0.3 (사실 기반 답변의 일관성을 위해 낮은 값 사용)

---

## 4. 멘토 피드백 반영 사항

### 4.1 세션 기반 대화 기록
```python
# Streamlit session_state 활용
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 히스토리를 LLM에 전달
for msg in chat_history[-10:]:
    messages.append({"role": msg["role"], "content": msg["content"]})
```
- 이전 대화 맥락 유지
- 후속 질문 처리 가능

### 4.2 스트리밍 응답
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    stream=True,  # 스트리밍 활성화
)

for chunk in response_stream:
    if chunk.choices[0].delta.content:
        full_response += chunk.choices[0].delta.content
        answer_placeholder.markdown(full_response + "▌")
```
- 실시간 답변 생성으로 UX 개선
- 긴 답변도 대기 없이 확인 가능

### 4.3 API 예외 처리 강화
```python
try:
    response = client.chat.completions.create(...)
except RateLimitError:
    st.error("⚠️ API 호출 한도를 초과했습니다.")
except APIConnectionError:
    st.error("⚠️ 네트워크 연결 오류가 발생했습니다.")
except APIError as e:
    st.error(f"⚠️ OpenAI API 오류: {str(e)}")
```
- 방어적 프로그래밍으로 앱 안정성 확보
- 사용자 친화적 에러 메시지 제공

### 4.4 인라인 출처 표시
프롬프트에서 출처 표시 규칙 명시:
```
- 답변 중 특정 ETF 정보를 인용할 때는 반드시 [ETF-001] 형식으로 출처를 표시해
```
- 답변의 신뢰성 향상
- 학술 논문 인용 방식 적용

### 4.5 사용자 피드백 수집
```python
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("👍 도움됨"):
        log_feedback(question, answer, "positive")
with col2:
    if st.button("👎 별로"):
        log_feedback(question, answer, "negative")
```
- 서비스 품질 모니터링
- 프롬프트 개선 데이터 수집

### 4.6 Edge Case 처리
```python
if not filtered_results:
    # 관련 문서를 찾지 못한 경우 명시적 안내
    user_message = """[시스템 알림] 질문과 직접적으로 관련된 ETF 문서를 찾지 못했습니다.
    일반적인 ETF 지식을 바탕으로 답변하되,
    "제공된 ETF 데이터에서는 관련 정보를 찾지 못했습니다"라고 먼저 안내해줘."""
```
- 검색 실패 시 투명한 안내
- 할루시네이션 방지

---

## 5. 데이터 구성

### 5.1 ETF 샘플 데이터 (8개 상품)

| 카테고리 | ETF 상품 | 위험등급 |
|----------|----------|----------|
| 국내 주식형 | KODEX 200 | 2등급 |
| 해외 주식형 | TIGER 미국S&P500, 미국나스닥100 | 2등급 |
| 섹터/테마형 | KODEX 2차전지산업 | 1등급 |
| 해외 테마형 | TIGER 차이나전기차 | 1등급 |
| 채권형 | KODEX 단기채권 | 5등급 |
| 배당형 | KODEX 고배당 | 3등급 |
| 인버스형 | KODEX 인버스 | 1등급 |

### 5.2 데이터 필드
- 기본 정보: 상품명, 티커, 카테고리, 추종지수
- 비용 정보: 총보수, NAV, AUM
- 투자 정보: 위험등급, 투자전략, 주요 보유종목
- 기타: 배당정책, 추적오차, 투자자 유의사항

---

## 6. 한계점 및 향후 고도화

### 6.1 현재 한계점

1. **데이터 한계**
   - 8개 ETF 샘플 데이터만 포함
   - 실시간 가격/수익률 정보 없음

2. **벡터 DB 휘발성**
   - 앱 재시작 시 임베딩 재생성 필요
   - 메모리 기반 저장

3. **보안 고려 부족**
   - 사용자 인증 없음
   - API 키 환경변수 의존

### 6.2 3주차 확장 계획

| 항목 | 구현 내용 |
|------|-----------|
| Vector DB 영구화 | Chroma persist 또는 Pinecone 전환 |
| 실제 데이터 연동 | 한국투자증권 API, 금융투자협회 API |
| 클라우드 배포 | Streamlit Cloud 또는 GCP Cloud Run |
| 모니터링 | LangSmith 또는 자체 대시보드 |
| PDF 업로드 | 사용자가 직접 ETF 문서 업로드 |

---

## 7. 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. API 키 설정
export OPENAI_API_KEY="your-api-key"

# 3. 앱 실행
streamlit run app.py
```

---

## 8. 결론

이번 2주차 프로토타입에서는 1주차에 설계한 ETF 질의응답 시스템의 핵심 기능을 구현했습니다.

**주요 성과:**
- LangChain + Chroma 기반 RAG 파이프라인 구축
- 멘토 피드백 6가지 항목 모두 반영
- 확장 가능한 모듈형 아키텍처 설계

**다음 단계:**
- 3주차에서 실제 ETF API 연동 및 클라우드 배포 진행
- 금융 규제(할루시네이션 방지, 출처 명시) 대응 강화
- 성능 모니터링 및 지속적 개선 체계 구축

---

*작성일: 2024년 12월*
*작성자: 정태민*
