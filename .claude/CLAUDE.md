# AI Agent 부트캠프 프로젝트

## 프로젝트 개요
코멘토 AI 에이전트 개발 부트캠프 과제

### 주제
**LLM 기반 ETF 질의응답 챗봇 구축**

---

## 과제 진행 현황

### 1주차: 고객 요구사항 분석 (완료)
- RFP 분석 → 고객 니즈 도출
- 시스템 아키텍처 설계
- 멀티 에이전트 구조 기획
  - Core LLM Agent
  - Retriever Agent
  - Verifier Agent
  - Security Wrapper
  - Feedback Agent

**멘토 피드백 요약:**
- 정량적 지표 부족 (응답 시간 목표 등)
- 사용자 그룹별 세분화 필요
- 구체적 일정/자원 계획 필요
- 금융 규제 대응 방안 보완 필요

### 2주차: 프로토타입 개발 (완료)
**위치:** `/Users/m2222n/AI_agent/2week_etf_chatbot/`

**기술 스택:**
- LLM: OpenAI GPT-4o
- Vector DB: FAISS (Chroma는 SQLite 버전 문제로 변경)
- Framework: LangChain
- UI: Streamlit
- 데이터: ETF 샘플 8개 (JSON)

**구현 기능 (멘토 피드백 반영):**
| 피드백 항목 | 구현 |
|-------------|------|
| 세션 기반 대화 기록 | st.session_state |
| 스트리밍 응답 | OpenAI streaming API |
| API 예외 처리 | RateLimitError 등 처리 |
| 인라인 출처 표시 | [ETF-001] 형식 |
| 사용자 피드백 수집 | 👍/👎 버튼 |
| Edge Case 처리 | 검색 결과 없을 때 안내 |

**파일 구조:**
```
2week_etf_chatbot/
├── app.py                 # 메인 애플리케이션 (3주차 고도화 완료)
├── test_scenarios.py      # 시나리오 테스트 스크립트 (3주차 추가)
├── test_report.json       # 테스트 결과 JSON (3주차 추가)
├── report_2week.md        # 2주차 과제 보고서
├── report_3week.md        # 3주차 과제 보고서 (3주차 추가)
├── requirements.txt       # 의존성
├── README.md              # 프로젝트 설명
├── data/
│   └── etf_data.json      # ETF 샘플 데이터 (8개)
└── logs/
    ├── chat_log.jsonl     # 질의응답 로그 (성능 메트릭 포함)
    └── feedback_log.jsonl # 피드백 로그
```

**실행 방법:**
```bash
cd /Users/m2222n/AI_agent
source .venv/bin/activate
cd 2week_etf_chatbot
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

### 2주차 배포 완료
- **GitHub**: https://github.com/m2222n/AI_agent.git
- **Streamlit Cloud**: 배포 완료 (Settings → Secrets에 OPENAI_API_KEY 설정)

### 3주차: 고도화 (완료 ✅)

**주제:** 생성형 AI 모델 조정, 최적화, 배포 및 실무 통합

#### 구현 내용

**1. 프롬프트 엔지니어링 적용**
| 기법 | 적용 내용 |
|------|----------|
| 역할 지정 | "10년 경력의 ETF 투자 전문 어드바이저" 역할 부여 |
| 형식 지정 | #역할, #제약조건, #출력형식 구조화 |
| CoT | 비교/추천 질문에 "차근차근 단계별로" 추론 유도 |
| Few-shot | 추천 질문에 질의응답 예시 제공 |

**2. 질문 유형 자동 분류**
```python
def classify_question_type(question: str) -> str:
    # 5가지 유형: simple, compare, recommend, risk, general
    # 우선순위: 비교 > 위험 > 단순정보 > 추천 > 일반
```

**3. 시나리오별 테스트 (17건)**
| 카테고리 | 테스트 수 | 평균 응답 시간 |
|----------|----------|----------------|
| simple | 3 | 4,882ms |
| compare | 3 | 10,242ms |
| recommend | 4 | 6,291ms |
| risk | 3 | 7,802ms |
| edge_case | 4 | 3,831ms |

**4. 테스트 결과**
| 지표 | 결과 |
|------|------|
| 성공률 | 100% (17/17) |
| 질문 유형 분류 정확도 | 88.2% (15/17) |
| 평균 응답 시간 | 6,428ms |

**5. 모니터링/로깅 강화**
- 검색 시간 / LLM 시간 / 전체 시간 측정
- 사이드바 실시간 성능 통계 대시보드
- chat_log.jsonl에 성능 메트릭 포함 로깅

**6. UX 개선**
- 질문 유형 표시 (AI가 질문을 어떻게 이해했는지)
- 응답 시간 표시 (투명성)
- 스트리밍 응답 유지

---

## 기술적 이슈 해결 기록

### SQLite 버전 문제
- **문제:** Chroma가 SQLite 3.35.0 이상 필요
- **해결:** FAISS로 변경 (멘토 참고 자료에서도 FAISS 사용)

---

## 참고 자료
- 멘토 제공 노트북: `2주차_코멘토_인공지능_Backend_이름.ipynb`
- 멘토 제공 노트북: `2주차_코멘토_인공지능_Frontend_이름.ipynb`

---

## 프롬프트 엔지니어링 Quick Reference

```
# 기본 템플릿
#명령문
당신은 [역할]입니다. 아래의 제약조건을 참고하여 입력문을 출력형식에 맞게 출력해주세요.

#제약조건
- [제약1]
- [제약2]

#입력문
[질문/요청]

#출력형식
[원하는 형식]
```

**정확도 향상 키워드:**
- "차근차근 생각해보자" (78.7%)
- "단계별로 나누어 해결해보자" (72.2%)
- "논리적으로 생각해보자" (74.5%)

**핵심 기법:**
| 기법 | 설명 |
|------|------|
| Few-shot | 예시를 먼저 제시하여 패턴 학습 유도 |
| 역할 지정 | 특정 전문가 역할 부여 |
| 형식 지정 | 명령문/제약조건/입력문/출력형식 구조화 |
| CoT | "단계별로 생각해보자"로 추론 유도 |
| 멀티 페르소나 | 가상 등장인물 토론 유도 |

---

## 다음 할 일
- [x] 2주차 과제 완료 (Streamlit Cloud 배포)
- [x] 3주차 과제: 프롬프트 엔지니어링 적용
- [x] 3주차 과제: 질문 유형 자동 분류 구현
- [x] 3주차 과제: 시나리오별 테스트 (17건, 100% 성공)
- [x] 3주차 과제: 모니터링/로깅 강화
- [x] 3주차 과제: UX 개선 (성능 대시보드)
- [x] 3주차 보고서 작성 (report_3week.md)
- [ ] 2주차/3주차 PPT 작성 (웹 Claude 활용)
- [ ] 실행 화면 스크린샷 캡처

---

_Last Updated: 2024-12-05_
