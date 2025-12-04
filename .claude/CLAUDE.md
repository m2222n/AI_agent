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
├── app.py              # 메인 애플리케이션
├── requirements.txt    # 의존성
├── README.md           # 프로젝트 설명
├── report_2week.md     # 과제 보고서
├── data/
│   └── etf_data.json   # ETF 샘플 데이터
└── logs/               # 질의응답/피드백 로그
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

### 3주차: 고도화 (진행 예정)

**멘토 참고 문서:**
1. 클라우드 배포 가이드 (AWS/GCP/Azure)
2. 프롬프트 엔지니어링 가이드

#### 3주차 구현 계획

**1. 프롬프트 엔지니어링 적용**
| 기법 | 적용 방안 |
|------|----------|
| 역할 지정 | "당신은 ETF 전문 금융 어드바이저입니다" |
| 형식 지정 | 제약조건 + 출력형식 명시 |
| CoT | 복잡한 비교 질문에 "단계별로 생각해보자" |
| Few-shot | ETF 질의응답 예시 추가 |

**2. 시나리오별 테스트**
- 단순 질문: "KODEX 200 수익률은?"
- 비교 질문: "KODEX 200과 TIGER 200 비교해줘"
- 복합 질문: "배당 많은 ETF 중 수수료 낮은 거 추천"
- Edge case: "비트코인 ETF 있어?", "주식 추천해줘"

**3. 모니터링/로깅 강화**
- 응답 시간 측정 및 로깅
- 사용자 피드백 분석
- 에러 발생률 추적

**4. (선택) 클라우드 고도화**
- AWS/GCP/Azure 배포 검토
- API Gateway 연동
- 오토스케일링 설정

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
- [ ] 2주차 PPT 작성 (웹 Claude 활용)
- [ ] 실행 화면 스크린샷 캡처
- [ ] 3주차 과제: 프롬프트 엔지니어링 적용
- [ ] 3주차 과제: 시나리오별 테스트
- [ ] 3주차 과제: 모니터링/로깅 강화

---

_Last Updated: 2024-12-05_
