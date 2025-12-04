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

### 3주차: 고도화 (예정)
- Vector DB 영구 저장
- 실제 ETF API 연동
- 클라우드 배포
- 모니터링 대시보드

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

## 다음 할 일
- [ ] 2주차 PPT 작성 (웹 Claude 활용)
- [ ] 실행 화면 스크린샷 캡처
- [ ] 3주차 과제 확인

---

_Last Updated: 2024-12-05_
