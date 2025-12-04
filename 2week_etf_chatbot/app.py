"""
ETF 질의응답 챗봇 - 3주차 MVP (Minimum Viable Product)

LLM 기반 ETF 질의응답 시스템
- RAG 파이프라인: LangChain + FAISS (Vector DB)
- LLM: OpenAI GPT-4o
- UI: Streamlit

[2주차] 멘토 피드백 반영:
1. 세션 기반 대화 기록 (st.session_state)
2. 스트리밍 응답 (실시간 답변 생성)
3. API 예외 처리 강화 (RateLimitError, APIConnectionError 등)
4. 인라인 출처 표시 ([ETF-001] 형식)
5. 사용자 피드백 수집 (좋아요/싫어요)
6. Edge Case 처리 (검색 결과 없을 때 명시적 안내)

[3주차] 고도화 적용:
1. 프롬프트 엔지니어링 (역할지정/형식지정/CoT/Few-shot)
2. 질문 유형별 분류 및 최적화 처리
3. 응답 시간 측정 및 성능 모니터링
4. 상세 로깅 시스템 (검색/LLM/전체 시간)
5. UX 개선 (로딩 상태, 성능 지표 표시)
"""

import os
import json
import time
from datetime import datetime
from typing import List, Tuple

import streamlit as st
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# -------------------------------------------------------------------
# 0. 환경 설정
# -------------------------------------------------------------------
def init_openai_client():
    """OpenAI 클라이언트 초기화"""
    # Streamlit Cloud secrets 또는 환경변수에서 API 키 로드
    api_key = None

    # 1. Streamlit secrets에서 먼저 확인 (Cloud 배포용)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

    # 2. 환경변수에서 확인 (로컬 개발용)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        st.info("Streamlit Cloud: Settings → Secrets에서 설정하세요.")
        st.stop()
    return OpenAI(api_key=api_key)

# -------------------------------------------------------------------
# 1. ETF 데이터 로드 및 벡터 DB 초기화
# -------------------------------------------------------------------
@st.cache_resource
def load_etf_data() -> List[dict]:
    """ETF 데이터 로드"""
    data_path = os.path.join(os.path.dirname(__file__), "data", "etf_data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def init_vector_db():
    """FAISS 벡터 DB 초기화"""
    etf_data = load_etf_data()

    # ETF 데이터를 Document 객체로 변환
    documents = []
    for etf in etf_data:
        # 모든 정보를 하나의 텍스트로 결합
        content = f"""
ETF ID: {etf['id']}
상품명: {etf['name']} ({etf['ticker']})
카테고리: {etf['category']}
추종지수: {etf['index']}
운용사: {etf['asset_manager']}
총보수: {etf['total_expense_ratio']}
순자산가치(NAV): {etf['nav']}
순자산총액(AUM): {etf['aum']}
상장일: {etf['listing_date']}
설명: {etf['description']}
위험등급: {etf['risk_level']}
투자전략: {etf['investment_strategy']}
주요 보유종목: {', '.join(etf['top_holdings'])}
배당정책: {etf['dividend_policy']}
추적오차: {etf['tracking_error']}
투자자 유의사항: {etf['investor_caution']}
"""
        doc = Document(
            page_content=content,
            metadata={"id": etf["id"], "name": etf["name"], "ticker": etf["ticker"]}
        )
        documents.append(doc)

    # OpenAI 임베딩으로 FAISS 벡터 DB 생성
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return vectorstore

# -------------------------------------------------------------------
# 2. RAG 검색 함수
# -------------------------------------------------------------------
def retrieve_relevant_docs(vectorstore, query: str, k: int = 3) -> Tuple[str, List[dict]]:
    """
    벡터 DB에서 관련 문서 검색

    Returns:
        context: 검색된 문서 내용 (문자열)
        sources: 출처 정보 리스트
    """
    results = vectorstore.similarity_search_with_score(query, k=k)

    # 유사도 점수 기준으로 필터링 (threshold: 1.5 이하만 사용)
    filtered_results = [(doc, score) for doc, score in results if score < 1.5]

    if not filtered_results:
        # Edge Case: 관련 문서를 찾지 못한 경우
        return None, []

    context_parts = []
    sources = []

    for doc, score in filtered_results:
        context_parts.append(f"[{doc.metadata['id']}] {doc.page_content}")
        sources.append({
            "id": doc.metadata["id"],
            "name": doc.metadata["name"],
            "ticker": doc.metadata["ticker"],
            "relevance_score": round(1 - score/2, 2)  # 점수를 0~1 범위로 변환
        })

    context = "\n\n---\n\n".join(context_parts)
    return context, sources

# -------------------------------------------------------------------
# 3. 질문 유형 분류 (3주차 추가)
# -------------------------------------------------------------------
def classify_question_type(question: str) -> str:
    """
    질문 유형을 분류하여 최적화된 프롬프트 적용

    유형:
    - simple: 단일 ETF 정보 질문 ("KODEX 200 수익률은?")
    - compare: 비교 질문 ("A와 B 비교해줘")
    - recommend: 추천 질문 ("배당 높은 ETF 추천")
    - risk: 위험/주의사항 질문 ("위험도", "주의")
    - general: 일반 ETF 지식 질문

    [3주차 개선] 분류 정확도 향상을 위한 우선순위 조정
    """
    question_lower = question.lower()

    # 특정 ETF 이름이 언급되면 우선 체크
    etf_names = ["kodex", "tiger", "코덱스", "타이거", "etf-"]
    has_specific_etf = any(name in question_lower for name in etf_names)

    # 1. 비교 질문 패턴 (최우선)
    compare_keywords = ["비교", "차이", "vs", "중에", "어떤게", "어떤 게", "둘 중"]
    # "와 ", "과 "는 비교 맥락에서만 사용
    compare_connectors = ["와 ", "과 "]
    has_compare_connector = any(conn in question_lower for conn in compare_connectors)
    has_compare_keyword = any(kw in question_lower for kw in compare_keywords)

    if has_compare_keyword or (has_compare_connector and has_specific_etf):
        return "compare"

    # 2. 위험/주의 질문 패턴
    risk_keywords = ["위험", "리스크", "주의", "손실", "안전", "변동성"]
    if any(kw in question_lower for kw in risk_keywords):
        return "risk"

    # 3. 특정 ETF 정보 질문 (단순 정보 요청)
    # "알려줘", "뭐야", "얼마" 등과 함께 특정 ETF 언급 시 simple
    info_keywords = ["알려줘", "뭐야", "뭐예요", "얼마", "무엇", "설명", "정보", "에 대해"]
    if has_specific_etf and any(kw in question_lower for kw in info_keywords):
        return "simple"

    # 4. 추천 질문 패턴 (조건부 추천)
    recommend_keywords = ["추천", "좋은", "괜찮은", "어떤 etf", "뭐가 좋", "골라", "선택"]
    if any(kw in question_lower for kw in recommend_keywords):
        return "recommend"

    # 5. 특정 ETF 이름만 있으면 simple
    if has_specific_etf:
        return "simple"

    return "general"

# -------------------------------------------------------------------
# 4. 프롬프트 엔지니어링 시스템 (3주차 핵심)
# -------------------------------------------------------------------
def build_system_prompt(question_type: str) -> str:
    """
    질문 유형별 최적화된 시스템 프롬프트 생성

    프롬프트 엔지니어링 기법 적용:
    - 역할 지정 (Role)
    - 형식 지정 (Format)
    - 제약조건 (Constraints)
    - Chain of Thought (CoT) - 비교/추천 질문
    - Few-shot 예시
    """

    # 기본 역할 정의 (역할 지정 기법)
    base_role = """#역할
당신은 10년 경력의 ETF 투자 전문 어드바이저입니다.
금융투자협회 인증 투자상담사 자격을 보유하고 있으며,
개인 투자자에게 ETF 상품 정보를 쉽고 정확하게 전달하는 것이 목표입니다."""

    # 공통 제약조건 (형식 지정 기법)
    base_constraints = """
#제약조건
- 제공된 ETF 문서 정보만 사용하여 답변합니다
- 문서에 없는 내용은 "해당 정보는 보유한 데이터에 없습니다"라고 안내합니다
- ETF 정보 인용 시 반드시 [ETF-XXX] 형식으로 출처를 표시합니다
- 투자 권유가 아닌 정보 제공임을 명확히 합니다
- 한국어로 친절하게 답변합니다
- 전문 용어는 쉬운 설명을 덧붙입니다"""

    # 질문 유형별 특화 프롬프트
    type_specific = {
        "simple": """
#답변 방식
단일 ETF에 대한 명확하고 구조화된 정보를 제공합니다.

#출력형식
1. **상품 개요**: 이름, 티커, 운용사
2. **핵심 정보**: 수수료, 위험등급, 배당정책
3. **투자 포인트**: 주요 특징 2-3개
4. ⚠️ **투자 유의사항**
5. 📎 **참고 ETF**: [ETF-XXX]""",

        "compare": """
#답변 방식
차근차근 단계별로 비교 분석합니다. (Chain of Thought)

먼저, 각 ETF의 핵심 특징을 파악합니다.
다음으로, 주요 항목별로 비교합니다.
마지막으로, 투자자 상황별 적합성을 정리합니다.

#출력형식
1. **비교 대상**: 각 ETF 간단 소개
2. **항목별 비교표**:
   | 항목 | ETF A | ETF B |
   |------|-------|-------|
   | 수수료 | | |
   | 위험등급 | | |
   | 배당 | | |
3. **분석 요약**: 각각의 장단점
4. **투자자 유형별 추천**
5. ⚠️ **투자 유의사항**
6. 📎 **참고 ETF**""",

        "recommend": """
#답변 방식
논리적으로 단계별 추론을 통해 추천합니다. (Chain of Thought)

우선, 사용자의 요구사항을 파악합니다.
그 다음, 조건에 맞는 ETF를 필터링합니다.
마지막으로, 적합한 순서대로 추천합니다.

#Few-shot 예시
Q: "배당 수익률 높은 ETF 추천해줘"
A: 배당 수익률을 중시하시는군요. 보유 데이터 중 배당 관련 ETF를 찾아보겠습니다.

[ETF-006] KODEX 고배당은 배당수익률 상위 50개 종목에 투자하며,
연 4~5%의 배당수익률을 기대할 수 있습니다. 분기 배당으로 정기적인
현금흐름을 원하시는 분께 적합합니다.

다만 금융주 비중이 높아 금리 변동에 민감한 점 참고해주세요.

#출력형식
1. **요구사항 파악**: 사용자가 원하는 조건
2. **추천 ETF**: 조건 부합 상품 (우선순위 순)
3. **추천 이유**: 각 상품별 장점
4. **대안**: 차선책 ETF
5. ⚠️ **투자 유의사항**
6. 📎 **참고 ETF**""",

        "risk": """
#답변 방식
투자 위험을 정확하고 균형있게 설명합니다.

#출력형식
1. **위험등급 설명**: 1~5등급 의미
2. **주요 위험 요소**: 해당 ETF의 리스크
3. **위험 관리 방안**: 분산투자 등 제안
4. ⚠️ **반드시 알아야 할 사항**
5. 📎 **참고 ETF**""",

        "general": """
#답변 방식
ETF 일반 지식을 쉽게 설명합니다.

#출력형식
1. **핵심 개념**: 질문에 대한 직접 답변
2. **상세 설명**: 추가 정보
3. **관련 ETF 예시**: 해당되는 경우
4. 📎 **참고 ETF** (해당 시)"""
    }

    return f"{base_role}\n{base_constraints}\n{type_specific.get(question_type, type_specific['general'])}"

# -------------------------------------------------------------------
# 5. LLM 호출 함수 (스트리밍 지원) - 3주차 개선
# -------------------------------------------------------------------
def call_llm_streaming(client: OpenAI, context: str, question: str, chat_history: list, question_type: str = "general"):
    """
    OpenAI API 스트리밍 호출

    [2주차] 멘토 피드백 반영:
    - 스트리밍 응답으로 UX 개선
    - 예외 처리 강화
    - 대화 히스토리 반영

    [3주차] 프롬프트 엔지니어링 적용:
    - 질문 유형별 최적화 프롬프트
    - 역할 지정 / 형식 지정 / CoT / Few-shot
    """
    # 질문 유형별 시스템 프롬프트 생성
    system_prompt = build_system_prompt(question_type)

    # 대화 히스토리를 메시지에 포함
    messages = [{"role": "system", "content": system_prompt}]

    # 최근 5개의 대화만 포함 (컨텍스트 길이 관리)
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 현재 질문과 컨텍스트
    if context:
        user_message = f"""[검색된 ETF 문서]
{context}

[사용자 질문]
{question}

위 문서를 참고하여 질문에 답변해줘. 답변 시 출처를 [ETF-XXX] 형식으로 표시해."""
    else:
        user_message = f"""[시스템 알림] 질문과 직접적으로 관련된 ETF 문서를 찾지 못했습니다.

[사용자 질문]
{question}

일반적인 ETF 지식을 바탕으로 답변하되, "제공된 ETF 데이터에서는 관련 정보를 찾지 못했습니다"라고 먼저 안내해줘."""

    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,  # 사실 기반 답변을 위해 낮은 temperature 사용
            stream=True,
            timeout=60
        )
        return response

    except RateLimitError:
        st.error("⚠️ API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
        return None
    except APIConnectionError:
        st.error("⚠️ 네트워크 연결 오류가 발생했습니다. 인터넷 연결을 확인해주세요.")
        return None
    except APIError as e:
        st.error(f"⚠️ OpenAI API 오류: {str(e)}")
        return None
    except Exception as e:
        st.error(f"⚠️ 예상치 못한 오류가 발생했습니다: {str(e)}")
        return None

# -------------------------------------------------------------------
# 6. 로깅 함수 (3주차 강화)
# -------------------------------------------------------------------
def log_interaction(
    question: str,
    answer: str,
    sources: list,
    question_type: str = "general",
    search_time: float = 0,
    llm_time: float = 0,
    total_time: float = 0,
    feedback: str = None
):
    """
    질의응답 로그 저장 (3주차: 성능 메트릭 추가)

    기록 항목:
    - 질문/답변 내용
    - 질문 유형 (simple/compare/recommend/risk/general)
    - 검색 시간, LLM 응답 시간, 전체 처리 시간
    - 사용된 ETF 출처
    - 사용자 피드백
    """
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "chat_log.jsonl")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "question_type": question_type,
        "answer": answer,
        "sources": [s["id"] for s in sources] if sources else [],
        "performance": {
            "search_time_ms": round(search_time * 1000, 2),
            "llm_time_ms": round(llm_time * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2)
        },
        "feedback": feedback
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def log_feedback(question: str, answer: str, feedback: str):
    """사용자 피드백 로그"""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    feedback_file = os.path.join(log_dir, "feedback_log.jsonl")

    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "feedback": feedback
    }

    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def get_performance_stats() -> dict:
    """
    로그에서 성능 통계 계산 (3주차 추가)
    """
    log_file = os.path.join(os.path.dirname(__file__), "logs", "chat_log.jsonl")

    if not os.path.exists(log_file):
        return None

    total_times = []
    search_times = []
    llm_times = []
    question_types = {}

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if "performance" in entry:
                    perf = entry["performance"]
                    total_times.append(perf.get("total_time_ms", 0))
                    search_times.append(perf.get("search_time_ms", 0))
                    llm_times.append(perf.get("llm_time_ms", 0))

                q_type = entry.get("question_type", "unknown")
                question_types[q_type] = question_types.get(q_type, 0) + 1

        if not total_times:
            return None

        return {
            "total_queries": len(total_times),
            "avg_total_time_ms": round(sum(total_times) / len(total_times), 2),
            "avg_search_time_ms": round(sum(search_times) / len(search_times), 2),
            "avg_llm_time_ms": round(sum(llm_times) / len(llm_times), 2),
            "question_types": question_types
        }
    except Exception:
        return None

# -------------------------------------------------------------------
# 7. Streamlit UI (3주차 개선)
# -------------------------------------------------------------------
def main():
    # 페이지 설정
    st.set_page_config(
        page_title="ETF 질의응답 챗봇",
        page_icon="📈",
        layout="wide"
    )

    # 헤더
    st.title("📈 ETF 질의응답 챗봇")
    st.caption("LLM 기반 ETF 투자 정보 검색 시스템 | 3주차 MVP")

    # 사이드바
    with st.sidebar:
        st.header("ℹ️ 서비스 안내")
        st.markdown("""
        이 챗봇은 **ETF 투자 정보**를 제공합니다.

        **주요 기능:**
        - ETF 상품 정보 검색
        - 투자 전략 설명
        - 위험도/수수료 비교
        - 배당 정책 안내

        **지원 ETF:**
        - 국내 주식형 (KODEX 200 등)
        - 해외 주식형 (S&P500, 나스닥100)
        - 섹터/테마형 (2차전지, 전기차)
        - 채권형 (단기채권)
        - 배당형, 인버스형
        """)

        st.divider()

        st.header("📊 ETF 목록")
        etf_data = load_etf_data()
        for etf in etf_data:
            with st.expander(f"{etf['name']} ({etf['ticker']})"):
                st.write(f"**카테고리:** {etf['category']}")
                st.write(f"**위험등급:** {etf['risk_level']}")
                st.write(f"**총보수:** {etf['total_expense_ratio']}")

        st.divider()

        st.warning("""
        ⚠️ **투자 유의사항**

        본 서비스는 정보 제공 목적이며,
        투자 권유가 아닙니다.
        투자 결정은 본인의 판단과
        책임 하에 이루어져야 합니다.
        """)

        # 3주차 추가: 성능 모니터링 대시보드
        st.divider()
        st.header("📊 성능 모니터링")
        stats = get_performance_stats()
        if stats:
            st.metric("총 질의 수", stats["total_queries"])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("평균 응답시간", f"{stats['avg_total_time_ms']:.0f}ms")
            with col2:
                st.metric("평균 검색시간", f"{stats['avg_search_time_ms']:.0f}ms")

            # 질문 유형 분포
            if stats["question_types"]:
                st.markdown("**질문 유형 분포:**")
                for q_type, count in stats["question_types"].items():
                    pct = count / stats["total_queries"] * 100
                    st.progress(pct / 100, text=f"{q_type}: {count}건 ({pct:.0f}%)")
        else:
            st.info("아직 통계 데이터가 없습니다.")

    # OpenAI 클라이언트 초기화
    client = init_openai_client()

    # 벡터 DB 초기화
    with st.spinner("ETF 데이터베이스 로딩 중..."):
        vectorstore = init_vector_db()

    # 세션 상태 초기화 (멘토 피드백: 세션 기반 대화 기록)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    # 대화 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 예시 질문 버튼
    if not st.session_state.messages:
        st.markdown("### 💡 이런 질문을 해보세요:")
        col1, col2 = st.columns(2)

        example_questions = [
            "KODEX 200 ETF에 대해 알려줘",
            "미국 주식에 투자하는 ETF 추천해줘",
            "2차전지 관련 ETF의 위험도는?",
            "배당 수익률이 높은 ETF는?"
        ]

        with col1:
            if st.button(example_questions[0], use_container_width=True):
                st.session_state.example_q = example_questions[0]
                st.rerun()
            if st.button(example_questions[2], use_container_width=True):
                st.session_state.example_q = example_questions[2]
                st.rerun()

        with col2:
            if st.button(example_questions[1], use_container_width=True):
                st.session_state.example_q = example_questions[1]
                st.rerun()
            if st.button(example_questions[3], use_container_width=True):
                st.session_state.example_q = example_questions[3]
                st.rerun()

    # 예시 질문 처리
    example_question = st.session_state.pop("example_q", None)

    # 채팅 입력
    user_input = st.chat_input("ETF에 대해 궁금한 점을 물어보세요...")

    # 입력 처리 (직접 입력 또는 예시 질문)
    question = example_question or user_input

    if question:
        # 전체 처리 시간 측정 시작
        total_start_time = time.time()

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 답변 생성
        with st.chat_message("assistant"):
            # [3주차] 질문 유형 분류
            question_type = classify_question_type(question)
            st.session_state.last_question_type = question_type

            # 1. 관련 문서 검색 (시간 측정)
            search_start_time = time.time()
            context, sources = retrieve_relevant_docs(vectorstore, question)
            search_time = time.time() - search_start_time

            st.session_state.last_sources = sources
            st.session_state.last_question = question

            # [3주차] 질문 유형 표시 (디버그용)
            type_labels = {
                "simple": "📝 단순 정보",
                "compare": "⚖️ 비교 분석",
                "recommend": "💡 추천",
                "risk": "⚠️ 위험 분석",
                "general": "📚 일반 질문"
            }
            st.caption(f"질문 유형: {type_labels.get(question_type, question_type)}")

            # 2. LLM 스트리밍 호출 (시간 측정)
            llm_start_time = time.time()
            response_stream = call_llm_streaming(
                client, context, question, st.session_state.messages, question_type
            )

            if response_stream:
                # 스트리밍 응답 표시
                answer_placeholder = st.empty()
                full_response = ""

                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        answer_placeholder.markdown(full_response + "▌")

                llm_time = time.time() - llm_start_time
                total_time = time.time() - total_start_time

                answer_placeholder.markdown(full_response)
                st.session_state.last_answer = full_response

                # 메시지 히스토리에 추가
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

                # 참고 ETF 표시
                if sources:
                    st.divider()
                    st.markdown("**🔍 검색된 ETF 정보:**")
                    for src in sources:
                        st.write(f"- **{src['id']}** {src['name']} ({src['ticker']}) - 관련도: {src['relevance_score']:.0%}")

                # [3주차] 성능 지표 표시
                st.caption(f"⏱️ 응답시간: {total_time*1000:.0f}ms (검색: {search_time*1000:.0f}ms, LLM: {llm_time*1000:.0f}ms)")

                # 로그 저장 (3주차: 성능 메트릭 포함)
                log_interaction(
                    question=question,
                    answer=full_response,
                    sources=sources,
                    question_type=question_type,
                    search_time=search_time,
                    llm_time=llm_time,
                    total_time=total_time
                )

    # 피드백 버튼 (멘토 피드백: 사용자 피드백 수집)
    if st.session_state.last_answer:
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 4])

        with col1:
            if st.button("👍 도움됨", key="feedback_positive"):
                log_feedback(
                    st.session_state.last_question,
                    st.session_state.last_answer,
                    "positive"
                )
                st.success("피드백 감사합니다!")

        with col2:
            if st.button("👎 별로", key="feedback_negative"):
                log_feedback(
                    st.session_state.last_question,
                    st.session_state.last_answer,
                    "negative"
                )
                st.info("개선에 참고하겠습니다!")

    # 대화 초기화 버튼
    if st.session_state.messages:
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.last_answer = ""
            st.session_state.last_question = ""
            st.rerun()

if __name__ == "__main__":
    main()
