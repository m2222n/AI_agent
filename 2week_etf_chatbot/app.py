"""
ETF 질의응답 챗봇 - 2주차 프로토타입

LLM 기반 ETF 질의응답 시스템
- RAG 파이프라인: LangChain + FAISS (Vector DB)
- LLM: OpenAI GPT-4o
- UI: Streamlit

멘토 피드백 반영사항:
1. 세션 기반 대화 기록 (st.session_state)
2. 스트리밍 응답 (실시간 답변 생성)
3. API 예외 처리 강화 (RateLimitError, APIConnectionError 등)
4. 인라인 출처 표시 ([ETF-001] 형식)
5. 사용자 피드백 수집 (좋아요/싫어요)
6. Edge Case 처리 (검색 결과 없을 때 명시적 안내)
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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
        st.info("터미널에서 다음 명령어로 설정하세요: export OPENAI_API_KEY='your-api-key'")
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
# 3. LLM 호출 함수 (스트리밍 지원)
# -------------------------------------------------------------------
def call_llm_streaming(client: OpenAI, context: str, question: str, chat_history: list):
    """
    OpenAI API 스트리밍 호출

    멘토 피드백 반영:
    - 스트리밍 응답으로 UX 개선
    - 예외 처리 강화
    - 대화 히스토리 반영
    """
    system_prompt = """너는 ETF 투자 전문 상담사야. 다음 규칙을 반드시 지켜:

1. 역할(Role): ETF 투자 정보를 정확하게 제공하는 전문가
2. 맥락(Context): 제공된 ETF 문서 정보를 기반으로 답변
3. 목표(Goal): 투자자가 ETF 상품을 이해하고 적절한 투자 결정을 내릴 수 있도록 도움
4. 제약조건(Constraint):
   - 문서에 없는 내용은 추측하지 말고 "해당 정보는 제공된 문서에 없습니다"라고 답해
   - 답변 중 특정 ETF 정보를 인용할 때는 반드시 [ETF-001] 형식으로 출처를 표시해
   - 투자 권유가 아닌 정보 제공임을 명시해
   - 한국어로 답변해
   - 3~5문단 이내로 핵심 위주로 설명해

5. 출력 형식:
   - 마지막에 "📎 참고 ETF" 섹션을 만들어 사용한 ETF ID를 bullet으로 정리해
   - 투자자 유의사항이 있으면 "⚠️ 투자 유의사항" 섹션에 포함해
"""

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
# 4. 로깅 함수
# -------------------------------------------------------------------
def log_interaction(question: str, answer: str, sources: list, feedback: str = None):
    """
    질의응답 로그 저장
    - 프롬프트 튜닝 및 서비스 개선을 위한 데이터 수집
    """
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "chat_log.jsonl")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "sources": [s["id"] for s in sources],
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

# -------------------------------------------------------------------
# 5. Streamlit UI
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
    st.caption("LLM 기반 ETF 투자 정보 검색 시스템 | 2주차 프로토타입")

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
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 답변 생성
        with st.chat_message("assistant"):
            # 1. 관련 문서 검색
            context, sources = retrieve_relevant_docs(vectorstore, question)
            st.session_state.last_sources = sources
            st.session_state.last_question = question

            # 2. LLM 스트리밍 호출
            response_stream = call_llm_streaming(
                client, context, question, st.session_state.messages
            )

            if response_stream:
                # 스트리밍 응답 표시 (멘토 피드백: 스트리밍 구현)
                answer_placeholder = st.empty()
                full_response = ""

                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        answer_placeholder.markdown(full_response + "▌")

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

                # 로그 저장
                log_interaction(question, full_response, sources)

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
