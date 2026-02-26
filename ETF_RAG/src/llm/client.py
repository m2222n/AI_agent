import os

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT, MAX_HISTORY_MESSAGES
from src.llm.prompts import build_system_prompt


class LLMError(Exception):
    """LLM 관련 기본 예외"""
    pass


class APIKeyMissingError(LLMError):
    pass


class RateLimitExceededError(LLMError):
    pass


class ConnectionFailedError(LLMError):
    pass


def get_api_key(streamlit_secrets=None) -> str:
    """Streamlit secrets 또는 환경변수에서 API 키 조회"""
    if streamlit_secrets:
        try:
            return streamlit_secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise APIKeyMissingError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return api_key


def create_client(api_key: str) -> OpenAI:
    """OpenAI 클라이언트 생성"""
    return OpenAI(api_key=api_key)


def call_llm_streaming(client: OpenAI, context, question: str,
                       chat_history: list, question_type: str = "general"):
    """
    OpenAI API 스트리밍 호출

    Returns: 스트리밍 응답 객체
    Raises: LLMError 하위 예외
    """
    system_prompt = build_system_prompt(question_type)

    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history[-MAX_HISTORY_MESSAGES:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

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
        return client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            stream=True,
            timeout=LLM_TIMEOUT
        )
    except RateLimitError:
        raise RateLimitExceededError("API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
    except APIConnectionError:
        raise ConnectionFailedError("네트워크 연결 오류가 발생했습니다. 인터넷 연결을 확인해주세요.")
    except APIError as e:
        raise LLMError(f"OpenAI API 오류: {str(e)}")
