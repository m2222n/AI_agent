import logging
import os

import tiktoken
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT, MAX_HISTORY_MESSAGES
from src.llm.prompts import build_system_prompt

logger = logging.getLogger(__name__)

# tiktoken 인코더 (GPT-4o 기준)
_encoder = None

MAX_HISTORY_TOKENS = 6000  # 대화 히스토리 최대 토큰 수


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            _encoder = tiktoken.encoding_for_model(LLM_MODEL)
        except KeyError:
            _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def _trim_history(chat_history: list, max_tokens: int = MAX_HISTORY_TOKENS) -> list:
    """대화 히스토리를 max_tokens 이내로 트리밍 (최신 메시지 우선 유지)."""
    # 먼저 MAX_HISTORY_MESSAGES로 슬라이스
    trimmed = chat_history[-MAX_HISTORY_MESSAGES:]

    # 토큰 수 계산 — 뒤에서부터(최신) 누적
    total = 0
    cutoff = 0
    for i in range(len(trimmed) - 1, -1, -1):
        msg_tokens = _count_tokens(trimmed[i].get("content", ""))
        if total + msg_tokens > max_tokens:
            cutoff = i + 1
            break
        total += msg_tokens

    if cutoff > 0:
        logger.info(f"히스토리 트리밍: {len(trimmed)}개 → {len(trimmed) - cutoff}개 (토큰 제한: {max_tokens})")

    return trimmed[cutoff:]


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

    for msg in _trim_history(chat_history):
        messages.append({"role": msg["role"], "content": msg["content"]})

    if context:
        user_message = f"""[검색된 ETF 문서]
{context}

[사용자 질문]
{question}

위 문서를 참고하여 질문에 답변해줘. 답변 시 출처를 [ETF-XXX] 형식으로 표시해."""
    else:
        user_message = f"""[시스템 알림] 질문과 관련된 ETF 문서를 찾지 못했습니다.

[사용자 질문]
{question}

보유한 ETF 데이터에서 관련 정보를 찾지 못했음을 안내하고, 추측하지 마세요.
- 사용자가 특정 ETF를 물었다면: "해당 ETF 데이터가 없습니다"라고 안내
- ETF 일반 개념 질문이라면: 간단히 개념만 설명하되, 구체적 수치나 종목 추천은 하지 마세요
- 다른 질문으로 안내해주세요 (예: "보유 중인 ETF 목록을 확인하시려면 사이드바를 참고해주세요")"""

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
