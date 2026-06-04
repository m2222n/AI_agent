"""LLM 클라이언트 테스트 (OpenAI API 호출 없이 순수 로직 검증)

get_api_key 우선순위, 히스토리 트리밍, 토큰 카운팅, 예외 분류를 모킹으로 검증.
"""
import os
from unittest.mock import patch, MagicMock

import pytest
from openai import APIError, RateLimitError, APIConnectionError

from src.llm.client import (
    get_api_key,
    create_client,
    call_llm_streaming,
    _trim_history,
    _count_tokens,
    APIKeyMissingError,
    RateLimitExceededError,
    ConnectionFailedError,
    LLMError,
)
from config import MAX_HISTORY_MESSAGES


# --- get_api_key ---

def test_get_api_key_from_secrets():
    """Streamlit secrets 우선"""
    secrets = {"OPENAI_API_KEY": "sk-from-secrets"}
    assert get_api_key(secrets) == "sk-from-secrets"


def test_get_api_key_from_env(monkeypatch):
    """secrets 없으면 환경변수 fallback"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert get_api_key(None) == "sk-from-env"


def test_get_api_key_secrets_missing_key_falls_back_to_env(monkeypatch):
    """secrets 객체는 있지만 키가 없으면 환경변수로 fallback"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    empty_secrets = {}  # KeyError 발생 → env fallback
    assert get_api_key(empty_secrets) == "sk-from-env"


def test_get_api_key_missing_raises(monkeypatch):
    """어디에도 키 없으면 APIKeyMissingError"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(APIKeyMissingError):
        get_api_key(None)


# --- _count_tokens ---

def test_count_tokens_nonempty():
    assert _count_tokens("안녕하세요") > 0


def test_count_tokens_empty():
    assert _count_tokens("") == 0


# --- _trim_history ---

def _msgs(n, content="안녕"):
    return [{"role": "user", "content": content} for _ in range(n)]


def test_trim_history_under_limit_keeps_all():
    """짧은 메시지는 그대로 유지 (MAX_HISTORY_MESSAGES 이내)"""
    msgs = _msgs(3)
    assert len(_trim_history(msgs, max_tokens=6000)) == 3


def test_trim_history_respects_max_messages():
    """MAX_HISTORY_MESSAGES 초과분은 슬라이스"""
    msgs = _msgs(MAX_HISTORY_MESSAGES + 5)
    result = _trim_history(msgs, max_tokens=100000)
    assert len(result) <= MAX_HISTORY_MESSAGES


def test_trim_history_drops_old_when_token_exceeded():
    """토큰 초과 시 오래된(앞쪽) 메시지부터 버림"""
    # 각 메시지가 큰 토큰을 갖도록
    msgs = [{"role": "user", "content": "가" * 3000} for _ in range(5)]
    result = _trim_history(msgs, max_tokens=6000)
    # 전부는 유지 못 함 (각 ~3000토큰, 6000 한도)
    assert 0 < len(result) < 5


def test_trim_history_keeps_latest_even_if_oversized():
    """회귀: 최신 메시지 하나가 max_tokens를 초과해도 통째로 비우지 않음.

    이전 버그: 최신 메시지가 한도 초과 시 cutoff=len → 빈 리스트 반환 →
    가장 최근 대화 맥락이 완전 소실.
    """
    big = {"role": "user", "content": "가" * 20000}  # ~20000 토큰
    small = {"role": "user", "content": "안녕"}
    result = _trim_history([small, big], max_tokens=6000)
    assert len(result) >= 1
    assert result[-1] is big  # 최신 메시지는 유지


def test_trim_history_empty():
    assert _trim_history([], max_tokens=6000) == []


def test_trim_history_missing_content_key():
    """content 키 없는 메시지도 크래시 없이 처리"""
    msgs = [{"role": "user"}, {"role": "user", "content": "안녕"}]
    result = _trim_history(msgs, max_tokens=6000)
    assert isinstance(result, list)


# --- create_client ---

def test_create_client_returns_openai():
    client = create_client("sk-test")
    assert client is not None


# --- call_llm_streaming 예외 분류 ---

def _fake_client(side_effect):
    client = MagicMock()
    client.chat.completions.create.side_effect = side_effect
    return client


def test_call_llm_streaming_rate_limit():
    """RateLimitError → RateLimitExceededError"""
    err = RateLimitError("rate", response=MagicMock(status_code=429), body=None)
    client = _fake_client(err)
    with pytest.raises(RateLimitExceededError):
        call_llm_streaming(client, "ctx", "질문", [])


def test_call_llm_streaming_connection_error():
    """APIConnectionError → ConnectionFailedError"""
    err = APIConnectionError(request=MagicMock())
    client = _fake_client(err)
    with pytest.raises(ConnectionFailedError):
        call_llm_streaming(client, "ctx", "질문", [])


def test_call_llm_streaming_api_error():
    """APIError → LLMError"""
    err = APIError("boom", request=MagicMock(), body=None)
    client = _fake_client(err)
    with pytest.raises(LLMError):
        call_llm_streaming(client, "ctx", "질문", [])


def test_call_llm_streaming_success_returns_stream():
    """정상 호출 시 create() 반환값 그대로 전달 + 모델/스트림 설정 확인"""
    client = MagicMock()
    sentinel = object()
    client.chat.completions.create.return_value = sentinel
    result = call_llm_streaming(client, "ctx", "질문", [])
    assert result is sentinel
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["stream"] is True
    assert "timeout" in kwargs


def test_call_llm_streaming_no_context_uses_fallback_prompt():
    """context 없으면 '문서를 찾지 못했습니다' 안내 메시지 사용"""
    client = MagicMock()
    client.chat.completions.create.return_value = object()
    call_llm_streaming(client, "", "질문", [])
    _, kwargs = client.chat.completions.create.call_args
    user_msg = kwargs["messages"][-1]["content"]
    assert "찾지 못했" in user_msg
