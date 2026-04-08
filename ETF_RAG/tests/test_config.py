"""config.py 테스트"""

import pytest
from unittest.mock import patch

from config import is_langsmith_enabled


def test_langsmith_enabled_with_both_vars():
    """TRACING_V2=true + API_KEY 있으면 활성화"""
    with patch.dict("os.environ", {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": "lsv2_test_key",
    }):
        assert is_langsmith_enabled() is True


def test_langsmith_disabled_without_api_key():
    """API_KEY 없으면 비활성화"""
    with patch.dict("os.environ", {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": "",
    }):
        assert is_langsmith_enabled() is False


def test_langsmith_disabled_without_tracing():
    """TRACING_V2=false면 비활성화"""
    with patch.dict("os.environ", {
        "LANGCHAIN_TRACING_V2": "false",
        "LANGCHAIN_API_KEY": "lsv2_test_key",
    }):
        assert is_langsmith_enabled() is False


def test_langsmith_disabled_no_env():
    """환경변수 없으면 비활성화"""
    with patch.dict("os.environ", {}, clear=True):
        assert is_langsmith_enabled() is False
