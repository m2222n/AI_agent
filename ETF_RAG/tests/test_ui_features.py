"""UI 기능 테스트 — 에러 메시지, 피드백 통계, CSS"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from src.ui.chat import _get_user_error_message
from src.utils.logging import get_feedback_stats


# ── 에러 메시지 테스트 ────────────────────────────────────

def test_error_message_rate_limit():
    msg = _get_user_error_message(Exception("Rate limit 429"))
    assert "호출 한도" in msg


def test_error_message_timeout():
    msg = _get_user_error_message(Exception("Request timed out"))
    assert "시간이 초과" in msg


def test_error_message_connection():
    msg = _get_user_error_message(Exception("network error"))
    assert "네트워크" in msg


def test_error_message_auth():
    msg = _get_user_error_message(Exception("Invalid API key"))
    assert "인증" in msg


def test_error_message_generic():
    msg = _get_user_error_message(Exception("unknown"))
    assert "일시적인 오류" in msg


# ── 피드백 통계 테스트 ────────────────────────────────────

def test_feedback_stats_empty():
    """피드백 파일 없을 때 None 반환"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = __import__("pathlib").Path(tmpdir)
        with patch("src.utils.logging.LOG_DIR", tmpdir_path):
            result = get_feedback_stats()
            assert result is None


def test_feedback_stats_positive_only():
    """긍정 피드백만 있는 경우"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = __import__("pathlib").Path(tmpdir)
        feedback_file = tmpdir_path / "feedback_log.jsonl"
        feedback_file.write_text(
            json.dumps({"feedback": "positive"}) + "\n"
            + json.dumps({"feedback": "positive"}) + "\n"
        )
        with patch("src.utils.logging.LOG_DIR", tmpdir_path):
            result = get_feedback_stats()
            assert result is not None
            assert result["positive"] == 2
            assert result["negative"] == 0
            assert result["satisfaction_rate"] == 100.0


def test_feedback_stats_mixed():
    """긍정+부정 피드백 혼합"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = __import__("pathlib").Path(tmpdir)
        feedback_file = tmpdir_path / "feedback_log.jsonl"
        entries = [
            json.dumps({"feedback": "positive"}),
            json.dumps({"feedback": "negative:정보가 부정확해요"}),
            json.dumps({"feedback": "positive"}),
            json.dumps({"feedback": "negative:기타 - 느려요"}),
        ]
        feedback_file.write_text("\n".join(entries) + "\n")
        with patch("src.utils.logging.LOG_DIR", tmpdir_path):
            result = get_feedback_stats()
            assert result["total"] == 4
            assert result["positive"] == 2
            assert result["negative"] == 2
            assert result["satisfaction_rate"] == 50.0
            assert "정보가 부정확해요" in result["negative_reasons"]


def test_feedback_stats_reasons_count():
    """부정 피드백 사유별 카운트"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = __import__("pathlib").Path(tmpdir)
        feedback_file = tmpdir_path / "feedback_log.jsonl"
        entries = [
            json.dumps({"feedback": "negative:정보가 부정확해요"}),
            json.dumps({"feedback": "negative:정보가 부정확해요"}),
            json.dumps({"feedback": "negative:원하는 답변이 아니에요"}),
        ]
        feedback_file.write_text("\n".join(entries) + "\n")
        with patch("src.utils.logging.LOG_DIR", tmpdir_path):
            result = get_feedback_stats()
            assert result["negative_reasons"]["정보가 부정확해요"] == 2
            assert result["negative_reasons"]["원하는 답변이 아니에요"] == 1


# ── CSS 스타일 테스트 ─────────────────────────────────────

def test_custom_css_importable():
    """styles 모듈 import 가능"""
    from src.ui.styles import CUSTOM_CSS, inject_custom_css
    assert "<style>" in CUSTOM_CSS
    assert "block-container" in CUSTOM_CSS
    assert "@media" in CUSTOM_CSS  # 반응형


def test_custom_css_contains_mobile():
    """모바일 반응형 CSS 포함"""
    from src.ui.styles import CUSTOM_CSS
    assert "768px" in CUSTOM_CSS
