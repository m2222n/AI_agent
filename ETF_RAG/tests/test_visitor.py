"""Supabase 방문자 카운터 테스트 (실제 HTTP 호출 없이 requests 모킹)

record_visit / get_visitor_counts의 삽입·증가·집계 분기와
미설정/네트워크 오류 시 graceful degradation을 검증.
"""
from unittest.mock import patch, MagicMock

import pytest

import src.data.visitor as visitor


@pytest.fixture(autouse=True)
def reset_config(monkeypatch):
    """모듈 레벨 캐시(_SUPABASE_URL/_KEY) 초기화 — 테스트 간 격리."""
    visitor._SUPABASE_URL = None
    visitor._SUPABASE_KEY = None
    # streamlit secrets 경로가 환경 따라 영향 주지 않도록 env로 고정
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    yield
    visitor._SUPABASE_URL = None
    visitor._SUPABASE_KEY = None


def _resp(status=200, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else []
    return r


# --- 미설정 graceful degradation ---

def test_record_visit_no_config(monkeypatch):
    """Supabase 미설정 시 (0, 0) 반환, HTTP 호출 안 함"""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    visitor._SUPABASE_URL = None
    visitor._SUPABASE_KEY = None
    with patch("src.data.visitor.requests") as mock_req:
        assert visitor.record_visit() == (0, 0)
        mock_req.get.assert_not_called()


def test_get_visitor_counts_no_config(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    visitor._SUPABASE_URL = None
    visitor._SUPABASE_KEY = None
    assert visitor.get_visitor_counts() == (0, 0)


# --- record_visit: 오늘 행 존재 → 증가 ---

def test_record_visit_existing_row_increments():
    """오늘 행이 있으면 count+1로 patch, 누적 합계 반환"""
    get_existing = _resp(200, [{"count": 9}])      # 오늘 행 조회
    get_total = _resp(200, [{"count": 10}, {"count": 5}])  # 누적
    with patch("src.data.visitor.requests") as mock_req:
        mock_req.get.side_effect = [get_existing, get_total]
        mock_req.patch.return_value = _resp(200)
        daily, total = visitor.record_visit()

    assert daily == 10           # 9 + 1
    assert total == 15           # 10 + 5
    # 증가는 PATCH로, POST(신규삽입)는 호출 안 함
    mock_req.patch.assert_called_once()
    mock_req.post.assert_not_called()


# --- record_visit: 오늘 행 없음 → 삽입 ---

def test_record_visit_new_row_inserts():
    """오늘 행이 없으면 count=1로 POST 삽입"""
    get_empty = _resp(200, [])                       # 오늘 행 없음
    get_total = _resp(200, [{"count": 1}])           # 누적
    with patch("src.data.visitor.requests") as mock_req:
        mock_req.get.side_effect = [get_empty, get_total]
        mock_req.post.return_value = _resp(200)
        daily, total = visitor.record_visit()

    assert daily == 1
    assert total == 1
    mock_req.post.assert_called_once()
    mock_req.patch.assert_not_called()


def test_record_visit_uses_timeout():
    """모든 요청에 timeout 지정 (hang 방지)"""
    with patch("src.data.visitor.requests") as mock_req:
        mock_req.get.side_effect = [_resp(200, []), _resp(200, [{"count": 1}])]
        mock_req.post.return_value = _resp(200)
        visitor.record_visit()
        for call in mock_req.get.call_args_list:
            assert call.kwargs.get("timeout") == 5


# --- 네트워크/예외 graceful degradation ---

def test_record_visit_network_error_returns_zero():
    """requests 예외 시 (0, 0) 반환 (크래시 안 함)"""
    with patch("src.data.visitor.requests") as mock_req:
        mock_req.get.side_effect = Exception("network down")
        assert visitor.record_visit() == (0, 0)


def test_get_visitor_counts_returns_daily_and_total():
    get_today = _resp(200, [{"count": 7}])
    get_total = _resp(200, [{"count": 7}, {"count": 3}])
    with patch("src.data.visitor.requests") as mock_req:
        mock_req.get.side_effect = [get_today, get_total]
        daily, total = visitor.get_visitor_counts()
    assert daily == 7
    assert total == 10


def test_get_visitor_counts_no_row_today():
    """오늘 방문 없으면 daily=0"""
    get_today = _resp(200, [])
    get_total = _resp(200, [{"count": 3}])
    with patch("src.data.visitor.requests") as mock_req:
        mock_req.get.side_effect = [get_today, get_total]
        daily, total = visitor.get_visitor_counts()
    assert daily == 0
    assert total == 3


def test_get_visitor_counts_network_error():
    with patch("src.data.visitor.requests") as mock_req:
        mock_req.get.side_effect = Exception("boom")
        assert visitor.get_visitor_counts() == (0, 0)


# --- 헬퍼 ---

def test_today_kst_format():
    """KST 날짜 YYYY-MM-DD 형식"""
    today = visitor._today_kst()
    assert len(today) == 10
    assert today[4] == "-" and today[7] == "-"


def test_headers_include_auth():
    """헤더에 apikey/Authorization 포함"""
    headers = visitor._headers()
    assert "apikey" in headers
    assert headers["Authorization"].startswith("Bearer ")


def test_get_config_caches():
    """_get_config는 첫 호출 후 캐시"""
    url1, key1 = visitor._get_config()
    url2, key2 = visitor._get_config()
    assert url1 == url2 == "https://test.supabase.co"
    assert key1 == key2 == "test-key"
