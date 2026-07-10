"""운영 cron 엔드포인트 테스트 — /admin/refresh-db (재부팅 없이 DB 새로고침).

test_push.py의 X-Cron-Token 검증 패턴을 따른다. 실제 DB 다운로드/재초기화는
절대 태우지 않고 ensure_db·run_init·reset·_purge를 전부 모킹한다.
"""

import os

os.environ["API_SKIP_INIT"] = "1"
os.environ["DATABASE_URL"] = "sqlite://"

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


# _reset_sse_global(autouse)은 tests/conftest.py에 공통 정의됨.


@pytest.fixture
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_refresh_db_no_token(client):
    """토큰 헤더 없음 → 403 (CRON_TOKEN 설정 상태에서)."""
    with patch("config.CRON_TOKEN", "secret"):
        r = client.post("/admin/refresh-db")
    assert r.status_code == 403


def test_refresh_db_cron_token_unset(client):
    """CRON_TOKEN 미설정 → 403 (비활성)."""
    with patch("config.CRON_TOKEN", ""):
        r = client.post("/admin/refresh-db", headers={"X-Cron-Token": "anything"})
    assert r.status_code == 403


def test_refresh_db_wrong_token(client):
    """토큰 불일치 → 403."""
    with patch("config.CRON_TOKEN", "secret"):
        r = client.post("/admin/refresh-db", headers={"X-Cron-Token": "nope"})
    assert r.status_code == 403


def test_refresh_db_success(client):
    """정상 토큰 → 파일정리·재다운로드·리셋·재init 순서대로 호출 후 refreshed=True."""
    with patch("config.CRON_TOKEN", "secret"), \
         patch("api.admin._purge_db_files") as m_purge, \
         patch("api.admin.ensure_db", return_value=True) as m_ensure, \
         patch("api.admin.technical.reset_db_connection") as m_reset, \
         patch("api.admin.run_init") as m_init:
        r = client.post("/admin/refresh-db", headers={"X-Cron-Token": "secret"})

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["refreshed"] is True
    m_purge.assert_called_once()
    m_ensure.assert_called_once()
    m_reset.assert_called_once()
    m_init.assert_called_once()


def test_refresh_db_download_fail(client):
    """ensure_db가 False(다운로드 실패) → 503, 이후 리셋/재init은 호출 안 함."""
    with patch("config.CRON_TOKEN", "secret"), \
         patch("api.admin._purge_db_files"), \
         patch("api.admin.ensure_db", return_value=False), \
         patch("api.admin.technical.reset_db_connection") as m_reset, \
         patch("api.admin.run_init") as m_init:
        r = client.post("/admin/refresh-db", headers={"X-Cron-Token": "secret"})

    assert r.status_code == 503
    m_reset.assert_not_called()
    m_init.assert_not_called()


def test_refresh_db_lock_busy(client):
    """이미 refresh 진행 중(Lock 점유) → 200 refreshed=False, 재init 호출 안 함."""
    import api.admin as admin

    admin._refresh_lock.acquire()
    try:
        with patch("config.CRON_TOKEN", "secret"), \
             patch("api.admin.run_init") as m_init:
            r = client.post("/admin/refresh-db", headers={"X-Cron-Token": "secret"})
    finally:
        admin._refresh_lock.release()

    assert r.status_code == 200
    assert r.json()["refreshed"] is False
    m_init.assert_not_called()


def test_reset_db_connection_clears_singleton_and_cache():
    """reset_db_connection: 싱글톤 close+None + TTL 캐시 clear."""
    from src.data import technical
    from src.data.technical import _data

    # 커넥션 싱글톤을 강제로 하나 열고 캐시에 값을 채운다.
    conn = _data._get_db_conn()
    assert _data._db_conn is conn
    _data._ohlcv_cache[("005930", 20)] = (0.0, ["dummy"])
    _data._closes_cache[("005930", 20)] = (0.0, ["dummy"])

    technical.reset_db_connection()

    assert _data._db_conn is None
    assert _data._ohlcv_cache == {}
    assert _data._closes_cache == {}
