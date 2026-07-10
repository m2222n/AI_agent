"""관리자 통계 엔드포인트 테스트 — GET /admin/stats.

가입자/방문자/나이대/가상투자 집계. DB가 필요하므로 conftest의 StaticPool
client(인메모리 sqlite)를 그대로 사용(로컬 client 오버라이드 없음).
X-Cron-Token 보호 — test_admin.py의 refresh-db와 동일 인증 패턴.
"""

import os

os.environ["API_SKIP_INIT"] = "1"
os.environ["DATABASE_URL"] = "sqlite://"

from unittest.mock import patch  # noqa: E402


def test_stats_no_token(client):
    """토큰 없음 → 403."""
    with patch("config.CRON_TOKEN", "secret"):
        r = client.get("/admin/stats")
    assert r.status_code == 403


def test_stats_wrong_token(client):
    """토큰 불일치 → 403."""
    with patch("config.CRON_TOKEN", "secret"):
        r = client.get("/admin/stats", headers={"X-Cron-Token": "nope"})
    assert r.status_code == 403


def test_stats_empty(client):
    """가입자 0명일 때 전부 0/빈 dict."""
    with patch("config.CRON_TOKEN", "secret"), \
         patch("src.data.visitor.get_visitor_counts", return_value=(0, 0)):
        r = client.get("/admin/stats", headers={"X-Cron-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_users"] == 0
    assert body["paper_players"] == 0
    assert body["age_groups"] == {}
    assert body["genders"] == {}


def test_stats_counts_users_and_age_groups(client):
    """가입 후 가입자 수·나이대 분포·닉네임 수 집계."""
    # 나이대 있는 유저 2명(20대), 나이대 없는 유저 1명
    client.post("/auth/signup", json={"email": "a@b.com", "password": "pw123456", "age_group": "20대", "gender": "남성"})
    client.post("/auth/signup", json={"email": "c@d.com", "password": "pw123456", "age_group": "20대", "gender": "여성"})
    client.post("/auth/signup", json={"email": "e@f.com", "password": "pw123456", "gender": "선택안함"})

    # 방문자 카운터는 외부 스토어라 모킹(집계에 영향 없음 확인용)
    with patch("config.CRON_TOKEN", "secret"), \
         patch("src.data.visitor.get_visitor_counts", return_value=(5, 42)):
        r = client.get("/admin/stats", headers={"X-Cron-Token": "secret"})

    assert r.status_code == 200
    body = r.json()
    assert body["total_users"] == 3
    assert body["age_groups"].get("20대") == 2
    assert body["age_groups"].get("미입력") == 1
    assert body["genders"].get("남성") == 1
    assert body["genders"].get("여성") == 1
    assert body["genders"].get("선택안함") == 1
    assert body["visitors_total"] == 42
