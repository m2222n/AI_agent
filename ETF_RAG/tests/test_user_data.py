"""유저별 저장(관심종목/대화이력) CRUD 테스트 (Phase F-1 B).

test_auth.py와 동일한 StaticPool 인메모리 sqlite 패턴.
"""

import os

os.environ["API_SKIP_INIT"] = "1"
os.environ["DATABASE_URL"] = "sqlite://"

import pytest  # noqa: E402

# client / _reset_sse_global 픽스처는 tests/conftest.py에 공통 정의됨.


@pytest.fixture
def auth(client):
    """가입 후 Authorization 헤더 반환."""
    r = client.post("/auth/signup", json={"email": "u@b.com", "password": "pw123456", "gender": "선택안함"})
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ── 관심종목 ────────────────────────────────────────────
def test_watchlist_add_list_remove(client, auth):
    assert client.get("/me/watchlist", headers=auth).json()["tickers"] == []

    r = client.put("/me/watchlist/005930", headers=auth)
    assert r.status_code == 200
    assert r.json()["tickers"] == ["005930"]

    client.put("/me/watchlist/000660", headers=auth)
    assert set(client.get("/me/watchlist", headers=auth).json()["tickers"]) == {
        "005930", "000660"
    }

    # 중복 추가 멱등
    client.put("/me/watchlist/005930", headers=auth)
    assert len(client.get("/me/watchlist", headers=auth).json()["tickers"]) == 2

    r = client.delete("/me/watchlist/005930", headers=auth)
    assert r.json()["tickers"] == ["000660"]


def test_watchlist_requires_auth(client):
    assert client.get("/me/watchlist").status_code == 401
    assert client.put("/me/watchlist/005930").status_code == 401


def test_watchlist_detail_includes_name(client, auth):
    from unittest.mock import patch
    client.put("/me/watchlist/005930", headers=auth)
    with patch("src.llm.tools._find_structured_data",
               return_value={"ticker": "005930", "name": "삼성전자"}):
        r = client.get("/me/watchlist/detail", headers=auth)
    assert r.status_code == 200
    items = r.json()["items"]
    assert items == [{"ticker": "005930", "name": "삼성전자"}]


def test_watchlist_detail_fallback_to_ticker(client, auth):
    """종목명 해석 실패 시 ticker로 fallback."""
    from unittest.mock import patch
    client.put("/me/watchlist/999999", headers=auth)
    with patch("src.llm.tools._find_structured_data", return_value=None):
        r = client.get("/me/watchlist/detail", headers=auth)
    assert r.json()["items"] == [{"ticker": "999999", "name": "999999"}]


def test_watchlist_isolated_per_user(client):
    t1 = client.post("/auth/signup", json={"email": "a@b.com", "password": "pw123456", "gender": "선택안함"}).json()["access_token"]
    t2 = client.post("/auth/signup", json={"email": "b@b.com", "password": "pw123456", "gender": "선택안함"}).json()["access_token"]
    client.put("/me/watchlist/005930", headers={"Authorization": f"Bearer {t1}"})
    # user2는 비어있어야
    r = client.get("/me/watchlist", headers={"Authorization": f"Bearer {t2}"})
    assert r.json()["tickers"] == []


# ── 대화 이력 ──────────────────────────────────────────
def test_history_append_list_clear(client, auth):
    assert client.get("/me/history", headers=auth).json()["messages"] == []

    payload = {"messages": [
        {"role": "user", "content": "삼성전자 어때?"},
        {"role": "assistant", "content": "PER 49.81배입니다.", "question_type": "simple", "model": "gpt-4o-mini"},
    ]}
    r = client.post("/me/history", headers=auth, json=payload)
    assert r.status_code == 201
    msgs = r.json()["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["question_type"] == "simple"

    # 순서 보존 + 누적
    client.post("/me/history", headers=auth, json={"messages": [{"role": "user", "content": "다음은?"}]})
    got = client.get("/me/history", headers=auth).json()["messages"]
    assert len(got) == 3
    assert got[-1]["content"] == "다음은?"

    assert client.delete("/me/history", headers=auth).status_code == 204
    assert client.get("/me/history", headers=auth).json()["messages"] == []


def test_history_requires_auth(client):
    assert client.get("/me/history").status_code == 401


def test_history_empty_payload_422(client, auth):
    r = client.post("/me/history", headers=auth, json={"messages": []})
    assert r.status_code == 422  # min_length=1
