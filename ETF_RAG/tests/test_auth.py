"""JWT 이메일 인증 엔드포인트 테스트 (Phase F-1 A).

sqlite :memory:는 커넥션마다 별도 DB → StaticPool로 단일 공유 커넥션.
import 전 API_SKIP_INIT=1 + DATABASE_URL=sqlite://(인메모리).
"""

import os

os.environ["API_SKIP_INIT"] = "1"
os.environ["DATABASE_URL"] = "sqlite://"  # 인메모리; conftest client 픽스처가 엔진 교체

# client / _reset_sse_global 픽스처는 tests/conftest.py에 공통 정의됨.


def _signup(client, email="a@b.com", pw="pw123456"):
    return client.post("/auth/signup", json={"email": email, "password": pw})


def test_signup_login_me_happy_path(client):
    r = _signup(client)
    assert r.status_code == 201
    token = r.json()["access_token"]
    assert token and r.json()["token_type"] == "bearer"

    # login
    r2 = client.post("/auth/login", json={"email": "a@b.com", "password": "pw123456"})
    assert r2.status_code == 200
    token2 = r2.json()["access_token"]

    # me
    r3 = client.get("/auth/me", headers={"Authorization": f"Bearer {token2}"})
    assert r3.status_code == 200
    body = r3.json()
    assert body["email"] == "a@b.com"
    assert isinstance(body["id"], int)
    # 닉네임 미설정 → 이메일 local-part로 fallback
    assert body["nickname"] == "a"


def test_signup_duplicate_email_400(client):
    assert _signup(client).status_code == 201
    r = _signup(client)  # 같은 이메일
    assert r.status_code == 400


def test_login_wrong_password_401(client):
    _signup(client)
    r = client.post("/auth/login", json={"email": "a@b.com", "password": "wrongpw1"})
    assert r.status_code == 401


def test_login_unknown_email_401(client):
    r = client.post("/auth/login", json={"email": "x@y.com", "password": "pw123456"})
    assert r.status_code == 401


def test_me_without_token_401(client):
    r = client.get("/auth/me")
    assert r.status_code == 401


def test_me_bad_token_401(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_signup_short_password_422(client):
    r = client.post("/auth/signup", json={"email": "a@b.com", "password": "short"})
    assert r.status_code == 422  # min_length=8


def test_signup_invalid_email_422(client):
    r = client.post("/auth/signup", json={"email": "notanemail", "password": "pw123456"})
    assert r.status_code == 422


# ── 계정 관리: 비밀번호 변경 / 닉네임 / 탈퇴 (2026-06-18) ──

def _token(client, email="a@b.com", pw="pw123456"):
    _signup(client, email=email, pw=pw)
    r = client.post("/auth/login", json={"email": email, "password": pw})
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_change_password_happy(client):
    token = _token(client)
    r = client.put(
        "/auth/password",
        json={"current_password": "pw123456", "new_password": "newpw7890"},
        headers=_auth(token),
    )
    assert r.status_code == 204
    # 기존 비번 로그인 실패, 새 비번 로그인 성공
    assert client.post("/auth/login",
                       json={"email": "a@b.com", "password": "pw123456"}).status_code == 401
    assert client.post("/auth/login",
                       json={"email": "a@b.com", "password": "newpw7890"}).status_code == 200


def test_change_password_wrong_current_400(client):
    token = _token(client)
    r = client.put(
        "/auth/password",
        json={"current_password": "WRONGpw1", "new_password": "newpw7890"},
        headers=_auth(token),
    )
    assert r.status_code == 400


def test_change_password_same_as_current_400(client):
    token = _token(client)
    r = client.put(
        "/auth/password",
        json={"current_password": "pw123456", "new_password": "pw123456"},
        headers=_auth(token),
    )
    assert r.status_code == 400


def test_change_password_requires_auth_401(client):
    r = client.put(
        "/auth/password",
        json={"current_password": "pw123456", "new_password": "newpw7890"},
    )
    assert r.status_code == 401


def test_change_password_short_new_422(client):
    token = _token(client)
    r = client.put(
        "/auth/password",
        json={"current_password": "pw123456", "new_password": "short"},
        headers=_auth(token),
    )
    assert r.status_code == 422  # min_length=8


def test_update_nickname_happy(client):
    token = _token(client)
    r = client.put("/auth/profile", json={"nickname": "투자왕"}, headers=_auth(token))
    assert r.status_code == 200
    assert r.json()["nickname"] == "투자왕"
    # me에도 반영
    assert client.get("/auth/me", headers=_auth(token)).json()["nickname"] == "투자왕"


def test_update_nickname_strips_whitespace(client):
    token = _token(client)
    r = client.put("/auth/profile", json={"nickname": "  태민  "}, headers=_auth(token))
    assert r.status_code == 200
    assert r.json()["nickname"] == "태민"


def test_update_nickname_blank_400(client):
    token = _token(client)
    r = client.put("/auth/profile", json={"nickname": "   "}, headers=_auth(token))
    assert r.status_code == 400


def test_update_nickname_too_long_422(client):
    token = _token(client)
    r = client.put("/auth/profile", json={"nickname": "가" * 41}, headers=_auth(token))
    assert r.status_code == 422  # max_length=40


def test_delete_account_happy(client):
    token = _token(client)
    r = client.request(
        "DELETE", "/auth/me", json={"password": "pw123456"}, headers=_auth(token)
    )
    assert r.status_code == 204
    # 토큰 무효(유저 없음) + 같은 이메일 재가입 가능
    assert client.get("/auth/me", headers=_auth(token)).status_code == 401
    assert _signup(client).status_code == 201


def test_delete_account_wrong_password_400(client):
    token = _token(client)
    r = client.request(
        "DELETE", "/auth/me", json={"password": "WRONGpw1"}, headers=_auth(token)
    )
    assert r.status_code == 400
    # 여전히 살아있음
    assert client.get("/auth/me", headers=_auth(token)).status_code == 200


def test_delete_account_purges_watchlist(client):
    token = _token(client)
    client.put("/me/watchlist/005930", headers=_auth(token))
    assert "005930" in client.get("/me/watchlist", headers=_auth(token)).json()["tickers"]
    # 탈퇴
    client.request("DELETE", "/auth/me", json={"password": "pw123456"}, headers=_auth(token))
    # 같은 이메일 재가입 → 관심종목이 비어 있어야(이전 유저 데이터 소거 확인)
    token2 = _token(client)
    assert client.get("/me/watchlist", headers=_auth(token2)).json()["tickers"] == []


def test_delete_account_requires_auth_401(client):
    r = client.request("DELETE", "/auth/me", json={"password": "pw123456"})
    assert r.status_code == 401


def test_delete_account_purges_paper_trading(client):
    """탈퇴 시 가상투자(PaperAccount/Holding/Trade/...) 데이터도 소거되어야 한다.

    회귀: 탈퇴 cascade에 Watchlist/ChatHistory/Push만 있고 Paper* 5개가 누락돼
    탈퇴 후 재가입 시 옛 보유종목/평가액이 남던 버그(2026-06-24 발견)."""
    from unittest.mock import patch

    token = _token(client)
    with patch("api.paper._resolve_price",
               return_value={"ticker": "005930", "name": "삼성전자", "price": 70000}):
        client.post("/me/paper/buy", json={"ticker": "005930", "qty": 10},
                    headers=_auth(token))
        pf = client.get("/me/paper/portfolio", headers=_auth(token)).json()
    assert any(h["ticker"] == "005930" for h in pf["holdings"])

    # 탈퇴
    assert client.request(
        "DELETE", "/auth/me", json={"password": "pw123456"}, headers=_auth(token)
    ).status_code == 204

    # 같은 이메일 재가입 → 가상투자가 1억 새 계좌로 초기화(이전 보유종목 없음)
    token2 = _token(client)
    pf2 = client.get("/me/paper/portfolio", headers=_auth(token2)).json()
    assert pf2["holdings"] == []
    assert pf2["cash"] == 100_000_000


# ── 나이대(age_group) 선택 수집 ──

def test_signup_with_age_group(client):
    r = client.post("/auth/signup", json={
        "email": "age@b.com", "password": "pw123456", "age_group": "30대"})
    assert r.status_code == 201
    token = r.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["age_group"] == "30대"


def test_signup_without_age_group_is_none(client):
    r = client.post("/auth/signup", json={"email": "noage@b.com", "password": "pw123456"})
    token = r.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["age_group"] is None


def test_signup_invalid_age_group_ignored(client):
    r = client.post("/auth/signup", json={
        "email": "bad@b.com", "password": "pw123456", "age_group": "백살"})
    token = r.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["age_group"] is None  # 미허용값은 무시


def test_profile_update_age_group(client):
    r = client.post("/auth/signup", json={"email": "p@b.com", "password": "pw123456"})
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    upd = client.put("/auth/profile", json={"nickname": "닉", "age_group": "40대"}, headers=h)
    assert upd.status_code == 200
    assert upd.json()["age_group"] == "40대"
    # 나이대 미전송(None) → 기존 유지
    upd2 = client.put("/auth/profile", json={"nickname": "닉2"}, headers=h)
    assert upd2.json()["age_group"] == "40대"
