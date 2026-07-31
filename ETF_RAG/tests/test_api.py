"""FastAPI 백엔드 엔드포인트 테스트 (Phase F-1).

실제 DB 다운로드/임베딩은 API_SKIP_INIT=1로 우회하고, run_agent/stream_agent는 patch.
lifespan은 `with TestClient(app)` 컨텍스트 매니저로만 트리거된다.
"""

import os

# api.main import 전에 설정해야 lifespan이 실제 init을 건너뛴다
os.environ["API_SKIP_INIT"] = "1"

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


# _reset_sse_global(autouse)은 tests/conftest.py에 공통 정의됨(sse-starlette의
# AppStatus.should_exit_event가 모듈 전역이라 테스트마다 리셋 — 'different loop' 방지).


@pytest.fixture
def client():
    # 컨텍스트 매니저 형태여야 lifespan(startup/shutdown)이 실행된다
    with TestClient(app) as c:
        yield c


def test_health_ready(client):
    """API_SKIP_INIT=1이면 ready=True, error=None."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["error"] is None


def test_chat_returns_agent_result(client):
    """/chat이 run_agent 결과를 ChatResponse로 반환."""
    fake = {"answer": "KODEX 200 수익률은 +2.91%입니다.",
            "question_type": "simple", "model": "gpt-4o-mini"}
    with patch("api.main.run_agent", return_value=fake) as m:
        r = client.post("/chat", json={"question": "KODEX 200 수익률?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == fake["answer"]
    assert body["question_type"] == "simple"
    assert body["model"] == "gpt-4o-mini"
    # run_in_threadpool(run_agent, question, history) — history는 None
    m.assert_called_once()
    args = m.call_args.args
    assert args[0] == "KODEX 200 수익률?"
    assert args[1] is None


def test_chat_passes_history(client):
    """chat_history가 [{"role","content"}, ...] dict 리스트로 전달되는지."""
    fake = {"answer": "a", "question_type": "simple", "model": "gpt-4o-mini"}
    history = [{"role": "user", "content": "안녕"},
               {"role": "assistant", "content": "안녕하세요"}]
    with patch("api.main.run_agent", return_value=fake) as m:
        r = client.post("/chat", json={"question": "그 다음은?", "chat_history": history})
    assert r.status_code == 200
    passed_history = m.call_args.args[1]
    assert passed_history == history


def test_chat_validates_empty_question(client):
    """빈 question은 422 (Pydantic min_length=1)."""
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422


def test_stream_passes_through_all_events(client):
    """/stream이 stream_agent 이벤트를 SSE로 전부 통과시키는지."""
    def fake_stream(question, history=None):
        yield {"event": "question_type", "data": "simple"}
        yield {"event": "tool_call", "data": {"name": "search_etf", "args": {"query": "x"}}}
        yield {"event": "token", "data": "삼성"}
        yield {"event": "token", "data": "삼성전자"}
        yield {"event": "done", "data": {"answer": "삼성전자", "model": "gpt-4o-mini",
                                         "question_type": "simple", "cov_applied": False}}

    with patch("api.main.stream_agent", side_effect=fake_stream):
        r = client.post("/stream", json={"question": "삼성전자?"})
    assert r.status_code == 200
    text = r.text
    # SSE 프레임에 각 이벤트 이름과 data가 들어있는지 (이름 화이트리스트 없이 전부 통과)
    assert "question_type" in text
    assert "tool_call" in text
    assert "search_etf" in text  # dict data가 JSON 직렬화됨
    assert "token" in text
    assert "done" in text
    assert "삼성전자" in text


def test_stream_dict_data_is_json(client):
    """dict data는 JSON 직렬화 (ensure_ascii=False로 한글 보존)."""
    def fake_stream(question, history=None):
        yield {"event": "done", "data": {"answer": "한글유지", "model": "gpt-4o"}}

    with patch("api.main.stream_agent", side_effect=fake_stream):
        r = client.post("/stream", json={"question": "x"})
    assert r.status_code == 200
    assert "한글유지" in r.text  # \uXXXX 이스케이프 안 됨


def test_feedback_anonymous(client):
    """피드백 — 익명(토큰 없음) 허용, 204."""
    with patch("api.main.log_feedback") as m:
        r = client.post("/feedback", json={
            "question": "삼성전자?", "answer": "PER 49.81배",
            "rating": "positive",
        })
    assert r.status_code == 204
    m.assert_called_once()


def test_feedback_negative_with_reason(client):
    with patch("api.main.log_feedback") as m:
        r = client.post("/feedback", json={
            "question": "x", "answer": "y",
            "rating": "negative", "reason": "정보가 부정확해요",
        })
    assert r.status_code == 204
    # log_feedback(question, answer, tag) — tag에 rating+reason 포함
    tag = m.call_args.args[2]
    assert "negative" in tag and "부정확" in tag


def test_visit_records_and_returns_counts(client):
    """POST /stats/visit이 record_visit 결과를 VisitorResponse로 반환."""
    with patch("api.main.record_visit", return_value=(3, 42)) as m:
        r = client.post("/stats/visit")
    assert r.status_code == 200
    assert r.json() == {"daily": 3, "total": 42}
    m.assert_called_once()


def test_visit_read_only(client):
    """GET /stats/visit은 get_visitor_counts만 호출(기록 없이)."""
    with patch("api.main.get_visitor_counts", return_value=(5, 100)) as m, \
            patch("api.main.record_visit") as rec:
        r = client.get("/stats/visit")
    assert r.status_code == 200
    assert r.json() == {"daily": 5, "total": 100}
    m.assert_called_once()
    rec.assert_not_called()


# --- CORS: iOS 앱(Capacitor) origin 허용 ---

def test_capacitor_origin_regex_when_restricted():
    """CORS_ORIGINS를 특정 웹 origin으로 좁히면 capacitor origin을 regex로 추가 허용."""
    from api.main import _capacitor_origin_regex

    regex = _capacitor_origin_regex(["https://myapp.example.com"])
    assert regex is not None
    import re
    # iOS(capacitor://localhost) + Android(http/https://localhost) 모두 허용
    assert re.match(regex, "capacitor://localhost")
    assert re.match(regex, "http://localhost")
    # Android 14 실측: 최신 WebView origin은 https://localhost (2026-07-31 에뮬 logcat 확인)
    assert re.match(regex, "https://localhost")
    # 임의의 다른 origin은 이 regex에 매칭되지 않아야(웹 origin은 allow_origins가 처리)
    assert not re.match(regex, "https://evil.example.com")
    # Next dev 서버(http://localhost:3000)는 포트가 있어 매칭 안 됨(별도 CORS_ORIGINS로 처리)
    assert not re.match(regex, "http://localhost:3000")
    assert not re.match(regex, "https://localhost:3000")


def test_capacitor_origin_regex_none_when_wildcard():
    """allow_origins가 '*'이면 이미 전부 허용 → regex 불필요(None)."""
    from api.main import _capacitor_origin_regex

    assert _capacitor_origin_regex(["*"]) is None


def test_capacitor_origin_allowed_in_cors_response():
    """제한된 CORS 설정에서 Capacitor origin의 preflight가 허용되는지 통합 검증."""
    import importlib
    import api.main as main_mod

    # 프로덕션처럼 특정 origin으로 제한한 앱을 별도 구성해 미들웨어를 재적용
    with patch.dict(os.environ, {"CORS_ORIGINS": "https://myapp.example.com"}):
        reloaded = importlib.reload(main_mod)
        try:
            with TestClient(reloaded.app) as c:
                r = c.options(
                    "/health",
                    headers={
                        "Origin": "capacitor://localhost",
                        "Access-Control-Request-Method": "GET",
                    },
                )
                assert r.headers.get("access-control-allow-origin") == "capacitor://localhost"
        finally:
            # 다른 테스트가 기본(*) 설정 앱을 쓰도록 원복
            importlib.reload(main_mod)
