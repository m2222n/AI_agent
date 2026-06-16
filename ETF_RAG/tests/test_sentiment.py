"""로컬 금융 감성 분류 (KR-FinBert-SC) + news.py 하이브리드 경로 테스트.

transformers/torch는 테스트 환경에 없으므로 pipeline을 mock하거나 미설치
경로(None)를 검증한다.
"""

from unittest.mock import patch, MagicMock

import pytest

from src.data import sentiment


@pytest.fixture(autouse=True)
def _reset_sentiment():
    sentiment.reset()
    yield
    sentiment.reset()


# ── 라벨 매핑 ─────────────────────────────────────────────
def test_to_korean_english_labels():
    assert sentiment._to_korean("positive") == "긍정"
    assert sentiment._to_korean("negative") == "부정"
    assert sentiment._to_korean("neutral") == "중립"


def test_to_korean_label_n_form():
    # id2label 없이 LABEL_n으로 나올 때 (0=neg,1=neu,2=pos)
    assert sentiment._to_korean("LABEL_2") == "긍정"
    assert sentiment._to_korean("LABEL_0") == "부정"
    assert sentiment._to_korean("LABEL_1") == "중립"


def test_to_korean_unknown_defaults_neutral():
    assert sentiment._to_korean("weird") == "중립"


# ── 미설치/비활성 → None (GPT fallback) ───────────────────
def test_classify_none_when_transformers_missing():
    # transformers import 실패 시뮬레이션 → _get_pipeline False
    with patch("src.data.sentiment._get_pipeline", return_value=None):
        assert sentiment.classify_sentiments(["삼성전자 신고가"]) is None


def test_classify_empty_returns_empty_list():
    assert sentiment.classify_sentiments([]) == []


def test_is_available_false_when_disabled():
    with patch("config.SENTIMENT", {"enabled": False, "model": "x"}):
        assert sentiment.is_available() is False


# ── pipeline mock으로 분류 동작 ───────────────────────────
def test_classify_with_mocked_pipeline():
    fake_pipe = MagicMock(return_value=[
        [{"label": "positive", "score": 0.9}],
        [{"label": "negative", "score": 0.8}],
        [{"label": "neutral", "score": 0.7}],
    ])
    with patch("src.data.sentiment._get_pipeline", return_value=fake_pipe):
        out = sentiment.classify_sentiments(["a", "b", "c"])
    assert out == ["긍정", "부정", "중립"]


def test_classify_dict_result_form():
    """일부 transformers 버전은 top_k=1에서 dict 1개를 반환."""
    fake_pipe = MagicMock(return_value=[{"label": "positive", "score": 0.9}])
    with patch("src.data.sentiment._get_pipeline", return_value=fake_pipe):
        out = sentiment.classify_sentiments(["삼성전자 신고가"])
    assert out == ["긍정"]


def test_classify_length_mismatch_returns_none():
    fake_pipe = MagicMock(return_value=[{"label": "positive"}])  # 입력 2개인데 1개
    with patch("src.data.sentiment._get_pipeline", return_value=fake_pipe):
        assert sentiment.classify_sentiments(["a", "b"]) is None


def test_classify_inference_error_returns_none():
    fake_pipe = MagicMock(side_effect=RuntimeError("oom"))
    with patch("src.data.sentiment._get_pipeline", return_value=fake_pipe):
        assert sentiment.classify_sentiments(["a"]) is None


# ── news.py 하이브리드 경로 ───────────────────────────────
def test_news_uses_local_sentiment_when_available():
    """로컬 분류 성공 → 감성은 로컬, GPT는 요약만. sentiment_source=local."""
    from src.data.news import analyze_sentiment_batch

    articles = [
        {"title": "삼성전자 신고가 경신", "source": "한경", "summary": "사상 최고"},
        {"title": "삼성전자 리콜 논란", "source": "매경", "summary": "결함"},
    ]
    # GPT는 요약/키워드만 반환
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = (
        '{"key_topics": ["반도체"], "summary": "엇갈린 뉴스."}'
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("src.data.news._local_classify", return_value=["긍정", "부정"]), \
         patch("src.llm.client.create_client", return_value=mock_client), \
         patch("src.llm.client.get_api_key", return_value="k"):
        r = analyze_sentiment_batch(articles, "삼성전자")

    assert r["sentiment_source"] == "local"
    assert r["positive_count"] == 1
    assert r["negative_count"] == 1
    assert r["overall_sentiment"] == "혼재"
    assert r["articles"][0]["sentiment"] == "긍정"
    assert r["summary"] == "엇갈린 뉴스."  # 요약은 GPT


def test_news_local_sentiment_survives_gpt_failure():
    """로컬 분류 성공 + GPT 실패 → 감성은 살고 요약만 누락."""
    from src.data.news import analyze_sentiment_batch

    articles = [{"title": "삼성전자 신고가", "source": "한경", "summary": ""}]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")

    with patch("src.data.news._local_classify", return_value=["긍정"]), \
         patch("src.llm.client.create_client", return_value=mock_client), \
         patch("src.llm.client.get_api_key", return_value="k"):
        r = analyze_sentiment_batch(articles, "삼성전자")

    assert r["sentiment_source"] == "local"
    assert r["positive_count"] == 1
    assert r["overall_sentiment"] == "긍정"
    assert "요약 생성 실패" in r["summary"]


def test_news_falls_back_to_gpt_when_local_unavailable():
    """로컬 미설치(None) → 기존 GPT 분류 경로. sentiment_source=gpt."""
    from src.data.news import analyze_sentiment_batch

    articles = [{"title": "삼성전자 신고가", "source": "한경", "summary": ""}]
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = (
        '{"sentiments": [{"index": 1, "sentiment": "긍정"}], '
        '"key_topics": ["반도체"], "summary": "호재."}'
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("src.data.news._local_classify", return_value=None), \
         patch("src.llm.client.create_client", return_value=mock_client), \
         patch("src.llm.client.get_api_key", return_value="k"):
        r = analyze_sentiment_batch(articles, "삼성전자")

    assert r["sentiment_source"] == "gpt"
    assert r["positive_count"] == 1
    assert r["overall_sentiment"] == "긍정"
