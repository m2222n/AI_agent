"""뉴스 수집 + 감성 분석 테스트"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.data.news import (
    fetch_google_news,
    analyze_sentiment_batch,
    get_stock_news_summary,
    GOOGLE_NEWS_RSS_URL,
)


# ── Google News RSS 테스트 ───────────────────────────────────

def _make_mock_feed(entries):
    """feedparser 결과를 모킹하는 헬퍼"""
    feed = MagicMock()
    feed.entries = entries
    return feed


def _make_entry(title, days_ago=0, source="한경", summary=""):
    """RSS 엔트리 모킹"""
    from datetime import timedelta
    pub_dt = datetime.now() - timedelta(days=days_ago)
    entry = MagicMock()
    entry.title = f"{title} - {source}" if source else title
    entry.link = f"https://example.com/news/{hash(title)}"
    entry.published_parsed = pub_dt.timetuple()[:9]
    entry.published = pub_dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    entry.summary = summary
    entry.get = lambda k, d="": {
        "title": entry.title,
        "link": entry.link,
        "published_parsed": entry.published_parsed,
        "published": entry.published,
        "summary": entry.summary,
    }.get(k, d)
    return entry


@patch("src.data.news._fetch_feed")
def test_fetch_google_news_basic(mock_parse):
    """기본 뉴스 수집"""
    entries = [
        _make_entry("삼성전자 주가 상승", days_ago=1),
        _make_entry("삼성전자 실적 발표", days_ago=2),
    ]
    mock_parse.return_value = _make_mock_feed(entries)

    articles = fetch_google_news("삼성전자")
    assert len(articles) == 2
    assert "삼성전자 주가 상승" in articles[0]["title"]
    assert articles[0]["source"] == "한경"


@patch("src.data.news._fetch_feed")
def test_fetch_google_news_date_filter(mock_parse):
    """오래된 기사 필터링"""
    entries = [
        _make_entry("최신 뉴스", days_ago=1),
        _make_entry("오래된 뉴스", days_ago=30),
    ]
    mock_parse.return_value = _make_mock_feed(entries)

    articles = fetch_google_news("테스트", days=7)
    assert len(articles) == 1
    assert "최신" in articles[0]["title"]


@patch("src.data.news._fetch_feed")
def test_fetch_google_news_max_articles(mock_parse):
    """최대 기사 수 제한"""
    entries = [_make_entry(f"뉴스 {i}", days_ago=i) for i in range(5)]
    mock_parse.return_value = _make_mock_feed(entries)

    articles = fetch_google_news("테스트", max_articles=3)
    assert len(articles) == 3


@patch("src.data.news._fetch_feed")
def test_fetch_google_news_empty(mock_parse):
    """뉴스 없음"""
    mock_parse.return_value = _make_mock_feed([])

    articles = fetch_google_news("존재하지않는종목xyz")
    assert articles == []


@patch("src.data.news._fetch_feed")
def test_fetch_google_news_parse_error(mock_parse):
    """RSS 파싱 에러 처리"""
    mock_parse.side_effect = Exception("Network error")

    articles = fetch_google_news("테스트")
    assert articles == []


@patch("src.data.news._fetch_feed")
def test_fetch_google_news_html_summary(mock_parse):
    """HTML 태그가 포함된 요약 정리"""
    entries = [
        _make_entry("테스트", days_ago=0, summary="<b>중요</b> <a href='#'>뉴스</a> 내용"),
    ]
    mock_parse.return_value = _make_mock_feed(entries)

    articles = fetch_google_news("테스트")
    assert "<" not in articles[0]["summary"]
    assert "중요" in articles[0]["summary"]


# ── 감성 분석 테스트 ──────────────────────────────────────────

def test_sentiment_empty_articles():
    """기사 없을 때"""
    result = analyze_sentiment_batch([], "삼성전자")
    assert result["overall_sentiment"] == "데이터 없음"
    assert result["positive_count"] == 0


@patch("src.llm.client.create_client")
def test_sentiment_positive(mock_client):
    """긍정 뉴스 분석"""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '''{
        "sentiments": [{"index": 1, "sentiment": "긍정"}, {"index": 2, "sentiment": "긍정"}],
        "key_topics": ["실적", "반도체"],
        "summary": "반도체 호황으로 실적 개선"
    }'''
    mock_client.return_value.chat.completions.create.return_value = mock_response

    articles = [
        {"title": "삼성 실적 호조", "source": "한경", "summary": "", "published": "2026-04-29"},
        {"title": "반도체 수출 증가", "source": "매경", "summary": "", "published": "2026-04-28"},
    ]
    result = analyze_sentiment_batch(articles, "삼성전자")
    assert result["overall_sentiment"] == "긍정"
    assert result["positive_count"] == 2
    assert "반도체" in result["key_topics"]


@patch("src.llm.client.create_client")
def test_sentiment_mixed(mock_client):
    """혼재 감성"""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '''{
        "sentiments": [{"index": 1, "sentiment": "긍정"}, {"index": 2, "sentiment": "부정"}],
        "key_topics": ["실적", "규제"],
        "summary": "실적은 좋으나 규제 리스크"
    }'''
    mock_client.return_value.chat.completions.create.return_value = mock_response

    articles = [
        {"title": "실적 개선", "source": "", "summary": "", "published": ""},
        {"title": "규제 리스크", "source": "", "summary": "", "published": ""},
    ]
    result = analyze_sentiment_batch(articles, "테스트")
    assert result["overall_sentiment"] == "혼재"
    assert result["positive_count"] == 1
    assert result["negative_count"] == 1


@patch("src.llm.client.create_client")
def test_sentiment_llm_failure(mock_client):
    """LLM 호출 실패 시 graceful fallback"""
    mock_client.side_effect = Exception("API error")

    articles = [{"title": "테스트", "source": "", "summary": "", "published": ""}]
    result = analyze_sentiment_batch(articles, "테스트")
    assert result["overall_sentiment"] == "분석 불가"
    assert len(result["articles"]) == 1


# ── 통합 테스트 ───────────────────────────────────────────────

@patch("src.data.news.analyze_sentiment_batch")
@patch("src.data.news.fetch_google_news")
def test_get_stock_news_summary(mock_fetch, mock_analyze):
    """통합 함수 테스트"""
    mock_fetch.return_value = [
        {"title": "뉴스1", "source": "한경", "summary": "", "published": ""},
        {"title": "뉴스2", "source": "매경", "summary": "", "published": ""},
        {"title": "뉴스3", "source": "연합", "summary": "", "published": ""},
    ]
    mock_analyze.return_value = {"overall_sentiment": "긍정", "articles": mock_fetch.return_value}

    result = get_stock_news_summary("삼성전자")
    # "삼성전자 주식"으로 검색
    mock_fetch.assert_called_once_with("삼성전자 주식", max_articles=8)
    mock_analyze.assert_called_once()


@patch("src.data.news.analyze_sentiment_batch")
@patch("src.data.news.fetch_google_news")
def test_get_stock_news_summary_retry(mock_fetch, mock_analyze):
    """기사 부족 시 재검색"""
    # 첫 검색: 2건 (< 3건), 재검색
    mock_fetch.side_effect = [
        [{"title": "뉴스A", "source": "", "summary": "", "published": ""}],
        [
            {"title": "뉴스A", "source": "", "summary": "", "published": ""},
            {"title": "뉴스B", "source": "", "summary": "", "published": ""},
            {"title": "뉴스C", "source": "", "summary": "", "published": ""},
        ],
    ]
    mock_analyze.return_value = {"overall_sentiment": "중립"}

    result = get_stock_news_summary("마이너종목")
    assert mock_fetch.call_count == 2  # 재검색 발생


# ── 엣지 케이스 테스트 ────────────────────────────────────────

@patch("src.data.news._fetch_feed")
def test_fetch_google_news_missing_published(mock_parse):
    """published_parsed 없는 엔트리 처리"""
    entry = MagicMock()
    entry.title = "제목만 있는 기사 - 출처"
    entry.link = "https://example.com/1"
    entry.published_parsed = None
    entry.published = ""
    entry.summary = "요약"
    entry.get = lambda k, d="": {
        "title": entry.title, "link": entry.link,
        "published_parsed": None, "published": "",
        "summary": "요약",
    }.get(k, d)
    mock_parse.return_value = _make_mock_feed([entry])

    articles = fetch_google_news("테스트")
    # published_parsed가 None이면 날짜 필터링 스킵하거나 포함
    assert isinstance(articles, list)


@patch("src.data.news._fetch_feed")
def test_fetch_google_news_source_extraction(mock_parse):
    """다양한 제목 형식에서 출처 추출"""
    entries = [
        _make_entry("제목만", days_ago=0, source=""),
    ]
    # source 없는 엔트리
    entry = entries[0]
    entry.title = "제목만 있는 기사"
    mock_parse.return_value = _make_mock_feed(entries)

    articles = fetch_google_news("테스트")
    assert len(articles) >= 0  # 에러 없이 처리


@patch("src.llm.client.create_client")
def test_sentiment_invalid_json(mock_client):
    """LLM이 비정상 JSON 반환 시 처리"""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "이것은 JSON이 아닙니다"
    mock_client.return_value.chat.completions.create.return_value = mock_response

    articles = [{"title": "테스트", "source": "", "summary": "", "published": ""}]
    result = analyze_sentiment_batch(articles, "테스트")
    # 에러 없이 fallback 반환
    assert "overall_sentiment" in result


@patch("src.llm.client.create_client")
def test_sentiment_partial_json(mock_client):
    """LLM이 일부 필드만 있는 JSON 반환"""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"sentiments": []}'
    mock_client.return_value.chat.completions.create.return_value = mock_response

    articles = [{"title": "테스트", "source": "", "summary": "", "published": ""}]
    result = analyze_sentiment_batch(articles, "테스트")
    assert "overall_sentiment" in result


@patch("src.data.news.analyze_sentiment_batch")
@patch("src.data.news.fetch_google_news")
def test_get_stock_news_summary_no_articles(mock_fetch, mock_analyze):
    """기사 0건일 때"""
    mock_fetch.return_value = []
    mock_analyze.return_value = {"overall_sentiment": "데이터 없음", "articles": []}

    result = get_stock_news_summary("존재하지않는종목")
    assert result["overall_sentiment"] == "데이터 없음"


# ── _fetch_feed: certifi 우선 + fallback (RSS SSL 검증 방어) ─────

@patch("src.data.news.urllib.request.urlopen")
def test_fetch_feed_uses_certifi_path(mock_urlopen):
    """certifi 컨텍스트로 직접 fetch 성공 시 그 bytes를 feedparser에 넘긴다.

    macOS Python 등 시스템 CA 미설치 환경에서 feedparser 기본 경로가
    SSL CERTIFICATE_VERIFY_FAILED로 0건 반환하던 문제(2026-06-17 RAGAS
    평가에서 뉴스 0건 → AR=0.0으로 발견)를 보완한 경로.
    """
    from src.data.news import _fetch_feed

    resp_cm = MagicMock()
    resp_cm.__enter__.return_value.read.return_value = b"<rss></rss>"
    mock_urlopen.return_value = resp_cm

    with patch("src.data.news.feedparser.parse") as mock_parse:
        mock_parse.return_value = _make_mock_feed([])
        _fetch_feed("https://news.google.com/rss/search?q=x")

    # urlopen이 호출되고(=certifi 경로), feedparser.parse는 bytes로 호출됨
    assert mock_urlopen.called
    mock_parse.assert_called_once_with(b"<rss></rss>")


@patch("src.data.news.urllib.request.urlopen", side_effect=Exception("SSL fail"))
def test_fetch_feed_fallback_on_certifi_failure(mock_urlopen):
    """certifi 직접 fetch 실패 시 feedparser 기본 경로(URL)로 fallback."""
    from src.data.news import _fetch_feed

    with patch("src.data.news.feedparser.parse") as mock_parse:
        mock_parse.return_value = _make_mock_feed([])
        _fetch_feed("https://news.google.com/rss/search?q=x")

    # fallback은 URL 문자열로 feedparser.parse 호출
    mock_parse.assert_called_once_with("https://news.google.com/rss/search?q=x")
