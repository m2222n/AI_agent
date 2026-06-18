"""
뉴스 수집 + 감성 분석 모듈

- Google News RSS로 종목 관련 뉴스 수집 (최근 7일)
- 감성 분류: 로컬 KR-FinBert-SC(설치 시) → GPT-4o-mini fallback (src/data/sentiment.py)
- 요약/키워드: GPT-4o-mini (로컬 분류 모델은 라벨만 → 요약은 생성 모델 필요)
- 네이버 크롤링 금지 (ToS 위반) → RSS/API만 사용
"""

import logging
import re
import ssl
import urllib.request
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote

import feedparser

logger = logging.getLogger(__name__)

# Google News RSS 엔드포인트
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"


def _fetch_feed(url: str):
    """RSS URL을 파싱. feedparser 기본 경로가 SSL 인증서를 못 찾는 환경
    (시스템 CA 미설치 macOS Python 등)을 대비해 certifi CA 번들로 직접
    fetch한 bytes를 넘긴다. 직접 fetch 실패 시 feedparser 기본 경로로 fallback."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            return feedparser.parse(resp.read())
    except Exception as e:
        logger.info(f"certifi fetch 실패, feedparser 기본 경로 사용: {e}")
        return feedparser.parse(url)


def fetch_google_news(
    query: str, max_articles: int = 10, days: int = 7
) -> list[dict]:
    """Google News RSS에서 뉴스 수집.

    Args:
        query: 검색어 (종목명 등)
        max_articles: 최대 기사 수
        days: 최근 N일 이내 기사만

    Returns:
        [{"title": str, "link": str, "source": str, "published": str, "summary": str}, ...]
    """
    url = GOOGLE_NEWS_RSS_URL.format(query=quote(query))
    try:
        feed = _fetch_feed(url)
    except Exception as e:
        logger.warning(f"Google News RSS 파싱 실패: {e}")
        return []

    cutoff = datetime.now() - timedelta(days=days)
    articles = []

    for entry in feed.entries[:max_articles * 2]:  # 날짜 필터용 여유분
        # 발행일 파싱
        published_parsed = entry.get("published_parsed")
        if published_parsed:
            pub_dt = datetime(*published_parsed[:6])
            if pub_dt < cutoff:
                continue
            pub_str = pub_dt.strftime("%Y-%m-%d %H:%M")
        else:
            pub_str = entry.get("published", "")

        # 출처 추출 (title에서 " - 출처" 패턴)
        title = entry.get("title", "")
        source = ""
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            title = parts[0].strip()
            source = parts[1].strip()

        # 요약 (HTML 태그 제거)
        summary = entry.get("summary", "")
        summary = re.sub(r"<[^>]+>", "", summary).strip()

        articles.append({
            "title": title,
            "link": entry.get("link", ""),
            "source": source,
            "published": pub_str,
            "summary": summary[:200] if summary else "",
        })

        if len(articles) >= max_articles:
            break

    logger.info(f"Google News: '{query}' → {len(articles)}건 수집")
    return articles


def _overall(pos: int, neg: int, neu: int) -> str:
    """긍/부/중 카운트 → 전체 감성."""
    if pos > neg and pos > neu:
        return "긍정"
    if neg > pos and neg > neu:
        return "부정"
    if pos == neg and pos > 0:
        return "혼재"
    return "중립"


def _local_classify(articles: list[dict]) -> Optional[list]:
    """로컬 모델(KR-FinBert-SC)로 기사 감성 분류. 미설치/실패 시 None.

    제목 + 요약 앞부분을 입력으로 사용. 반환 길이는 articles와 일치.
    """
    try:
        from src.data import sentiment
    except Exception:
        return None
    texts = []
    for a in articles:
        t = a.get("title", "")
        s = a.get("summary", "")
        texts.append(f"{t} {s[:100]}".strip() if s else t)
    labels = sentiment.classify_sentiments(texts)
    if labels is None or len(labels) != len(articles):
        return None
    return labels


def analyze_sentiment_batch(
    articles: list[dict], stock_name: str
) -> dict:
    """뉴스 감성 분석 (배치). 감성 분류는 로컬 모델 우선→GPT fallback, 요약은 GPT.

    Returns:
        {
            "overall_sentiment": "긍정"|"부정"|"중립"|"혼재",
            "positive_count": int,
            "negative_count": int,
            "neutral_count": int,
            "key_topics": [str, ...],
            "summary": str,
            "articles": [{...기존 + "sentiment": str}, ...]
        }
    """
    if not articles:
        return {
            "overall_sentiment": "데이터 없음",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "key_topics": [],
            "summary": f"'{stock_name}' 관련 최근 뉴스가 없습니다.",
            "articles": [],
        }

    # 1) 감성 분류: 로컬 모델(KR-FinBert-SC) 우선, 미설치 시 None → GPT가 담당
    local_sents = _local_classify(articles)

    # LLM 호출로 (요약/키워드 + 필요 시 감성 분류)
    try:
        from src.llm.client import create_client, get_api_key
        import json

        client = create_client(get_api_key())

        # 기사 목록 텍스트화
        articles_text = ""
        for i, a in enumerate(articles, 1):
            articles_text += f"\n[{i}] {a['title']}"
            if a["source"]:
                articles_text += f" ({a['source']})"
            if a["summary"]:
                articles_text += f"\n    {a['summary'][:100]}"

        if local_sents is not None:
            # 감성은 로컬이 결정 → GPT는 요약/키워드만
            prompt = f"""다음은 '{stock_name}' 관련 최근 뉴스 헤드라인입니다.
{articles_text}

아래 JSON 형식으로 분석해주세요 (한국어로):
{{
  "key_topics": ["주요 키워드 3-5개"],
  "summary": "전체 뉴스 흐름을 2-3문장으로 요약"
}}
JSON만 출력하세요"""
        else:
            prompt = f"""다음은 '{stock_name}' 관련 최근 뉴스 헤드라인입니다.
{articles_text}

아래 JSON 형식으로 분석해주세요 (한국어로):
{{
  "sentiments": [{{"index": 1, "sentiment": "긍정|부정|중립"}}],
  "key_topics": ["주요 키워드 3-5개"],
  "summary": "전체 뉴스 흐름을 2-3문장으로 요약"
}}

규칙:
- 주가 상승/실적 개선/신사업 긍정적 → "긍정"
- 주가 하락/실적 악화/규제/소송 → "부정"
- 단순 정보/중립적 보도 → "중립"
- JSON만 출력하세요"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )

        result_text = response.choices[0].message.content.strip()
        # JSON 추출 (```json ... ``` 또는 직접)
        json_match = re.search(r"\{[\s\S]*\}", result_text)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            raise ValueError("JSON 파싱 실패")

        # 감성 소스 결정: 로컬 우선, 없으면 GPT 응답
        if local_sents is not None:
            sentiments = {i + 1: s for i, s in enumerate(local_sents)}
        else:
            sentiments = {s["index"]: s["sentiment"]
                          for s in analysis.get("sentiments", [])}

        # 각 기사에 감성 태깅
        pos = neg = neu = 0
        for i, a in enumerate(articles, 1):
            sent = sentiments.get(i, "중립")
            a["sentiment"] = sent
            if sent == "긍정":
                pos += 1
            elif sent == "부정":
                neg += 1
            else:
                neu += 1

        return {
            "overall_sentiment": _overall(pos, neg, neu),
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "key_topics": analysis.get("key_topics", []),
            "summary": analysis.get("summary", ""),
            "sentiment_source": "local" if local_sents is not None else "gpt",
            "articles": articles,
        }

    except Exception as e:
        logger.warning(f"감성 분석 LLM 호출 실패: {e}")

        # GPT 실패해도 로컬 분류가 있으면 감성은 살린다(요약만 누락)
        if local_sents is not None:
            pos = neg = neu = 0
            for a, sent in zip(articles, local_sents):
                a["sentiment"] = sent
                if sent == "긍정":
                    pos += 1
                elif sent == "부정":
                    neg += 1
                else:
                    neu += 1
            return {
                "overall_sentiment": _overall(pos, neg, neu),
                "positive_count": pos,
                "negative_count": neg,
                "neutral_count": neu,
                "key_topics": [],
                "summary": f"로컬 감성 분석 완료 (요약 생성 실패, 뉴스 {len(articles)}건)",
                "sentiment_source": "local",
                "articles": articles,
            }
        # Fallback: 감성 분석 없이 기사만 반환
        for a in articles:
            a["sentiment"] = "분석 불가"
        return {
            "overall_sentiment": "분석 불가",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "key_topics": [],
            "summary": f"감성 분석 실패 (뉴스 {len(articles)}건 수집됨)",
            "articles": articles,
        }


def get_stock_news_summary(
    stock_name: str, max_articles: int = 8
) -> dict:
    """종목 관련 뉴스 수집 + 감성 분석 통합.

    Args:
        stock_name: 종목명 (예: "삼성전자")
        max_articles: 최대 기사 수

    Returns:
        감성 분석 결과 dict
    """
    # 검색어: "종목명 주식" (더 정확한 결과)
    query = f"{stock_name} 주식"
    articles = fetch_google_news(query, max_articles=max_articles)

    # 기사가 적으면 종목명만으로 재검색
    if len(articles) < 3:
        articles2 = fetch_google_news(stock_name, max_articles=max_articles)
        # 중복 제거 후 병합
        seen_titles = {a["title"] for a in articles}
        for a in articles2:
            if a["title"] not in seen_titles:
                articles.append(a)
                seen_titles.add(a["title"])
        articles = articles[:max_articles]

    return analyze_sentiment_batch(articles, stock_name)
