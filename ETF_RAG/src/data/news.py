"""
뉴스 수집 + LLM 감성 분석 모듈

- Google News RSS로 종목 관련 뉴스 수집 (최근 7일)
- GPT-4o-mini로 감성 분석 (긍정/부정/중립 + 요약)
- 네이버 크롤링 금지 (ToS 위반) → RSS/API만 사용
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote

import feedparser

logger = logging.getLogger(__name__)

# Google News RSS 엔드포인트
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"


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
        feed = feedparser.parse(url)
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


def analyze_sentiment_batch(
    articles: list[dict], stock_name: str
) -> dict:
    """LLM으로 뉴스 감성 분석 (배치).

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

    # LLM 호출로 감성 분석
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

        # 각 기사에 감성 태깅
        sentiments = {s["index"]: s["sentiment"] for s in analysis.get("sentiments", [])}
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

        # 전체 감성 판정
        if pos > neg and pos > neu:
            overall = "긍정"
        elif neg > pos and neg > neu:
            overall = "부정"
        elif pos == neg and pos > 0:
            overall = "혼재"
        else:
            overall = "중립"

        return {
            "overall_sentiment": overall,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "key_topics": analysis.get("key_topics", []),
            "summary": analysis.get("summary", ""),
            "articles": articles,
        }

    except Exception as e:
        logger.warning(f"감성 분석 LLM 호출 실패: {e}")
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
