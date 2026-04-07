import re


def _normalize(text: str) -> str:
    """공백 정규화 + 소문자 변환"""
    return re.sub(r"\s+", " ", text.lower()).strip()


def classify_question_type(question: str) -> str:
    """
    질문 유형을 분류하여 최적화된 프롬프트 적용

    유형:
    - simple: 단일 ETF 정보 질문 ("KODEX 200 수익률은?")
    - compare: 비교 질문 ("A와 B 비교해줘")
    - recommend: 추천 질문 ("배당 높은 ETF 추천")
    - risk: 위험/주의사항 질문 ("위험도", "주의")
    - general: 일반 ETF 지식 질문

    분류 우선순위: 비교 > 위험 > 단순정보 > 추천 > 일반
    """
    q = _normalize(question)

    # ETF 브랜드명 (주요 운용사 전체)
    etf_brands = [
        "kodex", "tiger", "ace", "arirang", "kbstar", "hanaro",
        "kosef", "sols", "plus", "timefolio", "woori",
        "코덱스", "타이거", "에이스", "아리랑", "케이비스타", "하나로",
    ]
    # ETF 티커 패턴 (6자리 숫자)
    has_ticker = bool(re.search(r"\b\d{6}\b", q))
    has_brand = any(name in q for name in etf_brands)
    has_etf_keyword = "etf" in q
    has_specific_etf = has_brand or has_ticker or "etf-" in q

    # 1. 비교 질문 패턴 (최우선)
    compare_keywords = [
        "비교", "차이", "vs", "versus",
        "중에", "중에서", "어떤게", "어떤 게", "어떤것", "어떤 것",
        "둘 중", "셋 중", "뭐가 더", "뭐가더",
    ]
    compare_connectors = ["와 ", "과 ", "이랑 ", "하고 ", "랑 "]
    has_compare_connector = any(conn in q for conn in compare_connectors)
    has_compare_keyword = any(kw in q for kw in compare_keywords)

    if has_compare_keyword or (has_compare_connector and (has_specific_etf or has_etf_keyword)):
        return "compare"

    # 2. 위험/주의 질문 패턴
    risk_keywords = ["위험", "리스크", "주의", "손실", "안전", "변동성", "하락", "폭락"]
    if any(kw in q for kw in risk_keywords):
        return "risk"

    # 3. 특정 ETF 정보 질문 (단순 정보 요청)
    info_keywords = [
        "알려줘", "알려주세요", "뭐야", "뭐예요", "얼마", "무엇",
        "설명", "정보", "에 대해", "수익률", "종가", "거래량", "보유종목",
    ]
    if has_specific_etf and any(kw in q for kw in info_keywords):
        return "simple"

    # 4. 추천 질문 패턴
    recommend_keywords = [
        "추천", "좋은", "괜찮은", "어떤 etf", "뭐가 좋", "골라", "선택",
        "적합한", "알맞은", "찾아줘",
    ]
    if any(kw in q for kw in recommend_keywords):
        return "recommend"

    # 5. 특정 ETF 이름만 있으면 simple
    if has_specific_etf:
        return "simple"

    return "general"
