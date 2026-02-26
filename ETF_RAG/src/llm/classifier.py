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
    question_lower = question.lower()

    etf_names = ["kodex", "tiger", "코덱스", "타이거", "etf-"]
    has_specific_etf = any(name in question_lower for name in etf_names)

    # 1. 비교 질문 패턴 (최우선)
    compare_keywords = ["비교", "차이", "vs", "중에", "어떤게", "어떤 게", "둘 중"]
    compare_connectors = ["와 ", "과 "]
    has_compare_connector = any(conn in question_lower for conn in compare_connectors)
    has_compare_keyword = any(kw in question_lower for kw in compare_keywords)

    if has_compare_keyword or (has_compare_connector and has_specific_etf):
        return "compare"

    # 2. 위험/주의 질문 패턴
    risk_keywords = ["위험", "리스크", "주의", "손실", "안전", "변동성"]
    if any(kw in question_lower for kw in risk_keywords):
        return "risk"

    # 3. 특정 ETF 정보 질문 (단순 정보 요청)
    info_keywords = ["알려줘", "뭐야", "뭐예요", "얼마", "무엇", "설명", "정보", "에 대해"]
    if has_specific_etf and any(kw in question_lower for kw in info_keywords):
        return "simple"

    # 4. 추천 질문 패턴
    recommend_keywords = ["추천", "좋은", "괜찮은", "어떤 etf", "뭐가 좋", "골라", "선택"]
    if any(kw in question_lower for kw in recommend_keywords):
        return "recommend"

    # 5. 특정 ETF 이름만 있으면 simple
    if has_specific_etf:
        return "simple"

    return "general"
