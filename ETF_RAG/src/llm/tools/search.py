"""
검색 도구 — search_etf, search_stock, get_etf_list, get_stock_list
"""

import logging

from langchain_core.tools import tool

from src.llm.tools import _state
from src.llm.tools._helpers import _enrich_with_structured_data

logger = logging.getLogger(__name__)


@tool
def search_etf(query: str) -> str:
    """ETF 관련 질문에 대해 데이터베이스를 검색합니다.
    ETF 가격, 수익률, NAV, 거래량, 보유종목 등 모든 ETF 정보를 검색할 수 있습니다.
    검색 결과가 없으면 빈 문자열을 반환합니다.

    Args:
        query: 검색할 ETF 관련 질문 또는 키워드
    """
    if _state._retriever is None:
        return "검색기가 초기화되지 않았습니다."

    from src.rag.retriever import retrieve_relevant_docs
    context, sources = retrieve_relevant_docs(_state._retriever, query)

    if not context:
        return ""

    # 출처 정보 포맷팅
    source_info = "\n".join(
        f"- [{s['ticker']}] {s['name']} (관련도: {s['relevance_score']:.0f}%)"
        for s in sources
    )
    result = f"{context}\n\n[검색된 ETF]\n{source_info}"

    # 구조화 데이터 보강
    enrichment = _enrich_with_structured_data(sources, _state._etf_data_index)
    if enrichment:
        result += enrichment

    return result


@tool
def search_stock(query: str) -> str:
    """주식(개별 종목) 관련 질문에 대해 데이터베이스를 검색합니다.
    주식 가격, PER, PBR, 시가총액, 배당, 수익률 등 주식 정보를 검색할 수 있습니다.
    ETF가 아닌 일반 주식(삼성전자, SK하이닉스 등)에 대한 질문에 사용합니다.
    검색 결과가 없으면 빈 문자열을 반환합니다.

    Args:
        query: 검색할 주식 관련 질문 또는 키워드
    """
    retriever = _state._stock_retriever or _state._retriever
    if retriever is None:
        return "검색기가 초기화되지 않았습니다."

    from src.rag.retriever import retrieve_relevant_docs
    context, sources = retrieve_relevant_docs(retriever, query)

    if not context:
        return ""

    source_info = "\n".join(
        f"- [{s['ticker']}] {s['name']} (관련도: {s['relevance_score']:.0f}%)"
        for s in sources
    )
    result = f"{context}\n\n[검색된 종목]\n{source_info}"

    # 구조화 데이터 보강
    enrichment = _enrich_with_structured_data(sources, _state._stock_data_index)
    if enrichment:
        result += enrichment

    return result


@tool
def get_etf_list(category: str = "") -> str:
    """특정 카테고리나 키워드에 해당하는 ETF 목록을 검색합니다.
    추천, 카테고리 탐색, 목록 조회 등에 사용합니다.

    Args:
        category: ETF 카테고리 또는 키워드 (예: "반도체", "2차전지", "배당", "인버스")
    """
    if _state._retriever is None:
        return "검색기가 초기화되지 않았습니다."

    from src.rag.retriever import retrieve_relevant_docs
    context, sources = retrieve_relevant_docs(_state._retriever, category, k=5)

    if not context:
        return f"'{category}' 관련 ETF를 찾지 못했습니다."

    source_info = "\n".join(
        f"- [{s['ticker']}] {s['name']} (관련도: {s['relevance_score']:.0f}%)"
        for s in sources
    )
    return f"{context}\n\n[검색된 ETF 목록]\n{source_info}"


@tool
def get_stock_list(category: str = "") -> str:
    """특정 카테고리나 키워드에 해당하는 주식 종목 목록을 검색합니다.
    업종, 테마, 특성 등으로 주식 종목을 탐색할 때 사용합니다.

    Args:
        category: 주식 카테고리 또는 키워드 (예: "반도체", "자동차", "배당", "대형주", "바이오")
    """
    retriever = _state._stock_retriever or _state._retriever
    if retriever is None:
        return "검색기가 초기화되지 않았습니다."

    from src.rag.retriever import retrieve_relevant_docs
    context, sources = retrieve_relevant_docs(retriever, category, k=5)

    if not context:
        return f"'{category}' 관련 주식 종목을 찾지 못했습니다."

    source_info = "\n".join(
        f"- [{s['ticker']}] {s['name']} (관련도: {s['relevance_score']:.0f}%)"
        for s in sources
    )
    result = f"{context}\n\n[검색된 종목 목록]\n{source_info}"

    # 구조화 데이터 보강
    enrichment = _enrich_with_structured_data(sources, _state._stock_data_index)
    if enrichment:
        result += enrichment

    return result
