"""
ETF/주식 RAG 도구 정의 — LangGraph Function Calling용

각 도구는 에이전트가 자동으로 호출할 수 있는 함수.
retriever와 documents는 모듈 레벨에서 set_retriever()로 주입.
"""

import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 모듈 레벨 retriever — app.py에서 초기화 후 주입
_retriever = None          # ETF용 (또는 ETF+주식 통합)
_stock_retriever = None    # 주식 전용
_documents = None


def set_retriever(retriever, documents=None, stock_retriever=None):
    """앱 초기화 시 retriever와 documents를 주입"""
    global _retriever, _documents, _stock_retriever
    _retriever = retriever
    _documents = documents or (retriever.documents if hasattr(retriever, "documents") else [])
    _stock_retriever = stock_retriever


@tool
def search_etf(query: str) -> str:
    """ETF 관련 질문에 대해 데이터베이스를 검색합니다.
    ETF 가격, 수익률, NAV, 거래량, 보유종목 등 모든 ETF 정보를 검색할 수 있습니다.
    검색 결과가 없으면 빈 문자열을 반환합니다.

    Args:
        query: 검색할 ETF 관련 질문 또는 키워드
    """
    if _retriever is None:
        return "검색기가 초기화되지 않았습니다."

    from src.rag.retriever import retrieve_relevant_docs
    context, sources = retrieve_relevant_docs(_retriever, query)

    if not context:
        return ""

    # 출처 정보 포맷팅
    source_info = "\n".join(
        f"- [{s['ticker']}] {s['name']} (관련도: {s['relevance_score']:.0f}%)"
        for s in sources
    )
    return f"{context}\n\n[검색된 ETF]\n{source_info}"


@tool
def compare_etfs(etf_name_1: str, etf_name_2: str) -> str:
    """두 ETF를 비교 분석합니다. 각 ETF의 가격, 수익률, 보유종목 등을 나란히 비교합니다.

    Args:
        etf_name_1: 첫 번째 ETF 이름 또는 티커 (예: "KODEX 200", "069500")
        etf_name_2: 두 번째 ETF 이름 또는 티커 (예: "TIGER 200", "102110")
    """
    if _retriever is None:
        return "검색기가 초기화되지 않았습니다."

    from src.rag.retriever import retrieve_relevant_docs

    # 각 ETF 개별 검색
    ctx1, src1 = retrieve_relevant_docs(_retriever, etf_name_1, k=1)
    ctx2, src2 = retrieve_relevant_docs(_retriever, etf_name_2, k=1)

    parts = []
    if ctx1:
        parts.append(f"[ETF 1: {etf_name_1}]\n{ctx1}")
    else:
        parts.append(f"[ETF 1: {etf_name_1}]\n해당 ETF 데이터를 찾지 못했습니다.")

    if ctx2:
        parts.append(f"[ETF 2: {etf_name_2}]\n{ctx2}")
    else:
        parts.append(f"[ETF 2: {etf_name_2}]\n해당 ETF 데이터를 찾지 못했습니다.")

    return "\n\n---\n\n".join(parts)


@tool
def get_etf_list(category: str = "") -> str:
    """특정 카테고리나 키워드에 해당하는 ETF 목록을 검색합니다.
    추천, 카테고리 탐색, 목록 조회 등에 사용합니다.

    Args:
        category: ETF 카테고리 또는 키워드 (예: "반도체", "2차전지", "배당", "인버스")
    """
    if _retriever is None:
        return "검색기가 초기화되지 않았습니다."

    from src.rag.retriever import retrieve_relevant_docs
    context, sources = retrieve_relevant_docs(_retriever, category, k=5)

    if not context:
        return f"'{category}' 관련 ETF를 찾지 못했습니다."

    source_info = "\n".join(
        f"- [{s['ticker']}] {s['name']} (관련도: {s['relevance_score']:.0f}%)"
        for s in sources
    )
    return f"{context}\n\n[검색된 ETF 목록]\n{source_info}"


@tool
def search_stock(query: str) -> str:
    """주식(개별 종목) 관련 질문에 대해 데이터베이스를 검색합니다.
    주식 가격, PER, PBR, 시가총액, 배당, 수익률 등 주식 정보를 검색할 수 있습니다.
    ETF가 아닌 일반 주식(삼성전자, SK하이닉스 등)에 대한 질문에 사용합니다.
    검색 결과가 없으면 빈 문자열을 반환합니다.

    Args:
        query: 검색할 주식 관련 질문 또는 키워드
    """
    retriever = _stock_retriever or _retriever
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
    return f"{context}\n\n[검색된 종목]\n{source_info}"


# 에이전트에 바인딩할 도구 목록
ALL_TOOLS = [search_etf, compare_etfs, get_etf_list, search_stock]
