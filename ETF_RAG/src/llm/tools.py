"""
ETF/주식 RAG 도구 정의 — LangGraph Function Calling용

각 도구는 에이전트가 자동으로 호출할 수 있는 함수.
retriever와 documents는 모듈 레벨에서 set_retriever()로 주입.
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 모듈 레벨 retriever — app.py에서 초기화 후 주입
_retriever = None          # ETF용 (또는 ETF+주식 통합)
_stock_retriever = None    # 주식 전용
_documents = None

# 구조화 데이터 인덱스 — 이름/티커로 원본 dict 직접 조회
_etf_data_index = {}       # {name_lower: dict, ticker: dict}
_stock_data_index = {}     # {name_lower: dict, ticker: dict}


def _build_data_index(data_list):
    """데이터 리스트에서 이름/티커 → dict 인덱스 구축"""
    index = {}
    for item in data_list:
        name = item.get("name", "")
        ticker = item.get("ticker", "")
        if name:
            index[name.lower()] = item
        if ticker:
            index[ticker] = item
    return index


def set_retriever(retriever, documents=None, stock_retriever=None,
                  etf_data=None, stock_data=None):
    """앱 초기화 시 retriever와 documents를 주입"""
    global _retriever, _documents, _stock_retriever
    global _etf_data_index, _stock_data_index
    _retriever = retriever
    _documents = documents or (retriever.documents if hasattr(retriever, "documents") else [])
    _stock_retriever = stock_retriever
    if etf_data is not None:
        _etf_data_index = _build_data_index(etf_data)
    if stock_data is not None:
        _stock_data_index = _build_data_index(stock_data)


def _enrich_with_structured_data(sources: list, index: dict) -> str:
    """검색 출처의 종목에 대해 구조화 데이터를 보강 텍스트로 반환"""
    enriched = []
    for s in sources:
        ticker = s.get("ticker", "")
        name = s.get("name", "")
        data = index.get(ticker) or index.get(name.lower()) if index else None
        if not data:
            continue

        returns = data.get("returns", {})
        returns_parts = []
        labels = {"1d": "1일", "1w": "1주", "1m": "1개월", "3m": "3개월", "1y": "1년"}
        for k, label in labels.items():
            v = returns.get(k)
            if v is not None:
                returns_parts.append(f"{label}: {v:+.2f}%")

        line = f"[{data['name']}] 종가: {data.get('close', 0):,}원, 등락률: {data.get('change_pct', 0):+.2f}%"
        if returns_parts:
            line += f", 수익률({', '.join(returns_parts)})"

        # ETF 전용
        if "nav" in data:
            line += f", NAV: {data.get('nav', 0):,.0f}원"
        # 주식 전용
        if "per" in data:
            line += f", PER: {data.get('per', 0):.2f}배"

        enriched.append(line)

    if not enriched:
        return ""
    return "\n\n[실시간 데이터 요약]\n" + "\n".join(enriched)


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
    result = f"{context}\n\n[검색된 ETF]\n{source_info}"

    # 구조화 데이터 보강
    enrichment = _enrich_with_structured_data(sources, _etf_data_index)
    if enrichment:
        result += enrichment

    return result


def _find_structured_data(name_or_ticker: str) -> Optional[dict]:
    """이름 또는 티커로 구조화 데이터 조회 (ETF → 주식 순)"""
    key = name_or_ticker.lower().strip()

    # ETF 인덱스에서 정확 매칭
    if key in _etf_data_index:
        return _etf_data_index[key]

    # 주식 인덱스에서 정확 매칭
    if key in _stock_data_index:
        return _stock_data_index[key]

    # 부분 매칭 (이름에 포함)
    for index in (_etf_data_index, _stock_data_index):
        for idx_key, data in index.items():
            if key in idx_key or idx_key in key:
                return data

    return None


def _extract_comparison_fields(data: dict) -> dict:
    """비교용 핵심 필드 추출 (ETF/주식 공통 + 개별)"""
    fields = {
        "name": data.get("name", ""),
        "ticker": data.get("ticker", ""),
        "close": data.get("close", 0),
        "change_pct": data.get("change_pct", 0),
        "volume": data.get("volume", 0),
        "trade_value": data.get("trade_value", 0),
    }

    # 수익률
    returns = data.get("returns", {})
    for period in ("1d", "1w", "1m", "3m", "1y"):
        fields[f"return_{period}"] = returns.get(period)

    # ETF 전용
    if "nav" in data:
        fields["nav"] = data.get("nav", 0)
        fields["deviation"] = data.get("deviation")
        fields["tracking_error"] = data.get("tracking_error")
        fields["asset_type"] = "etf"
        # 보유종목 상위 3개
        holdings = data.get("holdings", [])[:3]
        fields["top_holdings"] = [
            {"name": h.get("stock_name", ""), "weight": h.get("weight", 0)}
            for h in holdings
        ]

    # 주식 전용
    if "per" in data or "pbr" in data:
        fields["per"] = data.get("per", 0)
        fields["pbr"] = data.get("pbr", 0)
        fields["eps"] = data.get("eps", 0)
        fields["market_cap"] = data.get("market_cap", 0)
        fields["div"] = data.get("div", 0)
        fields["asset_type"] = "stock"

    if "asset_type" not in fields:
        fields["asset_type"] = "unknown"

    return fields


@tool
def compare_etfs(etf_name_1: str, etf_name_2: str) -> str:
    """두 ETF 또는 주식을 비교 분석합니다. 각 종목의 가격, 수익률, 보유종목 등을 나란히 비교합니다.

    Args:
        etf_name_1: 첫 번째 ETF/주식 이름 또는 티커 (예: "KODEX 200", "069500")
        etf_name_2: 두 번째 ETF/주식 이름 또는 티커 (예: "TIGER 200", "102110")
    """
    if _retriever is None:
        return "검색기가 초기화되지 않았습니다."

    # 구조화 데이터 직접 조회 시도
    d1 = _find_structured_data(etf_name_1)
    d2 = _find_structured_data(etf_name_2)

    if d1 and d2:
        comparison = {
            "__type__": "comparison_table",
            "items": [
                _extract_comparison_fields(d1),
                _extract_comparison_fields(d2),
            ],
        }
        # 구조화 JSON + 텍스트 컨텍스트 모두 반환
        # (LLM이 텍스트를 참조해서 자연어 답변 생성)
        text_parts = []
        for name, data in [(etf_name_1, d1), (etf_name_2, d2)]:
            text_parts.append(f"[{data['name']}] 종가: {data.get('close', 0):,}원, "
                              f"등락률: {data.get('change_pct', 0):+.2f}%")
        structured_json = json.dumps(comparison, ensure_ascii=False)
        return f"{structured_json}\n\n---\n\n" + "\n".join(text_parts)

    # 구조화 데이터 없으면 기존 텍스트 검색 fallback
    from src.rag.retriever import retrieve_relevant_docs

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
    result = f"{context}\n\n[검색된 종목]\n{source_info}"

    # 구조화 데이터 보강
    enrichment = _enrich_with_structured_data(sources, _stock_data_index)
    if enrichment:
        result += enrichment

    return result


@tool
def get_realtime_price(name_or_ticker: str) -> str:
    """ETF나 주식의 현재 가격을 조회합니다. 장중에는 실시간(15분 지연) 데이터를,
    장 마감 후에는 가장 최근 종가 데이터를 반환합니다.
    "현재 가격", "지금 얼마", "실시간 시세" 등의 질문에 사용합니다.

    Args:
        name_or_ticker: ETF/주식 이름 또는 티커 (예: "KODEX 200", "069500", "삼성전자")
    """
    from config import REALTIME_PRICE

    # 종목 조회
    data = _find_structured_data(name_or_ticker)
    if not data:
        return f"'{name_or_ticker}'에 해당하는 종목을 찾을 수 없습니다."

    ticker = data.get("ticker", "")
    name = data.get("name", "")
    asset_type = "etf" if "nav" in data else "stock"

    # 장중 실시간 조회 시도
    if REALTIME_PRICE.get("enabled", True):
        try:
            from src.data.realtime import get_realtime_price as _get_rt
            rt = _get_rt(ticker, asset_type,
                         cache_ttl=REALTIME_PRICE.get("cache_ttl", 300))
            if rt:
                line = f"[{name} ({ticker})] 현재가: {rt['price']:,}원"
                if rt["change"] is not None:
                    line += f", 전일대비: {rt['change']:+,}원 ({rt['change_pct']:+.2f}%)"
                if rt.get("volume"):
                    line += f", 거래량: {rt['volume']:,}주"
                line += f"\n(yfinance 15분 지연 데이터, 조회시각: {rt['timestamp']})"
                return line
        except Exception as e:
            logger.warning(f"실시간 가격 조회 실패: {e}")

    # Fallback: pykrx 구조화 데이터
    close = data.get("close", 0)
    change_pct = data.get("change_pct", 0)
    date = data.get("date", "")
    if len(date) == 8:
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    line = f"[{name} ({ticker})] 종가: {close:,}원, 등락률: {change_pct:+.2f}%"

    # 수익률 정보 추가
    returns = data.get("returns", {})
    if returns:
        labels = {"1d": "1일", "1w": "1주", "1m": "1개월", "3m": "3개월", "1y": "1년"}
        parts = []
        for k, label in labels.items():
            v = returns.get(k)
            if v is not None:
                parts.append(f"{label}: {v:+.2f}%")
        if parts:
            line += f"\n수익률: {', '.join(parts)}"

    # ETF 전용
    if "nav" in data:
        line += f"\nNAV: {data.get('nav', 0):,.0f}원"

    # 주식 전용
    if "per" in data:
        line += f"\nPER: {data.get('per', 0):.2f}배, PBR: {data.get('pbr', 0):.2f}배"

    try:
        from src.data.realtime import is_market_open
        if is_market_open():
            line += f"\n(실시간 데이터 조회 실패, 최근 수집 데이터 기준일: {date})"
        else:
            line += f"\n(장 마감 후 데이터, 기준일: {date})"
    except ImportError:
        line += f"\n(수집 데이터, 기준일: {date})"

    return line


# 에이전트에 바인딩할 도구 목록
ALL_TOOLS = [search_etf, compare_etfs, get_etf_list, search_stock, get_realtime_price]
