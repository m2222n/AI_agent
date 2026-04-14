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

# 역인덱스 — 보유종목 → ETF 매핑 (섹터 분석용)
_holdings_reverse_index = {}   # {stock_ticker: [{etf_name, etf_ticker, weight}]}

# 업종 인덱스 — 업종명 → 종목 리스트 매핑
_sector_index = {}             # {sector_name: [{name, ticker, per, pbr, market_cap, ...}]}


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


def _build_holdings_reverse_index(etf_data_list):
    """보유종목 → ETF 역인덱스 구축 (종목이 어떤 ETF에 담겨있는지)"""
    reverse = {}
    for etf in etf_data_list:
        etf_name = etf.get("name", "")
        etf_ticker = etf.get("ticker", "")
        for h in etf.get("holdings", []):
            stock_ticker = h.get("stock_ticker", "")
            stock_name = h.get("stock_name", "")
            if not stock_ticker:
                continue
            entry = {
                "etf_name": etf_name,
                "etf_ticker": etf_ticker,
                "weight": h.get("weight", 0),
                "stock_name": stock_name,
            }
            reverse.setdefault(stock_ticker, []).append(entry)
            # 종목명으로도 조회 가능하게
            if stock_name:
                reverse.setdefault(stock_name.lower(), []).append(entry)
    return reverse


def _build_sector_index(stock_data_list):
    """업종별 종목 인덱스 구축 — 업종명으로 종목 목록 조회"""
    index = {}
    for s in stock_data_list:
        sector = s.get("sector", "")
        if not sector:
            continue
        entry = {
            "name": s.get("name", ""),
            "ticker": s.get("ticker", ""),
            "close": s.get("close", 0),
            "change_pct": s.get("change_pct", 0),
            "market_cap": s.get("market_cap", 0),
            "trade_value": s.get("trade_value", 0),
            "per": s.get("per", 0),
            "pbr": s.get("pbr", 0),
            "eps": s.get("eps", 0),
            "div": s.get("div", 0),
        }
        index.setdefault(sector, []).append(entry)
    # 각 업종 내 시가총액 기준 정렬
    for sector in index:
        index[sector].sort(key=lambda x: x["market_cap"], reverse=True)
    return index


def set_retriever(retriever, documents=None, stock_retriever=None,
                  etf_data=None, stock_data=None):
    """앱 초기화 시 retriever와 documents를 주입"""
    global _retriever, _documents, _stock_retriever
    global _etf_data_index, _stock_data_index, _holdings_reverse_index, _sector_index
    _retriever = retriever
    _documents = documents or (retriever.documents if hasattr(retriever, "documents") else [])
    _stock_retriever = stock_retriever
    if etf_data is not None:
        _etf_data_index = _build_data_index(etf_data)
        _holdings_reverse_index = _build_holdings_reverse_index(etf_data)
    if stock_data is not None:
        _stock_data_index = _build_data_index(stock_data)
        _sector_index = _build_sector_index(stock_data)


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
            per = data.get("per", 0)
            pbr = data.get("pbr", 0)
            line += f", PER: {per:.2f}배, PBR: {pbr:.2f}배"
            mcap = data.get("market_cap", 0)
            if mcap:
                if mcap >= 1_0000_0000_0000:  # 조 단위
                    line += f", 시가총액: {mcap / 1_0000_0000_0000:.1f}조원"
                else:
                    line += f", 시가총액: {mcap / 1_0000_0000:,.0f}억원"
            div_rate = data.get("div", 0)
            if div_rate:
                line += f", 배당수익률: {div_rate:.2f}%"
            eps = data.get("eps", 0)
            if eps:
                line += f", EPS: {eps:,.0f}원"

        enriched.append(line)

        # 최근 분기 실적 추가 (DB에서 조회)
        if "per" in data and ticker:
            try:
                from src.data.database import get_latest_financial_summary, get_connection
                fin_conn = get_connection()
                fin = get_latest_financial_summary(fin_conn, ticker)
                fin_conn.close()
                if fin:
                    fy = fin.get("fiscal_year", "")
                    fq = fin.get("fiscal_quarter", "")
                    rev = fin.get("revenue")
                    op = fin.get("operating_profit")
                    om = fin.get("operating_margin")
                    rg = fin.get("revenue_growth_yoy")
                    parts = [f"최근 실적({fy}Q{fq})"]
                    if rev:
                        if abs(rev) >= 1_0000_0000_0000:
                            parts.append(f"매출 {rev / 1_0000_0000_0000:.1f}조")
                        else:
                            parts.append(f"매출 {rev / 1_0000_0000:,.0f}억")
                    if op is not None and om is not None:
                        if abs(op) >= 1_0000_0000_0000:
                            parts.append(f"영업이익 {op / 1_0000_0000_0000:.1f}조(마진 {om:.1f}%)")
                        else:
                            parts.append(f"영업이익 {op / 1_0000_0000:,.0f}억(마진 {om:.1f}%)")
                    if rg is not None:
                        parts.append(f"매출 YoY {rg:+.1f}%")
                    if len(parts) > 1:
                        enriched.append("  " + ", ".join(parts))
            except Exception:
                pass  # 재무 데이터 없으면 조용히 무시

        # 보유종목 정보 추가 (ETF)
        holdings = data.get("holdings", [])
        if holdings:
            top_h = holdings[:10]
            h_parts = [
                f"{h.get('stock_name', '?')}({h.get('weight', 0):.1f}%)"
                for h in top_h
            ]
            enriched.append(f"  보유종목(상위 {len(top_h)}개): " + ", ".join(h_parts))

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
        fields["bps"] = data.get("bps", 0)
        fields["market_cap"] = data.get("market_cap", 0)
        fields["div"] = data.get("div", 0)
        fields["dps"] = data.get("dps", 0)
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
def compare_stocks(stock_name_1: str, stock_name_2: str) -> str:
    """두 주식을 비교 분석합니다. PER, PBR, 시가총액, 배당수익률, 수익률 등을 나란히 비교합니다.
    주식 vs 주식 비교에 특화된 도구입니다.

    Args:
        stock_name_1: 첫 번째 주식 이름 또는 티커 (예: "삼성전자", "005930")
        stock_name_2: 두 번째 주식 이름 또는 티커 (예: "SK하이닉스", "000660")
    """
    # 구조화 데이터 직접 조회
    d1 = _find_structured_data(stock_name_1)
    d2 = _find_structured_data(stock_name_2)

    if d1 and d2:
        comparison = {
            "__type__": "comparison_table",
            "items": [
                _extract_comparison_fields(d1),
                _extract_comparison_fields(d2),
            ],
        }
        text_parts = []
        for name, data in [(stock_name_1, d1), (stock_name_2, d2)]:
            line = f"[{data['name']}] 종가: {data.get('close', 0):,}원"
            line += f", 등락률: {data.get('change_pct', 0):+.2f}%"
            if "per" in data:
                line += f", PER: {data.get('per', 0):.2f}배, PBR: {data.get('pbr', 0):.2f}배"
                mcap = data.get("market_cap", 0)
                if mcap:
                    if mcap >= 1_0000_0000_0000:
                        line += f", 시가총액: {mcap / 1_0000_0000_0000:.1f}조원"
                    else:
                        line += f", 시가총액: {mcap / 1_0000_0000:,.0f}억원"
                div_rate = data.get("div", 0)
                if div_rate:
                    line += f", 배당: {div_rate:.2f}%"
            text_parts.append(line)
        structured_json = json.dumps(comparison, ensure_ascii=False)
        return f"{structured_json}\n\n---\n\n" + "\n".join(text_parts)

    # 구조화 데이터 없으면 텍스트 검색 fallback
    retriever = _stock_retriever or _retriever
    if retriever is None:
        return "검색기가 초기화되지 않았습니다."

    from src.rag.retriever import retrieve_relevant_docs
    ctx1, src1 = retrieve_relevant_docs(retriever, stock_name_1, k=1)
    ctx2, src2 = retrieve_relevant_docs(retriever, stock_name_2, k=1)

    parts = []
    if ctx1:
        parts.append(f"[주식 1: {stock_name_1}]\n{ctx1}")
    else:
        parts.append(f"[주식 1: {stock_name_1}]\n해당 종목 데이터를 찾지 못했습니다.")
    if ctx2:
        parts.append(f"[주식 2: {stock_name_2}]\n{ctx2}")
    else:
        parts.append(f"[주식 2: {stock_name_2}]\n해당 종목 데이터를 찾지 못했습니다.")

    return "\n\n---\n\n".join(parts)


@tool
def get_stock_list(category: str = "") -> str:
    """특정 카테고리나 키워드에 해당하는 주식 종목 목록을 검색합니다.
    업종, 테마, 특성 등으로 주식 종목을 탐색할 때 사용합니다.

    Args:
        category: 주식 카테고리 또는 키워드 (예: "반도체", "자동차", "배당", "대형주", "바이오")
    """
    retriever = _stock_retriever or _retriever
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


def _format_cap(value: int) -> str:
    """시가총액을 조/억 단위 문자열로"""
    if value >= 1_0000_0000_0000:
        return f"{value / 1_0000_0000_0000:.1f}조"
    elif value >= 1_0000_0000:
        return f"{value / 1_0000_0000:,.0f}억"
    return f"{value:,}"


@tool
def analyze_sector(query: str) -> str:
    """특정 종목이 포함된 ETF를 찾거나, 업종별 종목 분석(PER/PBR 비교, 시가총액 상위)을 수행합니다.
    "삼성전자 담고 있는 ETF", "전기전자 업종 분석", "은행 업종 PER 비교", "반도체 관련 ETF" 등의 질문에 사용합니다.

    Args:
        query: 종목명/티커, 업종명, 또는 섹터 키워드 (예: "삼성전자", "전기·전자", "반도체", "은행")
    """
    query_lower = query.lower().strip()

    # 1. 업종 인덱스 검색 — 정확 매칭 또는 부분 매칭
    if _sector_index:
        matched_sector = None
        matched_stocks = None

        # 정확 매칭
        for sector_name, stocks in _sector_index.items():
            if sector_name.lower() == query_lower or sector_name == query:
                matched_sector = sector_name
                matched_stocks = stocks
                break

        # 부분 매칭
        if not matched_sector:
            for sector_name, stocks in _sector_index.items():
                if query_lower in sector_name.lower():
                    matched_sector = sector_name
                    matched_stocks = stocks
                    break

        if matched_sector and matched_stocks:
            return _format_sector_analysis(matched_sector, matched_stocks)

    # 2. 보유종목 역인덱스 — 종목→ETF 매핑
    if not _holdings_reverse_index:
        return "보유종목/업종 데이터가 없습니다. 데이터 수집 후 이용 가능합니다."

    # 2-1. 정확 매칭 — 종목명 또는 티커로 직접 조회
    matches = _holdings_reverse_index.get(query_lower) or _holdings_reverse_index.get(query)
    if matches:
        stock_name = matches[0].get("stock_name", query)
        sorted_matches = sorted(matches, key=lambda x: x["weight"], reverse=True)
        seen = set()
        unique = []
        for m in sorted_matches:
            if m["etf_ticker"] not in seen:
                seen.add(m["etf_ticker"])
                unique.append(m)

        lines = [f"[{stock_name}]을(를) 보유한 ETF ({len(unique)}개):\n"]
        for m in unique[:15]:
            lines.append(
                f"- [{m['etf_ticker']}] {m['etf_name']} (비중: {m['weight']:.2f}%)"
            )
        if len(unique) > 15:
            lines.append(f"  ... 외 {len(unique) - 15}개")

        avg_weight = sum(m["weight"] for m in unique) / len(unique)
        max_m = unique[0]
        lines.append(f"\n[통계] 평균 비중: {avg_weight:.2f}%, "
                      f"최대 비중: {max_m['etf_name']} ({max_m['weight']:.2f}%)")

        # 해당 종목의 업종 정보 + 밸류에이션 위치
        stock_data = _stock_data_index.get(query_lower) or _stock_data_index.get(query)
        if stock_data and stock_data.get("sector"):
            sector = stock_data["sector"]
            lines.append(f"\n[업종] {stock_name}: {sector}")

            if sector in _sector_index:
                sector_stocks = _sector_index[sector]
                # 밸류에이션 상대 위치
                val_info = _format_valuation_position(stock_data, sector_stocks)
                if val_info:
                    lines.append(val_info)

                # 동일 업종 종목
                peers = [s for s in sector_stocks
                         if s["ticker"] != stock_data.get("ticker", "")][:5]
                if peers:
                    peer_names = ", ".join(p["name"] for p in peers)
                    lines.append(f"[동일 업종] {peer_names}")

        return "\n".join(lines)

    # 2-2. 부분 매칭 — 키워드로 종목명 검색
    keyword_matches = {}
    for key, entries in _holdings_reverse_index.items():
        if not key.replace(" ", "").isalpha() and not any(
            '\uac00' <= c <= '\ud7a3' for c in key
        ):
            continue
        if query_lower in key:
            for e in entries:
                st = e.get("stock_name", "")
                if st not in keyword_matches:
                    keyword_matches[st] = {"stock_name": st, "etfs": []}
                if e["etf_ticker"] not in [x["etf_ticker"] for x in keyword_matches[st]["etfs"]]:
                    keyword_matches[st]["etfs"].append(e)

    if keyword_matches:
        lines = [f"'{query}' 관련 종목을 보유한 ETF:\n"]
        for stock_name, info in sorted(
            keyword_matches.items(),
            key=lambda x: len(x[1]["etfs"]),
            reverse=True,
        )[:5]:
            etfs = sorted(info["etfs"], key=lambda x: x["weight"], reverse=True)
            lines.append(f"**{stock_name}** ({len(etfs)}개 ETF에 편입)")
            for e in etfs[:5]:
                lines.append(
                    f"  - [{e['etf_ticker']}] {e['etf_name']} (비중: {e['weight']:.2f}%)"
                )
            if len(etfs) > 5:
                lines.append(f"    ... 외 {len(etfs) - 5}개")
            lines.append("")
        return "\n".join(lines)

    return f"'{query}'에 해당하는 업종/보유종목 정보를 찾지 못했습니다."


def _calc_percentile(value: float, values: list[float]) -> float:
    """값이 리스트에서 몇 번째 백분위인지 계산 (0~100)."""
    if not values or value is None:
        return 50.0
    below = sum(1 for v in values if v < value)
    return round(below / len(values) * 100, 1)


def _format_valuation_position(stock_data: dict, sector_stocks: list) -> str:
    """종목의 업종 내 밸류에이션 상대 위치를 포맷팅."""
    ticker = stock_data.get("ticker", "")
    parts = []

    # PER 위치
    per = stock_data.get("per", 0)
    pers = [s["per"] for s in sector_stocks if s["per"] and s["per"] > 0]
    if per and per > 0 and len(pers) >= 3:
        pctile = _calc_percentile(per, pers)
        avg_per = sum(pers) / len(pers)
        diff_pct = (per - avg_per) / avg_per * 100
        if diff_pct > 20:
            label = "고평가"
        elif diff_pct < -20:
            label = "저평가"
        else:
            label = "평균 수준"
        parts.append(f"PER {per:.1f}배 (업종 평균 {avg_per:.1f}, "
                     f"상위 {100 - pctile:.0f}%, {label})")

    # PBR 위치
    pbr = stock_data.get("pbr", 0)
    pbrs = [s["pbr"] for s in sector_stocks if s["pbr"] and s["pbr"] > 0]
    if pbr and pbr > 0 and len(pbrs) >= 3:
        pctile = _calc_percentile(pbr, pbrs)
        avg_pbr = sum(pbrs) / len(pbrs)
        parts.append(f"PBR {pbr:.2f}배 (업종 평균 {avg_pbr:.2f}, "
                     f"상위 {100 - pctile:.0f}%)")

    # 배당수익률 위치
    div_rate = stock_data.get("div", 0)
    divs = [s["div"] for s in sector_stocks if s["div"] and s["div"] > 0]
    if div_rate and div_rate > 0 and len(divs) >= 3:
        pctile = _calc_percentile(div_rate, divs)
        avg_div = sum(divs) / len(divs)
        parts.append(f"배당 {div_rate:.2f}% (업종 평균 {avg_div:.2f}%, "
                     f"상위 {100 - pctile:.0f}%)")

    # 시가총액 순위
    mcap = stock_data.get("market_cap", 0)
    if mcap:
        caps_sorted = sorted(
            [s["market_cap"] for s in sector_stocks if s["market_cap"]],
            reverse=True
        )
        rank = next((i + 1 for i, c in enumerate(caps_sorted) if c <= mcap), len(caps_sorted))
        parts.append(f"시가총액 업종 내 {rank}/{len(caps_sorted)}위")

    if not parts:
        return ""
    return "[업종 내 밸류에이션 위치] " + " | ".join(parts)


def _format_sector_analysis(sector: str, stocks: list) -> str:
    """업종 분석 결과를 포맷팅 — 시가총액 상위 + PER/PBR 통계"""
    lines = [f"[{sector}] 업종 분석 ({len(stocks)}종목)\n"]

    # 시가총액 상위 10개
    lines.append("**시가총액 상위:**")
    for i, s in enumerate(stocks[:10], 1):
        per_str = f"PER {s['per']:.1f}" if s["per"] else "PER -"
        pbr_str = f"PBR {s['pbr']:.2f}" if s["pbr"] else "PBR -"
        cap_str = _format_cap(s["market_cap"])
        lines.append(
            f"{i}. [{s['ticker']}] {s['name']} | "
            f"종가 {s['close']:,}원 ({s['change_pct']:+.2f}%) | "
            f"시총 {cap_str} | {per_str} | {pbr_str}"
        )

    # 업종 PER/PBR 통계
    pers = [s["per"] for s in stocks if s["per"] and s["per"] > 0]
    pbrs = [s["pbr"] for s in stocks if s["pbr"] and s["pbr"] > 0]
    divs = [s["div"] for s in stocks if s["div"] and s["div"] > 0]

    lines.append(f"\n**업종 밸류에이션 통계 ({sector}):**")
    if pers:
        avg_per = sum(pers) / len(pers)
        min_per = min(pers)
        max_per = max(pers)
        sorted_pers = sorted(pers)
        median_per = sorted_pers[len(sorted_pers) // 2]
        lines.append(f"- PER: 평균 {avg_per:.1f}배, 중간값 {median_per:.1f}배 "
                     f"(최저 {min_per:.1f} ~ 최고 {max_per:.1f})")
        # PER 분포 구간
        ranges = [(0, 10), (10, 20), (20, 50), (50, float("inf"))]
        labels = ["0~10배", "10~20배", "20~50배", "50배 이상"]
        dist = [sum(1 for p in pers if lo <= p < hi) for lo, hi in ranges]
        dist_str = ", ".join(f"{l}: {c}개" for l, c in zip(labels, dist) if c > 0)
        lines.append(f"  분포: {dist_str}")
    if pbrs:
        avg_pbr = sum(pbrs) / len(pbrs)
        low_pbr = [s for s in stocks if s["pbr"] and 0 < s["pbr"] < 1]
        lines.append(f"- PBR: 평균 {avg_pbr:.2f}배 (PBR<1 저평가 {len(low_pbr)}종목)")
    if divs:
        avg_div = sum(divs) / len(divs)
        high_div = [s for s in stocks if s["div"] and s["div"] >= 3.0]
        lines.append(f"- 배당수익률: 평균 {avg_div:.2f}% (3% 이상 고배당 {len(high_div)}종목)")

    # 업종 시가총액 합계
    total_cap = sum(s["market_cap"] for s in stocks)
    lines.append(f"- 업종 시가총액 합계: {_format_cap(total_cap)}")

    return "\n".join(lines)


@tool
def get_technical_indicators(name_or_ticker: str) -> str:
    """ETF/주식의 기술적 지표를 분석합니다. 이동평균(MA), RSI, MACD, 볼린저 밴드, 골든크로스/데드크로스 판정 등을 제공합니다.
    "삼성전자 골든크로스 났어?", "KODEX 200 기술적 분석", "SK하이닉스 RSI" 등의 질문에 사용합니다.

    Args:
        name_or_ticker: ETF/주식 이름 또는 티커 (예: "삼성전자", "005930", "KODEX 200")
    """
    # 종목 조회
    data = _find_structured_data(name_or_ticker)
    if not data:
        return f"'{name_or_ticker}'에 해당하는 종목을 찾을 수 없습니다."

    ticker = data.get("ticker", "")
    name = data.get("name", "")

    try:
        from src.data.technical import get_technical_summary
        summary = get_technical_summary(ticker)
    except Exception as e:
        logger.warning(f"기술적 지표 계산 실패: {e}")
        return f"'{name}'의 기술적 지표 계산에 실패했습니다. (데이터 부족 또는 오류)"

    if not summary:
        return f"'{name}'의 일봉 데이터가 부족합니다 (최소 20일 필요)."

    # 포맷팅
    lines = [f"[{name} ({ticker})] 기술적 분석 (기준일: {_fmt_date(summary['date'])}, "
             f"종가: {summary['close']:,}원, 분석 기간: {summary['data_days']}일)\n"]

    # 이동평균
    ma = summary["ma"]
    lines.append("**이동평균(MA):**")
    for label, key in [("5일", "ma5"), ("20일", "ma20"), ("60일", "ma60"), ("120일", "ma120")]:
        val = ma.get(key)
        if val:
            diff = summary["close"] - val
            pct = diff / val * 100
            position = "위" if diff > 0 else "아래"
            lines.append(f"  - {label} MA: {val:,}원 (현재가 {position} {abs(pct):.1f}%)")

    # 추세
    lines.append(f"  - 추세 판정: **{summary['trend']}** (MA5 vs MA20 vs MA60 정배열 기준)")

    # 크로스
    cross = summary["cross"]
    cross_msgs = []
    for label, key in [("5일/20일", "5_20"), ("20일/60일", "20_60"), ("60일/120일", "60_120")]:
        val = cross.get(key)
        if val == "golden_cross":
            cross_msgs.append(f"  - ⚡ **{label} 골든크로스** 발생!")
        elif val == "dead_cross":
            cross_msgs.append(f"  - ⚠️ **{label} 데드크로스** 발생!")
    if cross_msgs:
        lines.append("\n**크로스 시그널:**")
        lines.extend(cross_msgs)
    else:
        lines.append("\n**크로스 시그널:** 최근 교차 없음")

    # RSI
    rsi = summary.get("rsi")
    if rsi is not None:
        if rsi >= 70:
            rsi_label = "과매수 구간 (매도 신호)"
        elif rsi <= 30:
            rsi_label = "과매도 구간 (매수 신호)"
        else:
            rsi_label = "중립 구간"
        lines.append(f"\n**RSI(14):** {rsi:.1f} — {rsi_label}")

    # MACD
    macd = summary.get("macd")
    if macd:
        macd_signal = "매수 우위" if macd["histogram"] > 0 else "매도 우위"
        lines.append(f"\n**MACD(12,26,9):**")
        lines.append(f"  - MACD: {macd['macd']:,.0f}, Signal: {macd['signal']:,.0f}, "
                     f"Histogram: {macd['histogram']:,.0f} ({macd_signal})")

    # 볼린저 밴드
    bb = summary.get("bollinger")
    if bb:
        lines.append(f"\n**볼린저 밴드(20,2):**")
        lines.append(f"  - 상단: {bb['upper']:,.0f}원, 중심: {bb['middle']:,.0f}원, "
                     f"하단: {bb['lower']:,.0f}원")
        lines.append(f"  - 밴드폭: {bb['width']:.1f}%, %B: {bb['pct_b']:.1f}%")
        if bb["pct_b"] > 100:
            lines.append("  - 상단 돌파 (과매수 가능성)")
        elif bb["pct_b"] < 0:
            lines.append("  - 하단 이탈 (과매도 가능성)")

    return "\n".join(lines)


def _fmt_date(date_str: str) -> str:
    """YYYYMMDD → YYYY-MM-DD"""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


@tool
def get_stock_correlation(ticker1: str, ticker2: str) -> str:
    """두 종목 간의 상관관계와 베타 계수를 분석합니다.
    "삼성전자와 SK하이닉스 상관관계", "KODEX 200과 삼성전자 베타", "두 종목 연동성" 등의 질문에 사용합니다.

    Args:
        ticker1: 첫 번째 종목 이름 또는 티커 (예: "삼성전자", "005930")
        ticker2: 두 번째 종목 이름 또는 티커 (예: "SK하이닉스", "000660", "KODEX 200")
    """
    # 종목 조회
    d1 = _find_structured_data(ticker1)
    d2 = _find_structured_data(ticker2)

    if not d1:
        return f"'{ticker1}'에 해당하는 종목을 찾을 수 없습니다."
    if not d2:
        return f"'{ticker2}'에 해당하는 종목을 찾을 수 없습니다."

    t1 = d1.get("ticker", "")
    t2 = d2.get("ticker", "")
    n1 = d1.get("name", "")
    n2 = d2.get("name", "")

    try:
        from src.data.technical import calc_correlation, calc_beta, MARKET_BENCHMARK

        lines = [f"[{n1} ({t1}) vs {n2} ({t2})] 상관관계 분석\n"]

        # 상관계수 계산
        corr = calc_correlation(t1, t2, days=120)
        if corr:
            c = corr["correlation"]
            if c >= 0.7:
                label = "강한 양의 상관관계 (동반 등락 경향)"
            elif c >= 0.3:
                label = "약한 양의 상관관계"
            elif c >= -0.3:
                label = "거의 무관 (분산투자 효과)"
            elif c >= -0.7:
                label = "약한 음의 상관관계"
            else:
                label = "강한 음의 상관관계 (반대 방향 등락)"
            lines.append(f"**상관계수:** {c:.4f} — {label}")
            lines.append(f"  분석 기간: {corr['period']} ({corr['data_days']}일)")
        else:
            lines.append("**상관계수:** 데이터 부족으로 계산 불가")

        # 베타 계수 — 각 종목에 대해 시장 벤치마크 기준
        lines.append("")
        for ticker, name in [(t1, n1), (t2, n2)]:
            beta = calc_beta(ticker, days=250)
            if beta:
                b = beta["beta"]
                if b > 1.2:
                    b_label = "공격적 (시장 대비 변동성 큼)"
                elif b > 0.8:
                    b_label = "시장 평균 수준"
                elif b > 0:
                    b_label = "방어적 (시장 대비 변동성 작음)"
                else:
                    b_label = "역방향 (시장과 반대 움직임)"
                lines.append(f"**{name} 베타:** {b:.3f} — {b_label}")
                lines.append(f"  벤치마크: KODEX 200 ({beta['benchmark']}), "
                             f"분석 기간: {beta['data_days']}일")
            else:
                lines.append(f"**{name} 베타:** 데이터 부족으로 계산 불가")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"상관관계/베타 계산 실패: {e}")
        return f"상관관계 분석에 실패했습니다. (데이터 부족 또는 오류)"


@tool
def simulate_portfolio(tickers_and_weights: str, period: str = "1y") -> str:
    """포트폴리오를 구성하여 과거 데이터 기반 시뮬레이션(백테스트)을 수행합니다.
    총 수익률, 연환산 수익률, 최대 낙폭(MDD), 샤프 비율, 변동성을 계산합니다.
    "삼성전자 50% SK하이닉스 50% 1년 백테스트", "포트폴리오 시뮬레이션" 등의 질문에 사용합니다.

    Args:
        tickers_and_weights: 종목과 비중 (예: "삼성전자 50%, SK하이닉스 50%")
        period: 시뮬레이션 기간 (예: "6m", "1y", "2y", "3y", "5y"). 기본값 1y
    """
    import re

    # 기간 파싱
    period_map = {"6m": 125, "1y": 250, "2y": 500, "3y": 750, "5y": 1250}
    days = period_map.get(period.lower().strip(), 250)

    # 종목+비중 파싱: "삼성전자 50%, SK하이닉스 50%" 또는 "삼성전자:50 SK하이닉스:50"
    # 패턴: 종목명/티커 뒤에 숫자(비중)
    text = tickers_and_weights.replace(":", " ").replace(",", " ")
    # 숫자(비중) 추출 위치 기준으로 분리
    parts = re.split(r'(\d+(?:\.\d+)?)\s*%?\s*', text)

    resolved_tickers = []
    resolved_weights = []
    resolved_names = []

    i = 0
    while i < len(parts):
        name_part = parts[i].strip()
        if name_part and i + 1 < len(parts) and parts[i + 1].strip():
            # 종목명 뒤에 비중
            data = _find_structured_data(name_part)
            if data:
                resolved_tickers.append(data["ticker"])
                resolved_names.append(data["name"])
                resolved_weights.append(float(parts[i + 1].strip()))
                i += 2
                continue
        elif name_part:
            # 비중 없이 종목만 있는 경우
            data = _find_structured_data(name_part)
            if data:
                resolved_tickers.append(data["ticker"])
                resolved_names.append(data["name"])
                resolved_weights.append(0)  # 나중에 균등 배분
        i += 1

    if not resolved_tickers:
        return "종목을 찾을 수 없습니다. '삼성전자 50%, SK하이닉스 50%' 형식으로 입력해주세요."

    # 비중 없으면 균등 배분
    if all(w == 0 for w in resolved_weights):
        resolved_weights = [100 / len(resolved_tickers)] * len(resolved_tickers)

    # 비중 정규화 (합=1)
    w_sum = sum(resolved_weights)
    norm_weights = [w / w_sum for w in resolved_weights]

    try:
        from src.data.technical import simulate_portfolio as _sim

        result = _sim(resolved_tickers, norm_weights, days=days)
        if not result:
            return "시뮬레이션 데이터가 부족합니다 (최소 20영업일 필요)."

        p = result["portfolio"]
        lines = [f"[포트폴리오 시뮬레이션] 기간: {_fmt_date(result['period'].split('~')[0])}"
                 f" ~ {_fmt_date(result['period'].split('~')[1])}"
                 f" ({result['data_days']}영업일)\n"]

        # 구성
        lines.append("**포트폴리오 구성:**")
        for name, ticker, w in zip(resolved_names, resolved_tickers, norm_weights):
            lines.append(f"  - {name} ({ticker}): {w * 100:.1f}%")

        # 성과
        lines.append(f"\n**포트폴리오 성과:**")
        lines.append(f"  - 총 수익률: {p['total_return'] * 100:+.2f}%")
        lines.append(f"  - 연환산 수익률: {p['annualized_return'] * 100:+.2f}%")
        lines.append(f"  - 변동성 (연환산): {p['volatility'] * 100:.2f}%")
        lines.append(f"  - 샤프 비율: {p['sharpe_ratio']:.2f}")
        lines.append(f"  - 최대 낙폭(MDD): {p['max_drawdown'] * 100:.2f}%")

        # 개별
        lines.append(f"\n**개별 종목 수익률:**")
        for item, name in zip(result["individual"], resolved_names):
            lines.append(f"  - {name}: {item['total_return'] * 100:+.2f}%")

        lines.append("\n※ 과거 수익률은 미래 수익을 보장하지 않습니다. 참고용 시뮬레이션입니다.")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"포트폴리오 시뮬레이션 실패: {e}")
        return f"포트폴리오 시뮬레이션에 실패했습니다. (데이터 부족 또는 오류)"


@tool
def get_financial_statements(name_or_ticker: str, quarters: int = 4) -> str:
    """기업의 분기별 재무제표(매출액, 영업이익, 당기순이익)를 조회합니다.
    실적, 매출, 영업이익, 순이익, 영업이익률, 성장률 관련 질문에 사용합니다.

    Args:
        name_or_ticker: 기업명 또는 종목코드 (예: "삼성전자", "005930")
        quarters: 조회할 분기 수 (기본 4분기)
    """
    # 종목 식별
    data = _stock_data_index.get(name_or_ticker) or _stock_data_index.get(name_or_ticker.lower())
    if not data:
        # 부분 매칭
        for key, val in _stock_data_index.items():
            if name_or_ticker.lower() in key:
                data = val
                break
    if not data:
        return f"'{name_or_ticker}'에 대한 종목 정보를 찾을 수 없습니다."

    ticker = data.get("ticker", "")
    name = data.get("name", "")

    try:
        from src.data.database import get_financial_data, get_connection
        conn = get_connection()
        fin_data = get_financial_data(conn, ticker, quarters=quarters)
        conn.close()
    except Exception:
        fin_data = []

    if not fin_data:
        # deploy 데이터의 financial_summary fallback
        fs = data.get("financial_summary")
        if fs and (fs.get("revenue") or fs.get("operating_profit")):
            fy = fs.get("fiscal_year", "")
            fq = fs.get("fiscal_quarter", "")
            rev = fs.get("revenue")
            op = fs.get("operating_profit")
            ni = fs.get("net_income")
            om = fs.get("operating_margin")

            def _fmt(v):
                if v is None:
                    return "-"
                if abs(v) >= 1_0000_0000_0000:
                    return f"{v / 1_0000_0000_0000:.1f}조"
                if abs(v) >= 1_0000_0000:
                    return f"{v / 1_0000_0000:,.0f}억"
                return f"{v:,}"

            lines = [f"## {name}({ticker}) 최근 분기 재무제표\n"]
            lines.append("| 분기 | 매출액 | 영업이익 | 순이익 | 영업이익률 |")
            lines.append("|------|--------|----------|--------|-----------|")
            om_str = f"{om:+.1f}%" if om is not None else "-"
            lines.append(f"| {fy}Q{fq} | {_fmt(rev)} | {_fmt(op)} | {_fmt(ni)} | {om_str} |")
            lines.append(f"\n*deploy 데이터 기준 (최근 1분기만 표시)*")
            return "\n".join(lines)

        return (
            f"{name}({ticker})의 재무제표 데이터가 아직 수집되지 않았습니다.\n"
            f"(OpenDart API 키 설정 후 `python -m src.data.dart_collector`로 수집 가능)"
        )

    # 포맷팅
    lines = [f"## {name}({ticker}) 분기별 재무제표\n"]
    lines.append("| 분기 | 매출액 | 영업이익 | 순이익 | 영업이익률 | 매출 YoY | 영업이익 YoY |")
    lines.append("|------|--------|----------|--------|-----------|----------|-------------|")

    for d in fin_data:
        year = d.get("fiscal_year", "")
        q = d.get("fiscal_quarter", "")

        rev = d.get("revenue")
        op = d.get("operating_profit")
        ni = d.get("net_income")
        om = d.get("operating_margin")
        rg = d.get("revenue_growth_yoy")
        og = d.get("op_growth_yoy")

        def fmt_amount(v):
            if v is None:
                return "-"
            if abs(v) >= 1_0000_0000_0000:
                return f"{v / 1_0000_0000_0000:.1f}조"
            if abs(v) >= 1_0000_0000:
                return f"{v / 1_0000_0000:,.0f}억"
            return f"{v:,}"

        def fmt_pct(v):
            if v is None:
                return "-"
            return f"{v:+.1f}%"

        lines.append(
            f"| {year}Q{q} | {fmt_amount(rev)} | {fmt_amount(op)} | "
            f"{fmt_amount(ni)} | {fmt_pct(om)} | {fmt_pct(rg)} | {fmt_pct(og)} |"
        )

    return "\n".join(lines)


# 에이전트에 바인딩할 도구 목록
ALL_TOOLS = [search_etf, compare_etfs, get_etf_list, search_stock,
             compare_stocks, get_stock_list,
             get_realtime_price, analyze_sector, get_technical_indicators,
             get_stock_correlation, simulate_portfolio, get_financial_statements]
