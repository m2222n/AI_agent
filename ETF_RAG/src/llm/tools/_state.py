"""
도구 모듈 공유 상태 — retriever, 데이터 인덱스, 역인덱스

app.py에서 set_retriever()로 초기화한 뒤,
각 도구 모듈이 이 상태를 import하여 사용한다.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 모듈 레벨 retriever — app.py에서 초기화 후 주입
_retriever = None          # ETF용 (또는 ETF+주식 통합)
_stock_retriever = None    # 주식 전용
_documents = None

# 구조화 데이터 인덱스 — 이름/티커로 원본 dict 직접 조회
_etf_data_index = {}       # {name_lower: dict, ticker: dict}
_stock_data_index = {}     # {name_lower: dict, ticker: dict}
_data_initialized = False  # set_retriever()로 데이터 주입 여부

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


def get_sector_index() -> dict:
    """업종별 종목 인덱스 반환 (tabs.py 등 UI에서 사용)."""
    return _sector_index


def set_retriever(retriever, documents=None, stock_retriever=None,
                  etf_data=None, stock_data=None):
    """앱 초기화 시 retriever와 documents를 주입"""
    global _retriever, _documents, _stock_retriever, _data_initialized
    global _etf_data_index, _stock_data_index, _holdings_reverse_index, _sector_index
    _retriever = retriever
    _documents = documents or (retriever.documents if hasattr(retriever, "documents") else [])
    _stock_retriever = stock_retriever
    if etf_data is not None:
        _etf_data_index = _build_data_index(etf_data)
        _holdings_reverse_index = _build_holdings_reverse_index(etf_data)
        logger.info(f"[tools] ETF 인덱스: {len(_etf_data_index)}개 키 (원본 {len(etf_data)}종목)")
        _data_initialized = True
    if stock_data is not None:
        _stock_data_index = _build_data_index(stock_data)
        _sector_index = _build_sector_index(stock_data)
        logger.info(f"[tools] 주식 인덱스: {len(_stock_data_index)}개 키 (원본 {len(stock_data)}종목)")
        _data_initialized = True


def get_available_tickers(asset_type: Optional[str] = None) -> list[str]:
    """자동완성용 종목 옵션 리스트 반환 — 'name (ticker)' 형식, 정렬됨.

    asset_type: "stock"이면 주식만, "etf"면 ETF만, None이면 전체.
    (재무제표 탭처럼 주식만 의미 있는 화면에서 ETF를 자동완성에서 빼기 위함)
    """
    if asset_type == "stock":
        indices = (_stock_data_index,)
    elif asset_type == "etf":
        indices = (_etf_data_index,)
    else:
        indices = (_etf_data_index, _stock_data_index)
    seen = set()
    options = []
    for index in indices:
        for _key, data in index.items():
            ticker = data.get("ticker", "")
            name = data.get("name", "")
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            options.append(f"{name} ({ticker})")
    options.sort()
    return options


def get_data_indices():
    """ETF/주식 데이터 인덱스 반환 (읽기 전용)"""
    return _etf_data_index, _stock_data_index
