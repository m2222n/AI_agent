"""
ETF 데이터 로더 — 수집 데이터 또는 하드코딩 샘플에서 ETF 정보를 로드하고
LangChain Document 객체로 변환합니다.

로드 우선순위:
    1. collected/ 디렉토리의 최신 수집 파일 (etf_data_YYYYMMDD.json)
    2. etf_data.json 하드코딩 샘플 (8개 ETF, Phase 0 잔재)
"""

import json
import logging
from typing import List

from langchain_core.documents import Document

from config import ETF_DATA_PATH, ETF_SELECTION, get_latest_collected_path

logger = logging.getLogger(__name__)


def load_etf_data() -> List[dict]:
    """ETF 데이터 로드. 수집 데이터 우선, 없으면 하드코딩 fallback."""
    collected_path = get_latest_collected_path()

    if collected_path:
        logger.info(f"수집 데이터 로드: {collected_path.name}")
        with open(collected_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        etfs = _normalize_collected(raw)
        before = len(etfs)
        etfs = _filter_etfs(etfs)
        logger.info(f"ETF 필터링: {before}개 → {len(etfs)}개")
        return etfs

    logger.info("수집 데이터 없음, 하드코딩 샘플 로드")
    with open(ETF_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_collected(raw: dict) -> List[dict]:
    """수집 데이터를 통일된 ETF dict 리스트로 변환.

    수집 데이터 구조:
        {"metadata": {...}, "etfs": [{ticker, name, date, ohlcv, deviation, ...}]}

    반환 구조 (하드코딩과 호환 가능한 통일 포맷):
        [{ticker, name, date, close, nav, volume, trade_value, change_pct,
          deviation, tracking_error, holdings, ...}]
    """
    etfs = []
    for e in raw.get("etfs", []):
        ohlcv = e.get("ohlcv") or {}
        etf = {
            "ticker": e["ticker"],
            "name": e["name"],
            "date": e.get("date", raw.get("metadata", {}).get("collection_date", "")),
            # 시세
            "open": ohlcv.get("open", 0),
            "high": ohlcv.get("high", 0),
            "low": ohlcv.get("low", 0),
            "close": ohlcv.get("close", 0),
            "volume": ohlcv.get("volume", 0),
            "trade_value": ohlcv.get("trade_value", 0),
            "nav": ohlcv.get("nav", 0),
            "base_index": ohlcv.get("base_index", 0),
            "change": ohlcv.get("change", 0),
            "change_pct": ohlcv.get("change_pct", 0.0),
            # 괴리율/추적오차
            "deviation": e.get("deviation"),
            "tracking_error": e.get("tracking_error"),
            # 수익률
            "returns": e.get("returns", {}),
            # 보유종목
            "holdings": e.get("holdings", []),
        }
        etfs.append(etf)

    return etfs


def _filter_etfs(etfs: List[dict]) -> List[dict]:
    """ETF 선별 기준에 따라 필터링.

    거래대금/NAV/종가 기준으로 비활성 종목 제외.
    """
    min_tv = ETF_SELECTION["min_trade_value"]
    exclude_zero = ETF_SELECTION["exclude_zero_close"]

    filtered = []
    for etf in etfs:
        if exclude_zero and etf.get("close", 0) == 0:
            continue
        if etf.get("trade_value", 0) < min_tv:
            continue
        filtered.append(etf)

    return filtered


def _is_collected_format(etf: dict) -> bool:
    """수집 데이터 포맷인지 확인 (close 필드가 있으면 수집 데이터)"""
    return "close" in etf


def create_documents(etf_data: List[dict], include_pdfs: bool = True) -> List[Document]:
    """ETF 데이터를 LangChain Document 객체로 변환.

    하드코딩 포맷과 수집 포맷 모두 지원.
    include_pdfs가 True면 PDF 투자설명서 문서도 포함.
    """
    documents = []
    for etf in etf_data:
        if _is_collected_format(etf):
            doc = _create_doc_from_collected(etf)
        else:
            doc = _create_doc_from_hardcoded(etf)
        documents.append(doc)

    # PDF 투자설명서 문서 추가
    if include_pdfs:
        from src.data.pdf_loader import load_pdf_documents
        pdf_docs = load_pdf_documents()
        if pdf_docs:
            logger.info(f"PDF 문서 {len(pdf_docs)}개 추가")
            documents.extend(pdf_docs)

    return documents


def _create_doc_from_collected(etf: dict) -> Document:
    """수집 데이터에서 Document 생성"""
    # 보유종목 상위 5개 텍스트
    holdings_text = "정보 없음"
    if etf.get("holdings"):
        top5 = etf["holdings"][:5]
        holdings_text = ", ".join(
            f"{h['stock_name'] or h['stock_ticker']} ({h['weight']}%)"
            for h in top5
        )

    # 수익률 텍스트
    returns = etf.get("returns", {})
    if returns:
        return_labels = {"1d": "1일", "1w": "1주", "1m": "1개월", "3m": "3개월", "1y": "1년"}
        returns_text = ", ".join(
            f"{return_labels.get(k, k)}: {v:+.2f}%"
            for k, v in returns.items()
            if v is not None
        )
    else:
        returns_text = "정보 없음"

    content = f"""상품명: {etf['name']} ({etf['ticker']})
기준일: {etf['date']}
종가: {etf['close']:,}원
NAV: {etf['nav']:,}원
등락률: {etf['change_pct']:+.2f}%
수익률: {returns_text}
거래량: {etf['volume']:,}주
거래대금: {etf['trade_value']:,}원
기초지수: {etf['base_index']}
괴리율: {etf['deviation'] if etf['deviation'] is not None else '정보 없음'}%
추적오차율: {etf['tracking_error'] if etf['tracking_error'] is not None else '정보 없음'}%
주요 보유종목: {holdings_text}
"""

    return Document(
        page_content=content,
        metadata={
            "ticker": etf["ticker"],
            "name": etf["name"],
            "date": etf["date"],
            "source": "krx_collected",
        },
    )


def _create_doc_from_hardcoded(etf: dict) -> Document:
    """하드코딩 샘플에서 Document 생성 (기존 로직)"""
    content = f"""ETF ID: {etf['id']}
상품명: {etf['name']} ({etf['ticker']})
카테고리: {etf['category']}
추종지수: {etf['index']}
운용사: {etf['asset_manager']}
총보수: {etf['total_expense_ratio']}
순자산가치(NAV): {etf['nav']}
순자산총액(AUM): {etf['aum']}
상장일: {etf['listing_date']}
설명: {etf['description']}
위험등급: {etf['risk_level']}
투자전략: {etf['investment_strategy']}
주요 보유종목: {', '.join(etf['top_holdings'])}
배당정책: {etf['dividend_policy']}
추적오차: {etf['tracking_error']}
투자자 유의사항: {etf['investor_caution']}
"""

    return Document(
        page_content=content,
        metadata={
            "id": etf["id"],
            "name": etf["name"],
            "ticker": etf["ticker"],
            "source": "hardcoded",
        },
    )
