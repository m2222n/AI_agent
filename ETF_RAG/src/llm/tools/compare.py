"""
비교 도구 — compare_etfs, compare_stocks
"""

import json
import logging

from langchain_core.tools import tool

from src.llm.tools import _state
from src.llm.tools._helpers import _find_structured_data, _extract_comparison_fields

logger = logging.getLogger(__name__)


@tool
def compare_etfs(etf_name_1: str, etf_name_2: str) -> str:
    """두 ETF 또는 주식을 비교 분석합니다. 각 종목의 가격, 수익률, 보유종목 등을 나란히 비교합니다.

    Args:
        etf_name_1: 첫 번째 ETF/주식 이름 또는 티커 (예: "KODEX 200", "069500")
        etf_name_2: 두 번째 ETF/주식 이름 또는 티커 (예: "TIGER 200", "102110")
    """
    if _state._retriever is None:
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
        # 상대 수익률 차트 생성
        try:
            from src.data.chart_generator import generate_comparison_chart
            chart_b64 = generate_comparison_chart(
                [d1["ticker"], d2["ticker"]],
                [d1["name"], d2["name"]],
                days=120,
            )
            if chart_b64:
                comparison["comparison_chart_b64"] = chart_b64
        except Exception as e:
            logger.debug(f"비교 차트 생성 실패: {e}")

        # 구조화 JSON + 텍스트 컨텍스트 모두 반환
        text_parts = []
        for name, data in [(etf_name_1, d1), (etf_name_2, d2)]:
            text_parts.append(f"[{data['name']}] 종가: {data.get('close', 0):,}원, "
                              f"등락률: {data.get('change_pct', 0):+.2f}%")
        structured_json = json.dumps(comparison, ensure_ascii=False)
        return f"{structured_json}\n\n---\n\n" + "\n".join(text_parts)

    # 구조화 데이터 없으면 기존 텍스트 검색 fallback
    from src.rag.retriever import retrieve_relevant_docs

    ctx1, src1 = retrieve_relevant_docs(_state._retriever, etf_name_1, k=1)
    ctx2, src2 = retrieve_relevant_docs(_state._retriever, etf_name_2, k=1)

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
        # 상대 수익률 차트 생성
        try:
            from src.data.chart_generator import generate_comparison_chart
            chart_b64 = generate_comparison_chart(
                [d1["ticker"], d2["ticker"]],
                [d1["name"], d2["name"]],
                days=120,
            )
            if chart_b64:
                comparison["comparison_chart_b64"] = chart_b64
        except Exception as e:
            logger.debug(f"비교 차트 생성 실패: {e}")

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
            # 재무제표 요약 추가
            ticker = data.get("ticker", "")
            if ticker:
                try:
                    from src.data.database import get_connection, get_latest_financial_summary
                    conn = get_connection()
                    try:
                        fin = get_latest_financial_summary(conn, ticker)
                        if fin:
                            period = f"{fin.get('fiscal_year', '')}Q{fin.get('fiscal_quarter', '')}"
                            rev = fin.get("revenue")
                            op = fin.get("operating_profit")
                            margin = fin.get("operating_margin")
                            rev_g = fin.get("revenue_growth_yoy")
                            line += f"\n  실적({period}):"
                            if rev is not None:
                                if abs(rev) >= 1_0000_0000_0000:
                                    line += f" 매출 {rev / 1_0000_0000_0000:.1f}조"
                                else:
                                    line += f" 매출 {rev / 1_0000_0000:,.0f}억"
                            if op is not None:
                                if abs(op) >= 1_0000_0000_0000:
                                    line += f", 영업이익 {op / 1_0000_0000_0000:.1f}조"
                                else:
                                    line += f", 영업이익 {op / 1_0000_0000:,.0f}억"
                            if margin is not None:
                                line += f" (마진 {margin:.1f}%)"
                            if rev_g is not None:
                                line += f", 매출YoY {rev_g:+.1f}%"
                    finally:
                        conn.close()
                except Exception:
                    pass
            text_parts.append(line)
        structured_json = json.dumps(comparison, ensure_ascii=False)
        return f"{structured_json}\n\n---\n\n" + "\n".join(text_parts)

    # 구조화 데이터 없으면 텍스트 검색 fallback
    retriever = _state._stock_retriever or _state._retriever
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
