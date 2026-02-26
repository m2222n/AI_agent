import json
from typing import List

from langchain_core.documents import Document

from config import ETF_DATA_PATH


def load_etf_data() -> List[dict]:
    """ETF JSON 데이터 로드"""
    with open(ETF_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def create_documents(etf_data: List[dict]) -> List[Document]:
    """ETF 데이터를 LangChain Document 객체로 변환"""
    documents = []
    for etf in etf_data:
        content = f"""
ETF ID: {etf['id']}
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
        doc = Document(
            page_content=content,
            metadata={"id": etf["id"], "name": etf["name"], "ticker": etf["ticker"]}
        )
        documents.append(doc)
    return documents
