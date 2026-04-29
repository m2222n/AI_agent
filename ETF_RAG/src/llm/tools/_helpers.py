"""
도구 공통 헬퍼 함수

- 종목 조회 (_find_structured_data)
- 유사 종목 검색 (_find_similar_names)
- 구조화 데이터 보강 (_enrich_with_structured_data)
- 비교 필드 추출 (_extract_comparison_fields)
- deploy fallback
"""

import logging
from typing import Optional

from src.llm.tools import _state

logger = logging.getLogger(__name__)


def _find_structured_data(name_or_ticker: str) -> Optional[dict]:
    """이름 또는 티커로 구조화 데이터 조회 (ETF → 주식 순)"""
    key = name_or_ticker.lower().strip()

    # ETF 인덱스에서 정확 매칭
    if key in _state._etf_data_index:
        return _state._etf_data_index[key]

    # 주식 인덱스에서 정확 매칭
    if key in _state._stock_data_index:
        return _state._stock_data_index[key]

    # 부분 매칭 (이름에 포함)
    for index in (_state._etf_data_index, _state._stock_data_index):
        for idx_key, data in index.items():
            if key in idx_key or idx_key in key:
                return data

    # 인덱스 초기화 자체가 안 된 경우 deploy JSON에서 직접 검색 (초기화 실패 복구)
    if not _state._data_initialized and not _state._etf_data_index and not _state._stock_data_index:
        logger.warning(f"[tools] 인덱스 미초기화 — deploy JSON 직접 검색 시도: {name_or_ticker}")
        result = _fallback_deploy_lookup(key)
        if result:
            return result

    return None


def _fallback_deploy_lookup(key: str) -> Optional[dict]:
    """인덱스 초기화 실패 시 deploy JSON에서 직접 검색.

    정확 매칭 우선 → 부분 매칭 fallback. loader.py의 _normalize 포맷과 호환.
    """
    try:
        import json
        from config import DEPLOY_DIR
        for filename in ("stock_data.json", "etf_data.json"):
            path = DEPLOY_DIR / filename
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            items = raw.get("stocks", raw.get("etfs", []))

            # 1단계: 정확 매칭 (이름 또는 티커)
            for item in items:
                name = item.get("name", "").lower()
                ticker = item.get("ticker", "")
                if key == name or key == ticker:
                    return _normalize_deploy_item(item)

            # 2단계: 부분 매칭 (이름에 포함)
            for item in items:
                name = item.get("name", "").lower()
                if key in name or name in key:
                    return _normalize_deploy_item(item)

    except Exception as e:
        logger.error(f"[tools] deploy fallback 실패: {e}")
    return None


def _normalize_deploy_item(item: dict) -> dict:
    """deploy JSON 아이템을 loader.py 호환 포맷으로 변환."""
    ohlcv = item.get("ohlcv") or {}
    fund = item.get("fundamental") or {}
    result = {
        "ticker": item.get("ticker", ""),
        "name": item.get("name", ""),
        "date": item.get("date", ""),
        "close": ohlcv.get("close", item.get("close", 0)),
        "change_pct": ohlcv.get("change_pct", item.get("change_pct", 0)),
        "volume": ohlcv.get("volume", item.get("volume", 0)),
        "trade_value": ohlcv.get("trade_value", item.get("trade_value", 0)),
        "market_cap": item.get("market_cap", 0),
        "per": fund.get("per", item.get("per", 0)),
        "pbr": fund.get("pbr", item.get("pbr", 0)),
        "eps": fund.get("eps", item.get("eps", 0)),
        "bps": fund.get("bps", item.get("bps", 0)),
        "div": fund.get("div", item.get("div", 0)),
        "dps": fund.get("dps", item.get("dps", 0)),
        "returns": item.get("returns", {}),
        "sector": item.get("sector", ""),
    }
    # ETF 전용 필드
    if "nav" in item or "deviation" in item:
        result["nav"] = item.get("nav", ohlcv.get("nav", 0))
        result["deviation"] = item.get("deviation")
        result["tracking_error"] = item.get("tracking_error")
        result["holdings"] = item.get("holdings", [])
    # 재무 요약
    if item.get("financial_summary"):
        result["financial_summary"] = item["financial_summary"]
    return result


def _not_found_message(name_or_ticker: str) -> str:
    """종목을 찾을 수 없을 때 유사 종목 제안 포함 메시지 반환."""
    similar = _find_similar_names(name_or_ticker)
    if similar:
        suggestions = "\n".join(
            f"- {s['name']} ({s['ticker']})" for s in similar
        )
        return (
            f"'{name_or_ticker}'에 정확히 일치하는 종목이 없습니다. "
            f"혹시 다음 중 하나를 찾으시나요?\n{suggestions}\n\n"
            f"정확한 종목명이나 티커(숫자 6자리)로 다시 검색해주세요."
        )
    return f"'{name_or_ticker}'에 해당하는 종목을 찾을 수 없습니다."


def _find_similar_names(name_or_ticker: str, max_results: int = 5) -> list[dict]:
    """유사한 종목명 후보 리스트 반환 (정확 매칭 실패 시 사용).

    Returns:
        [{"name": str, "ticker": str, "asset_type": "etf"|"stock"}, ...]
    """
    key = name_or_ticker.lower().strip()
    if not key:
        return []

    candidates = []
    seen = set()

    for index, asset_type in [(_state._etf_data_index, "etf"), (_state._stock_data_index, "stock")]:
        for idx_key, data in index.items():
            name = data.get("name", "")
            ticker = data.get("ticker", "")
            if not name or ticker in seen:
                continue
            # 이름 키만 검사 (티커 키는 건너뛰기 — 중복 방지)
            if idx_key != name.lower():
                continue

            # 유사도 점수: 부분 포함 > 첫 글자 일치 > 기타
            score = 0
            name_lower = name.lower()
            if key in name_lower or name_lower in key:
                score = 100
            elif name_lower.startswith(key) or key.startswith(name_lower):
                score = 80
            elif any(c in name_lower for c in key if len(c.encode("utf-8")) > 1):
                # 한글 글자 단위 부분 매칭
                matched = sum(1 for c in key if c in name_lower and len(c.encode("utf-8")) > 1)
                score = matched * 20
            else:
                continue

            if score > 0:
                candidates.append({
                    "name": name,
                    "ticker": ticker,
                    "asset_type": asset_type,
                    "score": score,
                })
                seen.add(ticker)

    # 점수 높은 순, 같으면 이름 짧은 순
    candidates.sort(key=lambda x: (-x["score"], len(x["name"])))
    return candidates[:max_results]


def _enrich_with_structured_data(sources: list, index: dict) -> str:
    """검색 출처의 종목에 대해 구조화 데이터를 보강 텍스트로 반환"""
    # 1단계: 매칭되는 종목 + 재무 데이터 배치 조회
    matched = []
    stock_tickers = []
    for s in sources:
        ticker = s.get("ticker", "")
        name = s.get("name", "")
        data = index.get(ticker) or index.get(name.lower()) if index else None
        if not data:
            continue
        matched.append(data)
        if "per" in data and data.get("ticker"):
            stock_tickers.append(data["ticker"])

    # 재무 데이터 배치 조회 (단일 DB 연결)
    fin_cache = {}
    if stock_tickers:
        try:
            from src.data.database import get_financial_data, get_connection
            fin_conn = get_connection()
            try:
                for t in stock_tickers:
                    fin_cache[t] = get_financial_data(fin_conn, t, quarters=4)
            finally:
                fin_conn.close()
        except Exception as e:
            logger.debug(f"재무제표 배치 조회 실패: {e}")

    # 2단계: enrichment 텍스트 구성
    enriched = []
    for data in matched:
        ticker = data.get("ticker", "")

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

        # 최근 분기 실적 + 4분기 추세 추가 (배치 캐시에서 조회)
        if "per" in data and ticker:
            fin_list = fin_cache.get(ticker)
            if fin_list:
                # 최근 1분기 요약
                fin = fin_list[0]
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

                # 4분기 추세 + 가속/둔화 신호 (2개 이상일 때만)
                if len(fin_list) >= 2:
                    trend_parts = []
                    yoy_values = []
                    margin_values = []
                    for q in reversed(fin_list):  # 과거→최신 순
                        qy = q.get("fiscal_year", "")
                        qq = q.get("fiscal_quarter", "")
                        qr = q.get("revenue")
                        qo = q.get("operating_margin")
                        q_yoy = q.get("revenue_growth_yoy")
                        if qr is not None:
                            if abs(qr) >= 1_0000_0000_0000:
                                rev_s = f"{qr / 1_0000_0000_0000:.1f}조"
                            else:
                                rev_s = f"{qr / 1_0000_0000:,.0f}억"
                            margin_s = f"({qo:.1f}%)" if qo is not None else ""
                            trend_parts.append(f"{qy}Q{qq} {rev_s}{margin_s}")
                        if q_yoy is not None:
                            yoy_values.append(q_yoy)
                        if qo is not None:
                            margin_values.append(qo)
                    if len(trend_parts) >= 2:
                        enriched.append(f"  실적추세: {' → '.join(trend_parts)}")

                    # 성장 가속/둔화 판정
                    signals = []
                    if len(yoy_values) >= 2:
                        latest_yoy = yoy_values[-1]
                        prev_yoy = yoy_values[-2]
                        if latest_yoy > 0 and prev_yoy > 0 and latest_yoy > prev_yoy:
                            signals.append("매출 성장 가속")
                        elif latest_yoy > 0 and prev_yoy > 0 and latest_yoy < prev_yoy:
                            signals.append("매출 성장 둔화")
                        elif latest_yoy < 0 and prev_yoy >= 0:
                            signals.append("매출 역성장 전환")
                        elif latest_yoy >= 0 and prev_yoy < 0:
                            signals.append("매출 턴어라운드")

                    if len(margin_values) >= 2:
                        latest_m = margin_values[-1]
                        prev_m = margin_values[-2]
                        diff = latest_m - prev_m
                        if diff >= 3:
                            signals.append("수익성 개선")
                        elif diff <= -3:
                            signals.append("수익성 악화")

                    if signals:
                        enriched.append(f"  실적신호: {', '.join(signals)}")

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
    return "\n\n[구조화 데이터 — 아래 수치는 최신 수집 데이터이므로 답변에 반드시 활용하세요]\n" + "\n".join(enriched)


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

        # 최근 분기 재무제표 (DB 조회)
        ticker = data.get("ticker", "")
        if ticker:
            try:
                from src.data.database import get_connection, get_latest_financial_summary
                conn = get_connection()
                try:
                    fin = get_latest_financial_summary(conn, ticker)
                    if fin:
                        fields["revenue"] = fin.get("revenue")
                        fields["operating_profit"] = fin.get("operating_profit")
                        fields["net_income"] = fin.get("net_income")
                        fields["operating_margin"] = fin.get("operating_margin")
                        fields["revenue_growth_yoy"] = fin.get("revenue_growth_yoy")
                        fields["op_growth_yoy"] = fin.get("op_growth_yoy")
                        fields["fiscal_period"] = f"{fin.get('fiscal_year', '')}Q{fin.get('fiscal_quarter', '')}"
                finally:
                    conn.close()
            except Exception as e:
                logger.debug(f"비교용 재무제표 조회 실패 ({ticker}): {e}")

    if "asset_type" not in fields:
        fields["asset_type"] = "unknown"

    return fields


def _fmt_date(date_str: str) -> str:
    """YYYYMMDD → YYYY-MM-DD"""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str
