"""5개 데이터 탭 + 종목 자동완성 REST 엔드포인트 (Phase F).

Streamlit 탭(src/ui/tabs.py)이 호출하는 기존 동기 함수들을 Streamlit 없이 래핑한다.
src/ui/tabs.py는 streamlit 의존이라 import 금지 — _build_sector_stats는 복제.

모든 래핑 대상 함수는 동기이고 실패 시 None/[]를 반환(예외 아님) → run_in_threadpool로
이벤트 루프 보호, None이면 404. summary는 name을 안 주므로 _find_structured_data로 먼저 해석.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from api.deps import require_ready
from api.models import (
    ComparisonRequest,
    MoverItem,
    MoversResponse,
    TickerSearchResponse,
)
from src.data.technical import get_technical_summary
from src.data.predictor import build_price_outlook
from src.data.chart_generator import (
    generate_technical_chart,
    generate_comparison_chart,
    generate_valuation_chart,
    generate_financial_chart,
    generate_sector_overview_chart,
    generate_sector_detail_chart,
)
from src.data.database import get_connection, get_financial_data, DB_PATH
from src.llm.tools import (
    get_available_tickers,
    get_data_indices,
    get_sector_index,
    _find_structured_data,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tabs", tags=["tabs"], dependencies=[Depends(require_ready)])


# ── 섹터 집계 (src/ui/tabs.py:_build_sector_stats 복제 — streamlit 의존 없음) ──
def _build_sector_stats(sector_index: dict) -> list:
    stats = []
    for sector, stocks in sector_index.items():
        if not stocks:
            continue
        total_cap = sum(s.get("market_cap", 0) for s in stocks)
        weighted_change = 0.0
        if total_cap > 0:
            weighted_change = (
                sum(s.get("change_pct", 0) * s.get("market_cap", 0) for s in stocks)
                / total_cap
            )
        pers = [s["per"] for s in stocks if s.get("per") and s["per"] > 0]
        median_per = sorted(pers)[len(pers) // 2] if pers else 0
        stats.append({
            "sector": sector,
            "count": len(stocks),
            "market_cap": total_cap,
            "change_pct": round(weighted_change, 2),
            "median_per": round(median_per, 1),
            "up_count": sum(1 for s in stocks if s.get("change_pct", 0) > 0),
            "down_count": sum(1 for s in stocks if s.get("change_pct", 0) < 0),
        })
    stats.sort(key=lambda x: x["market_cap"], reverse=True)
    return stats


# ── Technical ──────────────────────────────────────────────────────
def _technical_blocking(query: str, days: int) -> Optional[dict]:
    data = _find_structured_data(query)
    if not data:
        return None
    ticker = data.get("ticker") or query
    name = data.get("name") or query
    summary = get_technical_summary(ticker)
    if summary is None:
        return None
    return {
        "ticker": ticker,
        "name": name,
        "summary": summary,
        "chart_b64": generate_technical_chart(ticker, name, days=days),
    }


@router.get("/technical", response_model=None)
async def technical(
    ticker: str = Query(..., min_length=1),
    days: int = Query(120, ge=20, le=2500),
):
    result = await run_in_threadpool(_technical_blocking, ticker, days)
    if result is None:
        raise HTTPException(404, f"'{ticker}' 기술적 데이터를 찾을 수 없습니다.")
    return result


# ── Financial ──────────────────────────────────────────────────────
def _financial_blocking(query: str, quarters: int) -> Optional[dict]:
    if not DB_PATH.exists():
        return None
    data = _find_structured_data(query)
    ticker = (data or {}).get("ticker") or query
    name = (data or {}).get("name") or query
    conn = get_connection()
    try:
        rows = get_financial_data(conn, ticker, quarters=quarters)
    finally:
        conn.close()
    if not rows:
        return None
    return {
        "ticker": ticker,
        "name": name,
        "rows": rows,
        "chart_b64": generate_financial_chart(rows, name),
    }


@router.get("/financial", response_model=None)
async def financial(
    ticker: str = Query(..., min_length=1),
    quarters: int = Query(8, ge=1, le=200),
):
    result = await run_in_threadpool(_financial_blocking, ticker, quarters)
    if result is None:
        raise HTTPException(404, f"'{ticker}' 재무 데이터를 찾을 수 없습니다.")
    return result


# ── Comparison ─────────────────────────────────────────────────────
def _comparison_blocking(tickers: list, days: int) -> Optional[dict]:
    resolved = [_find_structured_data(t) for t in tickers]
    if any(d is None for d in resolved):
        return None
    names = [d.get("name") or tickers[i] for i, d in enumerate(resolved)]
    codes = [d.get("ticker") or tickers[i] for i, d in enumerate(resolved)]

    # 밸류에이션 비교 차트 (tabs.py:581 형식)
    val_metrics = {}
    for key, label in [("per", "PER (배)"), ("pbr", "PBR (배)"), ("div", "배당수익률 (%)")]:
        v1, v2 = resolved[0].get(key), resolved[1].get(key)
        if v1 is not None or v2 is not None:
            val_metrics[label] = (v1 or 0, v2 or 0)

    return {
        "items": resolved,
        "comparison_chart_b64": generate_comparison_chart(codes, names, days=days),
        "valuation_chart_b64": (
            generate_valuation_chart(names[0], names[1], val_metrics)
            if val_metrics
            else None
        ),
    }


@router.post("/comparison", response_model=None)
async def comparison(req: ComparisonRequest):
    result = await run_in_threadpool(_comparison_blocking, req.tickers, req.days)
    if result is None:
        raise HTTPException(404, "비교할 종목을 찾을 수 없습니다.")
    return result


# ── Outlook ────────────────────────────────────────────────────────
def _outlook_blocking(query: str, horizon: str) -> Optional[dict]:
    data = _find_structured_data(query)
    ticker = (data or {}).get("ticker") or query
    name = (data or {}).get("name") or query
    summary = get_technical_summary(ticker)
    if summary is None:
        return None
    # structured_data(None 가능 — 펀더멘털 축만 degrade)
    return build_price_outlook(
        ticker, name, horizon=horizon, summary=summary, structured_data=data
    )


@router.get("/outlook", response_model=None)
async def outlook(
    ticker: str = Query(..., min_length=1),
    horizon: str = Query("1m"),
):
    result = await run_in_threadpool(_outlook_blocking, ticker, horizon)
    if not result:
        raise HTTPException(404, f"'{ticker}' 전망을 생성할 수 없습니다.")
    return result


# ── Sector ─────────────────────────────────────────────────────────
def _sector_blocking(sector: Optional[str]) -> Optional[dict]:
    sector_index = get_sector_index()
    if not sector_index:
        return None
    stats = _build_sector_stats(sector_index)
    out = {
        "stats": stats,
        "overview_chart_b64": generate_sector_overview_chart(stats),
    }
    if sector:
        stocks = sector_index.get(sector)
        if not stocks:
            return None  # 알 수 없는 섹터
        out["sector"] = sector
        out["detail_chart_b64"] = generate_sector_detail_chart(sector, stocks)
        out["stocks"] = stocks
    return out


@router.get("/sector", response_model=None)
async def sector(sector: Optional[str] = Query(None)):
    result = await run_in_threadpool(_sector_blocking, sector)
    if result is None:
        raise HTTPException(404, "섹터 데이터를 찾을 수 없습니다.")
    return result


# ── 종목 자동완성 / 해석 ────────────────────────────────────────────
@router.get("/tickers", response_model=TickerSearchResponse)
async def tickers(
    q: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=200),
):
    options = await run_in_threadpool(get_available_tickers)
    if q:
        ql = q.lower()
        options = [o for o in options if ql in o.lower()]
    return TickerSearchResponse(options=options[:limit])


@router.get("/tickers/resolve", response_model=None)
async def resolve(q: str = Query(..., min_length=1)):
    data = await run_in_threadpool(_find_structured_data, q)
    if not data:
        raise HTTPException(404, f"'{q}'을(를) 찾을 수 없습니다.")
    return data


# ── 동적 추천질문용 movers (오늘의 급등/급락/거래대금 TOP) ──────────
def _movers_blocking(n: int) -> dict:
    etf_idx, stock_idx = get_data_indices()
    # ticker 기준 dedup (인덱스는 name/ticker 둘 다 키라 중복됨)
    seen = {}
    for idx in (etf_idx, stock_idx):
        for data in idx.values():
            t = data.get("ticker")
            name = data.get("name")
            if not t or not name or t in seen:
                continue
            seen[t] = {
                "name": name,
                "ticker": t,
                "change_pct": data.get("change_pct", 0) or 0,
                "trade_value": data.get("trade_value", 0) or 0,
                "close": data.get("close", 0) or 0,
            }
    items = [v for v in seen.values() if v["close"] > 0]

    def _to(rows):
        return [MoverItem(name=r["name"], ticker=r["ticker"],
                          change_pct=round(r["change_pct"], 2)) for r in rows]

    gainers = sorted(items, key=lambda x: x["change_pct"], reverse=True)
    losers = sorted(items, key=lambda x: x["change_pct"])
    traded = sorted(items, key=lambda x: x["trade_value"], reverse=True)
    return {
        "gainers": _to([r for r in gainers if r["change_pct"] > 0][:n]),
        "losers": _to([r for r in losers if r["change_pct"] < 0][:n]),
        "most_traded": _to(traded[:n]),
    }


@router.get("/movers", response_model=MoversResponse)
async def movers(n: int = Query(3, ge=1, le=10)):
    result = await run_in_threadpool(_movers_blocking, n)
    return MoversResponse(**result)
