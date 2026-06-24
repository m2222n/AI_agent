"""5개 데이터 탭 + 종목 자동완성 REST 엔드포인트 (Phase F).

Streamlit 탭(src/ui/tabs.py)이 호출하는 기존 동기 함수들을 Streamlit 없이 래핑한다.
src/ui/tabs.py는 streamlit 의존이라 import 금지 — _build_sector_stats는 복제.

모든 래핑 대상 함수는 동기이고 실패 시 None/[]를 반환(예외 아님) → run_in_threadpool로
이벤트 루프 보호, None이면 404. summary는 name을 안 주므로 _find_structured_data로 먼저 해석.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from api.deps import require_ready
from api.models import (
    ComparisonRequest,
    InstrumentItem,
    MoverItem,
    MoversResponse,
    OrderbookResponse,
    OverviewResponse,
    PriceResponse,
    TickerSearchResponse,
)
from src.data.technical import get_technical_summary
from src.data.predictor import build_price_outlook
from src.data.chart_generator import (
    generate_technical_chart,
    generate_intraday_chart,
    generate_comparison_chart,
    generate_valuation_chart,
    generate_financial_chart,
    generate_sector_overview_chart,
    generate_sector_detail_chart,
    generate_sector_trend_chart,
)
from src.data.database import (
    get_connection,
    get_financial_data,
    get_closes_batch,
    get_low_history_tickers,
    DB_PATH,
)
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


# ── 장중 시세 차트 (yfinance 15분봉, 15분 지연) ────────────────────
def _intraday_blocking(query: str) -> Optional[dict]:
    data = _find_structured_data(query)
    if not data:
        return None
    ticker = data.get("ticker") or query
    name = data.get("name") or query
    prev_close = data.get("close") or None
    chart_b64 = generate_intraday_chart(ticker, name, prev_close)
    if not chart_b64:
        return None  # 장 외 시간/데이터 없음
    return {"ticker": ticker, "name": name, "chart_b64": chart_b64}


@router.get("/intraday", response_model=None)
async def intraday(ticker: str = Query(..., min_length=1)):
    result = await run_in_threadpool(_intraday_blocking, ticker)
    if result is None:
        raise HTTPException(404, "장중 시세를 불러올 수 없습니다. (장 외 시간이거나 데이터 없음)")
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
# 기간 라벨 → 캘린더 일수 (영업일 아님, 시작일 역산용 여유 포함)
_SECTOR_PERIOD_DAYS = {
    "1d": 1, "1w": 7, "1m": 31, "3m": 93, "6m": 186,
    "1y": 366, "2y": 731, "3y": 1096, "5y": 1827, "10y": 3653,
}
_SECTOR_TREND_TOP_N = 20  # 섹터 지수 구성 종목 수(시총 상위). 큰 섹터 성능 위해 제한.


def _sector_trend(sector: str, stocks: list, days: int) -> Optional[dict]:
    """섹터 시총가중 지수 시계열(기준일=100) 계산.

    시총 상위 N종목의 '기준일 대비 수익률'을 현재 시총 비중으로 가중 평균.
    과거 시총이 매일 있지 않아도 견고하도록 가중치는 현재 시총 고정(표준 가중수익률 지수).
    stocks는 시총 내림차순 정렬된 섹터 종목 리스트(get_sector_index 보장).
    """
    top = [s for s in stocks if s.get("market_cap", 0) > 0][:_SECTOR_TREND_TOP_N]
    if len(top) < 1:
        return None
    weights = {s["ticker"]: s["market_cap"] for s in top}
    total_w = sum(weights.values())
    if total_w <= 0:
        return None

    conn = get_connection(DB_PATH)
    try:
        latest = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0]
        if not latest:
            return None
        start = (datetime.strptime(latest, "%Y%m%d")
                 - timedelta(days=days)).strftime("%Y%m%d")
        by_date = get_closes_batch(conn, list(weights.keys()), start, latest)
    finally:
        conn.close()

    if len(by_date) < 2:
        return None

    dates = sorted(by_date.keys())
    # 기준일 = 첫 거래일. 그날 종가가 있는 종목만으로 지수를 고정 구성한다.
    # (중간 상장 종목을 도중에 끼우면 base가 뒤섞여 지수가 왜곡되므로 제외)
    base_day = by_date[dates[0]]
    members = {tk: w for tk, w in weights.items()
               if base_day.get(tk, 0) and base_day[tk] > 0}
    if not members:
        return None
    base_px = {tk: base_day[tk] for tk in members}

    out_dates = []
    index_values = []
    for d in dates:
        day = by_date[d]
        num = 0.0  # Σ(weight · 종목 정규화수익)
        den = 0.0  # Σ(weight)  — 그날 값이 있는 구성종목만
        for tk, w in members.items():
            px = day.get(tk)
            if not px or px <= 0:
                continue
            num += w * (px / base_px[tk])
            den += w
        if den <= 0:
            continue
        out_dates.append(d)
        index_values.append(round(num / den * 100.0, 2))

    if len(index_values) < 2:
        return None
    return {
        "dates": out_dates,
        "index_values": index_values,
        "return_pct": round(index_values[-1] - 100.0, 2),
        "constituents": len(members),
    }


def _sector_blocking(sector: Optional[str], period: str = "1d") -> Optional[dict]:
    sector_index = get_sector_index()
    if not sector_index:
        return None
    stats = _build_sector_stats(sector_index)
    out = {
        "stats": stats,
        "overview_chart_b64": generate_sector_overview_chart(stats),
        "period": period,
    }
    if sector:
        stocks = sector_index.get(sector)
        if not stocks:
            return None  # 알 수 없는 섹터
        out["sector"] = sector
        out["detail_chart_b64"] = generate_sector_detail_chart(sector, stocks)
        out["stocks"] = stocks
        # 기간 추이 차트 (1d는 스냅샷이므로 생략 — 기존 상세 차트로 충분)
        if period != "1d":
            days = _SECTOR_PERIOD_DAYS.get(period, 93)
            trend = _sector_trend(sector, stocks, days)
            if trend:
                out["trend_return_pct"] = trend["return_pct"]
                out["trend_constituents"] = trend["constituents"]
                out["trend_chart_b64"] = generate_sector_trend_chart(
                    sector, trend["dates"], trend["index_values"], period,
                )
    return out


@router.get("/sector", response_model=None)
async def sector(
    sector: Optional[str] = Query(None),
    period: str = Query(
        "1d",
        pattern="^(1d|1w|1m|3m|6m|1y|2y|3y|5y|10y)$",
    ),
):
    result = await run_in_threadpool(_sector_blocking, sector, period)
    if result is None:
        raise HTTPException(404, "섹터 데이터를 찾을 수 없습니다.")
    return result


# ── 종목 자동완성 / 해석 ────────────────────────────────────────────
# 시세 부족(min_days 미만) 제외 종목 집합 캐시 — 자동완성 매 호출 DB 조회 회피.
_low_hist_cache: dict = {"set": None, "min_days": None, "at": 0.0}
_LOW_HIST_TTL = 300  # 5분


def _low_history_set(min_days: int) -> set:
    import time
    now = time.time()
    c = _low_hist_cache
    if (c["set"] is not None and c["min_days"] == min_days
            and now - c["at"] < _LOW_HIST_TTL):
        return c["set"]
    conn = get_connection(DB_PATH)
    try:
        s = get_low_history_tickers(conn, min_days)
    finally:
        conn.close()
    c.update(set=s, min_days=min_days, at=now)
    return s


def _extract_ticker(opt: str) -> str:
    """'이름 (005930)' → '005930'."""
    if opt.endswith(")") and " (" in opt:
        return opt.rsplit(" (", 1)[1][:-1]
    return opt


@router.get("/tickers", response_model=TickerSearchResponse)
async def tickers(
    q: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=200),
    asset_type: Optional[str] = Query(None, pattern="^(stock|etf)$"),
    min_days: int = Query(0, ge=0, le=250),
):
    """종목 자동완성. min_days>0이면 시세 거래일이 그보다 적은 신규 종목 제외
    (기술적 분석 탭처럼 과거 데이터가 필요한 화면용 — 기본 0=제외 안 함)."""
    options = await run_in_threadpool(get_available_tickers, asset_type)
    if min_days > 0:
        low = await run_in_threadpool(_low_history_set, min_days)
        if low:
            options = [o for o in options if _extract_ticker(o) not in low]
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


# ── 실시간 시세 (KIS 우선 → yfinance, 장 외엔 수집 종가) ──────
def _price_blocking(q: str) -> Optional[dict]:
    """종목 해석 → 장중이면 실시간(KIS 우선) 조회, 실패/장외면 수집 종가 fallback."""
    data = _find_structured_data(q)
    if not data:
        return None
    ticker = data.get("ticker", "")
    name = data.get("name", "")
    asset_type = "etf" if "nav" in data else "stock"

    from src.data.realtime import get_realtime_price, is_market_open
    market_open = is_market_open()

    rt = get_realtime_price(ticker, asset_type) if market_open else None
    if rt:
        return {
            "name": name, "ticker": ticker,
            "price": rt["price"], "prev_close": rt.get("prev_close"),
            "change": rt.get("change"), "change_pct": rt.get("change_pct"),
            "volume": rt.get("volume"), "source": rt.get("source", "yfinance"),
            "is_live": True, "timestamp": rt.get("timestamp"),
            "market_open": True,
        }

    # Fallback: 수집 종가
    date = data.get("date", "")
    if len(date) == 8:
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    return {
        "name": name, "ticker": ticker,
        "price": data.get("close", 0) or 0, "prev_close": None,
        "change": None, "change_pct": data.get("change_pct"),
        "volume": data.get("volume"), "source": "close",
        "is_live": False, "timestamp": date or None,
        "market_open": market_open,
    }


@router.get("/price", response_model=PriceResponse)
async def price(ticker: str = Query(..., min_length=1)):
    """종목 현재가 — 장중엔 실시간(KIS 우선→yfinance), 장 외엔 수집 종가(source=close)."""
    result = await run_in_threadpool(_price_blocking, ticker)
    if not result:
        raise HTTPException(404, f"'{ticker}'을(를) 찾을 수 없습니다.")
    return PriceResponse(**result)


@router.get("/price/stream")
async def price_stream(request: Request, ticker: str = Query(..., min_length=1)):
    """체결 틱 실시간 SSE (KIS WebSocket). KIS 미연동/연결 실패 시 `unavailable`
    이벤트 1건 후 종료 → 프론트는 REST 폴링으로 fallback.

    이벤트: tick(체결 dict) / unavailable / (자동 close)
    """
    # 종목 해석 (이름/티커 → 6자리 티커)
    data = await run_in_threadpool(_find_structured_data, ticker)
    code = data.get("ticker") if data else ticker

    from src.data import kis_ws
    mgr = kis_ws.get_manager()

    async def _source():
        q = await mgr.subscribe(code)
        if q is None:
            yield {"event": "unavailable", "data": "KIS 실시간 미연동"}
            return
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    tick = await asyncio.wait_for(q.get(), timeout=15)
                    yield {"event": "tick",
                           "data": json.dumps(tick, ensure_ascii=False)}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}  # keep-alive
        finally:
            await mgr.unsubscribe(code, q)

    return EventSourceResponse(_source())


# ── 호가 10단계 (KIS 전용, 장중) ───────────────────────────
def _orderbook_blocking(q: str) -> Optional[dict]:
    """종목 해석 → KIS 호가 조회. KIS 비활성/장외/조회 실패 시 None.

    호가는 KIS만 제공(yfinance 미지원) → fallback 없음.
    """
    data = _find_structured_data(q)
    if not data:
        return None
    ticker = data.get("ticker", "")
    name = data.get("name", "")

    from src.data import kis_client
    if not kis_client.is_enabled():
        return None
    ob = kis_client.get_orderbook(ticker)
    if not ob:
        return None
    return {
        "name": name, "ticker": ticker,
        "asks": ob["asks"], "bids": ob["bids"],
        "total_ask_qty": ob["total_ask_qty"],
        "total_bid_qty": ob["total_bid_qty"],
        "timestamp": ob.get("timestamp"), "source": ob.get("source", "kis"),
    }


@router.get("/orderbook", response_model=OrderbookResponse)
async def orderbook(ticker: str = Query(..., min_length=1)):
    """종목 호가 10단계 (KIS 전용). KIS 미설정/장 외/조회 실패 시 404."""
    result = await run_in_threadpool(_orderbook_blocking, ticker)
    if not result:
        raise HTTPException(404, "호가를 가져올 수 없습니다. (KIS 미연동이거나 장 외 시간)")
    return OrderbookResponse(**result)


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


# ── 사이드바 개요 (데이터 현황 + ETF/주식 TOP 목록) ────────────────
def _dedup_items(idx: dict) -> list:
    """인덱스(name/ticker 둘 다 키)를 ticker 기준 dedup → 원본 dict 리스트."""
    seen = {}
    for data in idx.values():
        t = data.get("ticker")
        if t and t not in seen:
            seen[t] = data
    return list(seen.values())


def _to_instrument(d: dict) -> InstrumentItem:
    return InstrumentItem(
        name=d.get("name", ""),
        ticker=d.get("ticker", ""),
        close=d.get("close", 0) or 0,
        change_pct=round(d.get("change_pct", 0) or 0, 2),
        trade_value=d.get("trade_value", 0) or 0,
        sector=d.get("sector") or None,
        per=d.get("per") or None,
        market_cap=d.get("market_cap") or None,
    )


def _overview_blocking(top: int, sector: Optional[str] = None) -> dict:
    etf_idx, stock_idx = get_data_indices()
    etfs = _dedup_items(etf_idx)
    stocks = _dedup_items(stock_idx)

    # 기준일 — 아무 항목의 date
    as_of = None
    for d in etfs or stocks:
        if d.get("date"):
            as_of = d["date"]
            break

    # 업종 필터 — 지정 시 해당 업종 종목만(전체 목록 기준), 미지정 시 전체에서 거래대금 TOP
    filtered_stocks = stocks
    if sector:
        filtered_stocks = [d for d in stocks if d.get("sector") == sector]

    top_etfs = sorted(etfs, key=lambda x: x.get("trade_value", 0) or 0, reverse=True)[:top]
    top_stocks = sorted(filtered_stocks, key=lambda x: x.get("trade_value", 0) or 0, reverse=True)[:top]
    sectors = sorted({d.get("sector") for d in stocks if d.get("sector")})

    return {
        "etf_count": len(etfs),
        "stock_count": len(stocks),
        "as_of": as_of,
        "top_etfs": [_to_instrument(d) for d in top_etfs],
        "top_stocks": [_to_instrument(d) for d in top_stocks],
        "sectors": sectors,
    }


@router.get("/overview", response_model=OverviewResponse)
async def overview(
    top: int = Query(20, ge=1, le=50),
    sector: Optional[str] = Query(None),
):
    result = await run_in_threadpool(_overview_blocking, top, sector)
    return OverviewResponse(**result)
