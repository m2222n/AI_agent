"""가상투자(모의투자) — 1억 가상 현금으로 매수/매도, 평가손익, 거래내역, 랭킹.

체결가는 현재가(장중 KIS/yfinance, 장외 수집 종가) — api.tabs._price_blocking 재사용.
모든 핸들러 get_current_user 뒤, 동기 def(threadpool). 금액은 정수 원(KRW).
수수료/세금은 미반영(가상). 공매도/미수 불가(현금·보유 범위 내).
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.auth import get_current_user, _display_nickname
from api.db import get_db
from api.models import (
    HoldingItem,
    PortfolioResponse,
    RankingItem,
    RankingResponse,
    TradeHistoryItem,
    TradeHistoryResponse,
    TradeRequest,
    TradeResult,
)
from api.models_db import (
    INITIAL_CASH,
    PaperAccount,
    PaperHolding,
    PaperTrade,
    User,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/me/paper", tags=["paper-trading"])


# ── 헬퍼 ────────────────────────────────────────────────
def _resolve_price(ticker: str) -> dict:
    """종목 해석 + 현재가 조회. 실패 시 404. (api.tabs._price_blocking 재사용)"""
    from api.tabs import _price_blocking

    p = _price_blocking(ticker)
    if not p or not p.get("price"):
        raise HTTPException(404, f"'{ticker}' 종목 또는 현재가를 찾을 수 없습니다.")
    return p


def _get_or_create_account(db: Session, user_id: int) -> PaperAccount:
    acc = db.scalar(select(PaperAccount).where(PaperAccount.user_id == user_id))
    if acc is None:
        acc = PaperAccount(user_id=user_id, cash=INITIAL_CASH)
        db.add(acc)
        db.commit()
        db.refresh(acc)
    return acc


def _holdings(db: Session, user_id: int):
    return list(db.scalars(
        select(PaperHolding).where(PaperHolding.user_id == user_id)
    ))


def _build_portfolio(db: Session, user_id: int) -> PortfolioResponse:
    acc = _get_or_create_account(db, user_id)
    items = []
    holdings_value = 0
    for h in _holdings(db, user_id):
        try:
            p = _resolve_price(h.ticker)
        except HTTPException:
            continue  # 현재가 못 구하는 종목은 건너뜀(상폐 등)
        cur = float(p["price"])
        eval_value = int(round(cur * h.qty))
        cost_value = int(round(h.avg_price * h.qty))
        pnl = eval_value - cost_value
        pnl_pct = (pnl / cost_value * 100.0) if cost_value > 0 else 0.0
        holdings_value += eval_value
        items.append(HoldingItem(
            ticker=h.ticker, name=p["name"], qty=h.qty,
            avg_price=round(h.avg_price, 2), current_price=cur,
            eval_value=eval_value, cost_value=cost_value,
            pnl=pnl, pnl_pct=round(pnl_pct, 2),
            price_source=p.get("source", "close"),
        ))
    items.sort(key=lambda x: x.eval_value, reverse=True)
    total_value = acc.cash + holdings_value
    total_pnl = total_value - INITIAL_CASH
    return PortfolioResponse(
        cash=acc.cash, holdings=items,
        holdings_value=holdings_value, total_value=total_value,
        initial_cash=INITIAL_CASH,
        total_pnl=total_pnl,
        total_pnl_pct=round(total_pnl / INITIAL_CASH * 100.0, 2),
    )


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


# ── 엔드포인트 ──────────────────────────────────────────
@router.get("/portfolio", response_model=PortfolioResponse)
def portfolio(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> PortfolioResponse:
    return _build_portfolio(db, user.id)


@router.post("/buy", response_model=TradeResult)
def buy(
    req: TradeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TradeResult:
    p = _resolve_price(req.ticker)
    price = float(p["price"])
    ticker, name = p["ticker"], p["name"]
    amount = int(round(price * req.qty))

    acc = _get_or_create_account(db, user.id)
    if amount > acc.cash:
        raise HTTPException(
            400,
            f"잔고 부족: 필요 {amount:,}원 / 보유 {acc.cash:,}원",
        )

    acc.cash -= amount
    h = db.scalar(select(PaperHolding).where(
        PaperHolding.user_id == user.id, PaperHolding.ticker == ticker))
    if h is None:
        db.add(PaperHolding(user_id=user.id, ticker=ticker,
                            qty=req.qty, avg_price=price))
    else:
        # 평단가 = (기존 평가 + 신규 매입) / 총수량
        total_cost = h.avg_price * h.qty + price * req.qty
        h.qty += req.qty
        h.avg_price = total_cost / h.qty
    db.add(PaperTrade(user_id=user.id, ticker=ticker, name=name,
                      side="buy", qty=req.qty, price=price, amount=amount))
    db.commit()
    return TradeResult(
        ok=True, side="buy", ticker=ticker, name=name, qty=req.qty,
        price=price, amount=amount, cash=acc.cash,
        price_source=p.get("source", "close"),
    )


@router.post("/sell", response_model=TradeResult)
def sell(
    req: TradeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TradeResult:
    p = _resolve_price(req.ticker)
    price = float(p["price"])
    ticker, name = p["ticker"], p["name"]

    h = db.scalar(select(PaperHolding).where(
        PaperHolding.user_id == user.id, PaperHolding.ticker == ticker))
    if h is None or h.qty < req.qty:
        held = h.qty if h else 0
        raise HTTPException(400, f"보유 수량 부족: 매도 {req.qty} / 보유 {held}")

    amount = int(round(price * req.qty))
    realized = int(round((price - h.avg_price) * req.qty))  # 실현손익
    acc = _get_or_create_account(db, user.id)
    acc.cash += amount
    h.qty -= req.qty
    if h.qty == 0:
        db.delete(h)  # 전량 매도 → 보유 제거
    db.add(PaperTrade(user_id=user.id, ticker=ticker, name=name,
                      side="sell", qty=req.qty, price=price, amount=amount,
                      realized_pnl=realized))
    db.commit()
    return TradeResult(
        ok=True, side="sell", ticker=ticker, name=name, qty=req.qty,
        price=price, amount=amount, cash=acc.cash, realized_pnl=realized,
        price_source=p.get("source", "close"),
    )


@router.get("/trades", response_model=TradeHistoryResponse)
def trades(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> TradeHistoryResponse:
    rows = db.scalars(
        select(PaperTrade).where(PaperTrade.user_id == user.id)
        .order_by(PaperTrade.created_at.desc(), PaperTrade.id.desc())
        .limit(200)
    )
    return TradeHistoryResponse(trades=[
        TradeHistoryItem(
            ticker=t.ticker, name=t.name, side=t.side, qty=t.qty,
            price=t.price, amount=t.amount, realized_pnl=t.realized_pnl,
            created_at=t.created_at.isoformat() if t.created_at else "",
        ) for t in rows
    ])


@router.post("/reset", response_model=PortfolioResponse)
def reset(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> PortfolioResponse:
    """계좌 초기화 — 보유/내역 삭제 + 현금 1억으로 복구."""
    from sqlalchemy import delete
    db.execute(delete(PaperHolding).where(PaperHolding.user_id == user.id))
    db.execute(delete(PaperTrade).where(PaperTrade.user_id == user.id))
    acc = _get_or_create_account(db, user.id)
    acc.cash = INITIAL_CASH
    db.commit()
    return _build_portfolio(db, user.id)


@router.get("/ranking", response_model=RankingResponse)
def ranking(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> RankingResponse:
    """전 유저 총자산(수익률) 랭킹. 현재가는 _resolve_price 캐시(realtime 5분/종가) 활용."""
    accounts = list(db.scalars(select(PaperAccount)))
    # 종목별 현재가 1회만 조회(캐시) — 같은 종목 반복 회피
    price_cache: dict = {}

    def _price(tk: str):
        if tk not in price_cache:
            try:
                price_cache[tk] = float(_resolve_price(tk)["price"])
            except HTTPException:
                price_cache[tk] = None
        return price_cache[tk]

    rows = []
    for acc in accounts:
        value = acc.cash
        for h in _holdings(db, acc.user_id):
            cur = _price(h.ticker)
            if cur:
                value += int(round(cur * h.qty))
        u = db.get(User, acc.user_id)
        rows.append({
            "user_id": acc.user_id,
            "nickname": _display_nickname(u) if u else "?",
            "total_value": value,
            "total_pnl_pct": round((value - INITIAL_CASH) / INITIAL_CASH * 100.0, 2),
        })
    rows.sort(key=lambda r: r["total_value"], reverse=True)

    my_rank = None
    items = []
    for i, r in enumerate(rows, 1):
        is_me = r["user_id"] == user.id
        if is_me:
            my_rank = i
        if i <= 50 or is_me:  # 상위 50 + 본인
            items.append(RankingItem(
                rank=i, nickname=r["nickname"], total_value=r["total_value"],
                total_pnl_pct=r["total_pnl_pct"], is_me=is_me,
            ))
    return RankingResponse(
        rankings=items, my_rank=my_rank, total_players=len(rows),
    )
