"""가상투자(모의투자) — 1억 가상 현금으로 매수/매도, 평가손익, 거래내역, 랭킹.

체결가는 현재가(장중 KIS/yfinance, 장외 수집 종가) — api.tabs._price_blocking 재사용.
모든 핸들러 get_current_user 뒤, 동기 def(threadpool). 금액은 정수 원(KRW).
수수료/세금은 미반영(가상). 공매도/미수 불가(현금·보유 범위 내).
"""

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import delete, func as safunc, select
from sqlalchemy.orm import Session

from api.auth import get_current_user, _display_nickname
from api.db import get_db
from api.models import (
    HoldingItem,
    PaperHistoryPoint,
    PaperHistoryResponse,
    PaperRoundItem,
    PaperRoundsResponse,
    PortfolioResponse,
    RankingItem,
    RankingResponse,
    ResetRequest,
    RoundSymbolPnl,
    SnapshotAllResponse,
    TradeHistoryItem,
    TradeHistoryResponse,
    TradeRequest,
    TradeResult,
)
from src.data.chart_generator import generate_paper_trend_chart
from api.models_db import (
    INITIAL_CASH,
    PaperAccount,
    PaperHolding,
    PaperRound,
    PaperSnapshot,
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


def _today_kst() -> str:
    """KST 기준 YYYYMMDD (수집/거래일 기준 통일)."""
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(hours=9)).strftime("%Y%m%d")


def _user_total_value(db: Session, user_id: int, price_fn) -> int:
    """유저의 현재 총자산(현금+보유 평가액). price_fn(ticker)->float|None로 현재가 조회."""
    acc = _get_or_create_account(db, user_id)
    value = acc.cash
    for h in _holdings(db, user_id):
        cur = price_fn(h.ticker)
        if cur:
            value += int(round(cur * h.qty))
    return value


def _record_snapshot(db: Session, user_id: int, total_value: int) -> None:
    """오늘(KST) 스냅샷 upsert — 같은 날 여러 번이면 최신값으로 갱신. commit 안 함."""
    today = _today_kst()
    snap = db.scalar(select(PaperSnapshot).where(
        PaperSnapshot.user_id == user_id, PaperSnapshot.date == today))
    if snap is None:
        db.add(PaperSnapshot(user_id=user_id, date=today, total_value=total_value))
    else:
        snap.total_value = total_value


def _price_simple(ticker: str):
    """현재가만 (실패 시 None) — 스냅샷/랭킹용 경량 조회."""
    try:
        return float(_resolve_price(ticker)["price"])
    except HTTPException:
        return None


def _snapshot_after_trade(db: Session, user_id: int) -> None:
    """거래 직후 당일 스냅샷 갱신 — 실패해도 거래 자체는 영향 없게 예외 삼킴."""
    try:
        value = _user_total_value(db, user_id, _price_simple)
        _record_snapshot(db, user_id, value)
        db.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"거래 후 스냅샷 기록 실패(무시): {e}")
        db.rollback()


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
    _snapshot_after_trade(db, user.id)  # 당일 평가액 스냅샷 갱신
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
    _snapshot_after_trade(db, user.id)  # 당일 평가액 스냅샷 갱신
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


def _round_symbol_pnl(db: Session, user_id: int) -> list:
    """현재 라운드의 종목별 손익 = 실현(매도분 합) + 미실현(보유 평가손익).
    [{ticker,name,realized,unrealized,total}, ...] total 내림차순."""
    # 실현손익: 종목별 매도 realized_pnl 합 + 이름(가장 최근 거래명)
    agg: dict = {}
    for t in db.scalars(select(PaperTrade).where(PaperTrade.user_id == user_id)):
        e = agg.setdefault(t.ticker, {"ticker": t.ticker, "name": t.name or t.ticker,
                                      "realized": 0, "unrealized": 0})
        if t.name:
            e["name"] = t.name
        if t.side == "sell" and t.realized_pnl is not None:
            e["realized"] += t.realized_pnl
    # 미실현손익: 현재 보유 평가손익
    for h in _holdings(db, user_id):
        cur = _price_simple(h.ticker)
        e = agg.setdefault(h.ticker, {"ticker": h.ticker, "name": h.ticker,
                                      "realized": 0, "unrealized": 0})
        if cur:
            e["unrealized"] += int(round((cur - h.avg_price) * h.qty))
    out = []
    for e in agg.values():
        e["total"] = e["realized"] + e["unrealized"]
        out.append(e)
    out.sort(key=lambda x: x["total"], reverse=True)
    return out


@router.post("/reset", response_model=PortfolioResponse)
def reset(
    req: ResetRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PortfolioResponse:
    """계좌 초기화 — 직전 라운드 결산(기록 보존) 저장 후 1억으로 새 라운드 시작.

    confirm은 '초기화'여야 진행(실수 방지). 거래가 있었던 라운드만 결산 기록.
    """
    if req.confirm.strip() != "초기화":
        raise HTTPException(400, "확인 문구가 일치하지 않습니다. '초기화'를 입력하세요.")

    acc = _get_or_create_account(db, user.id)
    trade_count = db.scalar(
        select(safunc.count()).select_from(PaperTrade)
        .where(PaperTrade.user_id == user.id)
    ) or 0

    # 거래가 있었으면 라운드 결산 저장
    if trade_count > 0:
        final_value = _user_total_value(db, user.id, _price_simple)
        symbols = _round_symbol_pnl(db, user.id)
        last_no = db.scalar(
            select(safunc.max(PaperRound.round_no))
            .where(PaperRound.user_id == user.id)
        ) or 0
        db.add(PaperRound(
            user_id=user.id, round_no=last_no + 1,
            started_at=acc.started_at, initial_cash=INITIAL_CASH,
            final_value=final_value,
            return_pct=round((final_value - INITIAL_CASH) / INITIAL_CASH * 100.0, 2),
            trade_count=trade_count,
            summary=json.dumps(symbols, ensure_ascii=False),
        ))

    # 초기화
    db.execute(delete(PaperHolding).where(PaperHolding.user_id == user.id))
    db.execute(delete(PaperTrade).where(PaperTrade.user_id == user.id))
    db.execute(delete(PaperSnapshot).where(PaperSnapshot.user_id == user.id))
    acc.cash = INITIAL_CASH
    acc.started_at = datetime.now(timezone.utc)  # 새 라운드 시작
    db.commit()
    _record_snapshot(db, user.id, INITIAL_CASH)  # 새 라운드 첫 스냅샷(1억)
    db.commit()
    return _build_portfolio(db, user.id)


@router.get("/rounds", response_model=PaperRoundsResponse)
def rounds(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> PaperRoundsResponse:
    """지난 라운드 결산 목록 (최신순)."""
    rows = db.scalars(
        select(PaperRound).where(PaperRound.user_id == user.id)
        .order_by(PaperRound.round_no.desc())
    )
    items = []
    for r in rows:
        syms = []
        if r.summary:
            try:
                for s in json.loads(r.summary):
                    syms.append(RoundSymbolPnl(**s))
            except Exception:  # noqa: BLE001 — 손상 summary는 빈 목록
                pass
        items.append(PaperRoundItem(
            round_no=r.round_no,
            started_at=r.started_at.isoformat() if r.started_at else "",
            ended_at=r.ended_at.isoformat() if r.ended_at else "",
            initial_cash=r.initial_cash, final_value=r.final_value,
            return_pct=r.return_pct, trade_count=r.trade_count, symbols=syms,
        ))
    return PaperRoundsResponse(rounds=items)


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


@router.get("/history", response_model=PaperHistoryResponse)
def history(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> PaperHistoryResponse:
    """일별 평가액 스냅샷 → 수익률 추이. 스냅샷 2개 미만이면 차트 없음."""
    snaps = list(db.scalars(
        select(PaperSnapshot).where(PaperSnapshot.user_id == user.id)
        .order_by(PaperSnapshot.date)
    ))
    points = [
        PaperHistoryPoint(
            date=s.date, total_value=s.total_value,
            pnl_pct=round((s.total_value - INITIAL_CASH) / INITIAL_CASH * 100.0, 2),
        ) for s in snaps
    ]
    chart = None
    if len(points) >= 2:
        chart = generate_paper_trend_chart(
            [p.date for p in points], [p.pnl_pct for p in points],
        )
    return PaperHistoryResponse(points=points, chart_b64=chart)


@router.post("/snapshot-all", response_model=SnapshotAllResponse)
def snapshot_all(
    x_cron_token: str = Header(None, alias="X-Cron-Token"),
    db: Session = Depends(get_db),
) -> SnapshotAllResponse:
    """전 유저 당일 평가액 스냅샷 기록 (GitHub Actions 수집 후 호출). X-Cron-Token 보호.

    거래 안 한 날도 시세 변동을 추이에 반영하기 위함. 종목별 현재가는 1회만 조회(캐시).
    """
    from config import CRON_TOKEN
    if not CRON_TOKEN:
        raise HTTPException(403, "CRON_TOKEN 미설정 — 비활성")
    if x_cron_token != CRON_TOKEN:
        raise HTTPException(403, "잘못된 토큰")

    price_cache: dict = {}

    def _cached(tk: str):
        if tk not in price_cache:
            price_cache[tk] = _price_simple(tk)
        return price_cache[tk]

    accounts = list(db.scalars(select(PaperAccount)))
    for acc in accounts:
        value = _user_total_value(db, acc.user_id, _cached)
        _record_snapshot(db, acc.user_id, value)
    db.commit()
    return SnapshotAllResponse(
        ok=True, users_snapshotted=len(accounts), date=_today_kst(),
    )
