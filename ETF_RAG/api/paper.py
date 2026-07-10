"""가상투자(모의투자) — 1억 가상 현금으로 매수/매도, 평가손익, 거래내역, 랭킹.

체결가는 현재가(장중 KIS/yfinance, 장외 수집 종가) — api.tabs._price_blocking 재사용.
모든 핸들러 get_current_user 뒤, 동기 def(threadpool). 금액은 정수 원(KRW).
수수료/세금은 미반영(가상). 공매도/미수 불가(현금·보유 범위 내).
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

KST = timezone(timedelta(hours=9))


def _to_kst(dt: Optional[datetime]) -> Optional[datetime]:
    """저장된 created_at(UTC)을 KST로 변환. tz-naive면 UTC로 간주.

    created_at은 _utcnow(tz-aware UTC)로 저장되지만, sqlite(dev/test)는 tzinfo를
    벗겨 naive로 반환한다. naive에 그냥 .astimezone()을 하면 파이썬이 시스템 로컬
    시간으로 오해해 UTC 자정 근처 체결의 진입일이 하루 어긋난다(#93~108 점검). Postgres는
    timestamptz라 정상이지만, 방언 무관하게 안전하도록 naive는 UTC로 못박고 변환한다.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST)


from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func as safunc, select
from sqlalchemy.orm import Session

from api.auth import get_current_user, _display_nickname
from api.db import get_db
from api.deps import verify_cron_token
from api.models import (
    DividendItem,
    DividendResponse,
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
    TradeStatsResponse,
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


def _holding_since_map(db: Session, user_id: int) -> dict:
    """종목별 '현재 보유 구간의 진입일' {ticker: date(KST)}.

    거래를 시간순으로 누적 수량을 따라가다가, 수량이 0→양수가 되는 마지막
    시점이 현재 보유의 시작이다(전량 매도 후 재매수 시 재진입일 반영).
    """
    since: dict = {}
    running: dict = {}
    rows = db.scalars(
        select(PaperTrade).where(PaperTrade.user_id == user_id)
        .order_by(PaperTrade.created_at.asc(), PaperTrade.id.asc())
    )
    for t in rows:
        prev = running.get(t.ticker, 0)
        if t.side == "buy":
            if prev <= 0:  # 신규 진입(0에서 양수로)
                d = _to_kst(t.created_at)
                since[t.ticker] = d.strftime("%Y-%m-%d") if d else None
            running[t.ticker] = prev + t.qty
        else:  # sell
            running[t.ticker] = prev - t.qty
            if running[t.ticker] <= 0:
                since.pop(t.ticker, None)  # 전량 청산 → 진입일 리셋
    return since


def _build_portfolio(db: Session, user_id: int) -> PortfolioResponse:
    acc = _get_or_create_account(db, user_id)
    since_map = _holding_since_map(db, user_id)
    today_kst = datetime.now(KST).date()
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
        since = since_map.get(h.ticker)
        holding_days = None
        if since:
            try:
                d0 = datetime.strptime(since, "%Y-%m-%d").date()
                holding_days = (today_kst - d0).days
            except ValueError:
                holding_days = None
        items.append(HoldingItem(
            ticker=h.ticker, name=p["name"], qty=h.qty,
            avg_price=round(h.avg_price, 2), current_price=cur,
            eval_value=eval_value, cost_value=cost_value,
            pnl=pnl, pnl_pct=round(pnl_pct, 2),
            price_source=p.get("source", "close"),
            since=since, holding_days=holding_days,
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
    return datetime.now(KST).strftime("%Y%m%d")


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


@router.get("/stats", response_model=TradeStatsResponse)
def stats(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> TradeStatsResponse:
    """현재 라운드 거래 통계 — 매도(청산) 실현손익 기준 승률·평균손익·손익비."""
    rows = list(db.scalars(
        select(PaperTrade).where(PaperTrade.user_id == user.id)
    ))
    buys = [t for t in rows if t.side == "buy"]
    sells = [t for t in rows if t.side == "sell"]
    pnls = [t.realized_pnl for t in sells if t.realized_pnl is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return TradeStatsResponse(
        total_trades=len(rows),
        buy_count=len(buys),
        sell_count=len(sells),
        win_count=len(wins),
        loss_count=len(losses),
        win_rate=round(len(wins) / len(sells) * 100.0, 1) if sells else 0.0,
        realized_pnl=sum(pnls),
        avg_win=int(round(gross_profit / len(wins))) if wins else 0,
        avg_loss=int(round(sum(losses) / len(losses))) if losses else 0,
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
        best_trade=max(pnls) if pnls else None,
        worst_trade=min(pnls) if pnls else None,
    )


def _holding_dps(tickers: list) -> dict:
    """보유 종목별 최신 DPS(주당 연간 배당금) 조회. {ticker: dps}."""
    if not tickers:
        return {}
    from src.data.database import get_connection, get_latest_dps, DB_PATH
    if not DB_PATH.exists():
        return {}
    conn = get_connection(DB_PATH)
    try:
        return {t: get_latest_dps(conn, t) for t in tickers}
    finally:
        conn.close()


@router.post("/dividend", response_model=DividendResponse)
def dividend(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> DividendResponse:
    """배당 정산 — 보유 종목의 연 DPS × 수량을 현금으로 1회 지급(라운드당 1회).

    데이터 한계: pykrx DPS는 'TTM 연간 배당금'이라 정확한 배당락/지급일이 없음.
    따라서 '예상 연간 배당금'을 라운드당 한 번 현금에 더하는 단순 모델.
    """
    acc = _get_or_create_account(db, user.id)
    holdings = _holdings(db, user.id)
    dps_map = _holding_dps([h.ticker for h in holdings])

    items = []
    total = 0
    for h in holdings:
        dps = dps_map.get(h.ticker, 0.0)
        if dps <= 0:
            continue
        amount = int(round(dps * h.qty))
        if amount <= 0:
            continue
        # 종목명: 현재가 조회 실패해도 ticker로 표기(정산은 진행)
        try:
            name = _resolve_price(h.ticker).get("name", h.ticker)
        except HTTPException:
            name = h.ticker
        total += amount
        items.append(DividendItem(
            ticker=h.ticker, name=name, qty=h.qty, dps=round(dps, 2), amount=amount,
        ))
    items.sort(key=lambda x: x.amount, reverse=True)

    # 라운드당 1회 가드: 이미 이 라운드에서 정산했으면 지급 안 함(미리보기만)
    # started_at이 None인 경우(마이그레이션 직후 등) None 비교 TypeError 방어
    already = (
        acc.last_dividend_at is not None
        and acc.started_at is not None
        and acc.last_dividend_at >= acc.started_at
    )
    if already:
        return DividendResponse(
            ok=True, paid=False, total=total, cash=acc.cash, items=items,
            message="이번 라운드에서는 이미 배당을 정산했어요. 계좌 초기화 후 다시 받을 수 있어요.",
        )
    if total <= 0:
        return DividendResponse(
            ok=True, paid=False, total=0, cash=acc.cash, items=[],
            message="배당을 지급하는 보유 종목이 없어요.",
        )

    acc.cash += total
    acc.last_dividend_at = datetime.now(timezone.utc)
    db.commit()
    _snapshot_after_trade(db, user.id)
    return DividendResponse(
        ok=True, paid=True, total=total, cash=acc.cash, items=items,
        message=f"예상 연간 배당금 {total:,}원을 현금으로 지급했어요. (라운드당 1회)",
    )


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
    acc.last_dividend_at = None  # 새 라운드는 배당 다시 받을 수 있게
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

    # 유저를 한 번에 조회(계정 수만큼 개별 db.get 하던 N+1 제거).
    user_ids = [acc.user_id for acc in accounts]
    users = {
        u.id: u
        for u in db.scalars(select(User).where(User.id.in_(user_ids)))
    } if user_ids else {}

    rows = []
    for acc in accounts:
        value = acc.cash
        for h in _holdings(db, acc.user_id):
            cur = _price(h.ticker)
            if cur:
                value += int(round(cur * h.qty))
        u = users.get(acc.user_id)
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
    _: None = Depends(verify_cron_token),
    db: Session = Depends(get_db),
) -> SnapshotAllResponse:
    """전 유저 당일 평가액 스냅샷 기록 (GitHub Actions 수집 후 호출). X-Cron-Token 보호.

    거래 안 한 날도 시세 변동을 추이에 반영하기 위함. 종목별 현재가는 1회만 조회(캐시).
    """
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
