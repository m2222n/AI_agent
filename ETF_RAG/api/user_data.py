"""유저별 저장 — 관심종목/대화이력 CRUD (Phase F-1 B).

모든 엔드포인트 get_current_user 뒤. 동기 def 핸들러(threadpool).
"""

import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.db import get_db
from api.models import (
    ChatHistoryAppend,
    ChatHistoryItemDB,
    ChatHistoryResponse,
    WatchlistResponse,
)
from api.models_db import ChatHistory, User, Watchlist

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/me", tags=["user-data"])


# ── 관심종목 ────────────────────────────────────────────
@router.get("/watchlist", response_model=WatchlistResponse)
def get_watchlist(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> WatchlistResponse:
    rows = db.scalars(
        select(Watchlist.ticker)
        .where(Watchlist.user_id == user.id)
        .order_by(Watchlist.created_at)
    ).all()
    return WatchlistResponse(tickers=list(rows))


@router.put("/watchlist/{ticker}", response_model=WatchlistResponse,
            status_code=status.HTTP_200_OK)
def add_watchlist(
    ticker: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WatchlistResponse:
    exists = db.scalar(
        select(Watchlist).where(
            Watchlist.user_id == user.id, Watchlist.ticker == ticker
        )
    )
    if exists is None:
        db.add(Watchlist(user_id=user.id, ticker=ticker))
        db.commit()
    return get_watchlist(user, db)


@router.delete("/watchlist/{ticker}", response_model=WatchlistResponse)
def remove_watchlist(
    ticker: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WatchlistResponse:
    db.execute(
        delete(Watchlist).where(
            Watchlist.user_id == user.id, Watchlist.ticker == ticker
        )
    )
    db.commit()
    return get_watchlist(user, db)


# ── 대화 이력 (단순 append, 세션 구분 없음) ──────────────
@router.get("/history", response_model=ChatHistoryResponse)
def get_history(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 200,
) -> ChatHistoryResponse:
    rows = db.scalars(
        select(ChatHistory)
        .where(ChatHistory.user_id == user.id)
        .order_by(ChatHistory.created_at, ChatHistory.id)
        .limit(limit)
    ).all()
    return ChatHistoryResponse(
        messages=[
            ChatHistoryItemDB(
                role=r.role, content=r.content,
                question_type=r.question_type, model=r.model,
            )
            for r in rows
        ]
    )


@router.post("/history", response_model=ChatHistoryResponse,
             status_code=status.HTTP_201_CREATED)
def append_history(
    payload: ChatHistoryAppend,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatHistoryResponse:
    for m in payload.messages:
        db.add(ChatHistory(
            user_id=user.id, role=m.role, content=m.content,
            question_type=m.question_type, model=m.model,
        ))
    db.commit()
    return get_history(user, db)


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
def clear_history(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> None:
    db.execute(delete(ChatHistory).where(ChatHistory.user_id == user.id))
    db.commit()
