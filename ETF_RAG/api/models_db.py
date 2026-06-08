"""SQLAlchemy ORM 모델 (사용자 DB). Pydantic api/models.py와 분리.

Python 3.9: Mapped[Optional[str]] 사용(str|None 금지), datetime.now(timezone.utc).
Phase A는 User만. B에서 Watchlist/ChatHistory + relationship 추가 예정.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from api.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)  # 3.9: datetime.UTC 없음


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(
        String(320), unique=True, index=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )


class Watchlist(Base):
    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_watchlist_user_ticker"),
    )


class ChatHistory(Base):
    __tablename__ = "chat_histories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )
    __table_args__ = (
        Index("ix_chat_user_created", "user_id", "created_at"),
    )
