"""SQLAlchemy ORM 모델 (사용자 DB). Pydantic api/models.py와 분리.

Python 3.9: Mapped[Optional[str]] 사용(str|None 금지), datetime.now(timezone.utc).
Phase A는 User만. B에서 Watchlist/ChatHistory + relationship 추가 예정.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, func
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
