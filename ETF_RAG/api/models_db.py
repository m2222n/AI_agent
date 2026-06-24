"""SQLAlchemy ORM 모델 (사용자 DB). Pydantic api/models.py와 분리.

Python 3.9: Mapped[Optional[str]] 사용(str|None 금지), datetime.now(timezone.utc).
Phase A는 User만. B에서 Watchlist/ChatHistory + relationship 추가 예정.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
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
    # 표시용 닉네임(선택). 로그인은 이메일로, 이건 화면 표시·가상투자 랭킹 등에 사용.
    # 기존 유저/미입력은 NULL → 응답 시 이메일 local-part로 fallback.
    nickname: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
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


class PushSubscription(Base):
    """웹 푸시 구독 (브라우저 PushSubscription). user_id별, endpoint unique."""
    __tablename__ = "push_subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=False
    )
    endpoint: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    p256dh: Mapped[str] = mapped_column(String(255), nullable=False)  # 구독 공개키
    auth: Mapped[str] = mapped_column(String(255), nullable=False)    # 인증 시크릿
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
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


# ── 가상투자(모의투자) ─────────────────────────────────────
# 금액은 정수 원(KRW). 한국 주식은 1원 단위라 BigInteger로 충분(1억=1e8).
INITIAL_CASH = 100_000_000  # 가입 시 지급 가상 현금(1억 원)


class PaperAccount(Base):
    """유저별 가상투자 계좌 — 현금 잔고. 보유종목은 PaperHolding."""
    __tablename__ = "paper_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), unique=True, index=True, nullable=False
    )
    cash: Mapped[int] = mapped_column(BigInteger, nullable=False, default=INITIAL_CASH)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    # 현재 라운드 시작 시각 (초기화 시 갱신 → 라운드 결산 기간 산정)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )


class PaperHolding(Base):
    """보유 종목 — 평단가(avg_price)는 매수 시 가중평균으로 갱신."""
    __tablename__ = "paper_holdings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)        # 보유 수량
    avg_price: Mapped[float] = mapped_column(Float, nullable=False)  # 매입 평단가
    __table_args__ = (
        UniqueConstraint("user_id", "ticker", name="uq_holding_user_ticker"),
    )


class PaperTrade(Base):
    """체결 내역 — 매수/매도 1건. amount = price·qty (수수료 미반영, 가상)."""
    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    side: Mapped[str] = mapped_column(String(4), nullable=False)  # buy | sell
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)   # 체결 단가
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 체결 금액
    realized_pnl: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True  # 매도 시 실현손익(매수는 NULL)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True
    )
    __table_args__ = (
        Index("ix_trade_user_created", "user_id", "created_at"),
    )


class PaperSnapshot(Base):
    """일별 총 평가액 스냅샷 — 수익률 추이 차트/랭킹용. (user_id, date) 유일."""
    __tablename__ = "paper_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=False
    )
    date: Mapped[str] = mapped_column(String(8), nullable=False)  # YYYYMMDD
    total_value: Mapped[int] = mapped_column(BigInteger, nullable=False)  # 현금+평가액
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_snapshot_user_date"),
    )


class PaperRound(Base):
    """라운드 결산 — 계좌 초기화 시 직전 라운드의 성과를 1건 저장(기록 보존).

    summary: 종목별 손익 요약 JSON 문자열
      [{"ticker","name","realized","unrealized","total"}, ...] (total 내림차순).
    """
    __tablename__ = "paper_rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), index=True, nullable=False
    )
    round_no: Mapped[int] = mapped_column(Integer, nullable=False)  # 유저별 1,2,3...
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    initial_cash: Mapped[int] = mapped_column(BigInteger, nullable=False)
    final_value: Mapped[int] = mapped_column(BigInteger, nullable=False)
    return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 종목별 손익 JSON
    __table_args__ = (
        Index("ix_round_user_no", "user_id", "round_no"),
    )
