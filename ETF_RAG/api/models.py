"""FastAPI 요청/응답 Pydantic 모델 (v2). Python 3.9 호환 — typing.Optional/List 사용."""

from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field
from typing_extensions import Literal


class ChatMessage(BaseModel):
    """대화 히스토리 한 항목. agent의 [{"role","content"}, ...] 계약과 동일."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    answer: str
    question_type: str
    model: str


class HealthResponse(BaseModel):
    ready: bool
    error: Optional[str] = None


class TickerSearchResponse(BaseModel):
    options: List[str]


class VisitorResponse(BaseModel):
    """방문자 카운터 (당일/누적). Supabase 미설정 시 둘 다 0."""
    daily: int
    total: int


class ComparisonRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2, max_length=2)
    days: int = Field(120, ge=20, le=2500)


# ── 인증 ────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: EmailStr


# ── 유저별 저장 (관심종목 / 대화이력) ──────────────────────
class WatchlistResponse(BaseModel):
    tickers: List[str]


class ChatHistoryItemDB(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    question_type: Optional[str] = None
    model: Optional[str] = None


class ChatHistoryAppend(BaseModel):
    messages: List[ChatHistoryItemDB] = Field(..., min_length=1)


class ChatHistoryResponse(BaseModel):
    messages: List[ChatHistoryItemDB]


# ── 동적 추천질문 / 피드백 ──────────────────────────────
class MoverItem(BaseModel):
    name: str
    ticker: str
    change_pct: float


class MoversResponse(BaseModel):
    gainers: List[MoverItem]
    losers: List[MoverItem]
    most_traded: List[MoverItem]


# ── 실시간 시세 (KIS 우선 → yfinance, 장 외엔 종가) ──────
class PriceResponse(BaseModel):
    name: str
    ticker: str
    price: float                      # 현재가(장중) 또는 종가(장 외)
    prev_close: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    volume: Optional[int] = None
    source: str                       # "kis" | "yfinance" | "close"(수집 종가)
    is_live: bool                     # 장중 실시간 여부
    timestamp: Optional[str] = None    # 조회 시각(실시간) 또는 기준일(종가)
    market_open: bool                 # 현재 장 운영 여부


# ── 호가 10단계 (KIS, 장중 전용) ────────────────────────
class OrderbookLevel(BaseModel):
    price: int
    qty: int


class OrderbookResponse(BaseModel):
    name: str
    ticker: str
    asks: List[OrderbookLevel]      # 매도호가 1~10단계
    bids: List[OrderbookLevel]      # 매수호가 1~10단계
    total_ask_qty: int
    total_bid_qty: int
    timestamp: Optional[str] = None
    source: str                     # "kis"


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["positive", "negative"]
    reason: Optional[str] = None  # 부정 피드백 사유


# ── 사이드바 (데이터 현황 + 종목 목록) ──────────────────
class InstrumentItem(BaseModel):
    name: str
    ticker: str
    close: float
    change_pct: float
    trade_value: float
    sector: Optional[str] = None
    per: Optional[float] = None
    market_cap: Optional[float] = None


class OverviewResponse(BaseModel):
    etf_count: int
    stock_count: int
    as_of: Optional[str] = None  # 기준일 (YYYYMMDD 또는 YYYY-MM-DD)
    top_etfs: List[InstrumentItem]
    top_stocks: List[InstrumentItem]
    sectors: List[str]  # 주식 섹터 목록


# ── 웹 푸시 (VAPID) ─────────────────────────────────────
class PushKeys(BaseModel):
    p256dh: str
    auth: str


class PushSubscribeRequest(BaseModel):
    """브라우저 PushSubscription.toJSON() 형태."""
    endpoint: str
    keys: PushKeys


class PushUnsubscribeRequest(BaseModel):
    endpoint: str


class VapidPublicKeyResponse(BaseModel):
    public_key: str        # applicationServerKey (URL-safe base64). 미설정 시 ""
    enabled: bool


class PushStatusResponse(BaseModel):
    ok: bool
    detail: Optional[str] = None
