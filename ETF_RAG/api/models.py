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
    days: int = Field(120, ge=5, le=2500)  # 1주(5거래일)~10년


# ── 인증 ────────────────────────────────────────────────
# 허용 나이대 버킷 (분류정보, 선택). 빈 문자열은 '미설정'으로 처리.
AGE_GROUPS = ("10대", "20대", "30대", "40대", "50대", "60대", "70대 이상")


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    age_group: Optional[str] = None  # 선택 — AGE_GROUPS 중 하나(아니면 무시)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    nickname: str  # 미설정 시 이메일 local-part로 fallback (auth.py에서 채움)
    age_group: Optional[str] = None  # 나이대(선택, 미설정 시 None)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)


class ProfileUpdateRequest(BaseModel):
    # 1~40자. 공백만 입력 방지는 라우터에서 strip 후 검증.
    nickname: str = Field(..., min_length=1, max_length=40)
    # 나이대(선택). None=변경 안 함, "" 또는 미허용값=미설정으로 비움, 허용값=설정.
    age_group: Optional[str] = None


class AccountDeleteRequest(BaseModel):
    # 탈퇴는 되돌릴 수 없으므로 비밀번호 재확인.
    password: str = Field(..., min_length=1)


# ── 가상투자(모의투자) ─────────────────────────────────────
class TradeRequest(BaseModel):
    ticker: str = Field(..., min_length=1)  # 종목코드 또는 종목명(서버에서 해석)
    qty: int = Field(..., ge=1)


class HoldingItem(BaseModel):
    ticker: str
    name: str
    qty: int
    avg_price: float          # 평단가
    current_price: float      # 현재가
    eval_value: int           # 평가금액 = current_price·qty
    cost_value: int           # 매입금액 = avg_price·qty
    pnl: int                  # 평가손익 = eval - cost
    pnl_pct: float            # 평가수익률 %
    price_source: str         # "kis"|"yfinance"|"close"
    since: Optional[str] = None       # 현재 보유 시작일 YYYY-MM-DD (재진입 시 마지막 진입일)
    holding_days: Optional[int] = None  # 보유 일수(since~오늘)


class PortfolioResponse(BaseModel):
    cash: int                 # 현금 잔고
    holdings: List[HoldingItem]
    holdings_value: int       # 보유 종목 평가액 합
    total_value: int          # 총 자산 = cash + holdings_value
    initial_cash: int         # 기준 자본(1억)
    total_pnl: int            # 총 손익 = total_value - initial_cash
    total_pnl_pct: float      # 총 수익률 %


class TradeResult(BaseModel):
    ok: bool
    side: Literal["buy", "sell"]
    ticker: str
    name: str
    qty: int
    price: float
    amount: int
    cash: int                 # 체결 후 잔고
    realized_pnl: Optional[int] = None  # 매도 시 실현손익
    price_source: str


class TradeHistoryItem(BaseModel):
    ticker: str
    name: Optional[str]
    side: Literal["buy", "sell"]
    qty: int
    price: float
    amount: int
    realized_pnl: Optional[int]
    created_at: str


class TradeHistoryResponse(BaseModel):
    trades: List[TradeHistoryItem]


class RankingItem(BaseModel):
    rank: int
    nickname: str
    total_value: int
    total_pnl_pct: float
    is_me: bool = False


class RankingResponse(BaseModel):
    rankings: List[RankingItem]
    my_rank: Optional[int] = None
    total_players: int


class PaperHistoryPoint(BaseModel):
    date: str          # YYYYMMDD
    total_value: int
    pnl_pct: float     # 초기자본 대비 수익률 %


class PaperHistoryResponse(BaseModel):
    points: List[PaperHistoryPoint]
    chart_b64: Optional[str] = None  # 수익률 추이 라인 차트


class SnapshotAllResponse(BaseModel):
    ok: bool
    users_snapshotted: int
    date: str


class RoundSymbolPnl(BaseModel):
    ticker: str
    name: str
    realized: int      # 실현손익(매도분)
    unrealized: int    # 미실현손익(초기화 시점 보유분 평가)
    total: int


class PaperRoundItem(BaseModel):
    round_no: int
    started_at: str
    ended_at: str
    initial_cash: int
    final_value: int
    return_pct: float
    trade_count: int
    symbols: List[RoundSymbolPnl]  # 종목별 손익(total 내림차순)


class PaperRoundsResponse(BaseModel):
    rounds: List[PaperRoundItem]  # 최신 라운드부터


class TradeStatsResponse(BaseModel):
    """현재 라운드 거래 통계(실현 기준). 매도 체결만 손익 판정에 사용."""
    total_trades: int          # 전체 체결 수(매수+매도)
    buy_count: int
    sell_count: int            # 청산(손익 확정) 횟수
    win_count: int             # 이익 실현 매도 수
    loss_count: int            # 손실 실현 매도 수
    win_rate: float            # 승률 % = win/sell (매도 없으면 0)
    realized_pnl: int          # 실현손익 합(매도 realized_pnl)
    avg_win: int               # 이익 매도 평균 실현손익
    avg_loss: int              # 손실 매도 평균 실현손익(음수)
    profit_factor: Optional[float] = None  # 총이익/총손실 절댓값(손실 0이면 None)
    best_trade: Optional[int] = None       # 최대 단일 실현이익
    worst_trade: Optional[int] = None      # 최대 단일 실현손실


class DividendItem(BaseModel):
    ticker: str
    name: str
    qty: int
    dps: float          # 주당 연간 배당금(TTM)
    amount: int         # dps × qty (지급액)


class DividendResponse(BaseModel):
    """배당 정산 결과. 데이터 한계로 '예상 연간 배당금'을 라운드당 1회 현금 지급."""
    ok: bool
    paid: bool          # 이번 호출에서 실제 지급했는지(이미 정산했으면 False)
    total: int          # 지급(예정) 총액
    cash: int           # 정산 후 현금
    items: List[DividendItem]
    message: str


class ResetRequest(BaseModel):
    # 실수 방지 — 클라이언트에서 "초기화" 확인 입력. 서버도 재검증.
    confirm: str = Field(..., min_length=1)


# ── 유저별 저장 (관심종목 / 대화이력) ──────────────────────
class WatchlistResponse(BaseModel):
    tickers: List[str]


class WatchlistDetailItem(BaseModel):
    ticker: str
    name: str  # 종목명(해석 실패 시 ticker로 fallback)


class WatchlistDetailResponse(BaseModel):
    items: List[WatchlistDetailItem]


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


class WatchlistAlertResponse(BaseModel):
    ok: bool
    users_notified: int
    pushes_sent: int
    movers: int
