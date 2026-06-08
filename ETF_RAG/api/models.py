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
