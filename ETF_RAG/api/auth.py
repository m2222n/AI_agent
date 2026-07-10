"""JWT 이메일 인증 라우터 + 비밀번호 해싱 + get_current_user.

핸들러는 전부 동기 def → FastAPI가 threadpool에서 실행(blocking Session OK).
해싱은 bcrypt 직접 사용(passlib 1.7.4 + bcrypt 5.0 비호환 회피).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt  # PyJWT
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from config import (
    JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES,
    JWT_SECRET,
    RESET_TOKEN_EXPIRE_MINUTES,
    FRONTEND_BASE_URL,
)
from api.db import get_db
from api.models import (
    AGE_GROUPS,
    GENDERS,
    AccountDeleteRequest,
    LoginRequest,
    PasswordChangeRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    ProfileUpdateRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
)
from api.models_db import (
    ChatHistory,
    PaperAccount,
    PaperHolding,
    PaperRound,
    PaperSnapshot,
    PaperTrade,
    PushSubscription,
    User,
    Watchlist,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)  # 401은 직접 던진다

# bcrypt는 72바이트 초과 시 에러 → 직접 절단 (초과분은 보안상 무의미).
_BCRYPT_MAX = 72


def hash_password(pw: str) -> str:
    pw_bytes = pw.encode("utf-8")[:_BCRYPT_MAX]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:_BCRYPT_MAX], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_reset_token(user_id: int) -> str:
    """비밀번호 재설정 전용 단기 토큰(purpose=pwreset). 별도 DB 테이블 없이 JWT로.

    액세스 토큰과 구분하기 위해 purpose 클레임을 넣고, 짧게 만료시킨다.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "purpose": "pwreset",
        "iat": now,
        "exp": now + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_reset_token(token: str) -> Optional[int]:
    """재설정 토큰 검증. 유효하면 user_id, 아니면 None(만료·위조·purpose 불일치)."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("purpose") != "pwreset":
            return None
        return int(payload["sub"])
    except Exception:  # noqa: BLE001 — 만료/위조 전부 무효 처리
        return None


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="유효하지 않은 인증 정보입니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None:
        raise cred_exc
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise cred_exc
    user = db.get(User, user_id)
    if user is None:
        raise cred_exc
    return user


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    if db.scalar(select(User).where(User.email == req.email)) is not None:
        raise HTTPException(status_code=400, detail="이미 가입된 이메일입니다.")
    if req.gender not in GENDERS:  # 필수 — 허용값 아니면 거부
        raise HTTPException(status_code=400, detail="성별을 선택하세요.")
    user = User(
        email=req.email,
        password_hash=hash_password(req.password),
        gender=req.gender,
    )
    if req.age_group in AGE_GROUPS:  # 선택 — 허용값만 저장, 그 외 무시
        user.age_group = req.age_group
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == req.email))
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다."
        )
    return TokenResponse(access_token=create_access_token(user.id))


def _display_nickname(user: User) -> str:
    """닉네임 미설정 시 이메일 local-part(앞부분)로 fallback."""
    if user.nickname and user.nickname.strip():
        return user.nickname
    return user.email.split("@", 1)[0]


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id, email=user.email, nickname=_display_nickname(user),
        age_group=user.age_group or None, gender=user.gender or None,
    )


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return _user_response(user)


@router.put("/password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    req: PasswordChangeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """현재 비밀번호 확인 후 새 비밀번호로 변경."""
    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="현재 비밀번호가 올바르지 않습니다.")
    if req.new_password == req.current_password:
        raise HTTPException(status_code=400, detail="기존과 다른 비밀번호를 사용하세요.")
    user.password_hash = hash_password(req.new_password)
    db.commit()


@router.put("/profile", response_model=UserResponse)
def update_profile(
    req: ProfileUpdateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    """닉네임 변경(+선택 나이대). 공백만 입력은 거부."""
    nickname = req.nickname.strip()
    if not nickname:
        raise HTTPException(status_code=400, detail="닉네임을 입력하세요.")
    user.nickname = nickname
    # 나이대: None이면 미변경, 허용값이면 설정, 그 외(빈값 등)는 미설정으로 비움
    if req.age_group is not None:
        user.age_group = req.age_group if req.age_group in AGE_GROUPS else None
    # 성별: None이면 미변경, 허용값이면 설정, 그 외는 무시(비움 아님 — 필수값이라 보존)
    if req.gender is not None and req.gender in GENDERS:
        user.gender = req.gender
    db.commit()
    db.refresh(user)
    return _user_response(user)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(
    req: AccountDeleteRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """회원 탈퇴 — 비밀번호 재확인 후 계정 + 연관 데이터(관심종목/대화이력/푸시구독) 삭제.

    FK relationship/cascade를 모델에 안 걸었으므로 명시적으로 자식 행을 먼저 지운다."""
    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=400, detail="비밀번호가 올바르지 않습니다.")
    uid = user.id
    db.execute(delete(Watchlist).where(Watchlist.user_id == uid))
    db.execute(delete(ChatHistory).where(ChatHistory.user_id == uid))
    db.execute(delete(PushSubscription).where(PushSubscription.user_id == uid))
    # 가상투자(Phase F #69~71) — user_id FK를 가지나 모델 cascade 미설정이므로 명시 삭제.
    db.execute(delete(PaperTrade).where(PaperTrade.user_id == uid))
    db.execute(delete(PaperHolding).where(PaperHolding.user_id == uid))
    db.execute(delete(PaperSnapshot).where(PaperSnapshot.user_id == uid))
    db.execute(delete(PaperRound).where(PaperRound.user_id == uid))
    db.execute(delete(PaperAccount).where(PaperAccount.user_id == uid))
    db.delete(user)
    db.commit()


@router.post("/password-reset/request", status_code=status.HTTP_202_ACCEPTED)
def password_reset_request(
    req: PasswordResetRequest, db: Session = Depends(get_db)
) -> dict:
    """비밀번호 재설정 링크 이메일 발송 요청.

    보안: 이메일 존재 여부를 노출하지 않는다 — 가입 여부와 무관하게 항상 202.
    실제 발송은 RESEND_API_KEY 설정 시에만(미설정이면 no-op). 가입된 이메일이면
    재설정 토큰을 담은 링크를 발송.
    """
    from api.email import send_password_reset

    user = db.scalar(select(User).where(User.email == req.email))
    if user is not None:
        token = create_reset_token(user.id)
        base = FRONTEND_BASE_URL or ""  # 미설정 시 상대경로(운영은 env 권장)
        reset_url = f"{base}/reset?token={token}"
        send_password_reset(user.email, reset_url)
    # 유저 없어도 동일 응답(열거 공격 방지)
    return {"ok": True, "detail": "가입된 이메일이면 재설정 링크를 보냈어요."}


@router.post("/password-reset/confirm", status_code=status.HTTP_204_NO_CONTENT)
def password_reset_confirm(
    req: PasswordResetConfirm, db: Session = Depends(get_db)
) -> None:
    """재설정 토큰 + 새 비밀번호로 변경. 토큰 만료/위조면 400."""
    user_id = verify_reset_token(req.token)
    if user_id is None:
        raise HTTPException(status_code=400, detail="유효하지 않거나 만료된 링크예요.")
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=400, detail="유효하지 않은 링크예요.")
    user.password_hash = hash_password(req.new_password)
    db.commit()


# Phase C용 — 로그인 시 서버 영속, 비로그인 시 localStorage fallback.
def get_current_user_optional(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if creds is None:
        return None
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return db.get(User, int(payload["sub"]))
    except Exception:  # noqa: BLE001 — optional 경로는 실패 시 None
        return None
