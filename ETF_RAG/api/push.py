"""웹 푸시 알림 — 구독 등록/해제 + VAPID 공개키 + 발송 헬퍼 (Phase F 푸시 A).

구독은 로그인 유저별(user_id) 저장. 발송 헬퍼 send_push_to_user는 PR-B(관심종목
일일 알림)에서 재사용한다. VAPID 미설정 시 모든 발송은 graceful no-op.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.db import get_db
from api.deps import verify_cron_token
from api.models import (
    PushStatusResponse,
    PushSubscribeRequest,
    PushUnsubscribeRequest,
    VapidPublicKeyResponse,
    WatchlistAlertResponse,
)
from api.models_db import PushSubscription, User, Watchlist

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/push", tags=["push"])


def _vapid():
    from config import VAPID
    return VAPID


def send_push(subscription_info: dict, payload: dict) -> bool:
    """단일 구독에 푸시 전송. 성공 True. 410/404(만료)면 False(호출자가 삭제).

    VAPID 미설정/pywebpush 미설치 시 False (no-op).
    """
    cfg = _vapid()
    if not cfg.get("enabled"):
        return False
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        logger.warning("pywebpush 미설치 — 푸시 발송 불가")
        return False
    try:
        webpush(
            subscription_info=subscription_info,
            data=json.dumps(payload, ensure_ascii=False),
            vapid_private_key=cfg["private_key"],
            vapid_claims={"sub": cfg["subject"]},
        )
        return True
    except WebPushException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (404, 410):
            logger.info(f"푸시 구독 만료(삭제 대상): {status}")
            raise _SubscriptionGone()  # 호출자가 DB에서 삭제
        logger.warning(f"푸시 발송 실패: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"푸시 발송 예외: {e}")
        return False


class _SubscriptionGone(Exception):
    """구독이 만료(410/404) — DB에서 삭제해야 함."""


def send_push_to_user(db: Session, user_id: int, payload: dict) -> int:
    """유저의 모든 구독에 발송. 만료 구독은 삭제. 성공 발송 수 반환."""
    subs = db.execute(
        select(PushSubscription).where(PushSubscription.user_id == user_id)
    ).scalars().all()
    sent = 0
    for s in subs:
        info = {"endpoint": s.endpoint,
                "keys": {"p256dh": s.p256dh, "auth": s.auth}}
        try:
            if send_push(info, payload):
                sent += 1
        except _SubscriptionGone:
            db.delete(s)
    db.commit()
    return sent


# ── 엔드포인트 ────────────────────────────────────────────
@router.get("/vapid-public-key", response_model=VapidPublicKeyResponse)
def vapid_public_key() -> VapidPublicKeyResponse:
    """프론트 applicationServerKey용 공개키 (인증 불필요). 미설정 시 enabled=false."""
    cfg = _vapid()
    return VapidPublicKeyResponse(
        public_key=cfg.get("public_key", ""), enabled=bool(cfg.get("enabled"))
    )


@router.put("/subscribe", response_model=PushStatusResponse)
def subscribe(
    req: PushSubscribeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PushStatusResponse:
    """브라우저 푸시 구독 저장(멱등 — endpoint 기준 upsert)."""
    existing = db.execute(
        select(PushSubscription).where(PushSubscription.endpoint == req.endpoint)
    ).scalar_one_or_none()
    if existing:
        existing.user_id = user.id
        existing.p256dh = req.keys.p256dh
        existing.auth = req.keys.auth
    else:
        db.add(PushSubscription(
            user_id=user.id, endpoint=req.endpoint,
            p256dh=req.keys.p256dh, auth=req.keys.auth,
        ))
    db.commit()
    return PushStatusResponse(ok=True)


@router.post("/unsubscribe", response_model=PushStatusResponse)
def unsubscribe(
    req: PushUnsubscribeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PushStatusResponse:
    """구독 해제 (해당 유저의 endpoint만)."""
    db.execute(
        delete(PushSubscription).where(
            PushSubscription.endpoint == req.endpoint,
            PushSubscription.user_id == user.id,
        )
    )
    db.commit()
    return PushStatusResponse(ok=True)


@router.post("/test", response_model=PushStatusResponse)
def send_test(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PushStatusResponse:
    """내 구독 전체로 테스트 알림 발송 (구독 검증용)."""
    if not _vapid().get("enabled"):
        raise HTTPException(503, "푸시 미설정 (VAPID 키 없음)")
    sent = send_push_to_user(db, user.id, {
        "title": "투자 AI 알림 테스트",
        "body": "푸시 알림이 정상 동작합니다 🎉",
        "url": "/",
    })
    return PushStatusResponse(ok=sent > 0,
                              detail=f"{sent}개 기기로 발송")


# ── 관심종목 일일 알림 (배치) ─────────────────────────────
def run_watchlist_alerts(db: Session) -> dict:
    """구독 유저별 관심종목 중 당일 ±임계% 종목을 묶어 1건씩 push.

    종목 등락률은 _find_structured_data(현재 로드된 deploy/DB 데이터)로 조회.
    Returns {"users_notified", "pushes_sent", "movers"}.
    """
    from config import WATCHLIST_ALERT_THRESHOLD as THR
    from src.llm.tools import _find_structured_data

    # 구독이 있는 유저만 대상 (구독 없으면 보낼 곳 없음)
    user_ids = db.execute(
        select(PushSubscription.user_id).distinct()
    ).scalars().all()

    users_notified = 0
    pushes_sent = 0
    total_movers = 0

    for uid in user_ids:
        tickers = db.execute(
            select(Watchlist.ticker).where(Watchlist.user_id == uid)
        ).scalars().all()
        if not tickers:
            continue

        movers = []
        for t in tickers:
            data = _find_structured_data(t)
            if not data:
                continue
            pct = data.get("change_pct")
            if pct is None or abs(pct) < THR:
                continue
            movers.append((data.get("name", t), pct))

        if not movers:
            continue
        total_movers += len(movers)

        # 알림 본문: "삼성전자 +6.2%, SK하이닉스 -5.4%"
        movers.sort(key=lambda x: abs(x[1]), reverse=True)
        parts = [f"{n} {p:+.1f}%" for n, p in movers]
        title = f"관심종목 {len(movers)}개 급변동"
        body = ", ".join(parts[:5]) + ("…" if len(parts) > 5 else "")
        sent = send_push_to_user(db, uid, {
            "title": title, "body": body, "url": "/technical",
        })
        if sent:
            users_notified += 1
            pushes_sent += sent

    return {"users_notified": users_notified,
            "pushes_sent": pushes_sent,
            "movers": total_movers}


@router.post("/run-watchlist-alerts", response_model=WatchlistAlertResponse)
def run_alerts(
    _: None = Depends(verify_cron_token),
    db: Session = Depends(get_db),
) -> WatchlistAlertResponse:
    """관심종목 일일 알림 발송 (GitHub Actions 수집 후 호출). X-Cron-Token 보호.

    VAPID 미설정 시 발송은 no-op(0건).
    """
    result = run_watchlist_alerts(db)
    return WatchlistAlertResponse(ok=True, **result)
