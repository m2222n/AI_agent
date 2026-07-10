"""이메일 발송 — Resend API (비밀번호 재설정 링크).

RESEND_API_KEY 미설정 시 발송을 조용히 스킵(no-op)한다. 비번 재설정 요청은
유저 존재 여부를 노출하면 안 되므로, 발송 실패/비활성이어도 호출부는 항상
성공처럼 응답한다(이 모듈은 성공/스킵을 bool로만 알려줌).
"""

import logging

import httpx

from config import EMAIL_ENABLED, RESEND_API_KEY, RESET_EMAIL_FROM

logger = logging.getLogger(__name__)

_RESEND_URL = "https://api.resend.com/emails"


def send_email(to: str, subject: str, html: str) -> bool:
    """Resend로 이메일 발송. 성공 시 True, 비활성/실패 시 False(예외 안 던짐)."""
    if not EMAIL_ENABLED:
        logger.info("RESEND_API_KEY 미설정 — 이메일 발송 스킵(to=%s)", to)
        return False
    try:
        resp = httpx.post(
            _RESEND_URL,
            headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
            json={"from": RESET_EMAIL_FROM, "to": [to], "subject": subject, "html": html},
            timeout=10.0,
        )
        if resp.status_code >= 400:
            logger.warning("이메일 발송 실패 %s: %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as e:  # noqa: BLE001 — 발송 실패해도 호출부는 계속
        logger.warning("이메일 발송 예외: %s", e)
        return False


def send_password_reset(to: str, reset_url: str) -> bool:
    """비밀번호 재설정 링크 이메일."""
    html = (
        f"<p>비밀번호 재설정을 요청하셨습니다.</p>"
        f'<p><a href="{reset_url}">여기를 눌러 새 비밀번호를 설정</a>하세요. '
        f"(링크는 잠시 후 만료됩니다.)</p>"
        f"<p>요청하지 않으셨다면 이 메일을 무시하세요.</p>"
    )
    return send_email(to, "[투자 AI 어시스턴트] 비밀번호 재설정", html)
