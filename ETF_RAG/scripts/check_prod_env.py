"""
프로덕션 백엔드 env 설정 진단 스크립트

Railway 백엔드에 KIS / VAPID / CRON_TOKEN / 가상투자 스냅샷 자동화 env가
제대로 등록됐는지 라이브 엔드포인트로 점검한다. 보안상 백엔드가 비밀값을
직접 노출하지 않으므로, 공개 신호(활성 플래그·HTTP 상태)로 추론한다.

사용법:
    python scripts/check_prod_env.py                       # 기본 URL(프로덕션)
    python scripts/check_prod_env.py --base http://localhost:8000
    python scripts/check_prod_env.py --cron-token <TOKEN>   # CRON 토큰 실제 검증
    python scripts/check_prod_env.py --ticker 005930        # KIS 시세 source 확인

종료 코드: 모든 점검 통과(또는 정보성)면 0, 점검 자체 실패 시 1.
"""

import sys
import ssl
import json
import argparse
import urllib.request
import urllib.error

DEFAULT_BASE = "https://aiagent-production-75ca.up.railway.app"
TIMEOUT = 15

# macOS 시스템 Python은 CA 미설치라 HTTPS 검증이 깨질 수 있음(뉴스 SSL 이슈와 동일)
# → certifi 번들을 명시. certifi 없으면 시스템 기본 컨텍스트로 fallback.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = ssl.create_default_context()

OK = "✅"
NO = "❌"
WARN = "⚠️ "
INFO = "ℹ️ "


def _get(base, path, method="GET", token=None):
    """엔드포인트 호출 → (http_status, body_dict_or_text). 오류 시 status=None."""
    url = base.rstrip("/") + path
    headers = {}
    if token:
        headers["X-Cron-Token"] = token
    req = urllib.request.Request(url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=_SSL_CTX) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def check_health(base):
    status, body = _get(base, "/health")
    if status == 200 and isinstance(body, dict) and body.get("ready"):
        print(f"{OK} 백엔드 health: ready")
        return True
    print(f"{NO} 백엔드 health 비정상: status={status}, body={body}")
    return False


def check_kis(base, ticker):
    """장중에는 source=kis면 KIS 활성, 장외에는 추론 불가(close fallback 정상)."""
    status, body = _get(base, f"/tabs/price?ticker={ticker}")
    if status != 200 or not isinstance(body, dict):
        print(f"{NO} KIS 시세 점검 실패: status={status}, body={body}")
        return
    src = body.get("source")
    market_open = body.get("market_open")
    if src == "kis":
        print(f"{OK} KIS env 등록됨 — 실시간 시세 동작 (source=kis, {ticker})")
    elif market_open:
        print(f"{WARN}장중인데 source={src} — KIS env 미등록이거나 KIS 일시 오류 "
              f"(yfinance/close fallback 중)")
    else:
        print(f"{INFO}장 마감 — source={src} (close fallback 정상). "
              f"KIS 등록 여부는 장중(평일 09:00~15:30 KST)에 재실행해 확인")


def check_vapid(base):
    status, body = _get(base, "/push/vapid-public-key")
    if status == 200 and isinstance(body, dict):
        if body.get("enabled") and body.get("public_key"):
            print(f"{OK} VAPID env 등록됨 — 웹 푸시 발송 가능")
        else:
            print(f"{NO} VAPID 미등록 (enabled=false) — 웹 푸시 발송 불가. "
                  f"scripts/gen_vapid_keys.py로 키 생성 후 Railway env "
                  f"VAPID_PUBLIC_KEY/PRIVATE_KEY/SUBJECT 등록 필요")
    else:
        print(f"{NO} VAPID 점검 실패: status={status}, body={body}")


def check_cron(base, token):
    """CRON 보호 엔드포인트. 토큰 미설정이면 401/403, 일치하면 200."""
    if token:
        status, body = _get(base, "/push/run-watchlist-alerts", method="POST", token=token)
        if status == 200:
            print(f"{OK} CRON_TOKEN 일치 — 관심종목 알림 트리거 동작 "
                  f"(users_notified={body.get('users_notified') if isinstance(body, dict) else '?'})")
        elif status in (401, 403):
            print(f"{NO} CRON_TOKEN 불일치/미설정 (status={status}) — "
                  f"Railway env CRON_TOKEN과 전달 토큰이 다름")
        else:
            print(f"{WARN}CRON 트리거 응답 status={status}, body={body}")
    else:
        status, _ = _get(base, "/push/run-watchlist-alerts", method="POST")
        print(f"{INFO}CRON 보호 엔드포인트 status={status} (토큰 없이 호출 → 403 정상). "
              f"실제 설정 여부는 --cron-token <값>으로 검증")


def check_snapshot(base, token):
    if token:
        status, body = _get(base, "/me/paper/snapshot-all", method="POST", token=token)
        if status == 200:
            print(f"{OK} PAPER 스냅샷 트리거 동작 (CRON_TOKEN 일치)")
        elif status in (401, 403):
            print(f"{NO} PAPER 스냅샷 CRON_TOKEN 불일치/미설정 (status={status})")
        else:
            print(f"{WARN}PAPER 스냅샷 응답 status={status}, body={body}")
    else:
        status, _ = _get(base, "/me/paper/snapshot-all", method="POST")
        print(f"{INFO}PAPER 스냅샷 엔드포인트 status={status} (토큰 없이 → 403 정상)")


def main():
    p = argparse.ArgumentParser(description="프로덕션 백엔드 env 설정 진단")
    p.add_argument("--base", default=DEFAULT_BASE, help="백엔드 base URL")
    p.add_argument("--cron-token", default=None, help="CRON_TOKEN 실제 검증용")
    p.add_argument("--ticker", default="005930", help="KIS 시세 확인 종목코드")
    args = p.parse_args()

    print(f"=== 프로덕션 env 진단: {args.base} ===")
    if not check_health(args.base):
        print("백엔드가 준비되지 않아 나머지 점검을 건너뜁니다.")
        sys.exit(1)
    check_kis(args.base, args.ticker)
    check_vapid(args.base)
    check_cron(args.base, args.cron_token)
    check_snapshot(args.base, args.cron_token)
    print("\n※ 비밀값은 백엔드가 노출하지 않으므로 활성 플래그/HTTP 상태로 추론합니다.")
    print("  env 등록은 Railway 대시보드 → 백엔드 서비스(AI_agent) → Variables 에서.")


if __name__ == "__main__":
    main()
