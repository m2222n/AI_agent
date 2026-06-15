"""
한국투자증권(KIS) WebSocket 실시간 체결가 — 온디맨드 구독 매니저 (F-2)

설계: 프로세스당 KIS WS 단일 연결을 공유하고, 종목별 구독을 refcount로 관리한다.
- 첫 구독 시 lazy 연결, 마지막 구독 해제 시 연결 종료 → 유휴 시 연결 0
- H0STCNT0(실시간 체결) 파싱 → 최신 틱을 ticker별로 보관 + 구독자 큐로 broadcast
- KIS는 H0STCNT0+H0STASP0 합산 20종목 제한 → 온디맨드라 단일 종목 위주로 자연 회피
- 비활성(키 없음)이면 connect()가 False → 호출자(SSE)가 REST 폴링으로 fallback

asyncio 기반 — FastAPI 이벤트 루프 안에서 동작. websockets 패키지 사용.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# H0STCNT0 실시간 체결 — 레코드당 필드 수(메뉴 항목 수)
_CNT0_FIELDS = 46


def _ws_url() -> str:
    """KIS WebSocket URL (real/vps)."""
    from config import KIS
    return ("ws://ops.koreainvestment.com:31000"
            if KIS.get("env") == "vps"
            else "ws://ops.koreainvestment.com:21000")


def _get_approval_key() -> Optional[str]:
    """WebSocket 접속키(approval_key) 발급. REST 토큰과 별개 (oauth2/Approval)."""
    from config import KIS
    if not KIS.get("enabled"):
        return None
    import requests
    url = f"{KIS['base_url']}/oauth2/Approval"
    body = {
        "grant_type": "client_credentials",
        "appkey": KIS["app_key"],
        "secretkey": KIS["app_secret"],   # ⚠️ Approval은 secretkey (토큰은 appsecret)
    }
    try:
        resp = requests.post(url, json=body, timeout=KIS.get("timeout", 5))
        resp.raise_for_status()
        return resp.json().get("approval_key")
    except Exception as e:
        logger.warning(f"KIS approval_key 발급 실패: {e}")
        return None


def parse_cnt0_record(fields: list) -> Optional[dict]:
    """H0STCNT0 레코드(46필드, '^' split) → 체결 틱 dict.

    필드: 0=종목코드 1=체결시간 2=현재가 3=전일대비부호 4=전일대비 5=전일대비율
          13=누적거래량
    """
    if len(fields) < 14:
        return None

    def _f(i):
        try:
            return float(fields[i])
        except (TypeError, ValueError, IndexError):
            return None

    def _i(i):
        v = _f(i)
        return int(v) if v is not None else None

    price = _f(2)
    if price is None or price <= 0:
        return None
    sign = fields[3] if len(fields) > 3 else ""   # 4·5=하락
    change = _f(4)
    change_pct = _f(5)
    if change is not None and sign in ("4", "5"):
        change = -abs(change)
    if change_pct is not None and sign in ("4", "5"):
        change_pct = -abs(change_pct)

    hhmmss = fields[1] if len(fields) > 1 else ""
    ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    if len(hhmmss) == 6:
        ts = f"{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"

    return {
        "ticker": fields[0],
        "price": round(price),
        "change": round(change) if change is not None else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "volume": _i(13),
        "timestamp": ts,
        "source": "kis-ws",
    }


class KisWsManager:
    """KIS WebSocket 단일 연결 + 종목별 refcount 구독 매니저 (싱글턴)."""

    def __init__(self):
        self._ws = None
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._approval: Optional[str] = None
        # ticker → 구독자 asyncio.Queue 집합
        self._subscribers: dict = {}
        # ticker → 최신 틱 dict
        self._latest: dict = {}

    def is_enabled(self) -> bool:
        from config import KIS
        return bool(KIS.get("enabled"))

    async def _connect(self) -> bool:
        """WS 연결 + 수신 루프 시작 (이미 연결돼 있으면 True)."""
        if self._ws is not None:
            return True
        if not self.is_enabled():
            return False
        self._approval = await asyncio.get_running_loop().run_in_executor(
            None, _get_approval_key)
        if not self._approval:
            return False
        try:
            import websockets
            self._ws = await websockets.connect(
                _ws_url(), ping_interval=None, max_size=None)
        except Exception as e:
            logger.warning(f"KIS WS 연결 실패: {e}")
            self._ws = None
            return False
        self._task = asyncio.ensure_future(self._recv_loop())
        logger.info("KIS WS 연결됨")

        # 재연결인 경우: 수신 루프 death로 _ws만 None이 됐고 구독자(_subscribers)는
        # 남아 있다 → 새 연결에 기존 구독 종목을 전부 재등록(없으면 틱이 안 옴).
        for ticker in list(self._subscribers):
            try:
                await self._ws.send(self._sub_msg(ticker, True))
            except Exception as e:
                logger.warning(f"KIS WS 재구독 실패 ({ticker}): {e}")
        return True

    async def _disconnect(self):
        if self._task:
            self._task.cancel()
            self._task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._approval = None
        logger.info("KIS WS 연결 종료")

    def _sub_msg(self, ticker: str, register: bool) -> str:
        """H0STCNT0 구독/해제 메시지. tr_type 1=등록, 2=해제."""
        return json.dumps({
            "header": {
                "approval_key": self._approval,
                "custtype": "P",
                "tr_type": "1" if register else "2",
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": "H0STCNT0", "tr_key": ticker}},
        })

    async def _recv_loop(self):
        """WS 수신 → H0STCNT0 파싱 → 최신 틱 보관 + 구독자 broadcast. PINGPONG 응답."""
        try:
            while self._ws is not None:
                data = await self._ws.recv()
                if not data:
                    continue
                if data[0] in ("0", "1"):
                    # 실시간 데이터: '0|H0STCNT0|<count>|<fields^...>'
                    parts = data.split("|", 3)
                    if len(parts) < 4 or parts[1] != "H0STCNT0":
                        continue
                    try:
                        count = int(parts[2])
                    except ValueError:
                        count = 1
                    fields_all = parts[3].split("^")
                    for n in range(count):
                        rec = fields_all[n * _CNT0_FIELDS:(n + 1) * _CNT0_FIELDS]
                        tick = parse_cnt0_record(rec)
                        if tick:
                            self._latest[tick["ticker"]] = tick
                            self._broadcast(tick["ticker"], tick)
                else:
                    # JSON: 구독 응답 or PINGPONG
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    if obj.get("header", {}).get("tr_id") == "PINGPONG":
                        try:
                            await self._ws.pong(data)
                        except Exception:
                            pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"KIS WS 수신 루프 종료: {e}")
            # 연결이 끊김 — 정리 (재구독 시 _connect가 다시 연결)
            self._ws = None

    def _broadcast(self, ticker: str, tick: dict):
        for q in list(self._subscribers.get(ticker, ())):
            try:
                q.put_nowait(tick)
            except asyncio.QueueFull:
                pass  # 느린 구독자는 드롭(최신성 우선)

    async def subscribe(self, ticker: str) -> Optional[asyncio.Queue]:
        """종목 구독 → 틱 수신용 Queue 반환. 비활성/연결 실패 시 None."""
        async with self._lock:
            if not await self._connect():
                return None
            first = ticker not in self._subscribers
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            self._subscribers.setdefault(ticker, set()).add(q)
            if first:
                try:
                    await self._ws.send(self._sub_msg(ticker, True))
                except Exception as e:
                    logger.warning(f"KIS WS 구독 전송 실패 ({ticker}): {e}")
            # 최신 틱이 이미 있으면 즉시 1건 제공(첫 화면 공백 방지)
            if ticker in self._latest:
                try:
                    q.put_nowait(self._latest[ticker])
                except asyncio.QueueFull:
                    pass
            return q

    async def unsubscribe(self, ticker: str, q: asyncio.Queue):
        """구독 해제. 해당 종목 구독자 0이면 KIS 구독 해제, 전체 0이면 연결 종료."""
        async with self._lock:
            subs = self._subscribers.get(ticker)
            if subs and q in subs:
                subs.discard(q)
            if subs is not None and not subs:
                self._subscribers.pop(ticker, None)
                self._latest.pop(ticker, None)
                if self._ws is not None:
                    try:
                        await self._ws.send(self._sub_msg(ticker, False))
                    except Exception:
                        pass
            if not self._subscribers:
                await self._disconnect()


# 프로세스 싱글턴
_manager: Optional[KisWsManager] = None


def get_manager() -> KisWsManager:
    global _manager
    if _manager is None:
        _manager = KisWsManager()
    return _manager
