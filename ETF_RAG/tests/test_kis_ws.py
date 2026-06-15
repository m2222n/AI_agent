"""KIS WebSocket 실시간 체결 — 파서 + 온디맨드 구독 매니저 테스트."""

import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.data import kis_ws


# ── H0STCNT0 레코드 파서 ──────────────────────────────────

def _cnt0_fields(price="70000", sign="2", change="500", pct="0.72",
                 vol="12000000", t="103015", ticker="005930"):
    """46필드 레코드 생성 (필요 인덱스만 채움)."""
    f = [""] * kis_ws._CNT0_FIELDS
    f[0] = ticker
    f[1] = t
    f[2] = price
    f[3] = sign
    f[4] = change
    f[5] = pct
    f[13] = vol
    return f


def test_parse_cnt0_rising():
    p = kis_ws.parse_cnt0_record(_cnt0_fields())
    assert p["ticker"] == "005930"
    assert p["price"] == 70000
    assert p["change"] == 500
    assert p["change_pct"] == 0.72
    assert p["volume"] == 12000000
    assert p["timestamp"] == "10:30:15"
    assert p["source"] == "kis-ws"


def test_parse_cnt0_falling_sign():
    p = kis_ws.parse_cnt0_record(
        _cnt0_fields(sign="5", change="1000", pct="1.25"))
    assert p["change"] == -1000
    assert p["change_pct"] == -1.25


def test_parse_cnt0_zero_price_none():
    assert kis_ws.parse_cnt0_record(_cnt0_fields(price="0")) is None


def test_parse_cnt0_too_short_none():
    assert kis_ws.parse_cnt0_record(["005930", "1", "2"]) is None


# ── 매니저: 구독/해제/broadcast ───────────────────────────

ENABLED = {"enabled": True, "app_key": "k", "app_secret": "s",
           "env": "real", "base_url": "https://x", "timeout": 5}
DISABLED = {**ENABLED, "enabled": False}


@pytest.fixture
def mgr():
    m = kis_ws.KisWsManager()
    return m


async def _block_forever(*a, **k):
    """recv가 영원히 대기 → 수신 루프가 즉시 끝나지 않게."""
    await asyncio.sleep(3600)


def _fake_ws():
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.pong = AsyncMock()
    ws.recv = AsyncMock(side_effect=_block_forever)
    return ws


# pytest-asyncio 미사용 — 전용 이벤트 루프로 코루틴 구동.
# (asyncio.run은 Python 3.9에서 현재 루프를 None으로 남겨 다음 테스트 setup이
#  get_event_loop()에서 깨지므로, 직접 루프를 만들고 끝나면 새 루프로 교체한다.)
def _run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def test_subscribe_disabled_returns_none(mgr):
    async def body():
        with patch("config.KIS", DISABLED):
            return await mgr.subscribe("005930")
    assert _run(body()) is None


def test_subscribe_connect_fail_returns_none(mgr):
    async def body():
        with patch("config.KIS", ENABLED), \
             patch("src.data.kis_ws._get_approval_key", return_value=None):
            return await mgr.subscribe("005930")
    assert _run(body()) is None


def test_subscribe_sends_register_and_broadcast(mgr):
    fake = _fake_ws()

    async def body():
        with patch("config.KIS", ENABLED), \
             patch("src.data.kis_ws._get_approval_key", return_value="appr"), \
             patch("websockets.connect", AsyncMock(return_value=fake)):
            q = await mgr.subscribe("005930")
            assert q is not None
            sent = json.loads(fake.send.call_args[0][0])
            assert sent["body"]["input"]["tr_id"] == "H0STCNT0"
            assert sent["body"]["input"]["tr_key"] == "005930"
            assert sent["header"]["tr_type"] == "1"

            # broadcast → 구독자 큐에 도달
            mgr._broadcast("005930", {"ticker": "005930", "price": 71000})
            got = await asyncio.wait_for(q.get(), timeout=1)
            assert got["price"] == 71000
            await mgr._disconnect()

    _run(body())


def test_unsubscribe_last_disconnects(mgr):
    fake = _fake_ws()

    async def body():
        with patch("config.KIS", ENABLED), \
             patch("src.data.kis_ws._get_approval_key", return_value="appr"), \
             patch("websockets.connect", AsyncMock(return_value=fake)):
            q = await mgr.subscribe("005930")
            assert "005930" in mgr._subscribers
            await mgr.unsubscribe("005930", q)
        assert "005930" not in mgr._subscribers
        assert mgr._ws is None
        last = json.loads(fake.send.call_args[0][0])
        assert last["header"]["tr_type"] == "2"  # 해제 메시지

    _run(body())


def test_refcount_two_subscribers_one_register(mgr):
    fake = _fake_ws()

    async def body():
        with patch("config.KIS", ENABLED), \
             patch("src.data.kis_ws._get_approval_key", return_value="appr"), \
             patch("websockets.connect", AsyncMock(return_value=fake)):
            q1 = await mgr.subscribe("005930")
            q2 = await mgr.subscribe("005930")  # 같은 종목 2번째
            register_sends = [c for c in fake.send.call_args_list
                              if json.loads(c[0][0])["header"]["tr_type"] == "1"]
            assert len(register_sends) == 1   # refcount → 등록 1회
            assert len(mgr._subscribers["005930"]) == 2

            await mgr.unsubscribe("005930", q1)
            assert "005930" in mgr._subscribers  # q2 남음
            await mgr.unsubscribe("005930", q2)
            assert "005930" not in mgr._subscribers

    _run(body())
