"""한국투자증권(KIS) REST 클라이언트 테스트 — 토큰 캐싱 / 현재가 파싱 / fallback"""

import time
from unittest.mock import patch, MagicMock

import pytest

from src.data import kis_client


@pytest.fixture(autouse=True)
def _isolate_kis(tmp_path):
    """매 테스트마다 인메모리 캐시 초기화 + 토큰 디스크 캐시를 임시 경로로 격리."""
    kis_client.clear_cache()
    kis_client._price_cache.clear()
    kis_client._orderbook_cache.clear()
    kis_client._token_state.clear()
    with patch.object(kis_client, "_TOKEN_CACHE_PATH",
                      tmp_path / "kis_token.json"):
        yield
    kis_client.clear_cache()


ENABLED_CFG = {
    "enabled": True,
    "app_key": "appkey-x",
    "app_secret": "appsecret-y",
    "env": "real",
    "base_url": "https://openapi.koreainvestment.com:9443",
    "timeout": 5,
    "token_margin": 600,
}
DISABLED_CFG = {**ENABLED_CFG, "enabled": False, "app_key": "", "app_secret": ""}


def _mock_resp(json_data, status=200):
    m = MagicMock()
    m.json.return_value = json_data
    m.raise_for_status.return_value = None
    if status >= 400:
        m.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return m


# ── 활성화 여부 ───────────────────────────────────────────

def test_is_enabled_true():
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG):
        assert kis_client.is_enabled() is True


def test_is_enabled_false_no_keys():
    with patch.object(kis_client, "_kis_config", return_value=DISABLED_CFG):
        assert kis_client.is_enabled() is False


# ── 토큰 발급/캐싱 ────────────────────────────────────────

def test_get_token_disabled_returns_none():
    with patch.object(kis_client, "_kis_config", return_value=DISABLED_CFG):
        assert kis_client._get_access_token() is None


def test_get_token_issues_new():
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=_mock_resp(
             {"access_token": "tok-123", "expires_in": 86400})) as post:
        token = kis_client._get_access_token()
    assert token == "tok-123"
    post.assert_called_once()
    # tokenP 엔드포인트 호출 확인
    assert "/oauth2/tokenP" in post.call_args[0][0]


def test_get_token_inmemory_cache_hit():
    """두 번째 호출은 requests.post 미호출 (인메모리 캐시)."""
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=_mock_resp(
             {"access_token": "tok-abc", "expires_in": 86400})) as post:
        t1 = kis_client._get_access_token()
        t2 = kis_client._get_access_token()
    assert t1 == t2 == "tok-abc"
    assert post.call_count == 1


def test_get_token_expired_reissues():
    """만료 임박(margin 안) 토큰은 재발급."""
    kis_client._token_state.update({
        "access_token": "old", "expires_at": time.time() + 100,  # margin 600 안
    })
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=_mock_resp(
             {"access_token": "new", "expires_in": 86400})):
        token = kis_client._get_access_token()
    assert token == "new"


def test_get_token_disk_cache_survives_memory_clear():
    """디스크 캐시에서 복구 — 인메모리 비어도 post 미호출."""
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG):
        with patch("requests.post", return_value=_mock_resp(
                {"access_token": "disk-tok", "expires_in": 86400})) as post:
            kis_client._get_access_token()
        assert post.call_count == 1
        # 인메모리만 비우기 (디스크 캐시는 유지)
        kis_client._token_state.clear()
        with patch("requests.post") as post2:
            token = kis_client._get_access_token()
    assert token == "disk-tok"
    post2.assert_not_called()


def test_get_token_post_error_returns_none():
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", side_effect=Exception("network down")):
        assert kis_client._get_access_token() is None


def test_get_token_missing_field_returns_none():
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=_mock_resp({"msg": "no token"})):
        assert kis_client._get_access_token() is None


# ── 현재가 파싱 ───────────────────────────────────────────

def test_parse_price_output_rising():
    out = {
        "stck_prpr": "81200", "stck_sdpr": "80800",
        "prdy_vrss": "400", "prdy_ctrt": "0.50",
        "prdy_vrss_sign": "2", "acml_vol": "5000000",
    }
    p = kis_client._parse_price_output(out)
    assert p["price"] == 81200
    assert p["prev_close"] == 80800
    assert p["change"] == 400
    assert p["change_pct"] == 0.50
    assert p["volume"] == 5000000
    assert p["source"] == "kis"


def test_parse_price_output_falling_sign():
    """하락(sign=5)이면 change/change_pct 음수로 보정."""
    out = {
        "stck_prpr": "79000", "stck_sdpr": "80000",
        "prdy_vrss": "1000", "prdy_ctrt": "1.25",
        "prdy_vrss_sign": "5", "acml_vol": "3000000",
    }
    p = kis_client._parse_price_output(out)
    assert p["price"] == 79000
    assert p["change"] == -1000
    assert p["change_pct"] == -1.25


def test_parse_price_output_zero_price_none():
    assert kis_client._parse_price_output({"stck_prpr": "0"}) is None


def test_parse_price_output_missing_price_none():
    assert kis_client._parse_price_output({}) is None


# ── 현재가 조회 (통합) ────────────────────────────────────

def test_get_current_price_disabled_none():
    with patch.object(kis_client, "_kis_config", return_value=DISABLED_CFG):
        assert kis_client.get_current_price("005930") is None


def test_get_current_price_success():
    token_resp = _mock_resp({"access_token": "tok", "expires_in": 86400})
    price_resp = _mock_resp({
        "rt_cd": "0",
        "output": {
            "stck_prpr": "70000", "stck_sdpr": "69000",
            "prdy_vrss": "1000", "prdy_ctrt": "1.45",
            "prdy_vrss_sign": "2", "acml_vol": "12000000",
        },
    })
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=token_resp), \
         patch("requests.get", return_value=price_resp) as get:
        p = kis_client.get_current_price("005930")
    assert p["price"] == 70000
    assert p["change_pct"] == 1.45
    assert p["source"] == "kis"
    # 올바른 tr_id / 파라미터 확인
    _, kw = get.call_args
    assert kw["headers"]["tr_id"] == "FHKST01010100"
    assert kw["params"]["fid_input_iscd"] == "005930"
    assert kw["params"]["fid_cond_mrkt_div_code"] == "J"


def test_get_current_price_cache_hit():
    """두 번째 호출은 requests.get 미호출."""
    token_resp = _mock_resp({"access_token": "tok", "expires_in": 86400})
    price_resp = _mock_resp({
        "rt_cd": "0",
        "output": {"stck_prpr": "70000", "stck_sdpr": "69000",
                   "prdy_vrss": "1000", "prdy_ctrt": "1.45",
                   "prdy_vrss_sign": "2", "acml_vol": "100"},
    })
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=token_resp), \
         patch("requests.get", return_value=price_resp) as get:
        kis_client.get_current_price("005930", cache_ttl=300)
        kis_client.get_current_price("005930", cache_ttl=300)
    assert get.call_count == 1


def test_get_current_price_error_rt_cd_none():
    """rt_cd != '0' → None."""
    token_resp = _mock_resp({"access_token": "tok", "expires_in": 86400})
    err_resp = _mock_resp({"rt_cd": "1", "msg_cd": "EGW", "msg1": "오류"})
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=token_resp), \
         patch("requests.get", return_value=err_resp):
        assert kis_client.get_current_price("005930") is None


def test_get_current_price_http_error_none():
    token_resp = _mock_resp({"access_token": "tok", "expires_in": 86400})
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=token_resp), \
         patch("requests.get", side_effect=Exception("timeout")):
        assert kis_client.get_current_price("005930") is None


def test_get_current_price_no_token_none():
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", side_effect=Exception("token fail")):
        assert kis_client.get_current_price("005930") is None


# ── 호가 10단계 파싱 ──────────────────────────────────────

def _make_orderbook_output():
    """askp1~10/bidp1~10 + 잔량 + 총잔량 샘플."""
    out = {}
    for i in range(1, 11):
        out[f"askp{i}"] = str(70000 + i * 100)        # 매도호가 (i↑ = 고가)
        out[f"askp_rsqn{i}"] = str(100 * i)           # 매도 잔량
        out[f"bidp{i}"] = str(69900 - i * 100)        # 매수호가 (i↑ = 저가)
        out[f"bidp_rsqn{i}"] = str(200 * i)           # 매수 잔량
    out["total_askp_rsqn"] = "5500"
    out["total_bidp_rsqn"] = "11000"
    return out


def test_parse_orderbook_output_full():
    p = kis_client._parse_orderbook_output(_make_orderbook_output())
    assert len(p["asks"]) == 10
    assert len(p["bids"]) == 10
    assert p["asks"][0] == {"price": 70100, "qty": 100}
    assert p["asks"][9] == {"price": 71000, "qty": 1000}
    assert p["bids"][0] == {"price": 69800, "qty": 200}
    assert p["total_ask_qty"] == 5500
    assert p["total_bid_qty"] == 11000
    assert p["source"] == "kis"


def test_parse_orderbook_output_all_zero_none():
    out = {f"askp{i}": "0" for i in range(1, 11)}
    out.update({f"bidp{i}": "0" for i in range(1, 11)})
    assert kis_client._parse_orderbook_output(out) is None


def test_parse_orderbook_output_malformed_qty_zero():
    """잔량 필드가 비거나 비정상이면 0으로."""
    out = _make_orderbook_output()
    out["askp_rsqn1"] = ""
    out["bidp_rsqn1"] = None
    p = kis_client._parse_orderbook_output(out)
    assert p["asks"][0]["qty"] == 0
    assert p["bids"][0]["qty"] == 0
    assert p["asks"][0]["price"] == 70100  # 가격은 유효


# ── 호가 조회 (통합) ──────────────────────────────────────

def test_get_orderbook_disabled_none():
    with patch.object(kis_client, "_kis_config", return_value=DISABLED_CFG):
        assert kis_client.get_orderbook("005930") is None


def test_get_orderbook_success():
    token_resp = _mock_resp({"access_token": "tok", "expires_in": 86400})
    ob_resp = _mock_resp({"rt_cd": "0", "output1": _make_orderbook_output()})
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=token_resp), \
         patch("requests.get", return_value=ob_resp) as get:
        p = kis_client.get_orderbook("005930")
    assert p["asks"][0]["price"] == 70100
    assert p["total_bid_qty"] == 11000
    _, kw = get.call_args
    assert kw["headers"]["tr_id"] == "FHKST01010200"
    assert kw["params"]["fid_input_iscd"] == "005930"
    assert "inquire-asking-price-exp-ccn" in get.call_args[0][0]


def test_get_orderbook_cache_hit():
    token_resp = _mock_resp({"access_token": "tok", "expires_in": 86400})
    ob_resp = _mock_resp({"rt_cd": "0", "output1": _make_orderbook_output()})
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=token_resp), \
         patch("requests.get", return_value=ob_resp) as get:
        kis_client.get_orderbook("005930", cache_ttl=5)
        kis_client.get_orderbook("005930", cache_ttl=5)
    assert get.call_count == 1


def test_get_orderbook_rt_cd_error_none():
    token_resp = _mock_resp({"access_token": "tok", "expires_in": 86400})
    err = _mock_resp({"rt_cd": "1", "msg_cd": "X", "msg1": "오류"})
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=token_resp), \
         patch("requests.get", return_value=err):
        assert kis_client.get_orderbook("005930") is None


def test_get_orderbook_http_error_none():
    token_resp = _mock_resp({"access_token": "tok", "expires_in": 86400})
    with patch.object(kis_client, "_kis_config", return_value=ENABLED_CFG), \
         patch("requests.post", return_value=token_resp), \
         patch("requests.get", side_effect=Exception("timeout")):
        assert kis_client.get_orderbook("005930") is None


# ── realtime.py 통합: KIS 우선 → yfinance fallback ────────

def test_realtime_prefers_kis():
    """KIS 활성 + 성공 시 source=kis 반환, yfinance 미호출."""
    from src.data import realtime
    realtime.clear_cache()
    kis_data = {"price": 70000, "prev_close": 69000, "change": 1000,
                "change_pct": 1.45, "volume": 100,
                "timestamp": "2026-06-12 10:00", "source": "kis"}
    with patch("src.data.realtime.is_market_open", return_value=True), \
         patch("src.data.kis_client.is_enabled", return_value=True), \
         patch("src.data.kis_client.get_current_price", return_value=kis_data), \
         patch("yfinance.Ticker") as yf:
        result = realtime.get_realtime_price("005930", "stock")
    assert result["source"] == "kis"
    assert result["price"] == 70000
    yf.assert_not_called()


def test_realtime_fallback_to_yfinance_when_kis_fails():
    """KIS 활성이나 조회 실패 → yfinance로 fallback."""
    from src.data import realtime
    realtime.clear_cache()
    mock_info = MagicMock()
    mock_info.last_price = 70500.0
    mock_info.previous_close = 70000.0
    mock_info.last_volume = 999
    mock_ticker = MagicMock()
    mock_ticker.fast_info = mock_info
    with patch("src.data.realtime.is_market_open", return_value=True), \
         patch("src.data.kis_client.is_enabled", return_value=True), \
         patch("src.data.kis_client.get_current_price", return_value=None), \
         patch("yfinance.Ticker", return_value=mock_ticker):
        result = realtime.get_realtime_price("069500", "etf")
    assert result["source"] == "yfinance"
    assert result["price"] == 70500


def test_realtime_uses_yfinance_when_kis_disabled():
    """KIS 비활성 → yfinance 경로."""
    from src.data import realtime
    realtime.clear_cache()
    mock_info = MagicMock()
    mock_info.last_price = 80500.0
    mock_info.previous_close = 80000.0
    mock_info.last_volume = 1500000
    mock_ticker = MagicMock()
    mock_ticker.fast_info = mock_info
    with patch("src.data.realtime.is_market_open", return_value=True), \
         patch("src.data.kis_client.is_enabled", return_value=False), \
         patch("yfinance.Ticker", return_value=mock_ticker):
        result = realtime.get_realtime_price("069500", "etf")
    assert result["source"] == "yfinance"
    assert result["price"] == 80500
