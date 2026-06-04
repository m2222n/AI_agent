"""UI 모듈의 Streamlit 무관 순수 로직 테스트

st.* 모킹 없이 단위 테스트 가능한 함수만 대상:
- tabs._build_sector_stats: 업종별 집계 통계 (시총 가중 등락률, PER 중앙값)
- tabs._horizon_label: 기간 코드 → 한글 레이블
- chat._get_followup_suggestions: 도구/질문유형 기반 후속질문 생성
"""
from src.ui.tabs import _build_sector_stats, _horizon_label
from src.ui.chat import _get_followup_suggestions


# --- _horizon_label ---

def test_horizon_label_known():
    assert _horizon_label("1m") == "1개월"
    assert _horizon_label("1y") == "1년"
    assert _horizon_label("1w") == "1주"


def test_horizon_label_unknown_passthrough():
    """매핑에 없는 코드는 그대로 반환"""
    assert _horizon_label("99y") == "99y"


# --- _build_sector_stats ---

def test_sector_stats_market_cap_weighted_change():
    """시총 가중 등락률: (2.0*100 + -1.0*300)/400 = -0.25"""
    idx = {"반도체": [
        {"market_cap": 100, "change_pct": 2.0, "per": 10},
        {"market_cap": 300, "change_pct": -1.0, "per": 20},
    ]}
    stats = _build_sector_stats(idx)
    assert len(stats) == 1
    s = stats[0]
    assert s["sector"] == "반도체"
    assert s["count"] == 2
    assert s["market_cap"] == 400
    assert s["change_pct"] == -0.25
    assert s["up_count"] == 1
    assert s["down_count"] == 1


def test_sector_stats_sorted_by_market_cap_desc():
    """시총 큰 업종이 먼저"""
    idx = {
        "소형": [{"market_cap": 10, "change_pct": 0, "per": 5}],
        "대형": [{"market_cap": 1000, "change_pct": 0, "per": 5}],
    }
    stats = _build_sector_stats(idx)
    assert [s["sector"] for s in stats] == ["대형", "소형"]


def test_sector_stats_skips_empty_sector():
    """종목 없는 업종은 제외"""
    idx = {"빈업종": [], "반도체": [{"market_cap": 100, "change_pct": 1.0, "per": 10}]}
    stats = _build_sector_stats(idx)
    assert [s["sector"] for s in stats] == ["반도체"]


def test_sector_stats_zero_market_cap_no_div_error():
    """시총 합계 0이면 가중 등락률 0 (ZeroDivision 방어)"""
    idx = {"무시총": [{"market_cap": 0, "change_pct": 5.0, "per": 0}]}
    stats = _build_sector_stats(idx)
    assert stats[0]["change_pct"] == 0.0


def test_sector_stats_median_per_excludes_zero_and_none():
    """PER 중앙값은 0/None 제외 후 계산"""
    idx = {"섹터": [
        {"market_cap": 100, "change_pct": 0, "per": 0},      # 제외
        {"market_cap": 100, "change_pct": 0, "per": None},   # 제외
        {"market_cap": 100, "change_pct": 0, "per": 10},
        {"market_cap": 100, "change_pct": 0, "per": 30},
        {"market_cap": 100, "change_pct": 0, "per": 20},
    ]}
    stats = _build_sector_stats(idx)
    # 유효 PER [10,20,30] → sorted[3//2]=sorted[1]=20
    assert stats[0]["median_per"] == 20


def test_sector_stats_all_per_zero_median_zero():
    """유효 PER 없으면 median_per=0"""
    idx = {"섹터": [{"market_cap": 100, "change_pct": 0, "per": 0}]}
    stats = _build_sector_stats(idx)
    assert stats[0]["median_per"] == 0


def test_sector_stats_missing_keys_use_defaults():
    """market_cap/change_pct/per 키 누락 시 기본값(0)으로 안전 처리"""
    idx = {"섹터": [{}, {"market_cap": 50}]}
    stats = _build_sector_stats(idx)
    s = stats[0]
    assert s["count"] == 2
    assert s["market_cap"] == 50
    assert s["change_pct"] == 0.0
    assert s["up_count"] == 0 and s["down_count"] == 0


def test_sector_stats_empty_input():
    assert _build_sector_stats({}) == []


# --- _get_followup_suggestions ---

def test_followup_search_stock_with_target():
    """search_stock + 종목 → 기술적분석/전망 제안"""
    out = _get_followup_suggestions("삼성전자 어때?", ["search_stock"], "simple")
    assert "삼성전자 기술적 분석해줘" in out
    assert "삼성전자 앞으로 전망은?" in out


def test_followup_etf_target():
    """주식명 없고 ETF명만 있으면 ETF를 target으로"""
    out = _get_followup_suggestions("KODEX 200 알려줘", ["search_etf"], "simple")
    assert any("KODEX 200" in s for s in out)


def test_followup_stock_priority_over_etf():
    """주식명과 ETF명 모두 있으면 주식명 우선"""
    out = _get_followup_suggestions(
        "삼성전자랑 KODEX 200 비교", ["search_stock"], "simple"
    )
    assert all("삼성전자" in s for s in out)


def test_followup_technical_suggests_financials():
    """기술적 분석 도구 → 재무제표/실적 제안"""
    out = _get_followup_suggestions("삼성전자 기술적분석", ["get_technical_indicators"], "technical")
    assert "삼성전자 재무제표 보여줘" in out
    assert "삼성전자 최근 실적은 어때?" in out


def test_followup_financial_suggests_technical():
    out = _get_followup_suggestions("삼성전자 재무제표", ["get_financial_statements"], "simple")
    assert "삼성전자 기술적 분석해줘" in out


def test_followup_no_known_stock_empty():
    """알려진 종목명 없으면 빈 리스트"""
    out = _get_followup_suggestions("ETF가 뭐야?", ["search_etf"], "general")
    assert out == []


def test_followup_max_three():
    """최대 3개로 제한"""
    out = _get_followup_suggestions("삼성전자 어때?", ["search_stock"], "simple")
    assert len(out) <= 3


def test_followup_no_duplicate_technical():
    """simple+search_stock 시 기술적분석 제안이 중복되지 않음"""
    out = _get_followup_suggestions("삼성전자 어때?", ["search_stock"], "simple")
    assert out.count("삼성전자 기술적 분석해줘") == 1


def test_followup_fallback_when_no_tool_match():
    """매칭 도구 없어도 종목 있으면 기술적분석 fallback"""
    out = _get_followup_suggestions("삼성전자", ["unknown_tool"], "simple")
    assert "삼성전자 기술적 분석해줘" in out
