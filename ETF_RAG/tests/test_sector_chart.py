"""섹터 분석 차트 + 탭 헬퍼 테스트."""
import base64
import pytest


# ── chart_generator 테스트 ──


def test_generate_sector_overview_chart_basic():
    """업종 개요 차트 정상 생성."""
    from src.data.chart_generator import generate_sector_overview_chart

    stats = [
        {"sector": "전기·전자", "change_pct": 1.5, "market_cap": 500_000_000_000_000, "count": 50},
        {"sector": "반도체", "change_pct": -0.8, "market_cap": 300_000_000_000_000, "count": 30},
        {"sector": "화학", "change_pct": 0.3, "market_cap": 100_000_000_000_000, "count": 40},
    ]
    result = generate_sector_overview_chart(stats)
    assert result is not None
    # base64 디코딩 가능 확인
    decoded = base64.b64decode(result)
    assert decoded[:4] == b"\x89PNG"


def test_generate_sector_overview_chart_too_few():
    """2개 미만이면 None."""
    from src.data.chart_generator import generate_sector_overview_chart

    assert generate_sector_overview_chart([]) is None
    assert generate_sector_overview_chart([{"sector": "A", "change_pct": 0, "market_cap": 100, "count": 1}]) is None


def test_generate_sector_detail_chart_basic():
    """업종 상세 차트 정상 생성."""
    from src.data.chart_generator import generate_sector_detail_chart

    stocks = [
        {"name": "삼성전자", "ticker": "005930", "change_pct": 2.0, "market_cap": 400_000_000_000_000},
        {"name": "SK하이닉스", "ticker": "000660", "change_pct": -1.5, "market_cap": 100_000_000_000_000},
        {"name": "LG전자", "ticker": "066570", "change_pct": 0.5, "market_cap": 15_000_000_000_000},
    ]
    result = generate_sector_detail_chart("전기·전자", stocks)
    assert result is not None
    decoded = base64.b64decode(result)
    assert decoded[:4] == b"\x89PNG"


def test_generate_sector_detail_chart_too_few():
    """2개 미만이면 None."""
    from src.data.chart_generator import generate_sector_detail_chart

    assert generate_sector_detail_chart("A", []) is None
    assert generate_sector_detail_chart("A", [{"name": "X", "change_pct": 0}]) is None


# ── tabs.py 헬퍼 테스트 ──


def test_build_sector_stats():
    """업종별 통계 집계 확인."""
    from src.ui.tabs import _build_sector_stats

    sector_index = {
        "전기·전자": [
            {"name": "A", "market_cap": 100, "change_pct": 2.0, "per": 10, "pbr": 1.0},
            {"name": "B", "market_cap": 200, "change_pct": -1.0, "per": 20, "pbr": 0.8},
        ],
        "화학": [
            {"name": "C", "market_cap": 50, "change_pct": 0.5, "per": 15, "pbr": 1.2},
        ],
    }
    stats = _build_sector_stats(sector_index)
    assert len(stats) == 2
    # 시총 내림차순
    assert stats[0]["sector"] == "전기·전자"
    assert stats[0]["count"] == 2
    assert stats[0]["up_count"] == 1
    assert stats[0]["down_count"] == 1
    # 시총 가중 등락률: (2*100 + (-1)*200) / 300 = 0/300 = 0
    assert stats[0]["change_pct"] == 0.0


def test_build_sector_stats_empty():
    """빈 인덱스."""
    from src.ui.tabs import _build_sector_stats

    assert _build_sector_stats({}) == []


def test_get_sector_index_returns_dict():
    """get_sector_index는 dict 반환."""
    from src.llm.tools import get_sector_index

    result = get_sector_index()
    assert isinstance(result, dict)
