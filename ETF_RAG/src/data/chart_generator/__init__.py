"""
차트 생성 패키지 — matplotlib base64 PNG

서브모듈:
- _style.py    : 디자인 상수, 폰트 설정, 유틸리티
- _series.py   : 시계열 계산 (MA, RSI, MACD, 볼린저)
- technical.py : 기술적 분석 / 비교 / 장중 차트
- financial.py : 재무제표 / 밸류에이션 / 포트폴리오 차트
- sector.py    : 섹터(업종) 개요 / 상세 차트
"""

from src.data.chart_generator.technical import (
    generate_technical_chart,
    generate_comparison_chart,
    generate_intraday_chart,
)
from src.data.chart_generator.financial import (
    generate_financial_chart,
    generate_valuation_chart,
    generate_portfolio_chart,
    generate_paper_trend_chart,
)
from src.data.chart_generator.sector import (
    generate_sector_overview_chart,
    generate_sector_detail_chart,
    generate_sector_trend_chart,
)

__all__ = [
    "generate_technical_chart",
    "generate_comparison_chart",
    "generate_intraday_chart",
    "generate_financial_chart",
    "generate_valuation_chart",
    "generate_portfolio_chart",
    "generate_paper_trend_chart",
    "generate_sector_overview_chart",
    "generate_sector_detail_chart",
    "generate_sector_trend_chart",
]
