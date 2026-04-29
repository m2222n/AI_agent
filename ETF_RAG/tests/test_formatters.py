"""src/utils/formatters.py 단위 테스트"""

import pytest

from src.utils.formatters import (
    format_market_cap,
    format_large_number,
    format_change,
    format_percentage,
)


# ── format_market_cap ────────────────────────────

class TestFormatMarketCap:
    def test_trillion_with_suffix(self):
        assert format_market_cap(5_000_000_000_000) == "5.0조원"

    def test_trillion_without_suffix(self):
        assert format_market_cap(5_000_000_000_000, suffix=False) == "5.0조"

    def test_billion_with_suffix(self):
        assert format_market_cap(300_000_000_000) == "3,000억원"

    def test_billion_without_suffix(self):
        assert format_market_cap(300_000_000_000, suffix=False) == "3,000억"

    def test_small_with_suffix(self):
        assert format_market_cap(50_000_000) == "50,000,000원"

    def test_small_without_suffix(self):
        assert format_market_cap(50_000_000, suffix=False) == "50,000,000"

    def test_exact_boundary_trillion(self):
        assert format_market_cap(1_000_000_000_000) == "1.0조원"

    def test_exact_boundary_billion(self):
        assert format_market_cap(100_000_000) == "1억원"


# ── format_large_number ──────────────────────────

class TestFormatLargeNumber:
    def test_trillion(self):
        assert format_large_number(2_500_000_000_000) == "2.5조"

    def test_billion(self):
        assert format_large_number(500_000_000) == "5억"

    def test_ten_thousand(self):
        assert format_large_number(30_000) == "3만"

    def test_small(self):
        assert format_large_number(9_999) == "9,999"

    def test_zero(self):
        assert format_large_number(0) == "0"


# ── format_change ────────────────────────────────

class TestFormatChange:
    def test_positive(self):
        assert format_change(2.5) == "🔴 +2.50%"

    def test_negative(self):
        assert format_change(-1.3) == "🔵 -1.30%"

    def test_zero(self):
        assert format_change(0.0) == "⚪ 0.00%"


# ── format_percentage ────────────────────────────

class TestFormatPercentage:
    def test_positive(self):
        assert format_percentage(3.14) == "+3.14%"

    def test_negative(self):
        assert format_percentage(-2.0) == "-2.00%"

    def test_none_default(self):
        assert format_percentage(None) == "-"

    def test_none_custom(self):
        assert format_percentage(None, default="N/A") == "N/A"

    def test_zero(self):
        assert format_percentage(0.0) == "+0.00%"
