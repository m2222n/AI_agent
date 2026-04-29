"""
OpenDart 재무제표 수집기 테스트 — dart_collector.py + database.py CRUD
"""

import sqlite3
from datetime import datetime

import pytest

from src.data.database import (
    init_db,
    upsert_corp_codes,
    get_corp_code,
    get_all_corp_codes,
    upsert_financial_data,
    get_financial_data,
    get_latest_financial_summary,
    get_db_stats,
)


@pytest.fixture
def db(tmp_path):
    """테스트용 인메모리 DB"""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    yield conn
    conn.close()


# ── corp_code CRUD ───────────────────────────────────────────

class TestCorpCodes:
    def test_upsert_corp_codes(self, db):
        codes = [
            {"corp_code": "00126380", "ticker": "005930", "corp_name": "삼성전자"},
            {"corp_code": "00164779", "ticker": "000660", "corp_name": "SK하이닉스"},
        ]
        count = upsert_corp_codes(db, codes)
        assert count == 2

    def test_get_corp_code(self, db):
        upsert_corp_codes(db, [
            {"corp_code": "00126380", "ticker": "005930", "corp_name": "삼성전자"},
        ])
        assert get_corp_code(db, "005930") == "00126380"
        assert get_corp_code(db, "999999") is None

    def test_get_all_corp_codes(self, db):
        upsert_corp_codes(db, [
            {"corp_code": "00126380", "ticker": "005930", "corp_name": "삼성전자"},
            {"corp_code": "00164779", "ticker": "000660", "corp_name": "SK하이닉스"},
        ])
        result = get_all_corp_codes(db)
        assert result == {"005930": "00126380", "000660": "00164779"}

    def test_upsert_empty(self, db):
        assert upsert_corp_codes(db, []) == 0

    def test_upsert_replaces(self, db):
        upsert_corp_codes(db, [
            {"corp_code": "00126380", "ticker": "005930", "corp_name": "삼성전자"},
        ])
        upsert_corp_codes(db, [
            {"corp_code": "00126380", "ticker": "005930", "corp_name": "삼성전자(수정)"},
        ])
        row = db.execute(
            "SELECT corp_name FROM dart_corp_codes WHERE corp_code = '00126380'"
        ).fetchone()
        assert row["corp_name"] == "삼성전자(수정)"


# ── financial_data CRUD ──────────────────────────────────────

class TestFinancialData:
    def _sample_rows(self):
        return [
            {
                "ticker": "005930",
                "fiscal_year": "2025",
                "fiscal_quarter": 4,
                "report_code": "11011",
                "revenue": 74_000_000_000_000,
                "operating_profit": 6_500_000_000_000,
                "net_income": 5_000_000_000_000,
                "operating_margin": 8.78,
                "net_margin": 6.76,
                "revenue_growth_yoy": 12.3,
                "op_growth_yoy": 25.1,
            },
            {
                "ticker": "005930",
                "fiscal_year": "2025",
                "fiscal_quarter": 3,
                "report_code": "11014",
                "revenue": 70_000_000_000_000,
                "operating_profit": 6_000_000_000_000,
                "net_income": 4_500_000_000_000,
                "operating_margin": 8.57,
                "net_margin": 6.43,
            },
        ]

    def test_upsert_financial_data(self, db):
        count = upsert_financial_data(db, self._sample_rows())
        assert count == 2

    def test_upsert_empty(self, db):
        assert upsert_financial_data(db, []) == 0

    def test_get_financial_data(self, db):
        upsert_financial_data(db, self._sample_rows())
        result = get_financial_data(db, "005930", quarters=8)
        assert len(result) == 2
        # 최신순 정렬
        assert result[0]["fiscal_quarter"] == 4
        assert result[1]["fiscal_quarter"] == 3

    def test_get_financial_data_limit(self, db):
        upsert_financial_data(db, self._sample_rows())
        result = get_financial_data(db, "005930", quarters=1)
        assert len(result) == 1
        assert result[0]["fiscal_quarter"] == 4

    def test_get_financial_data_no_data(self, db):
        result = get_financial_data(db, "999999")
        assert result == []

    def test_get_latest_financial_summary(self, db):
        upsert_financial_data(db, self._sample_rows())
        summary = get_latest_financial_summary(db, "005930")
        assert summary is not None
        assert summary["fiscal_year"] == "2025"
        assert summary["fiscal_quarter"] == 4
        assert summary["revenue"] == 74_000_000_000_000

    def test_get_latest_financial_summary_no_data(self, db):
        assert get_latest_financial_summary(db, "999999") is None

    def test_upsert_replaces(self, db):
        upsert_financial_data(db, [{
            "ticker": "005930", "fiscal_year": "2025", "fiscal_quarter": 4,
            "report_code": "11011", "revenue": 100,
        }])
        upsert_financial_data(db, [{
            "ticker": "005930", "fiscal_year": "2025", "fiscal_quarter": 4,
            "report_code": "11011", "revenue": 200,
        }])
        result = get_financial_data(db, "005930")
        assert len(result) == 1
        assert result[0]["revenue"] == 200

    def test_margin_calculation_values(self, db):
        """마진율 값이 올바르게 저장되는지"""
        upsert_financial_data(db, self._sample_rows())
        result = get_financial_data(db, "005930")
        q4 = result[0]
        assert q4["operating_margin"] == 8.78
        assert q4["net_margin"] == 6.76

    def test_yoy_growth_values(self, db):
        """YoY 성장률 값이 올바르게 저장되는지"""
        upsert_financial_data(db, self._sample_rows())
        result = get_financial_data(db, "005930")
        q4 = result[0]
        assert q4["revenue_growth_yoy"] == 12.3
        assert q4["op_growth_yoy"] == 25.1


# ── DB stats 테이블 포함 확인 ────────────────────────────────

class TestDbStatsIncludesNewTables:
    def test_stats_includes_new_tables(self, db):
        stats = get_db_stats(db)
        assert "dart_corp_codes" in stats
        assert "stock_financials" in stats


# ── dart_collector 함수 단위 테스트 ──────────────────────────

class TestDartCollectorHelpers:
    def test_extract_account_value(self):
        from src.data.dart_collector import _extract_account_value
        accounts = [
            {"acc_nm": "매출액", "thstrm_amount": "74,000,000,000,000"},
            {"acc_nm": "영업이익", "thstrm_amount": "6,500,000,000,000"},
        ]
        assert _extract_account_value(accounts, ["매출액"]) == 74_000_000_000_000
        assert _extract_account_value(accounts, ["영업이익"]) == 6_500_000_000_000
        assert _extract_account_value(accounts, ["당기순이익"]) is None

    def test_extract_account_value_dash(self):
        from src.data.dart_collector import _extract_account_value
        accounts = [{"acc_nm": "매출액", "thstrm_amount": "-"}]
        assert _extract_account_value(accounts, ["매출액"]) is None

    def test_extract_account_value_empty(self):
        from src.data.dart_collector import _extract_account_value
        assert _extract_account_value([], ["매출액"]) is None

    def test_get_latest_quarter(self):
        from src.data.dart_collector import _get_latest_quarter
        year, quarter = _get_latest_quarter()
        assert isinstance(year, str)
        assert quarter in [1, 2, 3, 4]

    def test_report_codes(self):
        from src.data.dart_collector import REPORT_CODES
        assert REPORT_CODES[1] == "11013"
        assert REPORT_CODES[2] == "11012"
        assert REPORT_CODES[3] == "11014"
        assert REPORT_CODES[4] == "11011"


# ── tools.py 도구 등록 확인 ──────────────────────────────────

class TestToolsRegistration:
    def test_all_tools_count(self):
        from src.llm.tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 14

    def test_financial_tool_in_list(self):
        from src.llm.tools import ALL_TOOLS
        tool_names = [t.name for t in ALL_TOOLS]
        assert "get_financial_statements" in tool_names
