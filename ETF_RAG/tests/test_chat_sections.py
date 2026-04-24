"""tests/test_chat_sections.py — 답변 섹션 분리 테스트."""

import pytest

from src.ui.chat import split_into_sections, _MIN_LEN_FOR_SECTIONS, _MIN_SECTIONS_FOR_EXPANDER


class TestSplitIntoSections:
    """split_into_sections() 단위 테스트."""

    def test_no_headers(self):
        """헤더 없는 텍스트는 단일 섹션 반환."""
        text = "간단한 답변입니다.\n추가 설명입니다."
        result = split_into_sections(text)
        assert len(result) == 1
        assert result[0]["title"] is None
        assert "간단한 답변" in result[0]["body"]

    def test_single_h2_header(self):
        """## 헤더 1개 — 도입부 + 1섹션."""
        text = "도입부입니다.\n\n## 상세 분석\n분석 내용입니다."
        result = split_into_sections(text)
        assert len(result) == 2
        assert result[0]["title"] is None
        assert "도입부" in result[0]["body"]
        assert result[1]["title"] == "상세 분석"
        assert "분석 내용" in result[1]["body"]

    def test_multiple_h2_headers(self):
        """## 헤더 여러 개 — 섹션별 분리."""
        text = (
            "## 종합 판단\n강세입니다.\n\n"
            "## 기술적 분석\nRSI 65\n\n"
            "## 리스크 요인\n변동성 높음"
        )
        result = split_into_sections(text)
        assert len(result) == 3
        assert result[0]["title"] == "종합 판단"
        assert result[1]["title"] == "기술적 분석"
        assert result[2]["title"] == "리스크 요인"

    def test_h3_headers(self):
        """### 헤더도 분리됨."""
        text = "### 개요\n내용1\n\n### 상세\n내용2"
        result = split_into_sections(text)
        assert len(result) == 2
        assert result[0]["title"] == "개요"
        assert result[1]["title"] == "상세"

    def test_mixed_h2_h3(self):
        """## 와 ### 혼합."""
        text = "서론\n\n## 대분류\n내용1\n\n### 소분류\n내용2\n\n## 결론\n내용3"
        result = split_into_sections(text)
        assert len(result) == 4
        assert result[0]["title"] is None  # 서론
        assert result[1]["title"] == "대분류"
        assert result[2]["title"] == "소분류"
        assert result[3]["title"] == "결론"

    def test_h1_not_split(self):
        """# (h1) 헤더는 분리하지 않음."""
        text = "# 제목\n내용입니다."
        result = split_into_sections(text)
        assert len(result) == 1

    def test_no_intro_when_starts_with_header(self):
        """헤더로 시작하면 도입부(title=None) 없음."""
        text = "## 첫 번째\n내용1\n\n## 두 번째\n내용2"
        result = split_into_sections(text)
        assert len(result) == 2
        assert result[0]["title"] == "첫 번째"

    def test_empty_text(self):
        """빈 텍스트."""
        result = split_into_sections("")
        assert len(result) == 1
        assert result[0]["body"] == ""

    def test_header_with_emoji(self):
        """이모지 포함 헤더."""
        text = "## 📊 기술적 분석\nRSI 65\n\n## ⚠️ 리스크\n변동성"
        result = split_into_sections(text)
        assert len(result) == 2
        assert "기술적 분석" in result[0]["title"]
        assert "리스크" in result[1]["title"]

    def test_header_with_bold(self):
        """볼드 포함 헤더."""
        text = "## **종합 판단**\n내용\n\n## **리스크**\n내용2"
        result = split_into_sections(text)
        assert len(result) == 2

    def test_preserves_body_content(self):
        """본문 내용(테이블, 리스트 등)이 보존됨."""
        text = (
            "## 비교표\n"
            "| 항목 | A | B |\n"
            "|------|---|---|\n"
            "| PER | 10 | 20 |\n\n"
            "## 결론\n- 항목1\n- 항목2"
        )
        result = split_into_sections(text)
        assert "| PER | 10 | 20 |" in result[0]["body"]
        assert "- 항목1" in result[1]["body"]

    def test_real_world_predict_answer(self):
        """실제 예측 답변 형태 분리."""
        text = (
            "삼성전자의 향후 전망을 분석해 보겠습니다.\n\n"
            "## 종합 판단\n기술적·펀더멘털 종합 점수 65점, 중립~약강세\n\n"
            "## 기술적 분석 요약\nRSI 52, MACD 매수 신호\n\n"
            "## 재무제표 요약\n매출 74조원(+15.2%)\n\n"
            "## 리스크 요인\n반도체 사이클 둔화\n\n"
            "📌 위 내용은 참고 정보입니다."
        )
        result = split_into_sections(text)
        assert len(result) == 5  # 도입부 + 4개 헤더 섹션
        assert result[0]["title"] is None  # 도입부
        assert result[1]["title"] == "종합 판단"
        # 마지막 면책 문구는 리스크 섹션에 포함
        assert "참고 정보" in result[4]["body"]


class TestConstants:
    """설정 상수 검증."""

    def test_min_len_reasonable(self):
        assert 200 <= _MIN_LEN_FOR_SECTIONS <= 1000

    def test_min_sections_reasonable(self):
        assert 2 <= _MIN_SECTIONS_FOR_EXPANDER <= 5
