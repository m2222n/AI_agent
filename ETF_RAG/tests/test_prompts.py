from src.llm.prompts import build_system_prompt


class TestBuildSystemPrompt:
    def test_contains_role(self):
        prompt = build_system_prompt("simple")
        assert "투자 전문 어드바이저" in prompt

    def test_contains_constraints(self):
        prompt = build_system_prompt("simple")
        assert "[ETF-XXX]" in prompt

    def test_compare_has_cot(self):
        prompt = build_system_prompt("compare")
        assert "단계별" in prompt

    def test_recommend_has_few_shot(self):
        prompt = build_system_prompt("recommend")
        assert "KODEX 고배당" in prompt

    def test_risk_has_risk_grade(self):
        prompt = build_system_prompt("risk")
        assert "위험등급" in prompt

    def test_general_fallback(self):
        prompt = build_system_prompt("unknown_type")
        assert "핵심 개념" in prompt

    def test_all_types_return_nonempty(self):
        for q_type in ["simple", "compare", "recommend", "risk", "general"]:
            prompt = build_system_prompt(q_type)
            assert len(prompt) > 100
