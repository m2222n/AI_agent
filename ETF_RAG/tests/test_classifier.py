from src.llm.classifier import classify_question_type


class TestClassifyQuestionType:
    def test_simple_with_etf_name(self):
        assert classify_question_type("KODEX 200 ETF에 대해 알려줘") == "simple"

    def test_simple_with_ticker(self):
        assert classify_question_type("TIGER 미국S&P500의 수수료는 얼마야?") == "simple"

    def test_compare_with_keyword(self):
        assert classify_question_type("KODEX 200과 TIGER 미국S&P500 비교해줘") == "compare"

    def test_compare_with_vs(self):
        assert classify_question_type("국내 주식형 vs 해외 주식형 ETF") == "compare"

    def test_recommend(self):
        assert classify_question_type("배당 수익률 높은 ETF 추천해줘") == "recommend"

    def test_recommend_which(self):
        assert classify_question_type("안정적인 투자를 원하는데 어떤 ETF가 좋을까?") == "recommend"

    def test_risk(self):
        assert classify_question_type("KODEX 2차전지산업의 위험도는?") == "risk"

    def test_risk_caution(self):
        assert classify_question_type("인버스 ETF 투자할 때 주의사항이 뭐야?") == "risk"

    def test_general(self):
        assert classify_question_type("ETF란 무엇인가요?") == "general"

    def test_general_market(self):
        assert classify_question_type("내일 주식시장 어떻게 될까?") == "general"
