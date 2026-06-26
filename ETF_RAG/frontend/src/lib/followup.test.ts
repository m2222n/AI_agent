import { describe, it, expect } from "vitest";
import { getFollowupSuggestions } from "./followup";

describe("getFollowupSuggestions", () => {
  it("검색 도구 + 종목명 → 기술분석·전망 제안", () => {
    const out = getFollowupSuggestions("삼성전자 정보 알려줘", ["search_stock"], "simple");
    expect(out).toContain("삼성전자 기술적 분석해줘");
    expect(out).toContain("삼성전자 앞으로 전망은?");
  });

  it("기술적 지표 도구 → 재무제표·실적 제안", () => {
    const out = getFollowupSuggestions("SK하이닉스 기술적 분석", ["get_technical_indicators"], "technical");
    expect(out).toContain("SK하이닉스 재무제표 보여줘");
    expect(out).toContain("SK하이닉스 최근 실적은 어때?");
  });

  it("전망 도구 → 기술분석 제안", () => {
    const out = getFollowupSuggestions("현대차 전망", ["predict_price_outlook"], "forecast");
    expect(out).toContain("현대차 기술적 분석해줘");
  });

  it("최대 3개로 제한", () => {
    const out = getFollowupSuggestions("삼성전자 정보", ["search_stock"], "simple");
    expect(out.length).toBeLessThanOrEqual(3);
  });

  it("종목명 없으면 빈 배열(타겟 없음)", () => {
    const out = getFollowupSuggestions("시장 전체 어때?", ["search_etf"], "general");
    expect(out).toEqual([]);
  });

  it("중복 제안은 제거(simple 추가분이 기존과 안 겹침)", () => {
    const out = getFollowupSuggestions("삼성전자 정보", ["search_stock"], "simple");
    expect(new Set(out).size).toBe(out.length);
  });

  it("ETF명도 타겟으로 인식", () => {
    const out = getFollowupSuggestions("KODEX 200 알려줘", ["search_etf"], "simple");
    expect(out.some((s) => s.startsWith("KODEX 200"))).toBe(true);
  });
});
