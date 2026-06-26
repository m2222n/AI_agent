import { describe, it, expect } from "vitest";
import { questionTypeLabel, toolLabel } from "./labels";

describe("questionTypeLabel", () => {
  it("매핑된 유형은 한국어 라벨", () => {
    expect(questionTypeLabel("compare")).toContain("비교");
    expect(questionTypeLabel("technical")).toContain("기술");
  });
  it("undefined면 빈 문자열", () => {
    expect(questionTypeLabel(undefined)).toBe("");
  });
  it("미등록 유형은 원문 그대로 fallback", () => {
    // @ts-expect-error 의도적으로 미정의 유형 전달
    expect(questionTypeLabel("unknown_x")).toBe("unknown_x");
  });
});

describe("toolLabel", () => {
  it("등록 도구는 한국어 라벨", () => {
    expect(toolLabel("get_realtime_price")).toContain("실시간");
    expect(toolLabel("predict_price_outlook")).toContain("전망");
  });
  it("미등록 도구는 🔍 + 원문", () => {
    expect(toolLabel("brand_new_tool")).toBe("🔍 brand_new_tool");
  });
});
