import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Loading, ErrorText, Notice, EmptyState } from "./Feedback";

describe("Feedback 컴포넌트", () => {
  it("Loading: 문구 + role=status", () => {
    render(<Loading text="분석 중…" />);
    const el = screen.getByRole("status");
    expect(el).toHaveTextContent("분석 중…");
    expect(el).toHaveAttribute("aria-busy", "true");
  });

  it("ErrorText: role=alert + 메시지", () => {
    render(<ErrorText message="데이터를 찾을 수 없어요." />);
    const el = screen.getByRole("alert");
    expect(el).toHaveTextContent("데이터를 찾을 수 없어요.");
  });

  it("Notice ok: role=status + 초록색", () => {
    render(<Notice message="저장했어요" kind="ok" />);
    const el = screen.getByRole("status");
    expect(el).toHaveTextContent("저장했어요");
    expect(el.className).toContain("text-green-600");
  });

  it("Notice err: role=alert + 빨간색", () => {
    render(<Notice message="실패" kind="err" />);
    const el = screen.getByRole("alert");
    expect(el.className).toContain("text-red-600");
  });

  it("EmptyState: 아이콘 + 메시지 + 선택 action", () => {
    render(<EmptyState icon="📦" message="비어 있어요" action={<button>추가</button>} />);
    expect(screen.getByText("비어 있어요")).toBeInTheDocument();
    expect(screen.getByText("📦")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "추가" })).toBeInTheDocument();
  });
});
