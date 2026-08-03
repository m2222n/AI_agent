// 콜드스타트 재시도 "실제" 검증 — fetchEventSource를 모킹하지 않고,
// 로컬 가짜 백엔드(첫 2회 502 → 이후 정상 SSE)에 실제 HTTP로 붙어 확인한다.
// mock 테스트는 인자/실동작 누락을 못 잡으므로(과거 교훈) 이 경로를 함께 둔다.
//
// 가짜 백엔드가 안 떠 있으면 skip (CI에서 실패하지 않도록).
import { describe, expect, it, beforeAll } from "vitest";

const BASE = "http://127.0.0.1:8099";
let alive = false;

beforeAll(async () => {
  // 이 환경엔 localStorage 스텁이 없어 authHeader()가 터짐 → 최소 구현 주입
  if (typeof localStorage === "undefined" || !localStorage.getItem) {
    const store = new Map<string, string>();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: {
        getItem: (k: string) => store.get(k) ?? null,
        setItem: (k: string, v: string) => void store.set(k, v),
        removeItem: (k: string) => void store.delete(k),
      },
    });
  }
  try {
    const r = await fetch(`${BASE}/health`);
    alive = r.ok;
  } catch {
    alive = false;
  }
});

describe("streamChat 콜드스타트 — 실제 HTTP", () => {
  it("502 두 번 뒤 자동 재시도로 답변을 받아낸다", async () => {
    if (!alive) {
      console.warn("가짜 백엔드(8099) 없음 → skip");
      return;
    }
    process.env.NEXT_PUBLIC_API_BASE = BASE;
    const { streamChat } = await import("./api");

    const statuses: string[] = [];
    const errors: string[] = [];
    let answer = "";

    await new Promise<void>((resolve) => {
      streamChat("삼성전자 기술적 분석해줘", [], {
        onStatus: (m) => statuses.push(m),
        onError: (m) => {
          errors.push(m);
          resolve();
        },
        onToken: (t) => {
          answer = t;
        },
        onDone: () => resolve(),
      });
      setTimeout(resolve, 30000); // 안전망
    });

    // 실제로 재시도해서 답변까지 도달해야 한다
    expect(errors).toEqual([]);
    expect(statuses.length).toBeGreaterThanOrEqual(1);
    expect(statuses[0]).toContain("서버를 깨우고 있어요");
    expect(answer).toContain("재시도 성공");
  }, 40000);
});
