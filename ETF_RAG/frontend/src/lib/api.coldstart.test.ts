// 콜드스타트(유휴 재시작) 시 /stream 자동 재시도 동작 검증.
// Railway 백엔드가 재시작 중이면 502 등을 내는데, 예전엔 즉시 "연결 오류"로 끝나
// 스토어 신규 사용자의 첫 질문이 실패로 보였다. 이제 상태 안내 + 자동 재시도.
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

/** fetchEventSource 호출을 가로채 onopen에 임의 응답을 먹인다. */
const calls: Array<{ url: string; opts: Record<string, unknown> }> = [];
let statuses: number[] = [];

vi.mock("@microsoft/fetch-event-source", () => ({
  fetchEventSource: async (url: string, opts: Record<string, unknown>) => {
    calls.push({ url, opts });
    const status = statuses.shift() ?? 200;
    const onopen = opts.onopen as (r: Response) => Promise<void>;
    const onerror = opts.onerror as (e: unknown) => void;
    try {
      await onopen({ ok: status === 200, status } as Response);
    } catch (e) {
      onerror(e); // 라이브러리와 동일하게 onerror로 넘김
      throw e;
    }
  },
}));

vi.mock("./auth", () => ({ authHeader: () => ({}) }));

// 모킹 후 import (vitest hoisting)
const { streamChat } = await import("./api");

describe("streamChat 콜드스타트 재시도", () => {
  beforeEach(() => {
    calls.length = 0;
    statuses = [];
    vi.useFakeTimers();
  });
  afterEach(() => vi.useRealTimers());

  it("502면 에러 대신 상태 안내를 내고 자동 재시도한다", async () => {
    statuses = [502, 200];
    const onStatus = vi.fn();
    const onError = vi.fn();

    streamChat("삼성전자 분석", [], { onStatus, onError });
    await vi.advanceTimersByTimeAsync(0);

    // 첫 시도 실패 → 에러가 아니라 상태 안내
    expect(onError).not.toHaveBeenCalled();
    expect(onStatus).toHaveBeenCalledTimes(1);
    expect(onStatus.mock.calls[0][0]).toContain("서버를 깨우고 있어요");

    // 대기 후 재시도되어 2번째 호출 발생
    await vi.advanceTimersByTimeAsync(4000);
    expect(calls).toHaveLength(2);
    expect(onError).not.toHaveBeenCalled();
  });

  it("재시도를 다 쓰면 최종적으로 에러를 보고한다", async () => {
    statuses = [502, 502, 502, 502, 502, 502];
    const onStatus = vi.fn();
    const onError = vi.fn();

    streamChat("질문", [], { onStatus, onError });
    await vi.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 6; i++) await vi.advanceTimersByTimeAsync(4000);

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0]).toContain("서버가 응답하지 않아요");
  });

  it("401 같은 비콜드스타트 오류는 재시도하지 않는다", async () => {
    statuses = [401];
    const onStatus = vi.fn();
    const onError = vi.fn();

    streamChat("질문", [], { onStatus, onError });
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(4000);

    expect(calls).toHaveLength(1); // 재시도 없음
    expect(onStatus).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it("abort하면 예정된 재시도가 실행되지 않는다", async () => {
    statuses = [502, 200];
    const abort = streamChat("질문", [], {});
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toHaveLength(1);

    abort();
    await vi.advanceTimersByTimeAsync(4000);
    expect(calls).toHaveLength(1); // 재시도 안 됨
  });
});
