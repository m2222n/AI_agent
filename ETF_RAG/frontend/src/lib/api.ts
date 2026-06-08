// 백엔드 API 클라이언트.
import { fetchEventSource } from "@microsoft/fetch-event-source";
import type {
  ChatHistoryItem,
  ChatResponse,
  DonePayload,
  Health,
  QuestionType,
  StreamCallbacks,
  StructuredData,
  TechnicalResponse,
  TickerSearchResponse,
  FinancialResponse,
  ComparisonResponse,
  OutlookResponse,
  SectorResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`health ${res.status}`);
  return (await res.json()) as Health;
}

export async function chatOnce(
  question: string,
  history: ChatHistoryItem[],
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      chat_history: history.length ? history : null,
    }),
  });
  if (!res.ok) throw new Error(`chat ${res.status}`);
  return (await res.json()) as ChatResponse;
}

/**
 * POST /stream SSE 소비. 네이티브 EventSource는 GET only라 fetch-event-source 사용.
 * 라이브러리가 SSE 프레이밍(event/data 라인, `: ping` 주석 라인 스킵)을 처리한다.
 * token 이벤트의 data는 "누적 전체 답변" → onToken에서 replace.
 * 반환: abort() — 호출자가 스트림을 취소할 수 있음.
 */
export function streamChat(
  question: string,
  history: ChatHistoryItem[],
  cb: StreamCallbacks,
): () => void {
  const ctrl = new AbortController();

  fetchEventSource(`${API_BASE}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      chat_history: history.length ? history : null,
    }),
    signal: ctrl.signal,
    openWhenHidden: true, // 탭 백그라운드여도 스트림 유지

    onopen: async (res) => {
      if (!res.ok) throw new Error(`stream ${res.status}`);
    },

    onmessage: (ev) => {
      // ev.event / ev.data — 라이브러리가 `:` 주석(ping) 라인은 걸러줌
      const { event, data } = ev;
      switch (event) {
        case "question_type":
          cb.onQuestionType?.(data as QuestionType);
          break;
        case "tool_call":
          cb.onToolCall?.(JSON.parse(data));
          break;
        case "tool_result":
          cb.onToolResult?.(data);
          break;
        case "structured_data":
          cb.onStructuredData?.(JSON.parse(data) as StructuredData);
          break;
        case "token":
          cb.onToken?.(data); // 누적 → replace
          break;
        case "cov_revision":
          cb.onCovRevision?.(data);
          break;
        case "error":
          cb.onError?.(data);
          break;
        case "done":
          cb.onDone?.(JSON.parse(data) as DonePayload);
          break;
        default:
          break; // 알 수 없는 이벤트는 무시
      }
    },

    onerror: (err) => {
      cb.onError?.("연결 오류가 발생했어요. 잠시 후 다시 시도해주세요.");
      throw err; // re-throw → 라이브러리 자동 재시도(재POST) 중단
    },
  }).catch(() => {
    /* onError로 이미 표면화됨 */
  });

  return () => ctrl.abort();
}

// ── 탭 API ──────────────────────────────────────────────

/** 종목 자동완성 ("이름 (티커)" 옵션 리스트) */
export async function searchTickers(
  q: string,
  limit = 20,
): Promise<string[]> {
  const url = new URL(`${API_BASE}/tabs/tickers`);
  if (q) url.searchParams.set("q", q);
  url.searchParams.set("limit", String(limit));
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) return [];
  const body = (await res.json()) as TickerSearchResponse;
  return body.options;
}

/** 기술적 분석 — 지표 summary + 차트 base64. 404면 null. */
export async function getTechnical(
  ticker: string,
  days = 120,
): Promise<TechnicalResponse | null> {
  const url = new URL(`${API_BASE}/tabs/technical`);
  url.searchParams.set("ticker", ticker);
  url.searchParams.set("days", String(days));
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`technical ${res.status}`);
  return (await res.json()) as TechnicalResponse;
}

/** 재무제표 — 분기 rows + 차트. 404면 null. */
export async function getFinancial(
  ticker: string,
  quarters = 8,
): Promise<FinancialResponse | null> {
  const url = new URL(`${API_BASE}/tabs/financial`);
  url.searchParams.set("ticker", ticker);
  url.searchParams.set("quarters", String(quarters));
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`financial ${res.status}`);
  return (await res.json()) as FinancialResponse;
}

/** 비교 분석 — 2종목. 404면 null. */
export async function postComparison(
  tickers: [string, string],
  days = 120,
): Promise<ComparisonResponse | null> {
  const res = await fetch(`${API_BASE}/tabs/comparison`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tickers, days }),
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`comparison ${res.status}`);
  return (await res.json()) as ComparisonResponse;
}

/** 가격 전망 — 4축. 404면 null. */
export async function getOutlook(
  ticker: string,
  horizon = "1m",
): Promise<OutlookResponse | null> {
  const url = new URL(`${API_BASE}/tabs/outlook`);
  url.searchParams.set("ticker", ticker);
  url.searchParams.set("horizon", horizon);
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`outlook ${res.status}`);
  return (await res.json()) as OutlookResponse;
}

/** 섹터 분석 — 전체 또는 특정 섹터. 404면 null. */
export async function getSector(
  sector?: string,
): Promise<SectorResponse | null> {
  const url = new URL(`${API_BASE}/tabs/sector`);
  if (sector) url.searchParams.set("sector", sector);
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`sector ${res.status}`);
  return (await res.json()) as SectorResponse;
}
