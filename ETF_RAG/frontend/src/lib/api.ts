// 백엔드 API 클라이언트.
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { authHeader } from "./auth";
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
  NewsResponse,
  MoversResponse,
  OverviewResponse,
  VisitorResponse,
  PriceData,
  OrderbookData,
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
    headers: { "Content-Type": "application/json", ...authHeader() },
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
/** 콜드스타트(유휴 재시작) 중 게이트웨이가 내는 코드 — 잠시 후 되므로 재시도 대상. */
const COLD_START_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);
/** 콜드스타트 재시도 횟수/간격 — run_init(full DB+FAISS)에 수십초 걸림 */
const COLD_START_RETRIES = 5;
const COLD_START_DELAY_MS = 4000;

class ColdStartError extends Error {}

export function streamChat(
  question: string,
  history: ChatHistoryItem[],
  cb: StreamCallbacks,
): () => void {
  const ctrl = new AbortController();
  let attempt = 0;
  // 토큰을 한 번이라도 받았으면 재시도 금지(답변 중복/뒤섞임 방지)
  let gotData = false;

  const run = () => {
    fetchEventSource(`${API_BASE}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({
      question,
      chat_history: history.length ? history : null,
    }),
    signal: ctrl.signal,
    openWhenHidden: true, // 탭 백그라운드여도 스트림 유지

    onopen: async (res) => {
      if (res.ok) return;
      // 아직 데이터를 못 받았고 재시도 여력이 있으면 콜드스타트로 간주
      if (
        !gotData &&
        COLD_START_STATUS.has(res.status) &&
        attempt < COLD_START_RETRIES
      ) {
        throw new ColdStartError(`cold start ${res.status}`);
      }
      throw new Error(`stream ${res.status}`);
    },

    onmessage: (ev) => {
      // ev.event / ev.data — 라이브러리가 `:` 주석(ping) 라인은 걸러줌
      const { event, data } = ev;
      gotData = true; // 응답이 흐르기 시작 → 이후 실패는 재시도하지 않음
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
      // 콜드스타트로 판정된 경우: 서버가 깨어나길 기다렸다가 자동 재시도
      if (err instanceof ColdStartError) {
        attempt += 1;
        cb.onStatus?.(
          `서버를 깨우고 있어요… 잠시만 기다려주세요 (${attempt}/${COLD_START_RETRIES})`,
        );
        setTimeout(() => {
          if (!ctrl.signal.aborted) run();
        }, COLD_START_DELAY_MS);
        throw err; // 라이브러리 자체 재시도는 중단(위 setTimeout으로 직접 제어)
      }
      cb.onError?.(
        gotData
          ? "답변이 중간에 끊겼어요. 다시 시도해주세요."
          : "서버가 응답하지 않아요. 잠시 후 다시 시도해주세요.",
      );
      throw err; // re-throw → 라이브러리 자동 재시도(재POST) 중단
    },
    }).catch(() => {
      /* onError/onStatus로 이미 표면화됨 */
    });
  };

  run();
  return () => ctrl.abort();
}

// ── 탭 API ──────────────────────────────────────────────

/** 종목 자동완성 ("이름 (종목코드)" 옵션 리스트).
 * assetType="stock"이면 주식만, "etf"면 ETF만 (재무제표 탭처럼 주식 전용 화면용). */
export async function searchTickers(
  q: string,
  limit = 20,
  assetType?: "stock" | "etf",
  minDays?: number,
): Promise<string[]> {
  const url = new URL(`${API_BASE}/tabs/tickers`);
  if (q) url.searchParams.set("q", q);
  url.searchParams.set("limit", String(limit));
  if (assetType) url.searchParams.set("asset_type", assetType);
  if (minDays && minDays > 0) url.searchParams.set("min_days", String(minDays));
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

/** 실시간 시세 — 장중엔 KIS 우선(yfinance fallback), 장 외엔 수집 종가. 404면 null. */
export async function getPrice(ticker: string): Promise<PriceData | null> {
  const url = new URL(`${API_BASE}/tabs/price`);
  url.searchParams.set("ticker", ticker);
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`price ${res.status}`);
  return (await res.json()) as PriceData;
}

/**
 * 체결 틱 실시간 SSE (KIS WebSocket). 콜백으로 틱/unavailable 전달.
 * AbortController 반환 — .abort()로 연결 종료(구독 해제 트리거).
 * KIS 미연동/장외면 onUnavailable 호출(프론트가 REST 폴링으로 fallback).
 */
export function streamPrice(
  ticker: string,
  cb: {
    onTick: (t: PriceData) => void;
    onUnavailable: () => void;
    onError?: () => void;
  },
): AbortController {
  const ctrl = new AbortController();
  const url = new URL(`${API_BASE}/tabs/price/stream`);
  url.searchParams.set("ticker", ticker);
  fetchEventSource(url.toString(), {
    signal: ctrl.signal,
    openWhenHidden: true,
    onmessage(ev) {
      if (ev.event === "tick") {
        try {
          cb.onTick(JSON.parse(ev.data) as PriceData);
        } catch {
          /* ignore */
        }
      } else if (ev.event === "unavailable") {
        cb.onUnavailable();
      }
      // "ping"은 keep-alive — 무시
    },
    onerror(err) {
      cb.onError?.();
      throw err; // 자동 재연결 중단 (fallback은 PriceCard가 담당)
    },
  });
  return ctrl;
}

/** 호가 10단계 (KIS 전용). KIS 미연동/장 외/실패 시 null. */
export async function getOrderbook(ticker: string): Promise<OrderbookData | null> {
  const url = new URL(`${API_BASE}/tabs/orderbook`);
  url.searchParams.set("ticker", ticker);
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`orderbook ${res.status}`);
  return (await res.json()) as OrderbookData;
}

/** 장중 시세 차트 (yfinance 15분봉). 장 외/데이터 없으면 null. */
export async function getIntraday(
  ticker: string,
): Promise<{ ticker: string; name: string; chart_b64: string } | null> {
  const url = new URL(`${API_BASE}/tabs/intraday`);
  url.searchParams.set("ticker", ticker);
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`intraday ${res.status}`);
  return (await res.json()) as { ticker: string; name: string; chart_b64: string };
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

/** 섹터 분석 — 전체 또는 특정 섹터. period 지정 시 그 섹터의 기간 추이 차트 포함. 404면 null. */
export async function getSector(
  sector?: string,
  period?: string,
): Promise<SectorResponse | null> {
  const url = new URL(`${API_BASE}/tabs/sector`);
  if (sector) url.searchParams.set("sector", sector);
  if (period) url.searchParams.set("period", period);
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`sector ${res.status}`);
  return (await res.json()) as SectorResponse;
}

/** 뉴스 감성 + 일별 시계열 차트. 404면 null. */
export async function getNews(ticker: string): Promise<NewsResponse | null> {
  const url = new URL(`${API_BASE}/tabs/news`);
  url.searchParams.set("ticker", ticker);
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`news ${res.status}`);
  return (await res.json()) as NewsResponse;
}

/** 사이드바 개요 — 데이터 현황 + ETF/주식 TOP. sector 지정 시 해당 업종 종목만. 실패 시 null. */
export async function getOverview(
  top = 20,
  sector?: string,
): Promise<OverviewResponse | null> {
  const url = new URL(`${API_BASE}/tabs/overview`);
  url.searchParams.set("top", String(top));
  if (sector) url.searchParams.set("sector", sector);
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as OverviewResponse;
  } catch {
    return null;
  }
}

/** 오늘의 급등/급락/거래대금 TOP — 동적 추천질문용. 실패 시 null. */
export async function getMovers(n = 3): Promise<MoversResponse | null> {
  const url = new URL(`${API_BASE}/tabs/movers`);
  url.searchParams.set("n", String(n));
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as MoversResponse;
  } catch {
    return null;
  }
}

/** 방문 기록(POST, 세션당 1회) 또는 조회(GET). 실패 시 null. */
export async function getVisitor(record: boolean): Promise<VisitorResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/stats/visit`, {
      method: record ? "POST" : "GET",
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as VisitorResponse;
  } catch {
    return null;
  }
}

/** 답변 피드백 전송 (익명 허용, 로그인 시 Bearer 포함). 실패 무시. */
export async function sendFeedback(
  question: string,
  answer: string,
  rating: "positive" | "negative",
  reason?: string,
): Promise<void> {
  try {
    await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader() },
      body: JSON.stringify({ question, answer, rating, reason: reason ?? null }),
    });
  } catch {
    /* 피드백 실패는 무시 */
  }
}
