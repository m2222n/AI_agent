// 인증 토큰 저장 + 인증/유저데이터 API. 토큰은 localStorage.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const TOKEN_KEY = "etfrag.token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

/** 토큰이 있으면 Authorization 헤더 객체 반환 (없으면 빈 객체) */
export function authHeader(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

export interface AuthUser {
  id: number;
  email: string;
  nickname: string; // 미설정 시 백엔드가 이메일 앞부분으로 채워 보냄
}

async function postAuth(
  path: string,
  body: { email: string; password: string },
): Promise<string> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${path} ${res.status}`);
  }
  const data = await res.json();
  return data.access_token as string;
}

export async function signup(email: string, password: string): Promise<string> {
  return postAuth("/auth/signup", { email, password });
}

export async function login(email: string, password: string): Promise<string> {
  return postAuth("/auth/login", { email, password });
}

/** 현재 토큰으로 사용자 조회. 실패(401 등) 시 null. */
export async function fetchMe(): Promise<AuthUser | null> {
  const t = getToken();
  if (!t) return null;
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${t}` },
    cache: "no-store",
  });
  if (!res.ok) return null;
  return (await res.json()) as AuthUser;
}

// ── 계정 관리 (비번 변경 / 닉네임 / 탈퇴) ────────────────
async function readDetail(res: Response): Promise<string> {
  const d = await res.json().catch(() => ({}));
  return d.detail || `요청 실패 (${res.status})`;
}

export async function changePassword(
  currentPassword: string,
  newPassword: string,
): Promise<void> {
  const res = await fetch(`${API_BASE}/auth/password`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
  if (!res.ok) throw new Error(await readDetail(res));
}

export async function updateNickname(nickname: string): Promise<AuthUser> {
  const res = await fetch(`${API_BASE}/auth/profile`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({ nickname }),
  });
  if (!res.ok) throw new Error(await readDetail(res));
  return (await res.json()) as AuthUser;
}

export async function deleteAccount(password: string): Promise<void> {
  const res = await fetch(`${API_BASE}/auth/me`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({ password }),
  });
  if (!res.ok) throw new Error(await readDetail(res));
}

// ── 유저별 저장 (로그인 시) ──────────────────────────────
import type { ChatHistoryItem } from "./types";

export interface ServerChatMessage {
  role: "user" | "assistant";
  content: string;
  question_type?: string | null;
  model?: string | null;
}

export async function getServerHistory(): Promise<ServerChatMessage[]> {
  const res = await fetch(`${API_BASE}/me/history`, {
    headers: authHeader(),
    cache: "no-store",
  });
  if (!res.ok) return [];
  const data = await res.json();
  return (data.messages ?? []) as ServerChatMessage[];
}

export async function appendServerHistory(
  messages: ServerChatMessage[],
): Promise<void> {
  if (!messages.length) return;
  await fetch(`${API_BASE}/me/history`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({ messages }),
  });
}

export async function clearServerHistory(): Promise<void> {
  await fetch(`${API_BASE}/me/history`, {
    method: "DELETE",
    headers: authHeader(),
  });
}

// ChatHistoryItem(UI) → 서버 전송용 평탄화
export function toServerMessages(items: ChatHistoryItem[]): ServerChatMessage[] {
  return items.map((m) => ({ role: m.role, content: m.content }));
}

// ── 관심종목 (로그인 시) ─────────────────────────────────
export async function getWatchlist(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/me/watchlist`, {
    headers: authHeader(),
    cache: "no-store",
  });
  if (!res.ok) return [];
  const data = await res.json();
  return (data.tickers ?? []) as string[];
}

export async function addWatchlist(ticker: string): Promise<string[]> {
  const res = await fetch(`${API_BASE}/me/watchlist/${ticker}`, {
    method: "PUT",
    headers: authHeader(),
  });
  if (!res.ok) return [];
  const data = await res.json();
  return (data.tickers ?? []) as string[];
}

export async function removeWatchlist(ticker: string): Promise<string[]> {
  const res = await fetch(`${API_BASE}/me/watchlist/${ticker}`, {
    method: "DELETE",
    headers: authHeader(),
  });
  if (!res.ok) return [];
  const data = await res.json();
  return (data.tickers ?? []) as string[];
}

// ── 가상투자(모의투자) — 전부 로그인 필요 ─────────────────
import type {
  PaperPortfolio,
  PaperTradeResult,
  PaperTradeHistoryItem,
  PaperRanking,
  PaperHistory,
} from "./types";

async function paperGet<T>(path: string): Promise<T | null> {
  const res = await fetch(`${API_BASE}/me/paper${path}`, {
    headers: authHeader(),
    cache: "no-store",
  });
  if (!res.ok) return null;
  return (await res.json()) as T;
}

export function getPortfolio(): Promise<PaperPortfolio | null> {
  return paperGet<PaperPortfolio>("/portfolio");
}

export async function getTradeHistory(): Promise<PaperTradeHistoryItem[]> {
  const r = await paperGet<{ trades: PaperTradeHistoryItem[] }>("/trades");
  return r?.trades ?? [];
}

export function getRanking(): Promise<PaperRanking | null> {
  return paperGet<PaperRanking>("/ranking");
}

export function getPaperHistory(): Promise<PaperHistory | null> {
  return paperGet<PaperHistory>("/history");
}

/** 매수/매도 — 실패 시 detail 메시지로 throw(잔고부족 등 사용자에게 표시). */
async function paperTrade(
  side: "buy" | "sell",
  ticker: string,
  qty: number,
): Promise<PaperTradeResult> {
  const res = await fetch(`${API_BASE}/me/paper/${side}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({ ticker, qty }),
  });
  if (!res.ok) {
    const d = await res.json().catch(() => ({}));
    throw new Error(d.detail || `${side} 실패 (${res.status})`);
  }
  return (await res.json()) as PaperTradeResult;
}

export const buyStock = (ticker: string, qty: number) =>
  paperTrade("buy", ticker, qty);
export const sellStock = (ticker: string, qty: number) =>
  paperTrade("sell", ticker, qty);

export async function resetPaper(): Promise<PaperPortfolio | null> {
  const res = await fetch(`${API_BASE}/me/paper/reset`, {
    method: "POST",
    headers: authHeader(),
  });
  if (!res.ok) return null;
  return (await res.json()) as PaperPortfolio;
}
