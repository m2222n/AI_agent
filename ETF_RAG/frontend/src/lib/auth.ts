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
