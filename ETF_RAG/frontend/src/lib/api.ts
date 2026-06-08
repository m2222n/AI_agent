// 백엔드 API 클라이언트. 4a는 getHealth + chatOnce만. streamChat은 4b에서 추가.
import type { ChatHistoryItem, ChatResponse, Health } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

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
