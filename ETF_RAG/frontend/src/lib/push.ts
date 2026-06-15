// 웹 푸시 구독 — VAPID 공개키 조회 + 브라우저 구독/해제 + 백엔드 등록.
import { authHeader } from "./auth";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export interface VapidInfo {
  public_key: string;
  enabled: boolean;
}

export async function getVapidInfo(): Promise<VapidInfo> {
  const res = await fetch(`${API_BASE}/push/vapid-public-key`, { cache: "no-store" });
  if (!res.ok) return { public_key: "", enabled: false };
  return (await res.json()) as VapidInfo;
}

/** 브라우저가 웹 푸시를 지원하는지 */
export function pushSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    "serviceWorker" in navigator &&
    "PushManager" in window &&
    "Notification" in window
  );
}

function urlBase64ToUint8Array(base64: string): Uint8Array {
  const padding = "=".repeat((4 - (base64.length % 4)) % 4);
  const b64 = (base64 + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(b64);
  const arr = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
  return arr;
}

/** 현재 구독 상태 조회 (브라우저 기준) */
export async function getCurrentSubscription(): Promise<PushSubscription | null> {
  if (!pushSupported()) return null;
  const reg = await navigator.serviceWorker.ready;
  return reg.pushManager.getSubscription();
}

/** 푸시 구독 — 권한 요청 → SW 구독 → 백엔드 등록. 성공 true. */
export async function subscribePush(vapidPublicKey: string): Promise<boolean> {
  if (!pushSupported() || !vapidPublicKey) return false;
  const perm = await Notification.requestPermission();
  if (perm !== "granted") return false;

  const reg = await navigator.serviceWorker.ready;
  let sub = await reg.pushManager.getSubscription();
  if (!sub) {
    sub = await reg.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(vapidPublicKey) as BufferSource,
    });
  }
  const json = sub.toJSON() as {
    endpoint: string;
    keys: { p256dh: string; auth: string };
  };
  const res = await fetch(`${API_BASE}/push/subscribe`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({ endpoint: json.endpoint, keys: json.keys }),
  });
  return res.ok;
}

/** 푸시 해제 — 백엔드 삭제 + 브라우저 unsubscribe */
export async function unsubscribePush(): Promise<boolean> {
  const sub = await getCurrentSubscription();
  if (!sub) return true;
  await fetch(`${API_BASE}/push/unsubscribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({ endpoint: sub.endpoint }),
  }).catch(() => {});
  await sub.unsubscribe().catch(() => {});
  return true;
}

/** 테스트 알림 발송 (내 구독으로) */
export async function sendTestPush(): Promise<boolean> {
  const res = await fetch(`${API_BASE}/push/test`, {
    method: "POST",
    headers: authHeader(),
  });
  return res.ok;
}
