"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/lib/AuthContext";
import {
  getVapidInfo,
  getCurrentSubscription,
  subscribePush,
  unsubscribePush,
  sendTestPush,
  pushSupported,
} from "@/lib/push";

/**
 * 관심종목 푸시 알림 구독 토글. 로그인 + VAPID 활성 + 브라우저 지원 시에만 표시.
 * 일일 관심종목 급등/급락 알림(자동 발송)은 후속 PR이며, 이 토글로 구독을 등록한다.
 */
export default function PushToggle() {
  const { user } = useAuth();
  const [vapidKey, setVapidKey] = useState<string | null>(null);
  const [subscribed, setSubscribed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!user || !pushSupported()) return;
    let alive = true;
    (async () => {
      const info = await getVapidInfo();
      if (!alive) return;
      if (!info.enabled) {
        setVapidKey(null);
        return;
      }
      setVapidKey(info.public_key);
      const sub = await getCurrentSubscription();
      if (alive) setSubscribed(!!sub);
    })();
    return () => {
      alive = false;
    };
  }, [user]);

  // 로그인 안 함 / 미지원 / VAPID 미설정 → 숨김
  if (!user || !pushSupported() || !vapidKey) return null;

  const toggle = async () => {
    setBusy(true);
    setMsg(null);
    try {
      if (subscribed) {
        await unsubscribePush();
        setSubscribed(false);
        setMsg("알림 해제됨");
      } else {
        const ok = await subscribePush(vapidKey);
        setSubscribed(ok);
        setMsg(ok ? "알림 구독됨 🔔" : "알림 권한이 거부되었어요");
      }
    } finally {
      setBusy(false);
    }
  };

  const test = async () => {
    setBusy(true);
    setMsg(null);
    const ok = await sendTestPush();
    setMsg(ok ? "테스트 알림 발송됨" : "발송 실패");
    setBusy(false);
  };

  return (
    <div className="mt-3 border-t border-gray-100 pt-3">
      <div className="text-xs font-semibold text-gray-700">🔔 관심종목 알림</div>
      <button
        type="button"
        onClick={toggle}
        disabled={busy}
        className={[
          "mt-1 w-full rounded-lg px-2 py-1.5 text-xs",
          subscribed
            ? "bg-blue-600 text-white hover:bg-blue-700"
            : "border border-gray-300 dark:border-gray-700 text-gray-600 hover:bg-gray-100",
          busy ? "opacity-60" : "",
        ].join(" ")}
      >
        {busy ? "처리 중…" : subscribed ? "알림 켜짐 (끄기)" : "알림 받기"}
      </button>
      {subscribed && (
        <button
          type="button"
          onClick={test}
          disabled={busy}
          className="mt-1 w-full rounded-lg px-2 py-1 text-[11px] text-gray-400 hover:bg-gray-100"
        >
          테스트 알림 보내기
        </button>
      )}
      {msg && <div className="mt-1 text-[11px] text-gray-400">{msg}</div>}
      <p className="mt-1 text-[11px] leading-relaxed text-gray-400">
        관심종목 급등/급락 시 알림을 받습니다 (장 마감 후 1회).
      </p>
    </div>
  );
}
