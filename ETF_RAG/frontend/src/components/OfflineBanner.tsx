"use client";

import { useEffect, useState } from "react";
import { Capacitor } from "@capacitor/core";
import { Network } from "@capacitor/network";

/**
 * 오프라인 감지 배너.
 *
 * iOS 앱(Capacitor)에서 인터넷이 끊기면 흰 화면 대신 네이티브풍 안내를 띄운다 —
 * Apple 심사 가이드라인 4.2(Minimum Functionality)는 오프라인 시 빈 화면을 리젝 사유로 본다.
 * 네이티브에서는 @capacitor/network, 웹에서는 navigator.onLine으로 동작해 웹/앱 모두 안전.
 */
export default function OfflineBanner() {
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let removeListener: (() => void) | undefined;

    if (Capacitor.isNativePlatform()) {
      // 네이티브: Capacitor Network 플러그인
      Network.getStatus().then((s) => setOffline(!s.connected));
      const handle = Network.addListener("networkStatusChange", (s) => {
        setOffline(!s.connected);
      });
      removeListener = () => {
        handle.then((h) => h.remove());
      };
    } else if (typeof navigator !== "undefined") {
      // 웹: 브라우저 online/offline 이벤트
      const update = () => setOffline(!navigator.onLine);
      update();
      window.addEventListener("online", update);
      window.addEventListener("offline", update);
      removeListener = () => {
        window.removeEventListener("online", update);
        window.removeEventListener("offline", update);
      };
    }

    return () => removeListener?.();
  }, []);

  if (!offline) return null;

  return (
    <div
      role="alert"
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        background: "#b91c1c",
        color: "#fff",
        padding: "10px 16px",
        textAlign: "center",
        fontSize: "0.875rem",
        // iOS 노치/상태바 안전영역
        paddingTop: "calc(10px + env(safe-area-inset-top))",
      }}
    >
      📡 인터넷 연결이 끊겼어요. 연결을 확인한 뒤 다시 시도해 주세요.
    </div>
  );
}
