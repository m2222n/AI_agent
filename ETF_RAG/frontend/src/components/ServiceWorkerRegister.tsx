"use client";

import { useEffect } from "react";

// 프로덕션에서만 SW 등록 (dev는 HMR과 충돌 방지).
export default function ServiceWorkerRegister() {
  useEffect(() => {
    if (
      process.env.NODE_ENV === "production" &&
      "serviceWorker" in navigator
    ) {
      navigator.serviceWorker.register("/sw.js").catch(() => {
        /* 등록 실패는 무시 (PWA 미지원 환경 등) */
      });
    }
  }, []);
  return null;
}
