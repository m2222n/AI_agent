"use client";

import { useEffect } from "react";
import { Capacitor } from "@capacitor/core";

/**
 * iOS 앱(Capacitor) 네이티브 초기화.
 *
 * - 상태바 스타일을 앱 테마에 맞춤(밝은 배경 → 어두운 글자).
 * - 웹 자산 로드가 끝나면 스플래시 스크린을 숨김.
 *
 * 웹(브라우저/PWA)에서는 아무 것도 하지 않는다(네이티브 가드).
 * 플러그인은 앱 번들에만 실려 있으므로 동적 import로 웹 번들 오염을 피한다.
 */
export default function NativeAppInit() {
  useEffect(() => {
    if (!Capacitor.isNativePlatform()) return;

    (async () => {
      try {
        const { StatusBar, Style } = await import("@capacitor/status-bar");
        await StatusBar.setStyle({ style: Style.Light }); // 밝은 배경용 어두운 글자
      } catch {
        /* 상태바 설정 실패는 무시 */
      }
      try {
        const { SplashScreen } = await import("@capacitor/splash-screen");
        await SplashScreen.hide();
      } catch {
        /* 스플래시 숨김 실패는 무시 */
      }
    })();
  }, []);

  return null;
}
