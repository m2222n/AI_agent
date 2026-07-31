import type { CapacitorConfig } from "@capacitor/cli";

// Capacitor iOS 앱 설정.
// webDir=out: `npm run build:static`이 생성하는 Next.js 정적 번들을 앱에 내장한다.
// server.url을 두지 않는 것이 핵심 — 원격 URL 래퍼가 아니라 번들을 기기에 담아야
// Apple 심사 가이드라인 4.2(Minimum Functionality) 리젝을 피한다.
// 데이터는 앱 안의 JS가 라이브 백엔드(Railway)를 fetch로 호출한다.
const config: CapacitorConfig = {
  appId: "ai.jusunsaeng.app", // App Store 번들 ID (확정, 영구 고정)
  appName: "주선생",
  webDir: "out",
  ios: {
    // WKWebView가 스크롤 바운스를 앱처럼 처리
    contentInset: "always",
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 1500,
      backgroundColor: "#2563eb", // 매니페스트 theme_color와 통일
      showSpinner: false,
    },
  },
};

export default config;
