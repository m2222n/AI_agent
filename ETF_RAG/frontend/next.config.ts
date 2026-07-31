import type { NextConfig } from "next";

// 빌드 타깃 분기:
//  - 기본(웹/Railway): output: "standalone" — .next/standalone/server.js (Docker 린 runner).
//  - BUILD_TARGET=static (iOS/Capacitor 앱): output: "export" — out/ 정적 번들을 앱에 내장.
// 하나의 코드베이스로 웹 배포와 앱 빌드를 겸한다.
const isStatic = process.env.BUILD_TARGET === "static";

const nextConfig: NextConfig = {
  output: isStatic ? "export" : "standalone",
  // 정적 export는 서버 이미지 최적화를 쓸 수 없다. 앱은 어차피 일반 <img>만 쓰므로 무해.
  images: { unoptimized: true },
};

export default nextConfig;
