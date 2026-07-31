import type { MetadataRoute } from "next";

// 정적 export(iOS 앱 빌드) 시 매니페스트를 빌드 타임에 고정 생성. standalone에도 무해.
export const dynamic = "force-static";

// PWA 매니페스트 — 홈화면 설치 시 앱처럼 동작(standalone).
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "주선생",
    short_name: "주선생",
    description: "ETF·주식 RAG 기반 투자 질의응답 챗봇",
    start_url: "/",
    display: "standalone",
    background_color: "#ffffff",
    theme_color: "#2563eb",
    lang: "ko",
    icons: [
      {
        src: "/icons/icon-192.png",
        sizes: "192x192",
        type: "image/png",
        purpose: "any",
      },
      {
        src: "/icons/icon-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "any",
      },
      {
        src: "/icons/icon-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
  };
}
