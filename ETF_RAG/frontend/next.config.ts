import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Docker 배포용 — .next/standalone/server.js 생성 (린 runner 이미지)
  output: "standalone",
};

export default nextConfig;
