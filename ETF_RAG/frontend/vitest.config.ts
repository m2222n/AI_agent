import { defineConfig } from "vitest/config";
import path from "node:path";

// 프론트 단위 테스트 — 순수 로직(lib) + 가벼운 컴포넌트(Feedback 등).
// jsdom 환경 + @/ alias. 페이지 통합/E2E는 범위 밖(tsc+build로 커버).
export default defineConfig({
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
});
