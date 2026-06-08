# ETF RAG 프론트엔드 (Next.js 16, App Router, standalone) — Phase F-5 배포
# 멀티스테이지: deps → build → runner.
# NEXT_PUBLIC_API_BASE는 빌드타임 baked → build ARG로 주입 (백엔드 URL).

# 1) deps
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# 2) build — NEXT_PUBLIC_API_BASE가 번들에 baked됨
FROM node:20-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_BASE
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}
RUN npm run build

# 3) runner — standalone 최소 이미지
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000
# standalone은 필요한 node_modules + server.js만 포함.
# public, .next/static은 수동 복사 필요 (Next 16 standalone 사양).
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
# server.js는 PORT env를 네이티브로 읽음.
CMD ["node", "server.js"]
