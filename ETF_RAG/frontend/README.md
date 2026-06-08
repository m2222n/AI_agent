# 투자 AI 어시스턴트 — 프론트엔드 (Next.js)

ETF RAG 챗봇의 웹 프론트엔드. FastAPI 백엔드(`../api/`)를 호출한다. (Phase F-4)

- **F-4a (현재)**: 스캐폴딩 + `/health` 게이트 + 비스트리밍 `/chat` 채팅
- 후속: 4b SSE 스트리밍 · 4c 차트/표 · 4d 멀티턴/모바일/후속질문

## 기술 스택

Next.js 16 (App Router) · React 19 · TypeScript · Tailwind CSS v4

## 실행 (개발)

터미널 2개가 필요하다.

### 1) 백엔드 (repo root `ETF_RAG/`에서)

```bash
cd ..
uvicorn api.main:app --host 0.0.0.0 --port 8000   # 단일 워커만
```

- `.env`에 `OPENAI_API_KEY` 필요.
- 첫 부팅 시 DB 다운로드/임베딩으로 `/health`가 잠시 `ready:false` → 프론트가 자동 대기.

### 2) 프론트 (이 디렉토리에서)

```bash
cp .env.example .env.local     # 최초 1회. 백엔드 주소가 다르면 수정.
npm install                    # 최초 1회
npm run dev                    # http://localhost:3000
```

백엔드 CORS가 `localhost:3000`을 허용하므로 추가 설정 불필요.

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | 백엔드 FastAPI 주소 (client 노출 — `NEXT_PUBLIC_` 필수) |

## 주의

- Node v25에서 `next dev`/`next build`가 SWC/engine 에러를 내면 Node 22 LTS(nvm/fnm)로 폴백.
- `node_modules/`, `.next/`, `.env.local`은 커밋하지 않는다 (`.gitignore`로 제외).
