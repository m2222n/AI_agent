# 배포 가이드 (Phase F-5)

ETF RAG를 **Railway**에 배포하는 단계별 안내. 백엔드(FastAPI)와 프론트(Next.js)는
HTTP로만 결합된 **독립 2서비스**다. 기존 Streamlit 앱은 그대로 병행.

> 이 문서는 설정/도커화까지 준비된 상태 기준. 실제 계정 생성·배포 클릭은 직접 수행.
> **시크릿은 절대 커밋하지 않는다** (전부 Railway 대시보드 env로).

## 구성 요약

| 서비스 | Root Dir | 빌더 | 노출 |
|--------|----------|------|------|
| `etf-backend` | `ETF_RAG/` | Dockerfile | `/health`,`/chat`,`/stream`,`/tabs/*` |
| `etf-frontend` | `ETF_RAG/frontend/` | Dockerfile | Next.js UI |

(GitHub repo가 `m2222n/AI_agent`이고 ETF_RAG가 하위 디렉토리이므로 Root Dir은 위와 같이 지정.)

추가된 파일: `Dockerfile`, `.dockerignore`, `frontend/Dockerfile`, `frontend/.dockerignore`,
`frontend/next.config.ts`(standalone), `docker-compose.yml`, `api/main.py`(CORS env화).

## 사전 준비

- Railway 계정 + GitHub repo 연결
- `OPENAI_API_KEY` (필수). 선택: `COHERE_API_KEY`(rerank), `DART_API_KEY`(재무).
- DB는 현행 유지 — 백엔드 첫 부팅 시 GitHub Release `db-latest/etf_rag.db.zst`(~450MB→1.75GB) 자동 다운로드.

## 1단계: 백엔드 배포 (먼저)

1. Railway 프로젝트 생성 → 서비스 `etf-backend`, **Root Directory = `ETF_RAG`**. (Dockerfile 자동 감지)
2. **환경변수** (대시보드):
   - `OPENAI_API_KEY` (필수)
   - `CORS_ORIGINS` = 프론트 URL (2단계 후 설정 — 처음엔 비워두면 `*`)
   - **`DATABASE_URL`** = `postgresql://user:pass@host:5432/db` (인증/유저 DB — Railway Postgres 플러그인 추가 후 제공되는 URL). 미설정 시 로컬 sqlite(컨테이너 휘발 → 재시작 시 유저 데이터 소실, 프로덕션 비권장).
   - **`JWT_SECRET`** (필수) = 랜덤 긴 문자열. 미설정 시 dev 기본값 + 경고. `JWT_EXPIRE_MINUTES`(선택, 기본 7일).
   - 선택: `SUPABASE_URL` + `SUPABASE_KEY` (방문자 카운터 — 미설정 시 카운터 숨김). Streamlit과 같은 `visitor_stats` 테이블 공유.
   - 선택: `COHERE_API_KEY`, `DART_API_KEY`, `LANGCHAIN_*`, `VECTOR_DB_BACKEND`, `PINECONE_*`
   - `PORT` — Railway가 자동 주입 (Dockerfile CMD가 `${PORT}` 확장).
   - 인증/유저 DB 스키마는 부팅 시 `init_models()`가 Alembic으로 마이그레이션(`run_migrations`).
     기존 라이브 DB(가입자 보유)는 최초 배포 시 `stamp head`(DDL 0건, 데이터 무손상)로 버전만 각인,
     이후 배포부터 `upgrade head`. 테스트(`API_SKIP_INIT=1`)는 Alembic 없이 `create_all`.
     마이그레이션 실패 시 `create_all` fallback(가용성 우선). 스키마 변경 시 `alembic revision
     --autogenerate` → CI drift 게이트(`alembic check`)가 리비전 누락을 감지.
3. 배포 → **첫 부팅 동작**: lifespan `run_init()`이 DB 다운로드(3~5분) + FAISS 인덱스 빌드(~90s).
   그동안 `/health`는 `{ready:false}`, 완료되면 `{ready:true}`. 이후 부팅은 (볼륨 있으면) 빠름.
4. 백엔드 **public URL** 확보 (예: `https://etf-backend-production.up.railway.app`).

## 2단계: 프론트 배포 (백엔드 URL 확보 후)

1. 같은 프로젝트에 서비스 `etf-frontend`, **Root Directory = `ETF_RAG/frontend`**.
2. **빌드 변수** `NEXT_PUBLIC_API_BASE` = 백엔드 URL.
   - ⚠️ **빌드타임 baked** — 번들에 굳어진다. 백엔드 URL이 바뀌면 **프론트 재빌드** 필요.
3. 배포 후, 백엔드 `CORS_ORIGINS`를 프론트 URL로 설정하고 백엔드 재배포.

## 3단계: 검증

프론트 URL 열기 → health 게이트 통과 확인 → 채팅 한 번 → 데이터 탭(기술/재무/섹터) 확인.

## 영속 볼륨 — DB/FAISS 콜드스타트 제거 (코드 구현 완료)

영속 볼륨이 없으면 재배포(콜드스타트)마다 DB 재다운로드(3~5분, 1.8GB) + FAISS
재임베딩(~90s)이 반복된다. **코드는 `ETF_DATA_DIR` 환경변수를 지원**하도록 구현됨:
- `config.PERSIST_DIR = ETF_DATA_DIR`(미설정 시 기존 `src/data`) → DB_PATH·FAISS·BM25
  캐시가 모두 이 경로 기준(`_schema.py`/`api/deps.py`/`vectorstore.py`/`retriever.py` 통일).
- 미설정 시 동작 무변경(로컬·기존 배포 안전).

**Railway 설정 (사용자 1회):**
1. `etf-backend`(AI_agent) 서비스 → 우클릭/메뉴 → **Add Volume**
2. Mount path = **`/data`** (❌ `/app/src/data` 금지 — 소스 패키지/deploy fallback을 가림)
3. Variables에 **`ETF_DATA_DIR=/data`** 추가 → 재배포
→ 첫 부팅에 DB가 `/data`에 1회 저장되고, 이후 재배포는 그걸 재사용해 즉시 부팅.
   (deploy/ JSON·하드코딩 샘플은 소스(`src/data`)에 그대로 — 볼륨과 분리.)

## Render 대안 (비권장)

같은 Dockerfile 사용 가능. **단, 무료 티어는 디스크가 ephemeral** → 재시작마다 1.7GB 재다운로드 +
무료 인스턴스 유휴 슬립 → 반복 3~5분 콜드부팅. 영속 디스크는 유료 플랜 필요. 유료+영속 디스크면 가능.

## 로컬 테스트 (docker-compose)

```bash
# 이미지 빌드만 확인 (백엔드는 prophet 컴파일로 수분 소요)
docker build -t etf-backend .
docker build -t etf-frontend --build-arg NEXT_PUBLIC_API_BASE=http://localhost:8000 frontend/

# DB 다운로드 없이 백엔드 스모크:
docker run --rm -p 8000:8000 -e API_SKIP_INIT=1 etf-backend
#   → curl http://localhost:8000/health  (ready:true, 단 실제 retriever 없음 — 라우팅/CORS 확인용)

# 2서비스 함께 (실제 DB 다운로드 발생):
OPENAI_API_KEY=sk-... docker compose up --build
#   → 프론트 http://localhost:3000, 백엔드 http://localhost:8000
```

## 트러블슈팅

- `/health`가 계속 `ready:false` → `OPENAI_API_KEY` 확인, 로그에서 DB 다운로드 진행 확인(3~5분).
- CORS 에러 → 백엔드 `CORS_ORIGINS`를 프론트 URL로 설정했는지.
- 프론트가 `localhost:8000` 호출 → `NEXT_PUBLIC_API_BASE`를 올바른 백엔드 URL로 **재빌드**.

## 하지 말 것

- 시크릿 커밋 금지 (`.dockerignore`가 `.env*` 제외, 키는 대시보드로).
- 1.7GB DB 이미지 베이크 금지 (Release 다운로드 유지).
- **멀티 워커 금지** — `set_retriever`가 프로세스 전역 상태 (`api/main.py`). uvicorn 단일 워커.
- 볼륨을 `src/data`에 마운트 금지 (소스 패키지 shadow). `/data` + `ETF_DATA_DIR` 사용.
