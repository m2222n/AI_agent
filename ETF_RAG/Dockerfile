# ETF RAG FastAPI 백엔드 (Phase F-5 배포)
# 빌드 컨텍스트: repo 루트(ETF_RAG/). DB는 베이크하지 않고 런타임에 ensure_db()가 다운로드.
FROM python:3.11-slim

# 시스템 의존성:
#  - build-essential + cmake: prophet(C++ 컴파일), kiwipiepy(소스 빌드 가능성)
#  - fonts-nanum: matplotlib 한글 차트 글리프 (chart_generator)
#  - curl: HEALTHCHECK용
# pip install 전에 설치해야 prophet/kiwipiepy 컴파일이 컴파일러/cmake를 찾는다.
#  - fontconfig: fc-cache 제공 (slim 이미지에 미포함 → fonts-nanum 캐시 갱신용)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        fonts-nanum \
        fontconfig \
        curl \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# 의존성 레이어 먼저 (requirements.txt 변경 시에만 재실행 → 캐시 재사용)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 애플리케이션 코드만 (frontend/, tests/, eval/ 제외 — .dockerignore).
# src/data/*.py 소스 + src/data/deploy/ fallback JSON은 포함, etf_rag.db(1.7GB)는 제외.
COPY config.py .
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
# 사용자 DB 마이그레이션 — 부팅 시 api/db.py:run_migrations가 참조(없으면 런타임 실패).
COPY alembic.ini .
COPY alembic/ ./alembic/

EXPOSE 8000

# 첫 부팅 시 DB 다운로드(3~5분) 대기 위해 start-period 넉넉히.
HEALTHCHECK --interval=30s --timeout=5s --start-period=360s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

# 쉘 형식 — 앱이 PORT를 직접 안 읽으므로 ${PORT}를 쉘이 확장.
# 단일 워커 전용: set_retriever가 프로세스 전역 상태 (api/main.py 참조).
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
