import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")


def is_langsmith_enabled() -> bool:
    """LangSmith 트레이싱이 활성화되어 있는지 확인"""
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and bool(os.getenv("LANGCHAIN_API_KEY"))
    )

# Data paths
DATA_DIR = PROJECT_ROOT / "src" / "data"
ETF_DATA_PATH = DATA_DIR / "etf_data.json"  # 하드코딩 샘플 (fallback)
COLLECTED_DIR = DATA_DIR / "collected"       # 수집 데이터 디렉토리
DEPLOY_DIR = DATA_DIR / "deploy"             # 배포용 데이터 (Git 추적)
DB_PATH = DATA_DIR / "etf_rag.db"           # SQLite 데이터베이스 (주가 데이터, read-only)
LOG_DIR = PROJECT_ROOT / "logs"

# --- 인증 / 사용자 DB (Phase F-1 잔여) -------------------------------
# 사용자 DB(인증/관심종목/대화이력)는 주가용 stock DB(DB_PATH)와 별개 파일/엔진.
# 로컬/테스트는 sqlite, 프로덕션(Railway)은 DATABASE_URL=postgresql://...
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'etf_rag_users.db'}")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-insecure-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))  # 7일

import logging as _logging
if JWT_SECRET == "dev-insecure-change-me":
    _logging.getLogger(__name__).warning(
        "JWT_SECRET 미설정 — dev 기본값 사용 중. 프로덕션에서는 반드시 환경변수로 설정하세요."
    )

# 웹 푸시 알림 (VAPID) — scripts/gen_vapid_keys.py 로 1회 생성해 고정.
# 키 3종 모두 설정돼야 활성화. 미설정 시 푸시 비활성(구독/발송 비활성, 다른 기능 무관).
VAPID = {
    "public_key": os.getenv("VAPID_PUBLIC_KEY", ""),
    "private_key": os.getenv("VAPID_PRIVATE_KEY", ""),
    "subject": os.getenv("VAPID_SUBJECT", "mailto:admin@example.com"),
    "enabled": bool(os.getenv("VAPID_PUBLIC_KEY") and os.getenv("VAPID_PRIVATE_KEY")),
}


def get_latest_collected_path() -> Optional[Path]:
    """수집 디렉토리에서 가장 최근 ETF 데이터 파일 경로를 반환. 없으면 None."""
    if not COLLECTED_DIR.exists():
        return None
    files = sorted(COLLECTED_DIR.glob("etf_data_*.json"), reverse=True)
    return files[0] if files else None


def get_latest_stock_collected_path() -> Optional[Path]:
    """수집 디렉토리에서 가장 최근 주식 데이터 파일 경로를 반환. 없으면 None."""
    if not COLLECTED_DIR.exists():
        return None
    files = sorted(COLLECTED_DIR.glob("stock_data_*.json"), reverse=True)
    return files[0] if files else None


def get_deploy_etf_path() -> Optional[Path]:
    """배포용 ETF 데이터 경로. 없으면 None."""
    p = DEPLOY_DIR / "etf_data.json"
    return p if p.exists() else None


def get_deploy_stock_path() -> Optional[Path]:
    """배포용 주식 데이터 경로. 없으면 None."""
    p = DEPLOY_DIR / "stock_data.json"
    return p if p.exists() else None

# ETF 선별 기준
# 전종목 1084개 중 RAG 검색 대상을 필터링하는 기준
ETF_SELECTION = {
    "min_trade_value": 100_000_000,  # 최소 거래대금 1억원 (유동성 필터)
    "min_nav": 0,                     # NAV 0원 제외 (비정상 종목)
    "exclude_zero_close": True,       # 종가 0원 제외 (거래정지 등)
}

# RAG settings
SIMILARITY_THRESHOLD = 1.5
TOP_K_RESULTS = 3

# Hybrid search settings
HYBRID_SEARCH = {
    "dense_weight": 0.7,      # FAISS 벡터 검색 가중치
    "sparse_weight": 0.3,     # BM25 키워드 검색 가중치
    "bm25_k": 20,             # BM25 1차 후보 수
    "dense_k": 20,            # FAISS 1차 후보 수
    "final_k": 5,             # 최종 반환 문서 수
    "mmr_lambda": 0.7,        # MMR λ: 1.0=관련성만, 0.0=다양성만
    "min_rrf_score": 0.01,    # RRF 최소 점수 (이 이하는 무관한 결과로 판단)
}

# Rerank settings (Cohere)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
RERANK = {
    "enabled": bool(os.getenv("COHERE_API_KEY")),  # API 키 있으면 자동 활성화
    "model": "rerank-v3.5",         # Cohere Rerank 모델
    "top_n": 5,                     # Rerank 후 최종 반환 수 (= final_k)
}

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM settings
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT = 60
MAX_HISTORY_MESSAGES = 10

# Realtime price settings (yfinance)
# DART (OpenDart 재무제표 API)
DART_API_KEY = os.getenv("DART_API_KEY", "")
DART_COLLECTION = {
    "request_delay": 0.5,           # API 호출 간격 (초)
    "max_daily_requests": 39000,    # 일일 안전 한도 (실제 40,000)
    "min_trade_value": 1_000_000_000,  # 거래대금 10억 이상 종목만 수집
    "backfill_years": 3,            # 백필 기간 (년)
}

REALTIME_PRICE = {
    "cache_ttl": 300,           # 캐시 TTL (초) — 5분
    "market_open": "09:00",     # 장 시작 (KST)
    "market_close": "15:30",    # 장 마감 (KST)
    "enabled": True,            # 기능 활성화 플래그
}

# 한국투자증권 KIS Open API (F-2 실시간 시세)
# 키 3종이 모두 설정돼야 활성화. 미설정 시 realtime.py가 yfinance로 fallback.
# KIS_ENV: real(실전, openapi.koreainvestment.com:9443) | vps(모의, openapivts:29443)
# 시세 조회만 하므로 real로 충분 (주문 권한과 무관).
KIS_APP_KEY = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET", "")
KIS = {
    "enabled": bool(os.getenv("KIS_APP_KEY") and os.getenv("KIS_APP_SECRET")),
    "app_key": KIS_APP_KEY,
    "app_secret": KIS_APP_SECRET,
    "env": os.getenv("KIS_ENV", "real"),    # real | vps
    "base_url": (
        "https://openapivts.koreainvestment.com:29443"
        if os.getenv("KIS_ENV", "real") == "vps"
        else "https://openapi.koreainvestment.com:9443"
    ),
    "timeout": 5,               # REST 호출 타임아웃 (초)
    "token_margin": 600,        # 토큰 만료 N초 전 선제 갱신 (10분)
}

# Vector DB backend: "faiss" (default) or "pinecone"
VECTOR_DB_BACKEND = os.getenv("VECTOR_DB_BACKEND", "faiss")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "etf-rag")
