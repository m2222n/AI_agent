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
DB_PATH = DATA_DIR / "etf_rag.db"           # SQLite 데이터베이스
LOG_DIR = PROJECT_ROOT / "logs"


def get_latest_collected_path() -> Optional[Path]:
    """수집 디렉토리에서 가장 최근 데이터 파일 경로를 반환. 없으면 None."""
    if not COLLECTED_DIR.exists():
        return None
    files = sorted(COLLECTED_DIR.glob("etf_data_*.json"), reverse=True)
    return files[0] if files else None

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
    "min_rrf_score": 0.002,   # RRF 최소 점수 (이 이하는 무관한 결과로 판단)
}

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM settings
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT = 60
MAX_HISTORY_MESSAGES = 10

# Realtime price settings (yfinance)
REALTIME_PRICE = {
    "cache_ttl": 300,           # 캐시 TTL (초) — 5분
    "market_open": "09:00",     # 장 시작 (KST)
    "market_close": "15:30",    # 장 마감 (KST)
    "enabled": True,            # 기능 활성화 플래그
}
