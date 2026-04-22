"""
GitHub Release에서 SQLite DB 다운로드 (Streamlit Cloud 시작 시 1회)

DB가 이미 로컬에 있으면 건너뜀. 없으면 GitHub Release의 zstd 압축 DB를 다운로드/해제.
다운로드 실패 시 deploy/ JSON fallback이 동작하도록 예외를 삼킴.
"""

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Public repo → 인증 불필요
DB_RELEASE_URL = (
    "https://github.com/m2222n/AI_agent/releases/download/db-latest/etf_rag.db.zst"
)
DOWNLOAD_TIMEOUT = 300  # 5분


def ensure_db(db_path: Path) -> bool:
    """DB가 없으면 GitHub Release에서 다운로드. 성공 시 True, 실패/스킵 시 False."""
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        logger.info(f"DB 이미 존재: {db_path} ({size_mb:.0f}MB)")
        return True

    logger.info("DB 없음 — GitHub Release에서 다운로드 시작")
    zst_path = db_path.parent / "etf_rag.db.zst"

    try:
        # 1. 다운로드
        _download(DB_RELEASE_URL, zst_path)

        # 2. zstd 해제
        _decompress_zstd(zst_path, db_path)

        # 3. 압축 파일 정리
        zst_path.unlink(missing_ok=True)

        size_mb = db_path.stat().st_size / (1024 * 1024)
        logger.info(f"DB 다운로드 완료: {size_mb:.0f}MB")
        return True

    except Exception as e:
        logger.warning(f"DB 다운로드 실패 (deploy/ JSON fallback 사용): {e}")
        # 불완전 파일 정리
        zst_path.unlink(missing_ok=True)
        db_path.unlink(missing_ok=True)
        return False


def _download(url: str, dest: Path) -> None:
    """urllib로 파일 다운로드 (진행률 로깅)."""
    logger.info(f"다운로드: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        total_mb = total / (1024 * 1024) if total else 0
        logger.info(f"파일 크기: {total_mb:.0f}MB")

        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                # 50MB마다 진행률 로깅
                if downloaded % (50 * 1024 * 1024) < chunk_size:
                    logger.info(f"다운로드 진행: {downloaded / (1024*1024):.0f}/{total_mb:.0f}MB")

    logger.info(f"다운로드 완료: {downloaded / (1024*1024):.0f}MB")


def _decompress_zstd(src: Path, dest: Path) -> None:
    """zstandard으로 .zst 파일 해제."""
    import zstandard as zstd

    logger.info(f"zstd 해제: {src.name} → {dest.name}")
    dctx = zstd.ZstdDecompressor()
    with open(src, "rb") as ifh, open(dest, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)
    logger.info("zstd 해제 완료")
