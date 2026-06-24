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


def _is_valid_sqlite(db_path: Path) -> bool:
    """SQLite 파일이 열리고 손상되지 않았는지 빠르게 검증.

    PRAGMA quick_check는 integrity_check보다 빠르며(인덱스 무결성 생략) 대용량에서
    충분. 다운로드/압축해제가 중간에 끊긴 'malformed' 파일을 잡아낸다.
    """
    import sqlite3

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = con.execute("PRAGMA quick_check").fetchone()
            ok = bool(row) and row[0] == "ok"
            if ok:
                # 핵심 테이블 존재 + 최소 행 확인 (빈 껍데기 파일 방어)
                cnt = con.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
                ok = cnt > 0
            return ok
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001 — 손상 파일은 어떤 예외든 무효 처리
        logger.warning(f"SQLite 무결성 검사 실패(손상으로 간주): {e}")
        return False


# full DB(2014~, ~880만행)의 깊이 하한. 이보다 적으면 '얕은 DB'로 보고 재다운로드.
# 일일수집만으로 빈 DB에 쌓인 1년치(~105만행)를 잡아 full Release로 교체하기 위함.
# (full이 본 프로젝트 강점인 12년 시계열·재무제표를 담음 — 얕은 DB면 기간분석 불가)
_MIN_FULL_ROWS = 3_000_000


def _is_full_depth(db_path: Path) -> bool:
    """daily_prices 행수가 full DB 수준인지(얕은 1년치 DB 감지)."""
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cnt = con.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
            return cnt >= _MIN_FULL_ROWS
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return False


def ensure_db(db_path: Path) -> bool:
    """DB가 없으면 GitHub Release에서 다운로드. 성공 시 True, 실패/스킵 시 False.

    이미 존재하더라도 (1) 무결성 검사 — 손상(malformed)이면 재다운로드,
    (2) 깊이 검사 — 일일수집만으로 쌓인 얕은 1년치 DB면 full Release로 교체.
    """
    if db_path.exists():
        if _is_valid_sqlite(db_path):
            if _is_full_depth(db_path):
                size_mb = db_path.stat().st_size / (1024 * 1024)
                logger.info(f"DB 이미 존재(무결성·깊이 OK): {db_path} ({size_mb:.0f}MB)")
                return True
            logger.warning(
                "기존 DB가 얕음(full 미만, 일일수집 누적 추정) — full Release로 교체"
            )
            db_path.unlink(missing_ok=True)
        else:
            logger.warning("기존 DB 손상 감지 — 삭제 후 재다운로드")
            db_path.unlink(missing_ok=True)

    logger.info("DB 없음 — GitHub Release에서 다운로드 시작")
    zst_path = db_path.parent / "etf_rag.db.zst"

    try:
        # 1. 다운로드 (크기 검증 포함)
        _download(DB_RELEASE_URL, zst_path)

        # 2. zstd 해제
        _decompress_zstd(zst_path, db_path)

        # 3. 압축 파일 정리
        zst_path.unlink(missing_ok=True)

        # 4. 해제된 DB 무결성 검사 — 깨졌으면 실패로 처리(다음 부팅 재시도)
        if not _is_valid_sqlite(db_path):
            raise RuntimeError("다운로드한 DB 무결성 검사 실패 (malformed)")

        size_mb = db_path.stat().st_size / (1024 * 1024)
        logger.info(f"DB 다운로드 완료(무결성 OK): {size_mb:.0f}MB")
        return True

    except Exception as e:
        logger.warning(f"DB 다운로드 실패 (deploy/ JSON fallback 사용): {e}")
        # 불완전/손상 파일 정리 — 다음 부팅이 깨끗하게 재시도하도록
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

    # 다운로드 크기 검증 — Content-Length와 다르면 중간에 끊긴 것
    if total and downloaded != total:
        raise IOError(
            f"다운로드 불완전: {downloaded}/{total} bytes "
            f"(연결 중단 추정)"
        )
    logger.info(f"다운로드 완료: {downloaded / (1024*1024):.0f}MB")


def _decompress_zstd(src: Path, dest: Path) -> None:
    """zstandard으로 .zst 파일 해제."""
    import zstandard as zstd

    logger.info(f"zstd 해제: {src.name} → {dest.name}")
    dctx = zstd.ZstdDecompressor()
    with open(src, "rb") as ifh, open(dest, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)
    logger.info("zstd 해제 완료")
