"""
GitHub Release에서 SQLite DB 다운로드 (Streamlit Cloud 시작 시 1회)

DB가 이미 로컬에 있으면 건너뜀. 없으면 GitHub Release의 zstd 압축 DB를 다운로드/해제.
다운로드 실패 시 deploy/ JSON fallback이 동작하도록 예외를 삼킴.
"""

import logging
import os
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# DB 최신 데이터가 이 일수(달력일)보다 오래되면 stale로 보고 Release에서 재다운로드.
# 영속 볼륨 환경(Railway)에서 받은 DB가 굳어 날짜가 뒤처지던 문제 대응.
# 3일: 평일 1~2일 지연은 통과, 영업일 누락(3일+)이면 갱신. 주말(금→월=3일)은
# 경계라 드물게 재다운 가능하나 허용. 평소엔 재다운 안 함(콜드스타트 유지).
# DB_MAX_STALE_DAYS=0이면 비활성.
_STALE_DAYS_DEFAULT = 3

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


def _max_stale_days() -> int:
    """DB_MAX_STALE_DAYS 환경변수(없으면 기본 5, 0이면 비활성)."""
    try:
        return int(os.getenv("DB_MAX_STALE_DAYS", str(_STALE_DAYS_DEFAULT)))
    except (TypeError, ValueError):
        return _STALE_DAYS_DEFAULT


def _is_fresh_enough(db_path: Path) -> bool:
    """DB 최신 daily_prices.date가 stale 임계 안인지. 비활성(0)이면 항상 True.

    영속 볼륨에 굳은 DB가 Release 갱신을 못 따라가는 문제 감지용.
    """
    max_days = _max_stale_days()
    if max_days <= 0:
        return True
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT MAX(date) FROM daily_prices").fetchone()
        finally:
            con.close()
        latest = row[0] if row else None
        if not latest:
            return False
        latest_dt = datetime.strptime(str(latest), "%Y%m%d").date()
        age = (datetime.now(KST).date() - latest_dt).days
        if age > max_days:
            logger.warning(f"DB 최신일 {latest} ({age}일 전) — stale 임계 {max_days}일 초과")
            return False
        return True
    except Exception:  # noqa: BLE001 — 판단 불가 시 보수적으로 fresh 취급(재다운 안 함)
        return True


def ensure_db(db_path: Path) -> bool:
    """DB가 없으면 GitHub Release에서 다운로드. 성공 시 True, 실패/스킵 시 False.

    이미 존재하더라도 (1) 무결성 검사 — 손상(malformed)이면 재다운로드,
    (2) 깊이 검사 — 일일수집만으로 쌓인 얕은 1년치 DB면 full Release로 교체,
    (3) 신선도 검사 — 최신일이 stale 임계(DB_MAX_STALE_DAYS, 기본 5일) 초과면
        Release(매일 갱신)로 재다운로드. 영속 볼륨에서 DB가 굳는 문제 대응.
    """
    if db_path.exists():
        if _is_valid_sqlite(db_path):
            if not _is_full_depth(db_path):
                logger.warning(
                    "기존 DB가 얕음(full 미만, 일일수집 누적 추정) — full Release로 교체"
                )
                db_path.unlink(missing_ok=True)
            elif not _is_fresh_enough(db_path):
                logger.warning("기존 DB가 오래됨(stale) — 최신 Release로 교체")
                db_path.unlink(missing_ok=True)
            else:
                size_mb = db_path.stat().st_size / (1024 * 1024)
                logger.info(f"DB 이미 존재(무결성·깊이·신선도 OK): {db_path} ({size_mb:.0f}MB)")
                return True
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
