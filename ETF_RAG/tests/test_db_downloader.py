"""db_downloader 무결성 검사 테스트 (2026-06-19).

Railway에서 'database disk image is malformed'로 주가 DB 로드가 실패하던 문제 →
ensure_db가 파일 존재만 보고 손상 파일을 재사용하던 것이 원인. 무결성 검사 추가분 검증.
실제 네트워크 다운로드는 mock(여기선 무결성 로직만 검증).
"""

import sqlite3
from pathlib import Path

from src.data.db_downloader import _is_valid_sqlite, ensure_db


def _make_valid_db(path: Path) -> None:
    """daily_prices 테이블 + 1행을 가진 최소 정상 SQLite 생성."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE daily_prices (ticker TEXT, date TEXT, close INTEGER)")
    con.execute("INSERT INTO daily_prices VALUES ('005930', '20260618', 70000)")
    con.commit()
    con.close()


def test_valid_sqlite_passes(tmp_path):
    db = tmp_path / "ok.db"
    _make_valid_db(db)
    assert _is_valid_sqlite(db) is True


def test_corrupt_file_fails(tmp_path):
    db = tmp_path / "bad.db"
    db.write_bytes(b"SQLite format 3\x00" + b"\xde\xad\xbe\xef" * 500)
    assert _is_valid_sqlite(db) is False


def test_empty_file_fails(tmp_path):
    db = tmp_path / "empty.db"
    db.write_bytes(b"")
    assert _is_valid_sqlite(db) is False


def test_missing_file_fails(tmp_path):
    assert _is_valid_sqlite(tmp_path / "nope.db") is False


def test_valid_sqlite_without_daily_prices_fails(tmp_path):
    """SQLite로는 멀쩡하나 핵심 테이블이 없으면(빈 껍데기) 무효."""
    db = tmp_path / "shell.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE other (x INTEGER)")
    con.commit()
    con.close()
    assert _is_valid_sqlite(db) is False


def test_ensure_db_skips_when_valid_and_deep(tmp_path, monkeypatch):
    """유효 + 깊이 충분(full)이면 다운로드 없이 True."""
    db = tmp_path / "etf_rag.db"
    _make_valid_db(db)

    called = {"download": False}

    def _fake_download(url, dest):
        called["download"] = True

    monkeypatch.setattr("src.data.db_downloader._download", _fake_download)
    monkeypatch.setattr("src.data.db_downloader._is_full_depth", lambda p: True)
    assert ensure_db(db) is True
    assert called["download"] is False  # 재다운로드 안 함


def test_ensure_db_redownloads_when_shallow(tmp_path, monkeypatch):
    """유효하나 얕은 DB(일일수집 1년치)면 full Release로 재다운로드."""
    db = tmp_path / "etf_rag.db"
    _make_valid_db(db)  # 유효하지만 행수 적음 → _is_full_depth False

    attempted = {"download": False}

    def _fake_download(url, dest):
        attempted["download"] = True
        raise IOError("network down")  # 다운로드 분기 진입만 확인

    monkeypatch.setattr("src.data.db_downloader._download", _fake_download)
    # 실제 _is_full_depth로 검증(소수 행 DB는 얕음 판정)
    result = ensure_db(db)
    assert attempted["download"] is True  # 얕음 감지 → full 재다운로드 시도
    assert result is False  # 다운로드 실패 → fallback
    assert not db.exists()  # 얕은 DB 삭제됨


def test_is_full_depth(tmp_path):
    """행수 임계 미만이면 얕음(False)."""
    from src.data.db_downloader import _is_full_depth
    db = tmp_path / "shallow.db"
    _make_valid_db(db)  # 1행
    assert _is_full_depth(db) is False  # _MIN_FULL_ROWS 미만


def test_ensure_db_redownloads_when_corrupt(tmp_path, monkeypatch):
    """기존 파일이 손상이면 삭제 후 재다운로드 경로를 탄다."""
    db = tmp_path / "etf_rag.db"
    db.write_bytes(b"SQLite format 3\x00garbage")  # 손상

    attempted = {"download": False}

    def _fake_download(url, dest):
        attempted["download"] = True
        raise IOError("network down")  # 다운로드 자체는 실패시킴(무결성 분기만 확인)

    monkeypatch.setattr("src.data.db_downloader._download", _fake_download)
    result = ensure_db(db)
    assert attempted["download"] is True  # 손상 감지 → 재다운로드 시도함
    assert result is False  # 다운로드 실패 → JSON fallback
    assert not db.exists()  # 손상 파일 정리됨


def test_ensure_db_fails_if_downloaded_db_corrupt(tmp_path, monkeypatch):
    """다운로드/해제는 됐지만 결과가 손상이면 False + 파일 정리."""
    db = tmp_path / "etf_rag.db"

    def _fake_download(url, dest):
        Path(dest).write_bytes(b"compressed")  # 더미 zst

    def _fake_decompress(src, dest):
        Path(dest).write_bytes(b"SQLite format 3\x00broken")  # 손상된 결과물

    monkeypatch.setattr("src.data.db_downloader._download", _fake_download)
    monkeypatch.setattr("src.data.db_downloader._decompress_zstd", _fake_decompress)
    assert ensure_db(db) is False
    assert not db.exists()
