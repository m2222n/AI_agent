"""FAISS 벡터스토어 persist 테스트"""
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from src.rag.vectorstore import (
    _compute_docs_hash,
    create_vectorstore,
    FAISS_INDEX_DIR,
    _get_index_path,
    _read_hash_file,
    _write_hash_file,
)


# ── 해시 함수 테스트 ──────────────────────────────────────

def test_compute_docs_hash_consistent():
    """동일 문서 → 동일 해시"""
    docs = [Document(page_content="삼성전자"), Document(page_content="SK하이닉스")]
    h1 = _compute_docs_hash(docs)
    h2 = _compute_docs_hash(docs)
    assert h1 == h2
    assert len(h1) == 16


def test_compute_docs_hash_different():
    """다른 문서 → 다른 해시"""
    docs1 = [Document(page_content="삼성전자")]
    docs2 = [Document(page_content="LG전자")]
    assert _compute_docs_hash(docs1) != _compute_docs_hash(docs2)


def test_compute_docs_hash_empty():
    """빈 문서 목록도 해시 생성"""
    h = _compute_docs_hash([])
    assert isinstance(h, str)
    assert len(h) == 16


def test_compute_docs_hash_order_matters():
    """문서 순서가 다르면 해시 다름"""
    docs1 = [Document(page_content="A"), Document(page_content="B")]
    docs2 = [Document(page_content="B"), Document(page_content="A")]
    assert _compute_docs_hash(docs1) != _compute_docs_hash(docs2)


# ── 해시 파일 읽기/쓰기 ──────────────────────────────────

def test_hash_file_roundtrip(tmp_path):
    """해시 파일 쓰기 → 읽기"""
    _write_hash_file(tmp_path, "abc123")
    assert _read_hash_file(tmp_path) == "abc123"


def test_read_hash_file_missing(tmp_path):
    """해시 파일 없으면 None"""
    assert _read_hash_file(tmp_path / "nonexistent") is None


# ── 인덱스 경로 ──────────────────────────────────────────

def test_get_index_path():
    """prefix로 인덱스 경로 생성"""
    p = _get_index_path("etf")
    assert p == FAISS_INDEX_DIR / "etf"


# ── create_vectorstore 캐시 로직 ──────────────────────────

@patch("src.rag.vectorstore.FAISS")
@patch("src.rag.vectorstore.get_embeddings")
def test_create_vectorstore_cache_miss(mock_emb, mock_faiss, tmp_path):
    """캐시 미스 → from_documents 호출 + save_local"""
    with patch("src.rag.vectorstore.FAISS_INDEX_DIR", tmp_path):
        docs = [Document(page_content="테스트")]
        mock_vs = MagicMock()
        mock_faiss.from_documents.return_value = mock_vs

        result = create_vectorstore(docs, prefix="test")

        mock_faiss.from_documents.assert_called_once()
        mock_vs.save_local.assert_called_once()
        assert result == mock_vs


@patch("src.rag.vectorstore.FAISS")
@patch("src.rag.vectorstore.get_embeddings")
def test_create_vectorstore_cache_hit(mock_emb, mock_faiss, tmp_path):
    """캐시 히트 → load_local 호출, from_documents 미호출"""
    with patch("src.rag.vectorstore.FAISS_INDEX_DIR", tmp_path):
        docs = [Document(page_content="테스트")]
        h = _compute_docs_hash(docs)

        # 캐시 디렉토리 + 해시 파일 생성
        idx_dir = tmp_path / "test"
        idx_dir.mkdir()
        _write_hash_file(idx_dir, h)
        # index.faiss 파일도 있어야 exists() 통과
        (idx_dir / "index.faiss").touch()

        mock_vs = MagicMock()
        mock_faiss.load_local.return_value = mock_vs

        result = create_vectorstore(docs, prefix="test")

        mock_faiss.load_local.assert_called_once()
        mock_faiss.from_documents.assert_not_called()
        assert result == mock_vs


@patch("src.rag.vectorstore.FAISS")
@patch("src.rag.vectorstore.get_embeddings")
def test_create_vectorstore_hash_mismatch(mock_emb, mock_faiss, tmp_path):
    """해시 불일치 → from_documents 재생성"""
    with patch("src.rag.vectorstore.FAISS_INDEX_DIR", tmp_path):
        docs = [Document(page_content="새 데이터")]

        # 다른 해시로 캐시 생성
        idx_dir = tmp_path / "test"
        idx_dir.mkdir()
        _write_hash_file(idx_dir, "old_hash_value")

        mock_vs = MagicMock()
        mock_faiss.from_documents.return_value = mock_vs

        result = create_vectorstore(docs, prefix="test")

        mock_faiss.from_documents.assert_called_once()
        mock_faiss.load_local.assert_not_called()
