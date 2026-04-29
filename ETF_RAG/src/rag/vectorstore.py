"""
벡터스토어 생성 — FAISS (로컬) / Pinecone (서버리스) 선택 가능

- VECTOR_DB_BACKEND="faiss" (기본): 디스크 캐싱 + MD5 해시 무효화
- VECTOR_DB_BACKEND="pinecone": Pinecone 서버리스 (free tier)
- Pinecone 실패 시 FAISS로 자동 fallback
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    EMBEDDING_MODEL, DATA_DIR,
    VECTOR_DB_BACKEND, PINECONE_API_KEY, PINECONE_INDEX_NAME,
)

logger = logging.getLogger(__name__)

# FAISS 인덱스 저장 디렉토리
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"


def get_embeddings() -> OpenAIEmbeddings:
    """임베딩 모델 인스턴스 반환"""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _compute_docs_hash(documents: List[Document]) -> str:
    """문서 목록의 해시 계산 (캐시 무효화용)"""
    hasher = hashlib.md5()
    for doc in documents:
        hasher.update(doc.page_content.encode("utf-8"))
    return hasher.hexdigest()[:16]


def _get_index_path(prefix: str) -> Path:
    """인덱스 저장 경로 반환"""
    return FAISS_INDEX_DIR / prefix


def _read_hash_file(index_path: Path) -> Optional[str]:
    """저장된 해시 파일 읽기"""
    hash_file = index_path / "docs_hash.txt"
    if hash_file.exists():
        return hash_file.read_text().strip()
    return None


def _write_hash_file(index_path: Path, docs_hash: str):
    """해시 파일 쓰기"""
    hash_file = index_path / "docs_hash.txt"
    hash_file.write_text(docs_hash)


# ══════════════════════════════════════════════════════════════
# FAISS 백엔드
# ══════════════════════════════════════════════════════════════

def _create_faiss_vectorstore(
    documents: List[Document],
    prefix: str = "default",
) -> FAISS:
    """FAISS 벡터 DB 생성 (디스크 캐싱 포함)"""
    docs_hash = _compute_docs_hash(documents)
    index_path = _get_index_path(prefix)
    embeddings = get_embeddings()

    # 캐시 히트 확인
    if index_path.exists():
        saved_hash = _read_hash_file(index_path)
        if saved_hash == docs_hash:
            try:
                vs = FAISS.load_local(
                    str(index_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"FAISS 인덱스 캐시 히트: {prefix} ({len(documents)}문서)")
                return vs
            except Exception as e:
                logger.warning(f"FAISS 캐시 로드 실패, 재생성: {e}")
        else:
            logger.info(f"FAISS 해시 불일치 ({prefix}): {saved_hash} → {docs_hash}")

    # 캐시 미스: 새로 생성
    logger.info(f"FAISS 인덱스 생성: {prefix} ({len(documents)}문서)")
    vs = FAISS.from_documents(documents=documents, embedding=embeddings)

    # 디스크에 저장
    try:
        index_path.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(index_path))
        _write_hash_file(index_path, docs_hash)
        logger.info(f"FAISS 인덱스 저장: {index_path}")
    except Exception as e:
        logger.warning(f"FAISS 인덱스 저장 실패 (무시): {e}")

    return vs


# ══════════════════════════════════════════════════════════════
# Pinecone 백엔드
# ══════════════════════════════════════════════════════════════

def _create_pinecone_vectorstore(
    documents: List[Document],
    prefix: str = "default",
):
    """Pinecone 서버리스 벡터 DB 생성/연결.

    - 인덱스 없으면 자동 생성 (dimension=1536, cosine)
    - 벡터 수로 업데이트 필요 여부 판단
    """
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY 환경변수가 설정되지 않았습니다.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = f"{PINECONE_INDEX_NAME}-{prefix}"

    # 인덱스 존재 확인 / 생성
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        logger.info(f"Pinecone 인덱스 생성: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    embeddings = get_embeddings()
    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    # 네임스페이스에 데이터가 있고 벡터 수가 문서 수와 유사하면 스킵
    ns_vectors = stats.get("namespaces", {}).get(prefix, {}).get("vector_count", 0)
    if ns_vectors > 0 and abs(ns_vectors - len(documents)) < 10:
        logger.info(f"Pinecone 캐시 히트: {index_name}/{prefix} ({ns_vectors}벡터)")
        return PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=prefix,
        )

    # 기존 네임스페이스 삭제 후 재업로드
    if ns_vectors > 0:
        logger.info(f"Pinecone 데이터 갱신: {index_name}/{prefix} ({ns_vectors} → {len(documents)})")
        index.delete(delete_all=True, namespace=prefix)

    logger.info(f"Pinecone 문서 업로드: {index_name}/{prefix} ({len(documents)}문서)")
    vs = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
        namespace=prefix,
    )

    return vs


# ══════════════════════════════════════════════════════════════
# 통합 인터페이스
# ══════════════════════════════════════════════════════════════

def create_vectorstore(
    documents: List[Document],
    prefix: str = "default",
    backend: Optional[str] = None,
):
    """
    Document 목록으로 벡터 DB 생성 (백엔드 자동 선택)

    Args:
        documents: 문서 리스트
        prefix: 인덱스 접두사 (ETF/주식 구분)
        backend: "faiss" or "pinecone" (None이면 config에서 결정)
    """
    backend = backend or VECTOR_DB_BACKEND

    if backend == "pinecone" and PINECONE_API_KEY:
        try:
            return _create_pinecone_vectorstore(documents, prefix)
        except Exception as e:
            logger.warning(f"Pinecone 실패, FAISS로 fallback: {e}")
            return _create_faiss_vectorstore(documents, prefix)
    else:
        return _create_faiss_vectorstore(documents, prefix)
