"""
FAISS 벡터스토어 생성 + 디스크 persist

- 데이터 해시 기반 캐시 무효화: 데이터가 변경되면 인덱스 재생성
- save_local / load_local로 디스크 캐싱 → 앱 재시작 시 임베딩 API 재호출 없음
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL, DATA_DIR

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


def create_vectorstore(
    documents: List[Document],
    prefix: str = "default",
) -> FAISS:
    """
    Document 목록으로 FAISS 벡터 DB 생성 (디스크 캐싱 포함)

    1. 문서 해시 계산
    2. 캐시된 인덱스가 있고 해시 일치 → load_local
    3. 없거나 해시 불일치 → from_documents + save_local
    """
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
