"""RAG 모듈 공유 유틸리티"""

import hashlib
from typing import List

from langchain_core.documents import Document


def compute_docs_hash(documents: List[Document]) -> str:
    """문서 목록의 MD5 해시 계산 (캐시 무효화용).

    FAISS 벡터스토어와 BM25 인덱스 양쪽에서 공통 사용.
    """
    hasher = hashlib.md5()
    for doc in documents:
        hasher.update(doc.page_content.encode("utf-8"))
    return hasher.hexdigest()[:16]
