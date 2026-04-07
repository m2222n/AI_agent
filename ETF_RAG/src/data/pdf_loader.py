"""
ETF 투자설명서 PDF 파싱 파이프라인

사용법:
    from src.data.pdf_loader import load_pdf_documents
    docs = load_pdf_documents()  # PDF_DIR 내 모든 PDF → chunked Documents

파이프라인:
    1. PDF_DIR/*.pdf 스캔
    2. PyPDFLoader로 페이지별 로드
    3. RecursiveCharacterTextSplitter로 청킹
    4. 메타데이터 태깅 (ETF명, 문서유형, 파일명)
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# PDF 디렉토리 (ETF_RAG/src/data/pdfs/)
PDF_DIR = Path(__file__).parent / "pdfs"

# Chunking 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def load_pdf_documents(pdf_dir: Optional[Path] = None) -> List[Document]:
    """PDF 디렉토리의 모든 PDF를 로드하고 청킹하여 Document 리스트 반환.

    Args:
        pdf_dir: PDF 파일 디렉토리 (기본: src/data/pdfs/)

    Returns:
        청킹된 Document 리스트 (비어있으면 빈 리스트)
    """
    if pdf_dir is None:
        pdf_dir = PDF_DIR

    if not pdf_dir.exists():
        logger.info(f"PDF 디렉토리 없음: {pdf_dir}")
        return []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"PDF 파일 없음: {pdf_dir}")
        return []

    # lazy import — PDF 처리 시에만 의존성 필요
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as e:
        logger.warning(f"PDF 처리 패키지 미설치: {e}")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,  # 추후 tiktoken 기반으로 교체 가능
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_docs = []
    for pdf_path in pdf_files:
        try:
            docs = _load_single_pdf(pdf_path, splitter, PyPDFLoader)
            all_docs.extend(docs)
            logger.info(f"PDF 로드: {pdf_path.name} → {len(docs)}개 청크")
        except Exception as e:
            logger.warning(f"PDF 로드 실패: {pdf_path.name} — {e}")

    logger.info(f"총 PDF 문서: {len(all_docs)}개 청크 ({len(pdf_files)}개 파일)")
    return all_docs


def _load_single_pdf(pdf_path: Path, splitter, loader_cls) -> List[Document]:
    """단일 PDF를 로드하고 청킹.

    파일명에서 ETF 정보를 추출하여 메타데이터에 태깅.
    파일명 규칙: {ticker}_{ETF명}_{문서유형}.pdf
    예: 069500_KODEX200_투자설명서.pdf
    """
    loader = loader_cls(str(pdf_path))
    pages = loader.load()

    # 청킹
    chunks = splitter.split_documents(pages)

    # 파일명에서 메타데이터 추출
    file_meta = _extract_file_metadata(pdf_path.stem)

    # 각 청크에 메타데이터 추가
    for chunk in chunks:
        chunk.metadata.update({
            "source": "pdf",
            "file_name": pdf_path.name,
            **file_meta,
        })

    return chunks


def _extract_file_metadata(file_stem: str) -> dict:
    """파일명에서 ETF 메타데이터 추출.

    파일명 패턴: {ticker}_{name}_{doc_type}
    예: "069500_KODEX200_투자설명서" → {"ticker": "069500", "name": "KODEX200", "doc_type": "투자설명서"}
    """
    parts = file_stem.split("_", maxsplit=2)

    meta = {}
    if len(parts) >= 1 and re.match(r"^\d{6}$", parts[0]):
        meta["ticker"] = parts[0]
    if len(parts) >= 2:
        meta["name"] = parts[1]
    if len(parts) >= 3:
        meta["doc_type"] = parts[2]

    return meta
