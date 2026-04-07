from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL


def get_embeddings() -> OpenAIEmbeddings:
    """임베딩 모델 인스턴스 반환"""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def create_vectorstore(documents: List[Document]) -> FAISS:
    """Document 목록으로 FAISS 벡터 DB 생성"""
    return FAISS.from_documents(documents=documents, embedding=get_embeddings())
