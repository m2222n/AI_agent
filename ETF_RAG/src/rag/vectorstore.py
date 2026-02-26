from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vectorstore(documents: List[Document]) -> FAISS:
    """Document 목록으로 FAISS 벡터 DB 생성"""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents=documents, embedding=embeddings)
