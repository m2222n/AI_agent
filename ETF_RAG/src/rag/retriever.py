from typing import List, Tuple, Optional

from langchain_community.vectorstores import FAISS

from config import SIMILARITY_THRESHOLD, TOP_K_RESULTS


def retrieve_relevant_docs(
    vectorstore: FAISS, query: str, k: int = TOP_K_RESULTS
) -> Tuple[Optional[str], List[dict]]:
    """
    벡터 DB에서 관련 문서 검색

    Returns:
        context: 검색된 문서 내용 (문자열) 또는 None
        sources: 출처 정보 리스트
    """
    results = vectorstore.similarity_search_with_score(query, k=k)

    filtered_results = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]

    if not filtered_results:
        return None, []

    context_parts = []
    sources = []

    for doc, score in filtered_results:
        context_parts.append(f"[{doc.metadata['id']}] {doc.page_content}")
        sources.append({
            "id": doc.metadata["id"],
            "name": doc.metadata["name"],
            "ticker": doc.metadata["ticker"],
            "relevance_score": round(1 - score / 2, 2)
        })

    context = "\n\n---\n\n".join(context_parts)
    return context, sources
