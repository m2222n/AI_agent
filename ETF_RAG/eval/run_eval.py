"""
RAGAS 평가 파이프라인

사용법:
    cd ETF_RAG
    .venv/bin/python eval/run_eval.py              # 전체 평가
    .venv/bin/python eval/run_eval.py --no-llm     # 검색만 평가 (API 비용 없음)

평가 지표:
    - Context Precision: 검색된 문서 중 관련 문서 비율
    - Context Recall: 정답에 필요한 정보가 검색되었는지
    - Faithfulness: LLM 답변이 검색 결과에 근거하는지 (할루시네이션 방어)
    - Answer Relevancy: LLM 답변이 질문에 적절한지
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TOP_K_RESULTS
from src.data.loader import load_etf_data, load_stock_data, create_documents, create_stock_documents
from src.rag.vectorstore import create_vectorstore
from src.rag.retriever import HybridRetriever, retrieve_relevant_docs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_eval_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def init_retriever():
    """HybridRetriever 초기화 (ETF + 주식)"""
    logger.info("데이터 로드 + 벡터스토어 + BM25 인덱스 구축 중...")

    # ETF 문서
    etf_data = load_etf_data()
    etf_documents = create_documents(etf_data, include_pdfs=False)

    # 주식 문서
    stock_data = load_stock_data()
    stock_documents = create_stock_documents(stock_data)

    # ETF retriever
    etf_vectorstore = create_vectorstore(etf_documents)
    etf_retriever = HybridRetriever(etf_vectorstore, etf_documents)

    # 주식 retriever (주식 데이터 있을 때만)
    stock_retriever = None
    if stock_documents:
        stock_vectorstore = create_vectorstore(stock_documents)
        stock_retriever = HybridRetriever(stock_vectorstore, stock_documents)

    logger.info(f"초기화 완료: ETF {len(etf_documents)}개 + 주식 {len(stock_documents)}개 문서")
    return etf_retriever, stock_retriever


def evaluate_retrieval(etf_retriever, stock_retriever, dataset):
    """검색 품질 평가 (API 비용 없음)"""
    results = []

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_ticker = item.get("expected_ticker")
        expect_no_context = item.get("expect_no_context", False)
        asset_type = item.get("asset_type", "etf")

        # asset_type에 따라 적절한 retriever 선택
        if asset_type == "stock" and stock_retriever:
            context, sources = retrieve_relevant_docs(stock_retriever, question)
        elif asset_type == "mixed":
            # ETF + 주식 모두 검색, 결과 합산
            context_etf, sources_etf = retrieve_relevant_docs(etf_retriever, question)
            context_stock, sources_stock = (
                retrieve_relevant_docs(stock_retriever, question)
                if stock_retriever else (None, [])
            )
            sources = (sources_etf or []) + (sources_stock or [])
            parts = [c for c in [context_etf, context_stock] if c]
            context = "\n\n".join(parts) if parts else None
        else:
            context, sources = retrieve_relevant_docs(etf_retriever, question)

        # 검색된 티커 목록
        retrieved_tickers = [s["ticker"] for s in sources] if sources else []

        # 평가
        if expect_no_context:
            # "데이터 없음" 응답이 예상되는 경우
            hit = context is None or len(sources) == 0
            precision = 1.0 if hit else 0.0
            recall = 1.0 if hit else 0.0
        elif expected_ticker is None:
            # 특정 티커를 기대하지 않는 일반/추천 질문
            hit = context is not None
            precision = 1.0 if hit else 0.0
            recall = 1.0 if hit else 0.0
        elif isinstance(expected_ticker, list):
            # 복수 티커 (비교 질문)
            found = [t for t in expected_ticker if t in retrieved_tickers]
            hit = len(found) > 0
            precision = len(found) / len(retrieved_tickers) if retrieved_tickers else 0.0
            recall = len(found) / len(expected_ticker)
        else:
            # 단일 티커
            hit = expected_ticker in retrieved_tickers
            precision = 1.0 / len(retrieved_tickers) if hit and retrieved_tickers else 0.0
            recall = 1.0 if hit else 0.0

        result = {
            "question": question,
            "question_type": item["question_type"],
            "asset_type": asset_type,
            "expected_ticker": expected_ticker,
            "retrieved_tickers": retrieved_tickers,
            "hit": hit,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "num_sources": len(sources) if sources else 0,
        }
        results.append(result)

        status = "✓" if hit else "✗"
        logger.info(f"  [{status}] Q{i+1}: {question[:40]}... → {retrieved_tickers[:3]}")

    return results


def evaluate_with_ragas(etf_retriever, dataset):
    """RAGAS 전체 평가 (LLM API 사용, ETF만 대상)"""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
    except ImportError:
        logger.error("ragas 또는 datasets 패키지 미설치. pip install ragas datasets")
        return None

    from src.llm.client import get_api_key, create_client, call_llm_streaming
    from src.llm.classifier import classify_question_type

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in dataset:
        question = item["question"]

        # 검색
        context, sources = retrieve_relevant_docs(etf_retriever, question)

        # LLM 답변 생성 (비스트리밍)
        api_key = get_api_key()
        client = create_client(api_key)
        question_type = classify_question_type(question)

        from openai import OpenAI
        from src.llm.prompts import build_system_prompt
        from config import LLM_MODEL, LLM_TEMPERATURE

        system_prompt = build_system_prompt(question_type)
        if context:
            user_msg = f"[검색된 ETF 문서]\n{context}\n\n[사용자 질문]\n{question}\n\n위 문서를 참고하여 답변해줘."
        else:
            user_msg = f"[시스템 알림] 관련 ETF 문서를 찾지 못했습니다.\n\n[사용자 질문]\n{question}"

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=LLM_TEMPERATURE,
        )
        answer = response.choices[0].message.content

        questions.append(question)
        answers.append(answer)
        contexts.append([context] if context else ["관련 문서 없음"])
        ground_truths.append(item["ground_truth"])

        logger.info(f"  Q: {question[:40]}... → 답변 {len(answer)}자")

    # RAGAS Dataset 구성
    eval_data = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # RAGAS 평가 실행
    result = evaluate(
        eval_data,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return result


def save_results(retrieval_results, ragas_results=None):
    """평가 결과 저장"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 검색 평가 요약
    total = len(retrieval_results)
    hits = sum(1 for r in retrieval_results if r["hit"])
    avg_precision = sum(r["precision"] for r in retrieval_results) / total
    avg_recall = sum(r["recall"] for r in retrieval_results) / total

    # 유형별 통계
    type_stats = {}
    asset_stats = {}
    for r in retrieval_results:
        qt = r["question_type"]
        if qt not in type_stats:
            type_stats[qt] = {"total": 0, "hits": 0}
        type_stats[qt]["total"] += 1
        if r["hit"]:
            type_stats[qt]["hits"] += 1

        # asset_type별 통계
        at = r.get("asset_type", "etf")
        if at not in asset_stats:
            asset_stats[at] = {"total": 0, "hits": 0}
        asset_stats[at]["total"] += 1
        if r["hit"]:
            asset_stats[at]["hits"] += 1

    summary = {
        "timestamp": timestamp,
        "total_questions": total,
        "retrieval": {
            "hit_rate": round(hits / total, 3),
            "avg_precision": round(avg_precision, 3),
            "avg_recall": round(avg_recall, 3),
            "by_type": {
                k: {
                    "hit_rate": round(v["hits"] / v["total"], 3),
                    "total": v["total"],
                    "hits": v["hits"],
                }
                for k, v in type_stats.items()
            },
            "by_asset": {
                k: {
                    "hit_rate": round(v["hits"] / v["total"], 3),
                    "total": v["total"],
                    "hits": v["hits"],
                }
                for k, v in asset_stats.items()
            },
        },
        "details": retrieval_results,
    }

    if ragas_results:
        summary["ragas"] = {str(k): float(v) for k, v in ragas_results.items()}

    output_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_path, summary


def print_summary(summary):
    """평가 결과 요약 출력"""
    r = summary["retrieval"]
    print("\n" + "=" * 60)
    print("📊 ETF RAG 검색 평가 결과")
    print("=" * 60)
    print(f"  총 질문: {summary['total_questions']}개")
    print(f"  Hit Rate: {r['hit_rate']:.1%}")
    print(f"  Avg Precision: {r['avg_precision']:.3f}")
    print(f"  Avg Recall: {r['avg_recall']:.3f}")
    print()
    print("  유형별 Hit Rate:")
    for qt, stats in r["by_type"].items():
        print(f"    {qt:12s}: {stats['hit_rate']:.1%} ({stats['hits']}/{stats['total']})")

    if "by_asset" in r:
        print()
        print("  자산유형별 Hit Rate:")
        for at, stats in r["by_asset"].items():
            print(f"    {at:12s}: {stats['hit_rate']:.1%} ({stats['hits']}/{stats['total']})")

    if "ragas" in summary:
        print()
        print("  RAGAS 지표:")
        for metric, score in summary["ragas"].items():
            print(f"    {metric:25s}: {score:.3f}")

    print("=" * 60)


def main():
    no_llm = "--no-llm" in sys.argv

    logger.info("평가 데이터셋 로드...")
    dataset = load_eval_dataset()
    logger.info(f"  {len(dataset)}개 질문")

    logger.info("Retriever 초기화...")
    start = time.time()
    etf_retriever, stock_retriever = init_retriever()
    init_time = time.time() - start
    logger.info(f"  초기화 시간: {init_time:.1f}초")

    # 1. 검색 평가 (무료)
    logger.info("\n검색 품질 평가 시작...")
    retrieval_results = evaluate_retrieval(etf_retriever, stock_retriever, dataset)

    # 2. RAGAS 전체 평가 (유료)
    ragas_results = None
    if not no_llm:
        logger.info("\nRAGAS 전체 평가 시작 (LLM API 사용)...")
        ragas_results = evaluate_with_ragas(etf_retriever, dataset)

    # 결과 저장
    output_path, summary = save_results(retrieval_results, ragas_results)
    print_summary(summary)
    logger.info(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
