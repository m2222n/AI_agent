"""
RAGAS 평가 파이프라인

사용법:
    cd ETF_RAG
    .venv/bin/python eval/run_eval.py              # 전체 평가 (검색 + RAGAS)
    .venv/bin/python eval/run_eval.py --no-llm     # 검색만 평가 (API 비용 없음)
    .venv/bin/python eval/run_eval.py --sample 10  # 샘플 N개만 RAGAS 평가 (비용 절감)

평가 지표:
    - Hit Rate / Precision / Recall: 검색 품질 (API 비용 없음)
    - Faithfulness: LLM 답변이 검색 결과에 근거하는지 (할루시네이션 방어)
    - Answer Relevancy: LLM 답변이 질문에 적절한지
    - Context Recall: 정답에 필요한 정보가 검색되었는지
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


def evaluate_with_ragas(etf_retriever, stock_retriever, dataset, sample_size=None):
    """
    RAGAS 전체 평가 — 에이전트 기반 답변 생성 + Faithfulness/Answer Relevancy 측정

    Args:
        etf_retriever: ETF HybridRetriever
        stock_retriever: 주식 HybridRetriever (없으면 None)
        dataset: 평가 데이터셋 리스트
        sample_size: 평가할 질문 수 (None이면 전체)

    Returns:
        {"scores": {...}, "per_question": [...]}
    """
    try:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LCOpenAIEmbeddings
    except ImportError:
        logger.error("ragas 패키지 미설치. pip install ragas")
        return None

    from src.llm.agent import run_agent

    # 샘플링 — stratified: 각 question_type에서 균등하게 추출
    if sample_size and sample_size < len(dataset):
        from collections import defaultdict
        import random
        random.seed(42)
        by_type = defaultdict(list)
        for item in dataset:
            by_type[item["question_type"]].append(item)
        per_type = max(1, sample_size // len(by_type))
        eval_items = []
        for qt, items in sorted(by_type.items()):
            eval_items.extend(random.sample(items, min(per_type, len(items))))
        # 부족분 채우기
        remaining = [x for x in dataset if x not in eval_items]
        random.shuffle(remaining)
        eval_items.extend(remaining[:max(0, sample_size - len(eval_items))])
        eval_items = eval_items[:sample_size]
    else:
        eval_items = dataset

    samples = []
    per_question_details = []

    for i, item in enumerate(eval_items):
        question = item["question"]
        ground_truth = item["ground_truth"]
        asset_type = item.get("asset_type", "etf")

        logger.info(f"  [{i+1}/{len(eval_items)}] {question[:50]}...")

        # 1. 에이전트로 답변 생성 (실제 서비스와 동일 경로)
        #    도구 호출 결과를 context로 추출하기 위해 full state 사용
        #    차트 생성 비활성화 (base64가 context를 초과시킴)
        tool_contexts = []
        try:
            import src.data.chart_generator as _cg
            _cg_orig = _cg.generate_technical_chart
            _cg.generate_technical_chart = lambda *a, **kw: None

            from src.llm.agent import (
                get_agent, classify_with_llm, build_system_prompt,
                COMPLEX_TYPES, AgentState,
            )
            from langchain_core.messages import (
                SystemMessage, HumanMessage, AIMessage, ToolMessage,
            )

            agent = get_agent()
            question_type = classify_with_llm(question)
            system_prompt = build_system_prompt(question_type)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
            initial_state: AgentState = {
                "messages": messages,
                "question_type": question_type,
                "tool_call_count": 0,
            }
            final_state = agent.invoke(initial_state)

            # 최종 답변 추출
            answer = ""
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    answer = msg.content
                    break

            model_used = "gpt-4o" if question_type in COMPLEX_TYPES else "gpt-4o-mini"

            # 도구 호출 결과를 context로 추출
            for msg in final_state["messages"]:
                if isinstance(msg, ToolMessage) and msg.content:
                    tool_contexts.append(msg.content)

            # 차트 함수 복원
            _cg.generate_technical_chart = _cg_orig

        except Exception as e:
            logger.error(f"  에이전트 오류: {e}")
            answer = f"오류: {e}"
            question_type = "unknown"
            model_used = "error"
            try:
                _cg.generate_technical_chart = _cg_orig
            except Exception:
                pass

        # 2. 검색 컨텍스트 + 도구 호출 결과 통합
        #    에이전트가 실제로 본 정보 = RAG 검색 + 구조화 데이터 + 도구 호출 결과
        from src.llm.tools import _enrich_with_structured_data, _etf_data_index, _stock_data_index

        if asset_type == "stock" and stock_retriever:
            context, sources = retrieve_relevant_docs(stock_retriever, question)
            enriched = _enrich_with_structured_data(sources or [], _stock_data_index)
        elif asset_type == "mixed":
            ctx_etf, src_etf = retrieve_relevant_docs(etf_retriever, question)
            ctx_stock, src_stock = (
                retrieve_relevant_docs(stock_retriever, question)
                if stock_retriever else (None, [])
            )
            parts = [c for c in [ctx_etf, ctx_stock] if c]
            context = "\n\n".join(parts) if parts else None
            sources = (src_etf or []) + (src_stock or [])
            enriched = _enrich_with_structured_data(
                (src_etf or []) + (src_stock or []),
                {**_etf_data_index, **_stock_data_index},
            )
        else:
            context, sources = retrieve_relevant_docs(etf_retriever, question)
            enriched = _enrich_with_structured_data(sources or [], _etf_data_index)

        # 구조화 데이터를 context에 합산
        if context and enriched:
            context = context + enriched
        elif enriched:
            context = enriched

        # 도구 호출 결과를 context에 추가 (에이전트가 실제로 본 전체 정보)
        all_contexts = []
        if context:
            all_contexts.append(context)
        for tc in tool_contexts:
            # JSON 구조화 데이터(차트 등)는 제외, 텍스트만 포함
            if tc and not tc.startswith("{\"__type__\""):
                all_contexts.append(tc[:3000])  # 도구 결과는 3000자 제한

        retrieved_contexts = all_contexts if all_contexts else ["관련 문서 없음"]

        # 3. RAGAS SingleTurnSample 생성
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=ground_truth,
        )
        samples.append(sample)

        per_question_details.append({
            "question": question,
            "question_type": item["question_type"],
            "asset_type": asset_type,
            "answer_length": len(answer),
            "model": model_used,
            "has_context": context is not None,
            "answer_preview": answer[:200],
        })

        logger.info(f"    → {model_used}, 답변 {len(answer)}자")

    # 4. RAGAS 평가 실행 — 명시적으로 LLM/Embeddings 전달 (호환성)
    logger.info(f"\nRAGAS 평가 실행 중 ({len(samples)}개 샘플)...")
    eval_dataset = EvaluationDataset(samples=samples)

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    ragas_emb = LangchainEmbeddingsWrapper(LCOpenAIEmbeddings(model="text-embedding-3-small"))

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        LLMContextRecall(llm=ragas_llm),
    ]

    try:
        ragas_result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_emb,
            raise_exceptions=False,
        )
    except Exception as e:
        logger.error(f"RAGAS 평가 실패: {e}")
        return None

    # 5. 결과 정리 — DataFrame에서 점수 추출
    scores = {}
    try:
        df = ragas_result.to_pandas()
        metric_cols = [c for c in df.columns
                       if c not in ("user_input", "response", "retrieved_contexts", "reference")]

        # 전체 평균 점수
        for col in metric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                scores[col] = round(float(vals.mean()), 3)

        # 질문별 점수 추출
        for i, row in df.iterrows():
            if i < len(per_question_details):
                for col in metric_cols:
                    val = row[col]
                    if val is not None and not (isinstance(val, float) and val != val):
                        per_question_details[i][col] = round(float(val), 3)

        # general 유형 제외 (RAG 전용 — general은 LLM 지식 질문이므로 CR/F 평가 부적합)
        rag_types = [i for i in range(len(per_question_details))
                     if per_question_details[i].get("question_type") != "general"]

        rag_faith = [per_question_details[i].get("faithfulness")
                     for i in rag_types
                     if per_question_details[i].get("faithfulness") is not None]
        if rag_faith:
            scores["faithfulness_rag_only"] = round(sum(rag_faith) / len(rag_faith), 3)

        rag_cr = [per_question_details[i].get("context_recall")
                  for i in rag_types
                  if per_question_details[i].get("context_recall") is not None]
        if rag_cr:
            scores["context_recall_rag_only"] = round(sum(rag_cr) / len(rag_cr), 3)
    except Exception as e:
        logger.warning(f"결과 추출 실패: {e}")

    # 비용 정보
    try:
        tokens = ragas_result.total_tokens()
        if tokens:
            scores["total_tokens"] = int(tokens)
        cost = ragas_result.total_cost()
        if cost:
            scores["total_cost_usd"] = round(float(cost), 4)
    except Exception:
        pass

    return {
        "scores": scores,
        "per_question": per_question_details,
    }


def _json_safe(obj):
    """numpy/pandas 타입을 JSON 호환 타입으로 변환"""
    import numbers
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, numbers.Number) and not isinstance(obj, (int, float, bool)):
        return float(obj)
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    return obj


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
        "retrieval_details": retrieval_results,
    }

    if ragas_results:
        summary["ragas"] = ragas_results.get("scores", {})
        summary["ragas_per_question"] = ragas_results.get("per_question", [])

    # numpy/pandas 타입 변환
    summary = _json_safe(summary)

    output_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_path, summary


def print_summary(summary):
    """평가 결과 요약 출력"""
    r = summary["retrieval"]
    print("\n" + "=" * 60)
    print("  ETF RAG 평가 결과")
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
        print("  RAGAS 답변 품질 지표:")
        for metric, score in summary["ragas"].items():
            print(f"    {metric:25s}: {score:.3f}")

        # 낮은 점수 질문 하이라이트
        per_q = summary.get("ragas_per_question", [])
        low_faith = [q for q in per_q if q.get("faithfulness", 1.0) < 0.5]
        if low_faith:
            print()
            print(f"  Faithfulness 낮은 질문 ({len(low_faith)}개):")
            for q in low_faith[:5]:
                score = q.get("faithfulness", "N/A")
                print(f"    [{score}] {q['question'][:50]}...")

    print("=" * 60)


def main():
    no_llm = "--no-llm" in sys.argv

    # --sample N 옵션: RAGAS 평가 시 샘플 수 제한 (비용 절감)
    sample_size = None
    for i, arg in enumerate(sys.argv):
        if arg == "--sample" and i + 1 < len(sys.argv):
            sample_size = int(sys.argv[i + 1])

    logger.info("평가 데이터셋 로드...")
    dataset = load_eval_dataset()
    logger.info(f"  {len(dataset)}개 질문")

    logger.info("Retriever 초기화...")
    start = time.time()
    etf_retriever, stock_retriever = init_retriever()
    init_time = time.time() - start
    logger.info(f"  초기화 시간: {init_time:.1f}초")

    # 에이전트 도구에 retriever 주입 (run_agent가 도구를 사용할 수 있도록)
    from src.llm.tools import set_retriever
    from src.data.loader import load_etf_data as _load_etf, load_stock_data as _load_stock

    etf_data = _load_etf()
    stock_data = _load_stock()
    set_retriever(
        etf_retriever, None,
        stock_retriever=stock_retriever,
        etf_data=etf_data,
        stock_data=stock_data,
    )
    logger.info("에이전트 도구 초기화 완료")

    # 1. 검색 평가 (무료)
    logger.info("\n검색 품질 평가 시작...")
    retrieval_results = evaluate_retrieval(etf_retriever, stock_retriever, dataset)

    # 2. RAGAS 전체 평가 — 에이전트 기반 (유료)
    ragas_results = None
    if not no_llm:
        if sample_size:
            logger.info(f"\nRAGAS 평가 시작 (에이전트 기반, 샘플 {sample_size}개)...")
        else:
            logger.info(f"\nRAGAS 평가 시작 (에이전트 기반, 전체 {len(dataset)}개)...")
        ragas_results = evaluate_with_ragas(
            etf_retriever, stock_retriever, dataset, sample_size=sample_size,
        )

    # 결과 저장
    output_path, summary = save_results(retrieval_results, ragas_results)
    print_summary(summary)
    logger.info(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
