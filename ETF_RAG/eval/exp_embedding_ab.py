"""
임베딩 A/B 비교 실험 (격리·일회성) — text-embedding-3-small vs text-embedding-3-large

목적:
    현재 쓰는 OpenAI text-embedding-3-small(1536차원) vs text-embedding-3-large(3072차원)를
    한국어 검색 품질 기준으로 정량 비교. 둘 다 OpenAI API라 추가 설치 0 (BGE-M3 ~수GB 회피).

설계 (기존 코드 본체 불변, 전부 런타임 격리):
    - get_embeddings() monkeypatch로 모델 전환 (vectorstore.py 수정 안 함)
    - FAISS 인덱스 prefix를 모델별 분리 (차원 충돌·실서비스 캐시 오염 방지)
    - backend="faiss" 명시 (Pinecone dim=1536 하드코딩 충돌 회피)
    - 직접 매칭(_match_etf_by_name) off + dense-only(sparse_weight=0) 모드로 천장 효과 우회
    - 순위 민감 지표(MRR/Hit@1/@3/@5) 추가 — Hit@5만 보면 둘 다 천장이라 변별 불가

사용법 (.venv에서):
    .venv/bin/python eval/exp_embedding_ab.py            # 두 모델 모두, dense-only + full 둘 다
    .venv/bin/python eval/exp_embedding_ab.py --quick    # ETF만, 빠르게

주의:
    - OpenAI 임베딩 인덱싱 비용 발생(text-embedding-3 매우 저렴, 전체 1회 ≈ 수 센트).
    - LLM/RAGAS 미사용 → GPT 비용 0.
    - 이 스크립트는 배포 import 그래프 밖. requirements.txt 불변.
"""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import src.rag.vectorstore as vs_mod
from src.data.loader import (
    load_etf_data, load_stock_data, create_documents, create_stock_documents,
)
from src.rag.retriever import HybridRetriever, retrieve_relevant_docs

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("exp_embedding")
logger.setLevel(logging.INFO)

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

# 비교 대상 모델 정의. provider로 OpenAI / HuggingFace(로컬) 분기.
# bge-m3는 한국어 포함 다국어 특화(1024차원) — 로컬 모델이라 선택적 설치.
MODELS = {
    "small":  {"provider": "openai", "name": "text-embedding-3-small"},  # 현행 (1536)
    "large":  {"provider": "openai", "name": "text-embedding-3-large"},  # OpenAI 상위 (3072)
    # 한국어 특화 로컬(1024). revision: safetensors 포함 커밋 고정(torch<2.6 .bin 차단 회피).
    "bge-m3": {"provider": "hf", "name": "BAAI/bge-m3",
               "revision": "9a0624b896d81da7492a910ffa53731274b6cf3d"},
}

# dense 검색이 변별력을 갖는 "어려운" 유형 — 종목명이 명시 안 돼 직접 매칭이 안 먹는 질문군
HARD_TYPES = {"recommend", "risk", "general"}


def _hf_available() -> bool:
    """BGE-M3 실행에 필요한 langchain-huggingface + sentence-transformers 설치 여부."""
    try:
        import langchain_huggingface  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _patch_embeddings(tag: str):
    """get_embeddings()를 지정 모델로 런타임 치환. 본체 파일 불변.

    OpenAI는 API, HuggingFace(bge-m3)는 로컬 sentence-transformers.
    """
    spec = MODELS[tag]
    if spec["provider"] == "openai":
        from langchain_openai import OpenAIEmbeddings
        vs_mod.get_embeddings = lambda: OpenAIEmbeddings(model=spec["name"])
    else:  # hf — 로컬 임베딩 (정규화 권장: bge 계열은 cosine 사용)
        from langchain_huggingface import HuggingFaceEmbeddings
        # torch<2.6은 .bin(torch.load) 로드가 CVE-2025-32434로 차단됨.
        # Python 3.9는 torch 2.2.2가 최대라 업그레이드 불가 → safetensors가 포함된
        # revision을 명시해 .bin 경로를 회피한다. (BGE-M3 safetensors 커밋)
        rev = spec.get("revision")
        # device="cpu": Mac MPS(GPU) OOM 회피(대량 문서 임베딩 시). 느리지만 안정.
        # batch_size 작게 + normalize(cosine).
        mk = {"device": "cpu"}
        if rev:
            mk["revision"] = rev
        vs_mod.get_embeddings = lambda: HuggingFaceEmbeddings(
            model_name=spec["name"],
            model_kwargs=mk,
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )


def _build_retrievers(model_tag: str):
    """모델별 prefix로 격리된 retriever 구축 (ETF + 주식)."""
    _patch_embeddings(model_tag)

    etf_docs = create_documents(load_etf_data(), include_pdfs=False)
    stock_docs = create_stock_documents(load_stock_data())

    etf_vs = vs_mod.create_vectorstore(
        etf_docs, prefix=f"exp_{model_tag}_etf", backend="faiss"
    )
    etf_retriever = HybridRetriever(etf_vs, etf_docs)

    stock_retriever = None
    if stock_docs:
        stock_vs = vs_mod.create_vectorstore(
            stock_docs, prefix=f"exp_{model_tag}_stock", backend="faiss"
        )
        stock_retriever = HybridRetriever(stock_vs, stock_docs)

    logger.info(f"[{model_tag}] retriever 구축 완료 (ETF {len(etf_docs)} + 주식 {len(stock_docs)})")
    return etf_retriever, stock_retriever


def _disable_direct_match(retriever):
    """인스턴스 레벨로 직접 이름 매칭 off — dense/sparse 경로만 타게 함 (본체 불변)."""
    if retriever is not None:
        retriever._match_etf_by_name = lambda q: []


def _first_hit_rank(expected_ticker, retrieved):
    """정답 티커가 처음 등장하는 1-indexed rank. 없으면 None.
    복수 정답(list)이면 가장 앞선 rank."""
    if expected_ticker is None:
        return None
    targets = expected_ticker if isinstance(expected_ticker, list) else [expected_ticker]
    for rank, t in enumerate(retrieved, start=1):
        if t in targets:
            return rank
    return None


def _eval_once(etf_r, stock_r, dataset, k=5):
    """순위 민감 지표로 검색 평가. retrieve_relevant_docs는 순서 보존 → index=rank."""
    rows = []
    for item in dataset:
        q = item["question"]
        expected = item.get("expected_ticker")
        atype = item.get("asset_type", "etf")

        if atype == "stock" and stock_r:
            _, sources = retrieve_relevant_docs(stock_r, q, k=k)
        elif atype == "mixed":
            _, s_etf = retrieve_relevant_docs(etf_r, q, k=k)
            _, s_stk = retrieve_relevant_docs(stock_r, q, k=k) if stock_r else (None, [])
            sources = (s_etf or []) + (s_stk or [])
        else:
            _, sources = retrieve_relevant_docs(etf_r, q, k=k)

        retrieved = [s["ticker"] for s in sources] if sources else []
        rank = _first_hit_rank(expected, retrieved)
        rows.append({
            "type": item["question_type"],
            "expected": expected,
            "rank": rank,                       # None = 정답 미검색 또는 expected None
            "has_expected": expected is not None,
        })
    return rows


def _aggregate(rows):
    """MRR / Hit@1 / Hit@3 / Hit@5 — expected_ticker가 있는 질문만 대상."""
    scored = [r for r in rows if r["has_expected"]]
    n = len(scored)
    if n == 0:
        return {}
    def hit_at(k):
        return round(sum(1 for r in scored if r["rank"] and r["rank"] <= k) / n, 4)
    mrr = round(sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in scored) / n, 4)
    return {"n": n, "mrr": mrr, "hit@1": hit_at(1), "hit@3": hit_at(3), "hit@5": hit_at(5)}


def _aggregate_hard(rows):
    """어려운 유형(recommend/risk/general)만 — 단, 이들은 expected_ticker가 대부분 None이라
    직접 매칭 무관. 대신 종목명 비명시 + expected 있는 질문에서 dense 변별력이 드러남."""
    hard = [r for r in rows if r["type"] in HARD_TYPES and r["has_expected"]]
    return _aggregate(hard) if hard else {"n": 0}


def run(quick=False):
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if quick:
        dataset = [d for d in dataset if d.get("asset_type", "etf") == "etf"]
    logger.info(f"데이터셋 {len(dataset)}개 (quick={quick})")

    summary = {"config": {}, "models": {}, "skipped": []}
    orig_hybrid = copy.deepcopy(config.HYBRID_SEARCH)

    # 실행할 모델: OpenAI 2종은 항상, bge-m3는 의존성 있을 때만(없으면 skip 기록)
    tags = ["small", "large"]
    if _hf_available():
        tags.append("bge-m3")
    else:
        summary["skipped"].append("bge-m3 (langchain-huggingface/sentence-transformers 미설치)")
        logger.warning("bge-m3 skip — `pip install langchain-huggingface sentence-transformers`")

    for tag in tags:
        etf_r, stock_r = _build_retrievers(tag)
        model_result = {}

        # 모드 1: full 파이프라인 (실서비스 충실도 — 직접매칭+BM25+dense+Rerank)
        t0 = time.time()
        rows_full = _eval_once(etf_r, stock_r, dataset)
        model_result["full"] = _aggregate(rows_full)
        model_result["full_hard"] = _aggregate_hard(rows_full)

        # 모드 2: dense-only (직접매칭 off + sparse_weight=0) — 임베딩 순수 변별
        _disable_direct_match(etf_r)
        _disable_direct_match(stock_r)
        config.HYBRID_SEARCH["sparse_weight"] = 0.0
        config.HYBRID_SEARCH["dense_weight"] = 1.0
        rows_dense = _eval_once(etf_r, stock_r, dataset)
        model_result["dense_only"] = _aggregate(rows_dense)
        model_result["dense_only_hard"] = _aggregate_hard(rows_dense)
        config.HYBRID_SEARCH = copy.deepcopy(orig_hybrid)  # 원복

        model_result["elapsed_sec"] = round(time.time() - t0, 1)
        summary["models"][tag] = model_result
        logger.info(f"[{tag}] full={model_result['full']} dense_only={model_result['dense_only']}")

    # delta — small(현행) 대비 각 모델, dense-only 기준(임베딩 순수 변별이 드러나는 모드)
    base = summary["models"]["small"]["dense_only"]
    summary["delta_vs_small_dense_only"] = {}
    for tag in tags:
        if tag == "small":
            continue
        cmp = summary["models"][tag]["dense_only"]
        if base and cmp:
            summary["delta_vs_small_dense_only"][tag] = {
                kk: round(cmp[kk] - base[kk], 4) for kk in ("mrr", "hit@1", "hit@3", "hit@5")
            }
    summary["config"] = {
        "dataset_size": len(dataset),
        "models": {t: MODELS[t]["name"] for t in tags},
        "modes": ["full (실서비스)", "dense_only (직접매칭off + sparse=0)"],
        "note": "Hit@5 천장 효과 → dense_only MRR/Hit@1로 판정. bge-m3는 한국어 특화 비교용.",
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="ETF만 빠르게")
    args = ap.parse_args()

    summary = run(quick=args.quick)

    RESULTS_DIR.mkdir(exist_ok=True)
    # Date.now 회피: 파일명은 epoch 정수로 (스크립트 환경에 datetime 있으나 결정성 위해 time)
    stamp = int(time.time())
    out = RESULTS_DIR / f"exp_embedding_ab_{stamp}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("임베딩 비교 결과 (OpenAI small/large vs BGE-M3)")
    print("=" * 60)
    for tag, m in summary["models"].items():
        print(f"\n[{tag}] {summary['config']['models'].get(tag, '')}  ({m['elapsed_sec']}s)")
        print(f"  full       : {m['full']}")
        print(f"  dense_only : {m['dense_only']}")
        print(f"  dense_hard : {m['dense_only_hard']}")
    for tag, d in summary.get("delta_vs_small_dense_only", {}).items():
        print(f"\nΔ ({tag} - small, dense_only): {d}")
    for s in summary.get("skipped", []):
        print(f"\n⏭  skip: {s}")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
