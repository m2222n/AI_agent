# 임베딩 모델 비교 리포트 — OpenAI vs BGE-M3

ETF RAG 검색 품질을 임베딩 모델별로 정량 비교한 실험 기록.
재현: `python eval/exp_embedding_ab.py`(전체) / `--quick`(ETF만). BGE-M3는
`langchain-huggingface` + `sentence-transformers` 설치 시 자동 포함, 없으면 skip.

## 비교 대상

| 태그 | 모델 | 차원 | 비고 |
|------|------|------|------|
| small | `text-embedding-3-small` | 1536 | **현행** (OpenAI API) |
| large | `text-embedding-3-large` | 3072 | OpenAI 상위 (API) |
| bge-m3 | `BAAI/bge-m3` | 1024 | 한국어 포함 다국어 특화 (로컬, 선택 설치) |

## 평가 설계

- 데이터셋: `eval_dataset.json`(--quick=ETF 42문항, expected_ticker 보유분).
- **두 모드**로 측정:
  - **full (실서비스)**: 이름 직접매칭 + BM25 + dense + Cohere Rerank — 실제 파이프라인.
  - **dense_only**: 직접매칭 off + `sparse_weight=0` → **임베딩 순수 변별력**만 격리.
- 순위 민감 지표: **MRR / Hit@1 / Hit@3 / Hit@5** (Hit@5만 보면 천장이라 변별 불가 → MRR·Hit@1 중심).

## 결과 (dense_only, ETF 42문항)

| 모델 | MRR | Hit@1 | Hit@3 | Hit@5 |
|------|-----|-------|-------|-------|
| **small (현행)** | 0.495 | 0.429 | 0.524 | 0.619 |
| **large** | **0.874** | **0.857** | **0.881** | **0.905** |
| **bge-m3** | 0.576 | 0.571 | 0.571 | 0.595 |

**Δ vs small (dense_only):**
- large: MRR **+0.379**, Hit@1 **+0.428** — 압도적
- bge-m3: MRR +0.081, Hit@1 +0.143, Hit@5 −0.024 — small보다 약간 우위, large엔 크게 못 미침

### full 파이프라인 (실서비스)

| 모델 | MRR | Hit@1 | Hit@5 |
|------|-----|-------|-------|
| small / large / bge-m3 | **1.0** | **1.0** | **1.0** |

→ **세 모델 모두 천장.** 하이브리드 검색(이름 직접매칭 + BM25 + Rerank)이 임베딩 약점을 메워, **실서비스 체감 차이는 0**.

## 결론 / 의사결정

1. **현행 `text-embedding-3-small` 유지.** full 파이프라인이 천장이라 모델 교체의 실서비스 이득이 없음. small이 가장 싸고 빠름(인덱싱 38s).
2. **순수 dense 의존도가 커지면(예: PDF 투자설명서 등 비정형 문서 확대) `large`가 1순위.** dense-only MRR 0.49→0.87로 압도. 비용·인덱싱(2배) 감수 가치.
3. **BGE-M3(한국어 특화)는 이 도메인에서 기대만큼 강하지 않음.** 종목명·금융용어 위주라 한국어 일반 코퍼스 특화의 이점이 작고, small보다 약간 나은 수준. 게다가 **로컬 CPU 임베딩이 매우 느림**(3천여 문서 첫 인덱싱 ~18분, Mac MPS는 OOM) → 운영 부담 큼. 비용 0(외부 미전송)이 꼭 필요한 경우에만.

## 실행 함정 (재현 시 참고)

- **torch < 2.6 + .bin 가중치**: `torch.load` 보안 차단(CVE-2025-32434). Python 3.9는 torch 2.2.2가 최대라 업그레이드 불가 → BGE-M3는 **safetensors 포함 revision 고정**으로 회피(`exp_embedding_ab.py`의 MODELS revision).
- **Mac MPS OOM**: 대량 문서 임베딩 시 GPU 메모리 부족 → `model_kwargs={"device": "cpu"}` + 작은 batch_size로 CPU 강제(느리지만 안정).
- BGE-M3 의존성은 **배포 미포함**(requirements 주석). 실험 전용.

_측정: 2026-06-29, 결과 JSON: `eval/results/exp_embedding_ab_*.json`_
