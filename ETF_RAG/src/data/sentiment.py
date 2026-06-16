"""
로컬 금융 뉴스 감성 분류 (F-3) — KR-FinBert-SC 사전학습 모델.

선택적 의존: transformers/torch가 설치돼 있으면 로컬 분류 사용, 없으면 None을
반환 → 호출자(news.py)가 GPT-4o-mini로 fallback. 배포 이미지 경량 유지.

모델: snunlp/KR-FinBert-SC (3-class: negative/neutral/positive, 금융 코퍼스 파인튜닝).
분류만 로컬에서 하고, 요약/키워드는 호출자가 GPT로 처리(하이브리드).
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# 영문 라벨 → 한국어 (news.py 계약: 긍정/부정/중립)
_LABEL_MAP = {
    "positive": "긍정",
    "negative": "부정",
    "neutral": "중립",
    "LABEL_2": "긍정",   # id2label 없이 LABEL_n으로 나올 때 대비 (0=neg,1=neu,2=pos)
    "LABEL_0": "부정",
    "LABEL_1": "중립",
}

# 파이프라인 싱글턴: None=미초기화, False=사용불가(로드 실패), 그 외=pipeline 객체
_pipeline = None


def _model_name() -> str:
    from config import SENTIMENT
    return SENTIMENT.get("model", "snunlp/KR-FinBert-SC")


def _get_pipeline():
    """text-classification 파이프라인 lazy 로드. 실패/미설치 시 False 캐시."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline or None  # False면 None

    from config import SENTIMENT
    if not SENTIMENT.get("enabled", True):
        _pipeline = False
        return None
    try:
        from transformers import pipeline  # torch 포함 — 미설치면 ImportError
        _pipeline = pipeline(
            "text-classification",
            model=_model_name(),
            top_k=1,
            truncation=True,
            max_length=256,
        )
        logger.info(f"로컬 감성 모델 로드: {_model_name()}")
    except Exception as e:
        logger.info(f"로컬 감성 모델 미사용 (GPT fallback): {e}")
        _pipeline = False
        return None
    return _pipeline


def is_available() -> bool:
    """로컬 감성 분류 사용 가능 여부 (transformers 설치 + 모델 로드 성공)."""
    return _get_pipeline() is not None


def _to_korean(label: str) -> str:
    return _LABEL_MAP.get(str(label).lower(),
                          _LABEL_MAP.get(str(label), "중립"))


def classify_sentiments(texts: List[str]) -> Optional[List[str]]:
    """텍스트 목록 → 긍정/부정/중립 라벨 목록. 사용 불가/실패 시 None.

    호출자는 None이면 GPT 등 다른 경로로 fallback한다.
    """
    if not texts:
        return []
    pipe = _get_pipeline()
    if pipe is None:
        return None
    try:
        results = pipe(texts)
        labels = []
        for r in results:
            # top_k=1 → [{"label","score"}] 형태 (transformers 버전에 따라 dict일 수도)
            item = r[0] if isinstance(r, list) else r
            labels.append(_to_korean(item["label"]))
        if len(labels) != len(texts):
            return None
        return labels
    except Exception as e:
        logger.warning(f"로컬 감성 분류 실패 (GPT fallback): {e}")
        return None


def reset():
    """싱글턴 초기화 (테스트용)."""
    global _pipeline
    _pipeline = None
