import os
import json
from datetime import datetime

from config import LOG_DIR


def log_interaction(question: str, answer: str, sources: list,
                    question_type: str = "general", search_time: float = 0,
                    llm_time: float = 0, total_time: float = 0,
                    feedback: str = None):
    """질의응답 로그 저장 (성능 메트릭 포함)"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = LOG_DIR / "chat_log.jsonl"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "question_type": question_type,
        "answer": answer,
        "sources": [s["id"] for s in sources] if sources else [],
        "performance": {
            "search_time_ms": round(search_time * 1000, 2),
            "llm_time_ms": round(llm_time * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2)
        },
        "feedback": feedback
    }

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_feedback(question: str, answer: str, feedback: str):
    """사용자 피드백 로그"""
    os.makedirs(LOG_DIR, exist_ok=True)
    feedback_file = LOG_DIR / "feedback_log.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "feedback": feedback
    }

    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def get_performance_stats() -> dict:
    """로그에서 성능 통계 계산"""
    log_file = LOG_DIR / "chat_log.jsonl"

    if not os.path.exists(log_file):
        return None

    total_times = []
    search_times = []
    llm_times = []
    question_types = {}

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if "performance" in entry:
                    perf = entry["performance"]
                    total_times.append(perf.get("total_time_ms", 0))
                    search_times.append(perf.get("search_time_ms", 0))
                    llm_times.append(perf.get("llm_time_ms", 0))

                q_type = entry.get("question_type", "unknown")
                question_types[q_type] = question_types.get(q_type, 0) + 1

        if not total_times:
            return None

        return {
            "total_queries": len(total_times),
            "avg_total_time_ms": round(sum(total_times) / len(total_times), 2),
            "avg_search_time_ms": round(sum(search_times) / len(search_times), 2),
            "avg_llm_time_ms": round(sum(llm_times) / len(llm_times), 2),
            "question_types": question_types
        }
    except Exception:
        return None


def get_feedback_stats() -> dict:
    """피드백 로그에서 통계 계산"""
    feedback_file = LOG_DIR / "feedback_log.jsonl"

    if not os.path.exists(feedback_file):
        return None

    positive = 0
    negative = 0
    reasons = {}

    try:
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                fb = entry.get("feedback", "")
                if fb == "positive":
                    positive += 1
                elif fb.startswith("negative"):
                    negative += 1
                    if ":" in fb:
                        reason = fb.split(":", 1)[1].split(" - ")[0].strip()
                        reasons[reason] = reasons.get(reason, 0) + 1

        total = positive + negative
        if total == 0:
            return None

        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": round(positive / total * 100, 1),
            "negative_reasons": reasons,
        }
    except Exception:
        return None
