"""Supabase 기반 방문자 카운터 (REST API 직접 호출, 추가 패키지 불필요)."""

import datetime
import logging
import os
from typing import Tuple

import requests

logger = logging.getLogger(__name__)

_SUPABASE_URL = None
_SUPABASE_KEY = None


def _get_config() -> Tuple[str, str]:
    """Supabase URL과 anon key를 반환."""
    global _SUPABASE_URL, _SUPABASE_KEY
    if _SUPABASE_URL and _SUPABASE_KEY:
        return _SUPABASE_URL, _SUPABASE_KEY

    # Streamlit secrets 우선, 환경변수 fallback
    try:
        import streamlit as st
        _SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
        _SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        pass

    if not _SUPABASE_URL:
        _SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    if not _SUPABASE_KEY:
        _SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

    return _SUPABASE_URL, _SUPABASE_KEY


def _headers() -> dict:
    """Supabase REST API 헤더."""
    _, key = _get_config()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _today_kst() -> str:
    """KST 기준 오늘 날짜 (YYYY-MM-DD)."""
    kst = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(kst).strftime("%Y-%m-%d")


def record_visit() -> Tuple[int, int]:
    """방문 기록 + (당일 방문자, 누적 방문자) 반환.

    Supabase 미설정 시 (0, 0) 반환 (graceful degradation).
    """
    url, key = _get_config()
    if not url or not key:
        return 0, 0

    base = f"{url}/rest/v1/visitor_stats"
    headers = _headers()
    today = _today_kst()

    try:
        # 1) 오늘 행이 있는지 확인
        resp = requests.get(
            f"{base}?visit_date=eq.{today}&select=count",
            headers=headers,
            timeout=5,
        )
        rows = resp.json() if resp.status_code == 200 else []

        if rows:
            # 오늘 행 있음 → count + 1
            current = rows[0]["count"]
            new_count = current + 1
            requests.patch(
                f"{base}?visit_date=eq.{today}",
                headers=headers,
                json={"count": new_count},
                timeout=5,
            )
        else:
            # 오늘 행 없음 → 새로 삽입
            new_count = 1
            requests.post(
                base,
                headers=headers,
                json={"visit_date": today, "count": 1},
                timeout=5,
            )

        # 2) 누적 합계 조회
        resp = requests.get(
            f"{base}?select=count",
            headers=headers,
            timeout=5,
        )
        total = sum(r["count"] for r in resp.json()) if resp.status_code == 200 else new_count

        return new_count, total

    except Exception as e:
        logger.warning(f"방문자 카운터 오류: {e}")
        return 0, 0


def get_visitor_counts() -> Tuple[int, int]:
    """(당일 방문자, 누적 방문자) 조회만 (기록 없이).

    record_visit()과 별도로, 카운트만 확인할 때 사용.
    """
    url, key = _get_config()
    if not url or not key:
        return 0, 0

    base = f"{url}/rest/v1/visitor_stats"
    headers = _headers()
    today = _today_kst()

    try:
        # 오늘 카운트
        resp = requests.get(
            f"{base}?visit_date=eq.{today}&select=count",
            headers=headers,
            timeout=5,
        )
        rows = resp.json() if resp.status_code == 200 else []
        daily = rows[0]["count"] if rows else 0

        # 누적 합계
        resp = requests.get(
            f"{base}?select=count",
            headers=headers,
            timeout=5,
        )
        total = sum(r["count"] for r in resp.json()) if resp.status_code == 200 else 0

        return daily, total

    except Exception as e:
        logger.warning(f"방문자 카운터 조회 오류: {e}")
        return 0, 0
