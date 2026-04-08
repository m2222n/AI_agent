import os
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch
from src.data.loader import load_etf_data, create_documents


@pytest.fixture(scope="session")
def etf_data():
    """하드코딩 샘플 데이터 로드 (기존 테스트 호환용)"""
    from pathlib import Path
    fake_db = Path("/tmp/nonexistent.db")
    with patch("src.data.loader.DB_PATH", fake_db), \
         patch("src.data.loader.get_latest_collected_path", return_value=None):
        return load_etf_data()


@pytest.fixture(scope="session")
def documents(etf_data):
    return create_documents(etf_data)
