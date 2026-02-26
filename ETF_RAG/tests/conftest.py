import os
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.data.loader import load_etf_data, create_documents


@pytest.fixture(scope="session")
def etf_data():
    return load_etf_data()


@pytest.fixture(scope="session")
def documents(etf_data):
    return create_documents(etf_data)
