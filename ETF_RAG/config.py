from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

# Data paths
DATA_DIR = PROJECT_ROOT / "src" / "data"
ETF_DATA_PATH = DATA_DIR / "etf_data.json"
LOG_DIR = PROJECT_ROOT / "logs"

# RAG settings
SIMILARITY_THRESHOLD = 1.5
TOP_K_RESULTS = 3

# LLM settings
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT = 60
MAX_HISTORY_MESSAGES = 10
