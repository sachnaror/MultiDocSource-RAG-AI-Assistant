import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_DIMENSION = 1536
EMBEDDING_REQUEST_TIMEOUT_SEC = float(os.getenv("EMBEDDING_REQUEST_TIMEOUT_SEC", "20"))
DEFAULT_TOP_K = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-5.4-mini")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high")
CONFIDENCE_MODE = os.getenv("CONFIDENCE_MODE", "strict").strip().lower()
