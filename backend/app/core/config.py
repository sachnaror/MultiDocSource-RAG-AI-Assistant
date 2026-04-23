import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_DIMENSION = 1536
EMBEDDING_REQUEST_TIMEOUT_SEC = float(os.getenv("EMBEDDING_REQUEST_TIMEOUT_SEC", "20"))
DEFAULT_TOP_K = 5
CHUNK_DENSITY_MULTIPLIER = max(1, int(os.getenv("CHUNK_DENSITY_MULTIPLIER", "10")))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-5.4-mini")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high")
CONFIDENCE_MODE = os.getenv("CONFIDENCE_MODE", "strict").strip().lower()
TABLE_LOOKUP_MODE = os.getenv("TABLE_LOOKUP_MODE", "strict").strip().lower()
STRICT_LOOKUP_FAIL_MESSAGE = os.getenv(
    "STRICT_LOOKUP_FAIL_MESSAGE",
    "Exact answer not found in indexed content. Re-ingest and verify query terms.",
).strip()
