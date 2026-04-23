import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = "hash-embedding-v1"
EMBEDDING_DIMENSION = 384
DEFAULT_TOP_K = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4.1-mini")
