from __future__ import annotations

import math
import time
from dataclasses import dataclass

import httpx
import numpy as np

from app.core.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME, EMBEDDING_REQUEST_TIMEOUT_SEC, OPENAI_API_KEY


@dataclass
class EmbeddingResult:
    vector: np.ndarray
    duration_ms: float
    provider: str


class HashEmbeddingService:
    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.model_name = EMBEDDING_MODEL_NAME
        self.dimension = dimension
        self.runtime_model_name = EMBEDDING_MODEL_NAME if OPENAI_API_KEY else "hash-embedding-fallback"

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            idx = hash(token) % self.dimension
            vector[idx] += 1.0
        return self._normalize(vector)

    def _openai_embed(self, text: str) -> np.ndarray | None:
        if not OPENAI_API_KEY:
            return None

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": text,
            "dimensions": self.dimension,
        }

        try:
            with httpx.Client(timeout=EMBEDDING_REQUEST_TIMEOUT_SEC) as client:
                response = client.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception:
            return None

        items = data.get("data") or []
        if not items:
            return None

        embedding = items[0].get("embedding")
        if not isinstance(embedding, list):
            return None

        vector = np.array(embedding, dtype=np.float32)
        if vector.shape[0] != self.dimension:
            return None
        return self._normalize(vector)

    def embed_text(self, text: str) -> EmbeddingResult:
        start = time.perf_counter()
        vector = self._openai_embed(text)
        provider = "openai"
        if vector is None:
            vector = self._hash_embed(text)
            provider = "hash"
            self.runtime_model_name = "hash-embedding-fallback"
        else:
            self.runtime_model_name = self.model_name
        duration_ms = (time.perf_counter() - start) * 1000
        return EmbeddingResult(vector=vector, duration_ms=duration_ms, provider=provider)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if math.isclose(float(denom), 0.0):
            return 0.0
        return float(np.dot(a, b) / denom)
