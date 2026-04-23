from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from app.core.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME


@dataclass
class EmbeddingResult:
    vector: np.ndarray
    duration_ms: float


class HashEmbeddingService:
    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.model_name = EMBEDDING_MODEL_NAME
        self.dimension = dimension

    def embed_text(self, text: str) -> EmbeddingResult:
        start = time.perf_counter()
        vector = np.zeros(self.dimension, dtype=np.float32)

        tokens = text.lower().split()
        if not tokens:
            return EmbeddingResult(vector=vector, duration_ms=0.0)

        for token in tokens:
            idx = hash(token) % self.dimension
            vector[idx] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        duration_ms = (time.perf_counter() - start) * 1000
        return EmbeddingResult(vector=vector, duration_ms=duration_ms)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if math.isclose(float(denom), 0.0):
            return 0.0
        return float(np.dot(a, b) / denom)
