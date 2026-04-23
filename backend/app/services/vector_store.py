from __future__ import annotations

from dataclasses import dataclass
import re

from app.services.embeddings import HashEmbeddingService
from app.services.source_registry import ChunkRecord


@dataclass
class SearchResult:
    chunk: ChunkRecord
    similarity: float


class VectorStore:
    def __init__(self, embedder: HashEmbeddingService) -> None:
        self.embedder = embedder

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z]{3,}", text.lower()))

    def _hybrid_similarity(self, query: str, query_vector, chunk: ChunkRecord) -> float:
        semantic_raw = self.embedder.cosine_similarity(query_vector, chunk.vector)
        semantic = max(0.0, min(1.0, (semantic_raw + 1.0) / 2.0))

        q_terms = self._terms(query)
        c_terms = self._terms(chunk.content)
        lexical = (len(q_terms.intersection(c_terms)) / len(q_terms)) if q_terms else 0.0

        return (semantic * 0.85) + (lexical * 0.15)

    def search(self, query: str, chunks: list[ChunkRecord], top_k: int) -> list[SearchResult]:
        query_vector = self.embedder.embed_text(query).vector
        scored = [SearchResult(chunk=chunk, similarity=self._hybrid_similarity(query, query_vector, chunk)) for chunk in chunks]
        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:top_k]
