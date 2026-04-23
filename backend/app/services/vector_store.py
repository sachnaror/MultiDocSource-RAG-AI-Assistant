from __future__ import annotations

from dataclasses import dataclass

from app.services.embeddings import HashEmbeddingService
from app.services.source_registry import ChunkRecord


@dataclass
class SearchResult:
    chunk: ChunkRecord
    similarity: float


class VectorStore:
    def __init__(self, embedder: HashEmbeddingService) -> None:
        self.embedder = embedder

    def search(self, query: str, chunks: list[ChunkRecord], top_k: int) -> list[SearchResult]:
        query_vector = self.embedder.embed_text(query).vector
        scored = [
            SearchResult(chunk=chunk, similarity=self.embedder.cosine_similarity(query_vector, chunk.vector))
            for chunk in chunks
        ]
        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:top_k]
