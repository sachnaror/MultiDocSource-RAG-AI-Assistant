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
        return set(re.findall(r"[a-zA-Z0-9]{2,}", text.lower()))

    @staticmethod
    def _is_field_query(query: str) -> bool:
        q = query.lower()
        hints = [
            "name",
            "notary",
            "phone",
            "address",
            "objective",
            "purpose",
            "date",
            "email",
            "id",
            "value",
            "who is",
            "what is",
        ]
        return any(h in q for h in hints)

    def _hybrid_similarity(self, query: str, query_vector, chunk: ChunkRecord) -> float:
        semantic_raw = self.embedder.cosine_similarity(query_vector, chunk.vector)
        semantic = max(0.0, min(1.0, (semantic_raw + 1.0) / 2.0))

        q_terms = self._terms(query)
        c_terms = self._terms(chunk.content)
        lexical = (len(q_terms.intersection(c_terms)) / len(q_terms)) if q_terms else 0.0
        key_boost = 0.0
        row_data = chunk.metadata.get("row_data", {})
        if isinstance(row_data, dict) and row_data:
            key_terms = set()
            for key in row_data.keys():
                key_terms.update(self._terms(str(key)))
            if q_terms and key_terms:
                key_boost = len(q_terms.intersection(key_terms)) / max(1, len(q_terms))

        if self._is_field_query(query):
            score = (semantic * 0.55) + (lexical * 0.25) + (key_boost * 0.20)
        else:
            score = (semantic * 0.78) + (lexical * 0.18) + (key_boost * 0.04)
        return max(0.0, min(1.0, score))

    def upsert_source_chunks(self, source_id: str, chunks: list[ChunkRecord]) -> None:
        _ = source_id, chunks

    def delete_source(self, source_id: str) -> None:
        _ = source_id

    def search(
        self,
        query: str,
        chunks: list[ChunkRecord],
        top_k: int,
        source_id: str | None = None,
    ) -> list[SearchResult]:
        if source_id:
            chunks = [chunk for chunk in chunks if chunk.source_id == source_id]
        query_vector = self.embedder.embed_text(query).vector
        scored = [SearchResult(chunk=chunk, similarity=self._hybrid_similarity(query, query_vector, chunk)) for chunk in chunks]
        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:top_k]
