from __future__ import annotations

from app.services.source_registry import ChunkRecord
from app.services.vector_store import SearchResult, VectorStore


def vector_search(
    store: VectorStore,
    chunks: list[ChunkRecord],
    query: str,
    top_k: int,
) -> list[SearchResult]:
    return store.search(query=query, chunks=chunks, top_k=top_k)
