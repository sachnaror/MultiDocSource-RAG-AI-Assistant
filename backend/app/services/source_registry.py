from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np


@dataclass
class ChunkRecord:
    chunk_id: str
    source_id: str
    source_name: str
    source_type: Literal["pdf", "excel", "api"]
    locator: str
    page_or_row: str
    content: str
    vector: np.ndarray
    embedding_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceRecord:
    source_id: str
    source_name: str
    source_type: Literal["pdf", "excel", "api"]
    created_at: datetime
    pages_or_rows: int
    chunk_count: int
    indexed_percent: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryCounters:
    failed_retrievals: int = 0
    retries: int = 0
    empty_responses: int = 0
    total_queries: int = 0


class InMemoryRegistry:
    def __init__(self) -> None:
        self.sources: dict[str, SourceRecord] = {}
        self.chunks: list[ChunkRecord] = []
        self.counters = QueryCounters()
        self.last_updated = datetime.utcnow()

    def add_source(self, source: SourceRecord, chunk_records: list[ChunkRecord]) -> None:
        self.sources[source.source_id] = source
        self.chunks.extend(chunk_records)
        self.last_updated = datetime.utcnow()

    def list_sources(self) -> list[SourceRecord]:
        return sorted(self.sources.values(), key=lambda s: s.created_at, reverse=True)

    def all_chunks(self) -> list[ChunkRecord]:
        return self.chunks

    def remove_source(self, source_id: str) -> bool:
        if source_id not in self.sources:
            return False
        self.sources.pop(source_id, None)
        self.chunks = [chunk for chunk in self.chunks if chunk.source_id != source_id]
        self.last_updated = datetime.utcnow()
        return True
