from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkConfig:
    chunk_size: int = 450
    overlap: int = 80


class Chunker:
    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    def chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []

        size = self.config.chunk_size
        overlap = self.config.overlap
        chunks: list[str] = []
        step = max(1, size - overlap)

        for idx in range(0, len(words), step):
            chunk_words = words[idx : idx + size]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            if idx + size >= len(words):
                break
        return chunks
