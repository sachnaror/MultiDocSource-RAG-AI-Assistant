from __future__ import annotations

from dataclasses import dataclass

from app.core.config import CHUNK_DENSITY_MULTIPLIER


@dataclass
class ChunkConfig:
    chunk_size: int = 140
    overlap: int = 20
    density_multiplier: int = CHUNK_DENSITY_MULTIPLIER


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
        base_step = max(1, size - overlap)
        # Higher density means smaller stride, creating more overlapping chunks.
        step = max(1, base_step // max(1, self.config.density_multiplier))

        for idx in range(0, len(words), step):
            chunk_words = words[idx : idx + size]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            if idx + size >= len(words):
                break
        return chunks
