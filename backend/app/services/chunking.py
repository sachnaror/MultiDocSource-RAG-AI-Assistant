from __future__ import annotations

from dataclasses import dataclass
import re

from app.core.config import CHUNK_DENSITY_MULTIPLIER


@dataclass
class ChunkConfig:
    chunk_size: int = 260
    overlap: int = 40
    density_multiplier: int = CHUNK_DENSITY_MULTIPLIER


class Chunker:
    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    def chunk_text(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        words = normalized.split()
        if len(words) <= self.config.chunk_size:
            return [normalized]

        size = max(80, self.config.chunk_size)
        overlap = max(10, min(self.config.overlap, size // 2))
        base_step = max(1, size - overlap)

        # Avoid extreme duplication even if env asks for very high density.
        effective_density = max(1, min(self.config.density_multiplier, 3))
        step = max(60, base_step // effective_density)

        chunks: list[str] = []
        for idx in range(0, len(words), step):
            window = words[idx : idx + size]
            if not window:
                break

            # Keep chunk boundary near sentence end when possible.
            if idx + size < len(words):
                tail = " ".join(window[-20:])
                m = re.search(r"[.!?]\s+[A-Z]", tail)
                if m:
                    cut_tokens = max(1, len(" ".join(window[:-20]).split()) + len(tail[: m.start() + 1].split()))
                    window = window[:cut_tokens]

            chunk = " ".join(window).strip()
            if chunk:
                chunks.append(chunk)
            if idx + size >= len(words):
                break
        return chunks
