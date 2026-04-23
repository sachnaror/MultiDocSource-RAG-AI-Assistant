from __future__ import annotations

from typing import Callable

from app.services.vector_store import SearchResult


def build_context(
    results: list[SearchResult],
    citation_line: Callable[[SearchResult], str],
    clean_text: Callable[[str], str],
    limit: int = 3,
) -> str:
    return "\n\n".join(
        f"[{citation_line(item)}] {clean_text(item.chunk.content)}"
        for item in results[:limit]
    )
