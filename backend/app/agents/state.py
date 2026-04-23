from __future__ import annotations

from typing import Any, TypedDict

from app.services.vector_store import SearchResult


class GraphState(TypedDict, total=False):
    query: str
    resolved_question: str
    results: list[SearchResult]
    max_words: int
    max_lines: int
    concise: bool
    include_source: bool
    style_instruction: str
    answer: str
    draft_answer: str
    critique: str
    needs_retry: bool
    top_k: int
    metadata: dict[str, Any]
