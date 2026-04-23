from __future__ import annotations

from functools import lru_cache
from typing import Callable

from app.agents.graph import build_graph
from app.agents.state import GraphState
from app.services.source_registry import ChunkRecord
from app.services.vector_store import SearchResult, VectorStore


@lru_cache(maxsize=1)
def _compiled_graph():
    return build_graph()


def run_agents(
    *,
    query: str,
    resolved_question: str,
    results: list[SearchResult],
    max_words: int,
    max_lines: int,
    concise: bool,
    include_source: bool,
    style_instruction: str,
    top_k: int,
    store: VectorStore,
    chunks: list[ChunkRecord],
    llm_generate: Callable[..., str | None],
    local_short_answer: Callable[..., str],
    apply_constraints: Callable[[str, int, int], str],
) -> str | None:
    graph = _compiled_graph()

    initial_state: GraphState = {
        "query": query,
        "resolved_question": resolved_question,
        "results": results,
        "max_words": max_words,
        "max_lines": max_lines,
        "concise": concise,
        "include_source": include_source,
        "style_instruction": style_instruction,
        "top_k": top_k,
    }

    final_state = graph.invoke(
        initial_state,
        config={
            "configurable": {
                "store": store,
                "chunks": chunks,
                "top_k": top_k,
                "llm_generate": llm_generate,
                "local_short_answer": local_short_answer,
                "apply_constraints": apply_constraints,
            }
        },
    )

    answer = final_state.get("answer") if isinstance(final_state, dict) else None
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return None
