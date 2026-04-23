from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from app.agents.state import GraphState
from app.agents.tools.vector_search_tool import vector_search


def retrieval_agent(state: GraphState, config: RunnableConfig) -> GraphState:
    runtime = config.get("configurable", {})
    existing = state.get("results") or []
    if existing:
        return {"results": existing}

    store = runtime.get("store")
    chunks = runtime.get("chunks") or []
    if store is None or not chunks:
        return {"results": []}

    top_k = int(state.get("top_k") or runtime.get("top_k") or 3)
    query = state.get("query", "")
    return {"results": vector_search(store=store, chunks=chunks, query=query, top_k=top_k)}
