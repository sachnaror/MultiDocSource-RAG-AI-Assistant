from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from app.agents.prompts.system_prompts import REASONING_SYSTEM_PROMPT
from app.agents.state import GraphState


def reasoning_agent(state: GraphState, config: RunnableConfig) -> GraphState:
    runtime = config.get("configurable", {})
    llm_generate = runtime.get("llm_generate")
    local_short_answer = runtime.get("local_short_answer")

    query = state.get("query", "")
    resolved_question = state.get("resolved_question", query)
    results = state.get("results", [])
    max_words = int(state.get("max_words", 60))
    max_lines = int(state.get("max_lines", 3))
    concise = bool(state.get("concise", True))
    style_instruction = state.get("style_instruction", "")

    draft = None
    if callable(llm_generate):
        draft = llm_generate(
            question=query,
            resolved_question=resolved_question,
            results=results,
            max_words=max_words,
            max_lines=max_lines,
            concise=concise,
            style_instruction=(
                f"{style_instruction}. {REASONING_SYSTEM_PROMPT}" if style_instruction else REASONING_SYSTEM_PROMPT
            ),
        )

    if not draft and callable(local_short_answer):
        draft = local_short_answer(
            question=query,
            results=results,
            max_words=max_words,
            max_lines=max_lines,
            include_source=bool(state.get("include_source", False)),
        )

    return {"draft_answer": (draft or "").strip()}
