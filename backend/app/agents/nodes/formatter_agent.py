from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from app.agents.prompts.system_prompts import FORMATTER_SYSTEM_PROMPT
from app.agents.state import GraphState


def formatter_agent(state: GraphState, config: RunnableConfig) -> GraphState:
    runtime = config.get("configurable", {})
    local_short_answer = runtime.get("local_short_answer")
    apply_constraints = runtime.get("apply_constraints")

    answer = (state.get("draft_answer") or "").strip()
    results = state.get("results", [])

    if state.get("needs_retry") and callable(local_short_answer):
        answer = local_short_answer(
            question=state.get("query", ""),
            results=results,
            max_words=int(state.get("max_words", 60)),
            max_lines=int(state.get("max_lines", 3)),
            include_source=bool(state.get("include_source", False)),
        )

    if callable(apply_constraints):
        answer = apply_constraints(
            f"{answer}".strip(),
            int(state.get("max_words", 60)),
            int(state.get("max_lines", 3)),
        )

    if answer:
        answer = answer.strip()

    if not answer:
        answer = "Exact answer not found in indexed content."

    _ = FORMATTER_SYSTEM_PROMPT
    return {"answer": answer}
