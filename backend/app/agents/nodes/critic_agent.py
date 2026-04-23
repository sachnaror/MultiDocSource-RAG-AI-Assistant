from __future__ import annotations

import re

from langchain_core.runnables import RunnableConfig

from app.agents.prompts.system_prompts import CRITIC_SYSTEM_PROMPT
from app.agents.state import GraphState


_BAD_HEADINGS = (
    "direct answer",
    "grounded evidence",
    "question interpreted as",
)


def critic_agent(state: GraphState, config: RunnableConfig) -> GraphState:
    _ = config
    draft = (state.get("draft_answer") or "").strip()
    query = (state.get("query") or "").lower()

    if not draft:
        return {"needs_retry": True, "critique": f"{CRITIC_SYSTEM_PROMPT} Empty draft answer."}

    low = draft.lower()
    if any(h in low for h in _BAD_HEADINGS):
        return {
            "needs_retry": True,
            "critique": f"{CRITIC_SYSTEM_PROMPT} Contains forbidden heading/style artifacts.",
        }

    # Penalize long irrelevant responses for short factual questions.
    short_factual = ("?" in query and len(query.split()) <= 12) or bool(
        re.search(r"\b(address|email|phone|id|value|which|what is)\b", query)
    )
    if short_factual and len(draft.split()) > 70:
        return {
            "needs_retry": True,
            "critique": f"{CRITIC_SYSTEM_PROMPT} Answer is too broad for a focused factual query.",
        }

    return {"needs_retry": False, "critique": "answer accepted"}
