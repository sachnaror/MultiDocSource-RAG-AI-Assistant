from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QueryMode = Literal["auto", "strict_lookup", "table_only", "rag_generate"]
ResponseStyle = Literal["exact", "concise", "detailed", "analyst"]


@dataclass(frozen=True)
class StyleProfile:
    max_words: int
    max_lines: int
    concise: bool
    include_source_default: bool
    llm_instruction: str


STYLE_PROFILES: dict[ResponseStyle, StyleProfile] = {
    "exact": StyleProfile(
        max_words=24,
        max_lines=1,
        concise=True,
        include_source_default=False,
        llm_instruction=(
            "Return only the exact value or shortest direct answer. "
            "No explanation unless explicitly asked."
        ),
    ),
    "concise": StyleProfile(
        max_words=55,
        max_lines=3,
        concise=True,
        include_source_default=False,
        llm_instruction=(
            "Return a short direct answer in plain language. "
            "No headings, no bullet points, no filler."
        ),
    ),
    "detailed": StyleProfile(
        max_words=180,
        max_lines=10,
        concise=False,
        include_source_default=False,
        llm_instruction=(
            "Return a complete but focused explanation with clear reasoning. "
            "Still avoid headings unless user asks for structure."
        ),
    ),
    "analyst": StyleProfile(
        max_words=140,
        max_lines=7,
        concise=False,
        include_source_default=True,
        llm_instruction=(
            "Return a concise analyst-style answer with evidence-oriented language. "
            "Include source cue only when helpful."
        ),
    ),
}


def normalize_query_mode(mode: str | None) -> QueryMode:
    allowed: set[str] = {"auto", "strict_lookup", "table_only", "rag_generate"}
    normalized = (mode or "auto").strip().lower()
    return normalized if normalized in allowed else "auto"


def normalize_response_style(style: str | None) -> ResponseStyle:
    allowed: set[str] = {"exact", "concise", "detailed", "analyst"}
    normalized = (style or "concise").strip().lower()
    return normalized if normalized in allowed else "concise"
