from __future__ import annotations

import re
import time
from statistics import mean

import httpx

from app.agents.executor import run_agents
from app.core.guardrails import STYLE_PROFILES, normalize_query_mode, normalize_response_style
from app.core.config import (
    CONFIDENCE_MODE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL,
    OPENAI_API_KEY,
    REASONING_EFFORT,
    STRICT_LOOKUP_FAIL_MESSAGE,
    TABLE_LOOKUP_MODE,
)
from app.models.schemas import (
    ChunkDistribution,
    DebugPanel,
    EmbeddingInsights,
    QueryPerformance,
    QueryResponse,
    RetrievalMetrics,
    SourceAttribution,
    TokenUsage,
)
from app.services.source_registry import InMemoryRegistry
from app.services.vector_store import VectorStore


class RAGService:
    def __init__(self, store: VectorStore, registry: InMemoryRegistry) -> None:
        self.store = store
        self.registry = registry

    def _compute_confidence(self, similarities: list[float], top_k: int) -> float:
        if not similarities:
            return 0.0
        top = similarities[0]
        head = similarities[: min(3, len(similarities))]
        head_avg = mean(head)
        tail = similarities[: min(5, len(similarities))]
        tail_avg = mean(tail)
        coverage = min(1.0, len(similarities) / max(1, top_k))
        spread = max(0.0, top - head_avg)

        mode = (CONFIDENCE_MODE or "strict").strip().lower()
        if mode == "high":
            # Aggressive confidence mode: prioritizes best-match strength and
            # rewards strong top-hit retrieval.
            confidence = (top * 0.62) + (head_avg * 0.23) + (tail_avg * 0.10) + (coverage * 0.05)
            if top >= 0.75:
                confidence += 0.04
            if top >= 0.85:
                confidence += 0.04
            if head_avg >= 0.70:
                confidence += 0.03
            return max(0.35, min(0.995, confidence))

        if mode == "normal":
            confidence = (top * 0.60) + (head_avg * 0.30) + (coverage * 0.10)
            if top >= 0.80 and head_avg >= 0.65:
                confidence += 0.05
            return max(0.25, min(0.98, confidence))

        # Strict confidence mode (default): stronger penalty for uneven retrieval.
        confidence = (top * 0.50) + (head_avg * 0.30) + (tail_avg * 0.10) + (coverage * 0.10)
        confidence -= spread * 0.20
        if top >= 0.88 and head_avg >= 0.78:
            confidence += 0.03
        return max(0.12, min(0.93, confidence))

    def _detect_query_type(self, question: str) -> str:
        lower = question.lower()
        if any(token in lower for token in ["summarize", "summary", "overview"]):
            return "summary"
        if any(token in lower for token in ["compare", "why", "impact", "analyze"]):
            return "analytical"
        return "factual"

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _is_ambiguous_followup(self, question: str) -> bool:
        q = question.lower().strip()
        phrases = ["what does this mean", "what does that mean", "explain this", "explain that"]
        pronouns = {"this", "that", "it", "they", "these", "those"}
        short_question = len(q.split()) <= 8
        return any(phrase in q for phrase in phrases) or (short_question and any(p in q.split() for p in pronouns))

    def _resolve_question(self, question: str, chat_history: list[str], top_source_hint: str) -> str:
        if self._is_ambiguous_followup(question) and chat_history:
            return (
                f"{question}\n\n"
                f"Follow-up context from previous user question: {chat_history[-1]}\n"
                f"Focus on source hint: {top_source_hint}"
            )
        return question

    def _clean_text(self, text: str) -> str:
        cleaned = text.replace("|", ", ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _citation_line(self, item) -> str:
        return f"{item.chunk.source_name} ({item.chunk.locator}, chunk {item.chunk.chunk_id})"

    def _parse_constraints(self, question: str) -> tuple[int, int, bool]:
        q = question.lower()
        # Default behavior should feel ChatGPT-like: direct and succinct.
        max_words = 55
        max_lines = 3
        concise = True
        has_explicit_word_limit = False

        word_match = re.search(r"(\d{1,3})\s*words?", q)
        if word_match:
            max_words = max(8, min(int(word_match.group(1)), 120))
            concise = True
            has_explicit_word_limit = True

        if "one line" in q or "1 line" in q:
            max_lines = 1
            concise = True
        elif "two lines" in q or "2 lines" in q:
            max_lines = 2
            concise = True
        else:
            line_range = re.search(r"(\d)\s*[-to]+\s*(\d)\s*lines", q)
            if line_range:
                max_lines = max(1, min(6, int(line_range.group(2))))
                concise = True

        if "concise" in q or "brief" in q:
            concise = True
            max_words = min(max_words, 45)

        # If user asks for one line but no explicit word limit, keep it sharp.
        if max_lines == 1 and not has_explicit_word_limit:
            max_words = min(max_words, 20)

        if self._wants_detailed_output(question):
            concise = False
            max_words = max(max_words, 140)
            max_lines = max(max_lines, 8)

        return max_words, max_lines, concise

    def _apply_style_profile(self, question: str, response_style: str) -> tuple[int, int, bool, bool, str]:
        max_words, max_lines, concise = self._parse_constraints(question)
        style_key = normalize_response_style(response_style)
        profile = STYLE_PROFILES[style_key]

        if style_key == "exact":
            max_words = min(max_words, profile.max_words)
            max_lines = min(max_lines, profile.max_lines)
            concise = True
        elif style_key == "concise":
            max_words = min(max_words, profile.max_words)
            max_lines = min(max_lines, profile.max_lines)
            concise = True
        elif style_key == "detailed":
            max_words = max(max_words, profile.max_words)
            max_lines = max(max_lines, profile.max_lines)
            concise = False
        elif style_key == "analyst":
            max_words = min(max(max_words, 80), profile.max_words)
            max_lines = min(max(max_lines, 4), profile.max_lines)
            concise = False

        return max_words, max_lines, concise, profile.include_source_default, profile.llm_instruction

    def _apply_constraints(self, text: str, max_words: int, max_lines: int) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return text.strip()

        lines = lines[:max_lines]
        words: list[str] = []
        for line in lines:
            for token in line.split():
                words.append(token)
                if len(words) >= max_words:
                    return " ".join(words)
        return "\n".join(lines)

    def _focus_terms(self, question: str) -> list[str]:
        terms = re.findall(r"[a-zA-Z]{3,}", question.lower())
        stop = {
            "what",
            "does",
            "this",
            "that",
            "mean",
            "explain",
            "line",
            "lines",
            "with",
            "from",
            "about",
            "max",
            "word",
            "words",
        }
        return [t for t in terms if t not in stop][:8]

    def _question_terms(self, question: str) -> set[str]:
        return set(self._focus_terms(question))

    def _is_small_info_request(self, question: str) -> bool:
        q = question.lower().strip()
        if "?" in q and len(q.split()) <= 12:
            return True
        small_intents = [
            "what is",
            "which is",
            "who is",
            "when is",
            "where is",
            "how much",
            "how many",
            "only",
            "just",
        ]
        return any(intent in q for intent in small_intents)

    @staticmethod
    def _normalize_key(name: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(name).strip().lower())
        cleaned = re.sub(r"[^a-z0-9 _-]", "", cleaned)
        return cleaned

    @staticmethod
    def _tokenize_key(text: str) -> list[str]:
        return [t for t in re.split(r"[ _-]+", text) if t]

    @staticmethod
    def _alias_token(token: str) -> str:
        alias = {
            "emp": "employee",
            "dept": "department",
            "dob": "birth",
            "amt": "amount",
            "qty": "quantity",
            "num": "number",
            "no": "number",
        }
        return alias.get(token, token)

    def _normalized_tokens(self, text: str) -> list[str]:
        key = self._normalize_key(text)
        return [self._alias_token(t) for t in self._tokenize_key(key)]

    def _column_variants(self, column: str) -> set[str]:
        key = self._normalize_key(column)
        tokens = self._tokenize_key(key)
        compact = "".join(tokens)
        variants = {key, compact, " ".join(tokens), "_".join(tokens), "-".join(tokens)}
        if tokens:
            variants.add("".join(t[0] for t in tokens))
        # Alias-expanded variant for terms like emp->employee.
        alias_tokens = [self._alias_token(t) for t in tokens]
        variants.add(" ".join(alias_tokens))
        variants.add("".join(alias_tokens))
        return {v.strip() for v in variants if v.strip()}

    def _is_table_query(self, question: str) -> bool:
        q = question.lower()
        hints = [
            "sheet",
            "row",
            "column",
            "table",
            "cell",
            "value in",
            "how many rows",
            "number of rows",
            "total rows",
        ]
        return any(h in q for h in hints)

    def _excel_chunks(self) -> list:
        return [chunk for chunk in self.registry.all_chunks() if chunk.source_type == "excel"]

    def _sheet_hint(self, question: str) -> str | None:
        q = question.lower()
        match = re.search(r"\bsheet\s+([a-z0-9 _-]+)", q)
        if not match:
            return None
        return self._normalize_key(match.group(1))

    def _row_hint(self, question: str) -> int | None:
        match = re.search(r"\brow\s+(\d+)\b", question.lower())
        if not match:
            return None
        return int(match.group(1))

    def _requested_column(self, question: str, all_columns: set[str]) -> str | None:
        q = self._normalize_key(question)
        q_tokens = set(self._normalized_tokens(question))
        best_col: str | None = None
        best_score = 0.0

        for col in all_columns:
            if not col:
                continue
            variants = self._column_variants(col)
            if any(v and v in q for v in variants):
                score = 1.0 + (len(col) / 100.0)
                if score > best_score:
                    best_score = score
                    best_col = col
                continue

            col_tokens = set(self._normalized_tokens(col))
            if not col_tokens:
                continue
            overlap = len(q_tokens.intersection(col_tokens))
            if overlap <= 0:
                continue
            score = overlap / len(col_tokens)
            if score > best_score:
                best_score = score
                best_col = col

        if best_score >= 0.5:
            return best_col
        return None

    def _row_filter_terms(self, question: str, requested_column: str | None) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9]{2,}", question.lower())
        stop = {
            "what",
            "which",
            "who",
            "when",
            "where",
            "how",
            "much",
            "many",
            "is",
            "the",
            "a",
            "an",
            "for",
            "in",
            "on",
            "of",
            "from",
            "please",
            "tell",
            "me",
            "value",
            "column",
            "row",
            "sheet",
            "table",
            "show",
            "find",
            "give",
            "only",
            "just",
        }
        if requested_column:
            stop.update(requested_column.split())
        return [t for t in tokens if t not in stop]

    def _table_direct_answer(self, question: str, force_strict: bool | None = None) -> tuple[str | None, bool]:
        excel_chunks = self._excel_chunks()
        if not excel_chunks:
            return None, False

        strict_mode = (TABLE_LOOKUP_MODE or "strict").strip().lower() == "strict"
        if force_strict is not None:
            strict_mode = force_strict
        is_table_query = self._is_table_query(question)

        has_structured_rows = any(
            isinstance(chunk.metadata.get("row_data"), dict) and chunk.metadata.get("row_data")
            for chunk in excel_chunks
        )
        if is_table_query and not has_structured_rows:
            if strict_mode:
                return STRICT_LOOKUP_FAIL_MESSAGE, True
            return None, False

        all_columns: set[str] = set()
        for chunk in excel_chunks:
            row_data = chunk.metadata.get("row_data", {})
            if isinstance(row_data, dict):
                all_columns.update(self._normalize_key(k) for k in row_data.keys())
            cols = chunk.metadata.get("columns", [])
            if isinstance(cols, list):
                all_columns.update(self._normalize_key(c) for c in cols if c)

        requested_col = self._requested_column(question, all_columns)
        sheet_hint = self._sheet_hint(question)
        row_hint = self._row_hint(question)
        q = question.lower()
        if requested_col:
            is_table_query = True

        if "column" in q and any(t in q for t in ["what", "which", "list", "show"]):
            candidate_cols = sorted(c for c in all_columns if c)
            if candidate_cols:
                return ", ".join(candidate_cols), False

        if any(phrase in q for phrase in ["how many rows", "number of rows", "total rows"]):
            rows = set()
            for chunk in excel_chunks:
                row = chunk.metadata.get("row")
                sheet = self._normalize_key(chunk.metadata.get("sheet", ""))
                if isinstance(row, int):
                    if sheet_hint and sheet_hint not in sheet:
                        continue
                    rows.add((sheet, row))
            if rows:
                return str(len(rows)), False

        if requested_col and row_hint is not None:
            for chunk in excel_chunks:
                row = chunk.metadata.get("row")
                sheet = self._normalize_key(chunk.metadata.get("sheet", ""))
                if not isinstance(row, int) or row != row_hint:
                    continue
                if sheet_hint and sheet_hint not in sheet:
                    continue
                row_data = chunk.metadata.get("row_data", {})
                if isinstance(row_data, dict):
                    value = row_data.get(requested_col)
                    if isinstance(value, str) and value.strip():
                        return value.strip(), False

        terms = self._row_filter_terms(question, requested_col)
        if not terms and not requested_col:
            if is_table_query and strict_mode:
                return STRICT_LOOKUP_FAIL_MESSAGE, True
            return None, False

        best_chunk = None
        best_score = -1
        for chunk in excel_chunks:
            sheet = self._normalize_key(chunk.metadata.get("sheet", ""))
            if sheet_hint and sheet_hint not in sheet:
                continue
            row_data = chunk.metadata.get("row_data", {})
            if not isinstance(row_data, dict) or not row_data:
                continue

            haystack = f"{sheet} " + " ".join(str(v).lower() for v in row_data.values())
            score = sum(1 for t in terms if t in haystack)
            if requested_col and requested_col in row_data:
                score += 2
            if score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk is None or best_score <= 0:
            if is_table_query and strict_mode:
                return STRICT_LOOKUP_FAIL_MESSAGE, True
            return None, False

        row_data = best_chunk.metadata.get("row_data", {})
        if not isinstance(row_data, dict):
            if is_table_query and strict_mode:
                return STRICT_LOOKUP_FAIL_MESSAGE, True
            return None, False

        if requested_col:
            value = row_data.get(requested_col)
            if isinstance(value, str) and value.strip():
                return value.strip(), False
            if strict_mode:
                return STRICT_LOOKUP_FAIL_MESSAGE, True
            return None, False

        # If no specific requested column, return the most relevant fields only.
        compact_pairs = []
        for key, value in row_data.items():
            if not isinstance(value, str) or not value.strip():
                continue
            if any(term in value.lower() or term in key for term in terms):
                compact_pairs.append(f"{key}: {value.strip()}")
            if len(compact_pairs) >= 3:
                break

        if compact_pairs:
            return "; ".join(compact_pairs), False
        if is_table_query and strict_mode:
            return STRICT_LOOKUP_FAIL_MESSAGE, True
        return None, False

    def _requested_key(self, question: str) -> str:
        q = question.lower()
        q = re.sub(r"[?.,]+", " ", q)
        q = re.sub(
            r"\b(tell|me|what|is|give|show|only|just|in|one|two|three|line|lines|max|word|words|please|the)\b",
            " ",
            q,
        )
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _is_field_lookup_query(self, question: str) -> bool:
        q = question.lower()
        if any(tok in q for tok in ["exact", "value", "address", "email", "phone", "id", "location"]):
            return True
        return bool(re.search(r"\bwhat\s+is\s+.+\bof\b.+", q))

    def _field_target_terms(self, question: str) -> set[str]:
        cleaned = self._requested_key(question)
        terms = re.findall(r"[a-zA-Z]{2,}", cleaned)
        stop = {"of", "for", "and", "with", "from", "that", "this", "company"}
        return {t for t in terms if t not in stop}

    def _global_pdf_field_lookup(self, question: str) -> str | None:
        if not self._is_field_lookup_query(question):
            return None

        target_terms = self._field_target_terms(question)
        wants_address = "address" in target_terms or "location" in target_terms
        if not wants_address:
            return None

        chunks = [c for c in self.registry.all_chunks() if c.source_type == "pdf"]
        if not chunks:
            return None

        best_value: str | None = None
        best_score = 0.0

        pattern = re.compile(
            r"(?:registered\s+address|address)\s*:\s*(.+?)(?=\s+(?:purpose|official|structure|name\s+position|7\s+|6\s+)|$)",
            flags=re.I,
        )

        for chunk in chunks:
            text = self._clean_text(chunk.content)
            for m in pattern.finditer(text):
                candidate = re.sub(r"\s+", " ", m.group(1)).strip(" -.;,")
                if not candidate:
                    continue
                low = candidate.lower()
                score = 2.0
                if any(tok in low for tok in ["street", "st", "road", "building", "city", "jakarta", "province"]):
                    score += 2.0
                if 4 <= len(candidate.split()) <= 40:
                    score += 1.0
                if score > best_score:
                    best_score = score
                    best_value = candidate

        return best_value

    def _rerank_results_for_field_lookup(self, question: str, results: list) -> list:
        target_terms = self._field_target_terms(question)
        if not target_terms:
            return results

        def score(item) -> float:
            text = self._clean_text(item.chunk.content).lower()
            pattern_hits = text.count(":")
            term_hits = sum(1 for t in target_terms if t in text)
            exact_key_like = 1.0 if any(f"{t}:" in text for t in target_terms) else 0.0
            return (term_hits * 3.0) + (pattern_hits * 0.2) + exact_key_like + float(item.similarity)

        ranked = sorted(results, key=score, reverse=True)
        return ranked

    def _extract_field_value(self, question: str, results: list) -> str | None:
        q = question.lower()
        if "only" not in q and "just" not in q and not self._is_field_lookup_query(question):
            return None

        results = self._rerank_results_for_field_lookup(question, results)
        target = self._requested_key(question)
        if len(target) < 3:
            return None

        # Parse "Field: Value" spans and stop before the next field label.
        pattern = re.compile(
            r"([A-Za-z][A-Za-z0-9 /()_-]{2,50}?)\s*\d*\s*:\s*(.+?)(?=\s+[A-Z][A-Za-z0-9 /()_-]{2,40}\d*\s*:|$)"
        )

        best_value: str | None = None
        best_score = 0.0
        target_terms = self._field_target_terms(question)

        for item in results:
            text = self._clean_text(item.chunk.content)
            for key, value in pattern.findall(text):
                key_n = re.sub(r"\s+", " ", key.lower()).strip()
                value_n = re.sub(r"\s+", " ", value).strip()
                if not value_n:
                    continue

                key_terms = set(key_n.split())
                score = float(len(target_terms.intersection(key_terms)))

                if target in key_n:
                    score += 3
                if any(term in key_n for term in target_terms):
                    score += 1.5
                # Prefer concise values for exact field lookup.
                word_len = len(value_n.split())
                if 1 <= word_len <= 12:
                    score += 0.8
                if "address" in target_terms and any(tok in value_n.lower() for tok in ["road", "street", "st", "ave", "city", "india", "usa"]):
                    score += 1.2
                if score > best_score:
                    best_score = score
                    best_value = value_n

        if best_score <= 0 or not best_value:
            return None

        max_words_allowed = 40 if "address" in target_terms else 14
        if len(best_value.split()) > max_words_allowed:
            return None

        # Return only the value when user asks for "only".
        return best_value

    def _best_sentence(self, text: str, question: str) -> str:
        cleaned = self._clean_text(text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        if not sentences:
            return cleaned

        terms = self._focus_terms(question)
        if not terms:
            return sentences[0]

        best_sentence = sentences[0]
        best_score = -1
        for sent in sentences:
            sent_l = sent.lower()
            score = sum(1 for t in terms if t in sent_l)
            if score > best_score:
                best_score = score
                best_sentence = sent
        return best_sentence

    def _best_span(self, text: str, question: str, window_words: int = 26) -> str:
        cleaned = self._clean_text(text)
        tokens = cleaned.split()
        if len(tokens) <= window_words:
            return cleaned

        terms = self._question_terms(question)
        if not terms:
            return " ".join(tokens[:window_words])

        best_start = 0
        best_score = -1.0
        for i in range(0, len(tokens) - window_words + 1):
            window = tokens[i : i + window_words]
            window_text = " ".join(window).lower()
            hits = sum(1 for t in terms if t in window_text)
            density = hits / max(1, window_words)
            score = (hits * 2.0) + density
            if score > best_score:
                best_score = score
                best_start = i

        return " ".join(tokens[best_start : best_start + window_words]).strip()

    def _focused_snippet(self, content: str, question: str, max_words: int = 36) -> str:
        span = self._best_span(content, question, window_words=max_words)
        span = re.sub(r"\s+", " ", span).strip()
        return self._apply_constraints(span, max_words=max_words, max_lines=1)

    def _local_short_answer(
        self,
        question: str,
        results: list,
        max_words: int,
        max_lines: int,
        include_source: bool = False,
    ) -> str:
        top = results[0]
        summary = self._focused_snippet(top.chunk.content, question, max_words=min(max_words, 28))
        summary = re.sub(r"\s+", " ", summary).strip()
        if not summary:
            summary = "The selected section describes the candidate's core backend and automation responsibilities."

        if max_lines <= 1 or not include_source:
            return self._apply_constraints(summary, max_words, 1)

        line2 = f"Source: {top.chunk.source_name}, {top.chunk.locator}."
        combined = f"{summary}\n{line2}"
        return self._apply_constraints(combined, max_words, max_lines)

    def _generate_local_refined_answer(self, question: str, resolved_question: str, results: list, query_type: str) -> str:
        _ = resolved_question, query_type
        return self._local_short_answer(question, results, max_words=120, max_lines=5, include_source=self._wants_source_in_answer(question))

    def _candidate_models(self) -> list[str]:
        env_model = (GENERATION_MODEL or "").strip()
        candidates: list[str] = []
        if env_model:
            candidates.append(env_model)
            candidates.append(env_model.replace(" ", "-"))

        # Reliable fallbacks if provided model alias is invalid.
        candidates.extend(["gpt-5.4", "gpt-5.4-mini", "gpt-4.1-mini", "gpt-4o-mini"])

        seen: set[str] = set()
        ordered: list[str] = []
        for model in candidates:
            if model and model not in seen:
                ordered.append(model)
                seen.add(model)
        return ordered

    def _generate_llm_answer(
        self,
        question: str,
        resolved_question: str,
        results: list,
        max_words: int,
        max_lines: int,
        concise: bool,
        style_instruction: str,
    ) -> str | None:
        if not OPENAI_API_KEY:
            return None

        context_lines = []
        for item in results[:3]:
            snippet = self._focused_snippet(item.chunk.content, question, max_words=44)
            context_lines.append(f"[{self._citation_line(item)}] {snippet}")

        brevity_block = (
            f"- Output max {max_words} words.\n- Output max {max_lines} line(s).\n- Do not include headings or bullet points."
            if concise
            else "- Keep response focused and grounded. Use short paragraphs, not section headers."
        )

        prompt = "\n\n".join(
            [
                f"User question: {question}",
                f"Resolved question with conversation context: {resolved_question}",
                "Answer requirements:",
                "- Return only the answer text.",
                "- No headings like 'Direct Answer' or 'Grounded Evidence'.",
                "- No preamble or meta commentary.",
                "- Stay grounded strictly in provided context.",
                "- Explain meaning clearly in plain language.",
                "- Prefer a single short paragraph unless user asks for detail.",
                f"- Style guardrail: {style_instruction}",
                brevity_block,
                "Context:",
                "\n".join(context_lines),
            ]
        )

        for model in self._candidate_models():
            try:
                with httpx.Client(timeout=20.0) as client:
                    response = client.post(
                        "https://api.openai.com/v1/responses",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "input": [
                                {
                                    "role": "system",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "You are a grounded RAG assistant. "
                                                "Write natural ChatGPT-style answers that are specific, clear, and concise. "
                                                "Output only the direct answer. No headings, no extra sections, no filler. "
                                                "If user asks for a single value, return only that value. "
                                                "Use the provided context only."
                                            ),
                                        }
                                    ],
                                },
                                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                            ],
                            "reasoning": {"effort": REASONING_EFFORT},
                            "temperature": 0.1,
                            "max_output_tokens": min(180, max(40, max_words * 2)),
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
            except Exception:
                continue

            output_text = payload.get("output_text", "")
            if isinstance(output_text, str) and output_text.strip():
                return self._apply_constraints(output_text.strip(), max_words, max_lines)

            for item in payload.get("output", []):
                for content_item in item.get("content", []):
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        return self._apply_constraints(text.strip(), max_words, max_lines)

        return None

    def ask(
        self,
        question: str,
        top_k: int,
        source_id: str | None = None,
        chat_history: list[str] | None = None,
        query_mode: str = "auto",
        response_style: str = "concise",
    ) -> QueryResponse:
        started = time.perf_counter()
        self.registry.counters.total_queries += 1
        chat_history = chat_history or []
        normalized_mode = normalize_query_mode(query_mode)
        normalized_style = normalize_response_style(response_style)

        retrieval_start = time.perf_counter()
        all_chunks = self.registry.all_chunks()
        if source_id:
            all_chunks = [chunk for chunk in all_chunks if chunk.source_id == source_id]
        results = self.store.search(question, all_chunks, top_k=top_k, source_id=source_id)
        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        if not results:
            self.registry.counters.failed_retrievals += 1
            self.registry.counters.empty_responses += 1
            return QueryResponse(
                answer="I could not retrieve relevant information. Please ingest a document first.",
                query_type=self._detect_query_type(question),
                applied_query_mode=normalized_mode,
                applied_response_style=normalized_style,
                confidence_score=0.0,
                retrieval_metrics=RetrievalMetrics(top_k=top_k, similarity_scores=[], avg_similarity_score=0.0),
                token_usage=TokenUsage(input_tokens=self._estimate_tokens(question), output_tokens=12, total_tokens=12 + self._estimate_tokens(question)),
                chunk_distribution=ChunkDistribution(total_chunks=0, avg_chunk_size=0, min_chunk_size=0, max_chunk_size=0, overlap_percent=14.29),
                embedding_insights=EmbeddingInsights(
                    model=self.store.embedder.runtime_model_name,
                    vector_dimension=EMBEDDING_DIMENSION,
                    avg_embedding_time_ms=0.0,
                ),
                query_performance=QueryPerformance(retrieval_time_ms=retrieval_time_ms, llm_response_time_ms=0.0, total_latency_ms=(time.perf_counter() - started) * 1000),
                source_attribution=[],
                debug=DebugPanel(raw_prompt=question, retrieved_context=""),
            )

        query_type = self._detect_query_type(question)
        top_source_hint = self._citation_line(results[0])
        resolved_question = self._resolve_question(question, chat_history, top_source_hint)
        max_words, max_lines, concise, include_source_default, style_instruction = self._apply_style_profile(question, normalized_style)
        include_source = include_source_default or self._wants_source_in_answer(question)

        raw_similarities = [item.similarity for item in results]
        similarities = [round(score, 4) for score in raw_similarities]
        avg_similarity = round(mean(raw_similarities), 4)
        context = "\n\n".join([f"[{self._citation_line(item)}] {self._clean_text(item.chunk.content)}" for item in results])

        llm_start = time.perf_counter()
        answer: str | None = None
        table_strict_block = False

        if normalized_mode in {"auto", "strict_lookup"}:
            answer = self._global_pdf_field_lookup(question)
            if answer:
                answer = self._apply_constraints(answer, max_words=max_words, max_lines=max_lines)

        if normalized_mode in {"auto", "strict_lookup", "table_only"}:
            force_strict = normalized_mode in {"strict_lookup", "table_only"}
            table_answer, table_strict_block = self._table_direct_answer(question, force_strict=force_strict)
            if not answer and table_answer:
                answer = table_answer
                answer = self._apply_constraints(answer, max_words=max_words, max_lines=max_lines)

        if not answer and normalized_mode in {"auto", "strict_lookup"}:
            answer = self._extract_field_value(question, results)
            if answer:
                answer = self._apply_constraints(answer, max_words, max_lines)

        # For exact field-style questions, avoid vague generative fallback.
        if not answer and normalized_mode == "auto" and self._is_field_lookup_query(question):
            answer = self._apply_constraints(STRICT_LOOKUP_FAIL_MESSAGE, max_words=max_words, max_lines=max_lines)
            table_strict_block = True

        if not answer and normalized_mode in {"strict_lookup", "table_only"}:
            answer = self._apply_constraints(STRICT_LOOKUP_FAIL_MESSAGE, max_words=max_words, max_lines=max_lines)
            table_strict_block = True

        if not answer and normalized_mode in {"auto", "rag_generate"} and not table_strict_block:
            answer = run_agents(
                query=question,
                resolved_question=resolved_question,
                results=results,
                max_words=max_words,
                max_lines=max_lines,
                concise=concise,
                include_source=include_source,
                style_instruction=style_instruction,
                top_k=top_k,
                store=self.store,
                chunks=all_chunks,
                llm_generate=self._generate_llm_answer,
                local_short_answer=self._local_short_answer,
                apply_constraints=self._apply_constraints,
            )

        if (
            not answer
            and normalized_mode in {"auto", "rag_generate"}
            and not table_strict_block
            and self._is_small_info_request(question)
        ):
            answer = self._local_short_answer(question, results, max_words=min(max_words, 20), max_lines=1, include_source=False)
        llm_response_time_ms = (time.perf_counter() - llm_start) * 1000

        all_chunk_lengths = [len(c.content.split()) for c in self.registry.all_chunks()]
        avg_embed_time = mean([c.embedding_time_ms for c in self.registry.all_chunks()]) if self.registry.all_chunks() else 0.0

        input_tokens = self._estimate_tokens(question + " " + context)
        output_tokens = self._estimate_tokens(answer)
        confidence = self._compute_confidence(raw_similarities, top_k)

        return QueryResponse(
            answer=answer,
            query_type=query_type,
            applied_query_mode=normalized_mode,
            applied_response_style=normalized_style,
            confidence_score=round(confidence * 100, 2),
            retrieval_metrics=RetrievalMetrics(top_k=top_k, similarity_scores=similarities, avg_similarity_score=avg_similarity),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens),
            chunk_distribution=ChunkDistribution(
                total_chunks=len(all_chunk_lengths),
                avg_chunk_size=round(mean(all_chunk_lengths), 2) if all_chunk_lengths else 0,
                min_chunk_size=min(all_chunk_lengths) if all_chunk_lengths else 0,
                max_chunk_size=max(all_chunk_lengths) if all_chunk_lengths else 0,
                overlap_percent=14.29,
            ),
            embedding_insights=EmbeddingInsights(
                model=self.store.embedder.runtime_model_name,
                vector_dimension=EMBEDDING_DIMENSION,
                avg_embedding_time_ms=round(avg_embed_time, 2),
            ),
            query_performance=QueryPerformance(
                retrieval_time_ms=round(retrieval_time_ms, 2),
                llm_response_time_ms=round(llm_response_time_ms, 2),
                total_latency_ms=round((time.perf_counter() - started) * 1000, 2),
            ),
            source_attribution=[
                SourceAttribution(
                    source_name=item.chunk.source_name,
                    source_type=item.chunk.source_type,
                    page_or_row=item.chunk.page_or_row,
                    chunk_id=item.chunk.chunk_id,
                    similarity=round(item.similarity, 4),
                )
                for item in results
            ],
            debug=DebugPanel(raw_prompt=resolved_question, retrieved_context=context[:5000]),
        )
    def _wants_detailed_output(self, question: str) -> bool:
        q = question.lower()
        detail_hints = [
            "in detail",
            "detailed",
            "elaborate",
            "step by step",
            "deep dive",
            "comprehensive",
            "thorough",
            "long answer",
        ]
        return any(hint in q for hint in detail_hints)

    def _wants_source_in_answer(self, question: str) -> bool:
        q = question.lower()
        source_hints = [
            "source",
            "citation",
            "cite",
            "evidence",
            "where did you get",
            "which document",
        ]
        return any(hint in q for hint in source_hints)
