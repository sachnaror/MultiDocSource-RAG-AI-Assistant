from __future__ import annotations

import re
import time
from statistics import mean

import httpx

from app.core.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL,
    OPENAI_API_KEY,
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

    def _plain_language(self, text: str, max_words: int = 90) -> str:
        cleaned = self._clean_text(text)
        words = cleaned.split()
        if not words:
            return ""
        snippet = " ".join(words[:max_words])
        substitutions = {
            "shall": "must",
            "herein": "in this document",
            "thereof": "of it",
            "pursuant": "according",
            "commence": "start",
            "terminate": "end",
        }
        for src, dst in substitutions.items():
            snippet = re.sub(rf"\b{src}\b", dst, snippet, flags=re.IGNORECASE)
        return snippet

    def _citation_line(self, item) -> str:
        return f"{item.chunk.source_name} ({item.chunk.locator}, chunk {item.chunk.chunk_id})"

    def _generate_local_refined_answer(self, question: str, resolved_question: str, results: list, query_type: str) -> str:
        top = results[0]
        top_explanation = self._plain_language(top.chunk.content)

        key_points = []
        for item in results[:3]:
            snippet = self._plain_language(item.chunk.content, max_words=55)
            key_points.append(f"- {snippet} [{self._citation_line(item)}]")

        direct_answer = (
            f"Based on the retrieved context, this is primarily about: {top_explanation}."
            if top_explanation
            else "I found relevant context, but it is too sparse to explain confidently."
        )

        if "mean" in question.lower() or "explain" in question.lower():
            explanation_header = "What this means in simple terms"
            explanation_body = (
                f"In this specific reference, it is saying that {top_explanation.lower()}"
                if top_explanation
                else "The source text is limited, so I need a slightly more specific question."
            )
        else:
            explanation_header = "Interpretation"
            explanation_body = (
                f"Interpreting your question ({query_type}), the strongest evidence points to this reading: {top_explanation}."
                if top_explanation
                else "I need a little more context to interpret this precisely."
            )

        return "\n".join(
            [
                "Direct Answer",
                direct_answer,
                "",
                explanation_header,
                explanation_body,
                "",
                "Grounded Evidence",
                *key_points,
                "",
                f"Question interpreted as: {resolved_question}",
            ]
        )

    def _generate_llm_answer(self, question: str, resolved_question: str, results: list) -> str | None:
        if not OPENAI_API_KEY:
            return None

        context_lines = []
        for item in results[:5]:
            context_lines.append(f"[{self._citation_line(item)}] {self._clean_text(item.chunk.content)[:800]}")

        prompt = "\n\n".join(
            [
                f"User question: {question}",
                f"Resolved question with conversation context: {resolved_question}",
                "Answer requirements:",
                "- Answer in a refined, ChatGPT-like style.",
                "- Stay grounded strictly in provided context.",
                "- If user asks 'what does this mean', explain in simple terms with reference-specific interpretation.",
                "- Include short citations like [source (page/row, chunk)].",
                "Context:",
                "\n".join(context_lines),
            ]
        )

        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.post(
                    "https://api.openai.com/v1/responses",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": GENERATION_MODEL,
                        "input": [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "You are a grounded RAG assistant. Be clear, concise, and context-faithful.",
                                    }
                                ],
                            },
                            {"role": "user", "content": [{"type": "text", "text": prompt}]},
                        ],
                        "temperature": 0.2,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return None

        output_text = payload.get("output_text", "")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        for item in payload.get("output", []):
            for content_item in item.get("content", []):
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

        return None

    def ask(self, question: str, top_k: int, chat_history: list[str] | None = None) -> QueryResponse:
        started = time.perf_counter()
        self.registry.counters.total_queries += 1
        chat_history = chat_history or []

        retrieval_start = time.perf_counter()
        results = self.store.search(question, self.registry.all_chunks(), top_k=top_k)
        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        if not results:
            self.registry.counters.failed_retrievals += 1
            self.registry.counters.empty_responses += 1
            return QueryResponse(
                answer="I could not retrieve relevant information. Please ingest a document first.",
                query_type=self._detect_query_type(question),
                confidence_score=0.0,
                retrieval_metrics=RetrievalMetrics(top_k=top_k, similarity_scores=[], avg_similarity_score=0.0),
                token_usage=TokenUsage(input_tokens=self._estimate_tokens(question), output_tokens=12, total_tokens=12 + self._estimate_tokens(question)),
                chunk_distribution=ChunkDistribution(total_chunks=0, avg_chunk_size=0, min_chunk_size=0, max_chunk_size=0, overlap_percent=17.78),
                embedding_insights=EmbeddingInsights(model=EMBEDDING_MODEL_NAME, vector_dimension=EMBEDDING_DIMENSION, avg_embedding_time_ms=0.0),
                query_performance=QueryPerformance(retrieval_time_ms=retrieval_time_ms, llm_response_time_ms=0.0, total_latency_ms=(time.perf_counter() - started) * 1000),
                source_attribution=[],
                debug=DebugPanel(raw_prompt=question, retrieved_context=""),
            )

        query_type = self._detect_query_type(question)
        top_source_hint = self._citation_line(results[0])
        resolved_question = self._resolve_question(question, chat_history, top_source_hint)

        similarities = [round(item.similarity, 4) for item in results]
        avg_similarity = round(mean(similarities), 4)
        context = "\n\n".join([f"[{self._citation_line(item)}] {self._clean_text(item.chunk.content)}" for item in results])

        llm_start = time.perf_counter()
        answer = self._generate_llm_answer(question, resolved_question, results)
        if not answer:
            answer = self._generate_local_refined_answer(question, resolved_question, results, query_type)
        llm_response_time_ms = (time.perf_counter() - llm_start) * 1000

        all_chunk_lengths = [len(c.content.split()) for c in self.registry.all_chunks()]
        avg_embed_time = mean([c.embedding_time_ms for c in self.registry.all_chunks()]) if self.registry.all_chunks() else 0.0

        input_tokens = self._estimate_tokens(question + " " + context)
        output_tokens = self._estimate_tokens(answer)
        confidence = max(0.0, min(1.0, (avg_similarity * 0.75) + 0.15))

        return QueryResponse(
            answer=answer,
            query_type=query_type,
            confidence_score=round(confidence * 100, 2),
            retrieval_metrics=RetrievalMetrics(top_k=top_k, similarity_scores=similarities, avg_similarity_score=avg_similarity),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens),
            chunk_distribution=ChunkDistribution(
                total_chunks=len(all_chunk_lengths),
                avg_chunk_size=round(mean(all_chunk_lengths), 2) if all_chunk_lengths else 0,
                min_chunk_size=min(all_chunk_lengths) if all_chunk_lengths else 0,
                max_chunk_size=max(all_chunk_lengths) if all_chunk_lengths else 0,
                overlap_percent=17.78,
            ),
            embedding_insights=EmbeddingInsights(
                model=EMBEDDING_MODEL_NAME,
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
