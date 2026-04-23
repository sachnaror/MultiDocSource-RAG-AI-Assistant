from __future__ import annotations

import time
from statistics import mean

from app.core.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
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

    def ask(self, question: str, top_k: int) -> QueryResponse:
        started = time.perf_counter()
        self.registry.counters.total_queries += 1

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

        similarities = [round(item.similarity, 4) for item in results]
        avg_similarity = round(mean(similarities), 4)
        context = "\n\n".join([f"[{item.chunk.locator}] {item.chunk.content}" for item in results])

        llm_start = time.perf_counter()
        answer = self._build_answer(question, results)
        llm_response_time_ms = (time.perf_counter() - llm_start) * 1000

        all_chunk_lengths = [len(c.content.split()) for c in self.registry.all_chunks()]
        avg_embed_time = mean([c.embedding_time_ms for c in self.registry.all_chunks()]) if self.registry.all_chunks() else 0.0

        input_tokens = self._estimate_tokens(question + " " + context)
        output_tokens = self._estimate_tokens(answer)
        confidence = max(0.0, min(1.0, (avg_similarity * 0.75) + 0.15))

        return QueryResponse(
            answer=answer,
            query_type=self._detect_query_type(question),
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
            debug=DebugPanel(raw_prompt=question, retrieved_context=context[:5000]),
        )

    def _build_answer(self, question: str, results: list) -> str:
        lines = [
            f"Question: {question}",
            "",
            "Answer based on retrieved sources:",
        ]
        for item in results:
            lines.append(
                f"- {item.chunk.source_name} ({item.chunk.locator}, chunk {item.chunk.chunk_id}): {item.chunk.content[:220]}"
            )
        lines.append("")
        lines.append("Use the source attribution panel for exact origin details.")
        return "\n".join(lines)
