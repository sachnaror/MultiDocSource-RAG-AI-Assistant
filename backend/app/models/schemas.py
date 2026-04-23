from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class APIIngestionRequest(BaseModel):
    source_name: str = Field(..., min_length=2)
    url: str
    headers: dict[str, str] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    message: str
    created_at: datetime
    updated_at: datetime
    source_id: str | None = None


class SourceAttribution(BaseModel):
    source_name: str
    source_type: Literal["pdf", "excel", "api"]
    page_or_row: str
    chunk_id: str
    similarity: float


class RetrievalMetrics(BaseModel):
    top_k: int
    similarity_scores: list[float]
    avg_similarity_score: float


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ChunkDistribution(BaseModel):
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    overlap_percent: float


class EmbeddingInsights(BaseModel):
    model: str
    vector_dimension: int
    avg_embedding_time_ms: float


class QueryPerformance(BaseModel):
    retrieval_time_ms: float
    llm_response_time_ms: float
    total_latency_ms: float


class DebugPanel(BaseModel):
    raw_prompt: str
    retrieved_context: str


class QueryResponse(BaseModel):
    answer: str
    query_type: Literal["factual", "analytical", "summary"]
    applied_query_mode: Literal["auto", "strict_lookup", "table_only", "rag_generate"]
    applied_response_style: Literal["exact", "concise", "detailed", "analyst"]
    confidence_score: float
    retrieval_metrics: RetrievalMetrics
    token_usage: TokenUsage
    chunk_distribution: ChunkDistribution
    embedding_insights: EmbeddingInsights
    query_performance: QueryPerformance
    source_attribution: list[SourceAttribution]
    debug: DebugPanel


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=1, ge=1, le=10)
    source_id: str | None = None
    chat_history: list[str] = Field(default_factory=list)
    query_mode: Literal["auto", "strict_lookup", "table_only", "rag_generate"] = "auto"
    response_style: Literal["exact", "concise", "detailed", "analyst"] = "exact"


class DocumentStats(BaseModel):
    source_name: str
    source_type: str
    pages_or_rows: int
    chunk_count: int
    indexed_percent: float


class DashboardStats(BaseModel):
    documents: list[DocumentStats]
    failed_retrievals: int
    retries: int
    empty_responses: int
    total_queries: int
    embeddings_model: str
    embedding_dimension: int
    avg_embedding_time_ms: float
    last_updated: datetime


class SourceSummary(BaseModel):
    source_id: str
    source_name: str
    source_type: str
    chunk_count: int
    metadata: dict[str, Any]
