from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from statistics import mean

from app.core.config import EMBEDDING_DIMENSION, UPLOAD_DIR
from app.models.schemas import (
    APIIngestionRequest,
    DashboardStats,
    DocumentStats,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    SourceSummary,
)
from app.services.rag import RAGService
from app.state import job_manager, registry

router = APIRouter(prefix="/v1", tags=["rag"])


@router.post("/ingest/file", response_model=JobStatusResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source_name: str = Form(...),
    source_type: str = Form(...),
) -> JobStatusResponse:
    source_type = source_type.lower().strip()
    if source_type not in {"pdf", "excel"}:
        raise HTTPException(status_code=400, detail="source_type must be either 'pdf' or 'excel'")

    destination = UPLOAD_DIR / file.filename
    payload = await file.read()
    destination.write_bytes(payload)

    job = job_manager.create_job(message=f"Queued {source_type} ingestion")
    asyncio.create_task(job_manager.process_file(job.job_id, source_name, source_type, Path(destination)))

    return JobStatusResponse(**job.__dict__)


@router.post("/ingest/api", response_model=JobStatusResponse)
async def ingest_api(request: APIIngestionRequest) -> JobStatusResponse:
    job = job_manager.create_job(message="Queued API ingestion")
    asyncio.create_task(job_manager.process_api(job.job_id, request.source_name, request.url, request.headers))
    return JobStatusResponse(**job.__dict__)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job.__dict__)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    rag = RAGService(store=job_manager.vector_store, registry=registry)
    return rag.ask(
        question=request.question,
        top_k=request.top_k,
        source_id=request.source_id,
        chat_history=request.chat_history,
        query_mode=request.query_mode,
        response_style=request.response_style,
    )


@router.get("/sources", response_model=list[SourceSummary])
async def sources() -> list[SourceSummary]:
    return [
        SourceSummary(
            source_id=s.source_id,
            source_name=s.source_name,
            source_type=s.source_type,
            chunk_count=s.chunk_count,
            metadata=s.metadata,
        )
        for s in registry.list_sources()
    ]


@router.delete("/sources/{source_id}")
async def delete_source(source_id: str) -> dict[str, str]:
    deleted = registry.remove_source(source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Source not found")
    job_manager.vector_store.delete_source(source_id)
    return {"status": "deleted", "source_id": source_id}


@router.get("/dashboard", response_model=DashboardStats)
async def dashboard() -> DashboardStats:
    docs = [
        DocumentStats(
            source_name=s.source_name,
            source_type=s.source_type,
            pages_or_rows=s.pages_or_rows,
            chunk_count=s.chunk_count,
            indexed_percent=s.indexed_percent,
        )
        for s in registry.list_sources()
    ]

    embed_times = [c.embedding_time_ms for c in registry.all_chunks()]

    return DashboardStats(
        documents=docs,
        failed_retrievals=registry.counters.failed_retrievals,
        retries=registry.counters.retries,
        empty_responses=registry.counters.empty_responses,
        total_queries=registry.counters.total_queries,
        embeddings_model=job_manager.embedder.runtime_model_name,
        embedding_dimension=EMBEDDING_DIMENSION,
        avg_embedding_time_ms=round(mean(embed_times), 2) if embed_times else 0.0,
        last_updated=registry.last_updated,
    )
