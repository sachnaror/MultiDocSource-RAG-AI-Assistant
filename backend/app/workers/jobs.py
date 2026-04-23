from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from app.core.config import UPLOAD_DIR
from app.services.chunking import Chunker
from app.services.embeddings import HashEmbeddingService
from app.services.parsers import ParserService
from app.services.source_registry import ChunkRecord, InMemoryRegistry, SourceRecord


@dataclass
class Job:
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    message: str
    created_at: datetime
    updated_at: datetime
    source_id: str | None = None


class JobManager:
    def __init__(self, registry: InMemoryRegistry) -> None:
        self.registry = registry
        self.jobs: dict[str, Job] = {}
        self.parser = ParserService()
        self.chunker = Chunker()
        self.embedder = HashEmbeddingService()

    def create_job(self, message: str = "Queued") -> Job:
        now = datetime.utcnow()
        job = Job(
            job_id=str(uuid.uuid4()),
            status="queued",
            message=message,
            created_at=now,
            updated_at=now,
        )
        self.jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    async def process_file(self, job_id: str, source_name: str, source_type: str, file_path: Path) -> None:
        job = self.jobs[job_id]
        job.status = "processing"
        job.message = "Ingestion started"
        job.updated_at = datetime.utcnow()

        try:
            if source_type == "pdf":
                records = await self.parser.parse_pdf(file_path)
            elif source_type == "excel":
                records = await self.parser.parse_excel(file_path)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            await self._index_records(job, source_name, source_type, records)
            job.status = "completed"
            job.message = f"Ingestion completed for {source_name}"
        except Exception as exc:
            job.status = "failed"
            job.message = f"Ingestion failed: {exc}"
        finally:
            job.updated_at = datetime.utcnow()

    async def process_api(self, job_id: str, source_name: str, url: str, headers: dict[str, str]) -> None:
        job = self.jobs[job_id]
        job.status = "processing"
        job.message = "API ingestion started"
        job.updated_at = datetime.utcnow()

        try:
            records = await self.parser.parse_api(url, headers)
            await self._index_records(job, source_name, "api", records)
            job.status = "completed"
            job.message = f"API ingestion completed for {source_name}"
        except Exception as exc:
            job.status = "failed"
            job.message = f"API ingestion failed: {exc}"
        finally:
            job.updated_at = datetime.utcnow()

    async def _index_records(self, job: Job, source_name: str, source_type: str, records: list[dict]) -> None:
        source_id = str(uuid.uuid4())
        job.source_id = source_id

        chunk_records: list[ChunkRecord] = []
        for record_idx, record in enumerate(records, start=1):
            record_type = str(record.get("metadata", {}).get("record_type", ""))
            if source_type == "excel" or record_type == "table_row":
                chunks = [record["content"]]
            else:
                chunks = self.chunker.chunk_text(record["content"])
            if not chunks:
                continue

            for chunk_idx, chunk in enumerate(chunks, start=1):
                result = self.embedder.embed_text(chunk)
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=f"{record_idx}-{chunk_idx}",
                        source_id=source_id,
                        source_name=source_name,
                        source_type=source_type,
                        locator=record["locator"],
                        page_or_row=record["page_or_row"],
                        content=chunk,
                        vector=result.vector,
                        embedding_time_ms=result.duration_ms,
                        metadata=record.get("metadata", {}),
                    )
                )

            await asyncio.sleep(0)

        indexed_percent = 100.0 if records else 0.0
        source_record = SourceRecord(
            source_id=source_id,
            source_name=source_name,
            source_type=source_type,
            created_at=datetime.utcnow(),
            pages_or_rows=len(records),
            chunk_count=len(chunk_records),
            indexed_percent=indexed_percent,
            metadata={"upload_path": str(UPLOAD_DIR)},
        )
        self.registry.add_source(source_record, chunk_records)
