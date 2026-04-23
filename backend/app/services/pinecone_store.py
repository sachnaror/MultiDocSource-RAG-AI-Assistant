from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np

from app.core.config import (
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_NAMESPACE_MODE,
    PINECONE_REGION,
)
from app.services.embeddings import HashEmbeddingService
from app.services.source_registry import ChunkRecord
from app.services.vector_store import SearchResult, VectorStore


class PineconeVectorStore(VectorStore):
    def __init__(self, embedder: HashEmbeddingService) -> None:
        super().__init__(embedder)
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is required when VECTOR_BACKEND=pinecone")

        try:
            from pinecone import Pinecone, ServerlessSpec
        except Exception as exc:
            raise RuntimeError("pinecone package is not installed") from exc

        self._pc = Pinecone(api_key=PINECONE_API_KEY)
        self._index_name = PINECONE_INDEX_NAME
        self._namespace_mode = PINECONE_NAMESPACE_MODE
        self._default_namespace = PINECONE_NAMESPACE or "default"

        listed = self._pc.list_indexes()
        if hasattr(listed, "names"):
            existing_indexes = set(listed.names())
        else:
            existing_indexes = {item["name"] for item in listed}
        if self._index_name not in existing_indexes:
            self._pc.create_index(
                name=self._index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )

        self._index = self._pc.Index(self._index_name)
        index_info = self._pc.describe_index(self._index_name)
        index_dimension = int(index_info.dimension)
        if self.embedder.dimension != index_dimension:
            # Keep runtime embedding size aligned with existing Pinecone index.
            warnings.warn(
                f"Pinecone index dimension is {index_dimension}, "
                f"but embedder is {self.embedder.dimension}; aligning embedder to index dimension.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.embedder.dimension = index_dimension

    def _namespace(self, source_id: str | None = None) -> str:
        if self._namespace_mode == "per_source" and source_id:
            return source_id
        return self._default_namespace

    @staticmethod
    def _serialize_metadata(record: ChunkRecord) -> dict[str, Any]:
        md = record.metadata or {}

        payload: dict[str, Any] = {
            "source_id": record.source_id,
            "source_name": record.source_name,
            "source_type": record.source_type,
            "chunk_id": record.chunk_id,
            "locator": record.locator,
            "page_or_row": record.page_or_row,
            "content": record.content,
            "record_type": str(md.get("record_type", "")),
            "sheet": str(md.get("sheet", "")),
        }

        row = md.get("row")
        if isinstance(row, int):
            payload["row"] = row

        cols = md.get("columns")
        if isinstance(cols, list):
            payload["columns_csv"] = ",".join(str(c) for c in cols if c)

        row_data = md.get("row_data")
        if isinstance(row_data, dict):
            payload["row_data_json"] = json.dumps(row_data, ensure_ascii=True)

        return payload

    @staticmethod
    def _chunk_from_match(match: dict[str, Any]) -> ChunkRecord:
        md = match.get("metadata") or {}

        row_data: dict[str, Any] = {}
        row_data_raw = md.get("row_data_json", "")
        if isinstance(row_data_raw, str) and row_data_raw:
            try:
                parsed = json.loads(row_data_raw)
                if isinstance(parsed, dict):
                    row_data = parsed
            except Exception:
                row_data = {}

        columns_csv = str(md.get("columns_csv", ""))
        columns = [c.strip() for c in columns_csv.split(",") if c.strip()]

        metadata = {
            "record_type": str(md.get("record_type", "")),
            "sheet": str(md.get("sheet", "")),
            "columns": columns,
            "row_data": row_data,
        }
        if isinstance(md.get("row"), int):
            metadata["row"] = md["row"]

        return ChunkRecord(
            chunk_id=str(md.get("chunk_id", match.get("id", ""))),
            source_id=str(md.get("source_id", "")),
            source_name=str(md.get("source_name", "")),
            source_type=str(md.get("source_type", "pdf")),
            locator=str(md.get("locator", "")),
            page_or_row=str(md.get("page_or_row", "")),
            content=str(md.get("content", "")),
            vector=np.array(match.get("values") or [], dtype=np.float32),
            embedding_time_ms=0.0,
            metadata=metadata,
        )

    def upsert_source_chunks(self, source_id: str, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return

        namespace = self._namespace(source_id)
        vectors = [
            {
                "id": f"{chunk.source_id}:{chunk.chunk_id}",
                "values": chunk.vector.tolist(),
                "metadata": self._serialize_metadata(chunk),
            }
            for chunk in chunks
        ]

        # Batch upsert for reliability with large documents.
        for i in range(0, len(vectors), 100):
            self._index.upsert(vectors=vectors[i : i + 100], namespace=namespace)

    def delete_source(self, source_id: str) -> None:
        namespace = self._namespace(source_id)
        self._index.delete(filter={"source_id": {"$eq": source_id}}, namespace=namespace)

    def search(
        self,
        query: str,
        chunks: list[ChunkRecord],
        top_k: int,
        source_id: str | None = None,
    ) -> list[SearchResult]:
        _ = chunks
        query_vector = self.embedder.embed_text(query).vector

        namespace = self._namespace(source_id)
        filters = {"source_id": {"$eq": source_id}} if source_id else None
        candidate_k = max(top_k * 6, 12)

        response = self._index.query(
            vector=query_vector.tolist(),
            top_k=candidate_k,
            include_metadata=True,
            include_values=True,
            namespace=namespace,
            filter=filters,
        )

        matches = getattr(response, "matches", None) or response.get("matches", [])
        best_by_content: dict[str, SearchResult] = {}
        for match in matches:
            payload = match.to_dict() if hasattr(match, "to_dict") else match
            chunk = self._chunk_from_match(payload)
            pinecone_score = float(payload.get("score", 0.0))
            if pinecone_score < 0:
                pinecone_score = max(0.0, min(1.0, (pinecone_score + 1.0) / 2.0))
            else:
                pinecone_score = max(0.0, min(1.0, pinecone_score))

            # Blend Pinecone score with local lexical+semantic hybrid for better precision.
            hybrid_score = self._hybrid_similarity(query, query_vector, chunk)
            blended = (hybrid_score * 0.75) + (pinecone_score * 0.25)
            result = SearchResult(chunk=chunk, similarity=blended)

            content_key = " ".join(chunk.content.lower().split())
            prev = best_by_content.get(content_key)
            if prev is None or result.similarity > prev.similarity:
                best_by_content[content_key] = result
        results = sorted(best_by_content.values(), key=lambda item: item.similarity, reverse=True)
        return results[:top_k]
