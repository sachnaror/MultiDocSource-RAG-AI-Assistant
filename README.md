# MultiDocSource-RAG-AI-Assistant

Multi-source RAG assistant for **PDF + Excel + API** data with:
- FastAPI backend
- PyQt desktop client
- In-memory vector search
- Deterministic table lookup for precise row/column questions
- LangGraph multi-agent orchestration for generation path

## What This App Does

1. Ingests documents and API payloads.
2. Parses content into normalized records.
3. Generates embeddings for each record/chunk.
4. Stores vectors in either memory or Pinecone.
5. Answers questions using:
   - deterministic table lookup first (for table-style queries)
   - then multi-agent retrieval -> reasoning -> critic -> formatter pipeline.

## Multi-Agent Layer

`backend/app/agents/` is a dedicated orchestration layer (separate from `services/`):

- `retrieval_agent`: gets top relevant chunks
- `reasoning_agent`: drafts grounded answer
- `critic_agent`: rejects vague or noisy drafts
- `formatter_agent`: enforces concise final style

This keeps core RAG deterministic logic clean while making generative answering modular and extensible.

## Architecture Overview

### Vector Storage

- Supports 2 backends:
- `memory` (default): vectors kept in-process RAM.
- `pinecone`: persistent, scalable vector storage.
- Recommended production strategy: single Pinecone index + metadata filtering by `source_id` (and optional namespace strategy).

Relevant files:
- [embeddings.py](backend/app/services/embeddings.py)
- [vector_store.py](backend/app/services/vector_store.py)
- [pinecone_store.py](backend/app/services/pinecone_store.py)
- [source_registry.py](backend/app/services/source_registry.py)

### Embedding Behavior

- Primary embedding provider: OpenAI embeddings (`text-embedding-3-large` by default).
- Fallback provider: local hash embedding if OpenAI embedding call fails or key missing.

### Retrieval Behavior

- Query embedding is computed at request time.
- Similarity uses hybrid scoring (semantic + lexical signal).
- Top-K results feed answer generation unless deterministic table lookup returns first.

### Deterministic Table Lookup

Excel ingestion stores table metadata (`sheet`, `row`, `columns`, `row_data`).

In `TABLE_LOOKUP_MODE=strict`:
- exact table answer found -> return exact value
- exact answer not found -> return explicit not-found message
- avoids vague LLM fallback for table-style questions

### Query Mode And Response Style

Each query can explicitly control routing and answer persona:

- `query_mode`
- `auto`: deterministic lookup first, then RAG/generation fallback
- `strict_lookup`: deterministic extraction only
- `table_only`: deterministic table extraction only
- `rag_generate`: generation-first flow (no strict lookup block)

- `response_style`
- `exact`: one-line direct value/answer
- `concise`: short plain-language answer
- `detailed`: fuller explanation
- `analyst`: evidence-oriented concise answer

Guardrail profile definitions are in:
- [guardrails.py](backend/app/core/guardrails.py)

## Project Structure

```text
MultiDocSource-RAG-AI-Assistant/
├── backend/
│   └── app/
│       ├── agents/
│       │   ├── graph.py
│       │   ├── state.py
│       │   ├── executor.py
│       │   ├── nodes/
│       │   │   ├── retrieval_agent.py
│       │   │   ├── reasoning_agent.py
│       │   │   ├── critic_agent.py
│       │   │   └── formatter_agent.py
│       │   ├── tools/
│       │   │   ├── vector_search_tool.py
│       │   │   └── parser_tool.py
│       │   └── prompts/
│       │       └── system_prompts.py
│       ├── api/
│       │   └── routes.py
│       ├── core/
│       │   ├── config.py
│       │   └── guardrails.py
│       ├── models/
│       │   └── schemas.py
│       ├── services/
│       │   ├── chunking.py
│       │   ├── embeddings.py
│       │   ├── parsers.py
│       │   ├── pinecone_store.py
│       │   ├── rag.py
│       │   ├── source_registry.py
│       │   └── vector_store.py
│       ├── workers/
│       │   └── jobs.py
│       ├── state.py
│       └── main.py
├── desktop/
│   ├── api_client.py
│   └── app.py
├── scripts/
│   ├── run_backend.sh
│   └── run_desktop.sh
├── data/uploads/
├── requirements.txt
└── README.md
```

## Setup

```bash
cd /Users/homesachin/Desktop/zoneone/practice/MultiDocSource-RAG-AI-Assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Recommended:

```bash
export OPENAI_API_KEY=your_real_key
export GENERATION_MODEL=gpt-5.4
export CONFIDENCE_MODE=strict
export TABLE_LOOKUP_MODE=strict

# Vector backend
export VECTOR_BACKEND=memory
# or
export VECTOR_BACKEND=pinecone

# Pinecone (required only when VECTOR_BACKEND=pinecone)
export PINECONE_API_KEY=your_pinecone_key
export PINECONE_INDEX_NAME=rag-docs
export PINECONE_CLOUD=aws
export PINECONE_REGION=us-east-1
export PINECONE_NAMESPACE_MODE=single
export PINECONE_NAMESPACE=default
```

Optional:

- `EMBEDDING_MODEL_NAME` (default `text-embedding-3-large`)
- `EMBEDDING_REQUEST_TIMEOUT_SEC` (default `20`)
- `REASONING_EFFORT` (default `high`)
- `STRICT_LOOKUP_FAIL_MESSAGE` (custom strict-mode not-found text)
- Per-query options are passed in request body: `query_mode`, `response_style`
- Optional query filter: `source_id`

## Run Backend

```bash
source .venv/bin/activate
./scripts/run_backend.sh
```

Backend URL: `http://127.0.0.1:8000`

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

## Run Desktop App

Open another terminal:

```bash
cd /Users/homesachin/Desktop/zoneone/practice/MultiDocSource-RAG-AI-Assistant
source .venv/bin/activate
./scripts/run_desktop.sh
```

## API Endpoints

- `POST /v1/ingest/file` - upload PDF/Excel
- `POST /v1/ingest/api` - ingest API JSON source
- `GET /v1/jobs/{job_id}` - check ingestion job status
- `POST /v1/query` - ask a question
- `GET /v1/sources` - list indexed sources
- `DELETE /v1/sources/{source_id}` - delete source vectors + registry data
- `GET /v1/dashboard` - summary metrics

Example `/v1/query` payload:

```json
{
  "question": "What is value in row 4 column revenue?",
  "top_k": 2,
  "source_id": "optional-source-id-filter",
  "chat_history": [],
  "query_mode": "strict_lookup",
  "response_style": "exact"
}
```

## Query Behavior

### Best For Exact Table Answers

Use specific prompts like:
- `What is value in row 4 column revenue?`
- `How many rows are in sheet Sales?`
- `Show columns in sheet Sales`
- `What is employee_id for row 12 in sheet HR?`

Tips:
- keep `Top K` low (`1` or `2`) for precision
- include sheet/row/column names explicitly

### General QA

For non-table questions, the app uses retrieval + generation with concise-answer constraints.

## Important Re-Index Rule

Re-ingestion is required after parser/index logic changes.

If you indexed files before latest table-aware updates, old chunks may lack `row_data` metadata and exact lookup will underperform.

Action:
1. restart backend
2. re-upload/re-ingest files
3. retest queries

## Troubleshooting

### “Answers are vague”

- Ensure `OPENAI_API_KEY` is set before backend start.
- Set `TABLE_LOOKUP_MODE=strict`.
- Re-ingest Excel after updates.
- Use precise row/column wording.

### “Not found” for values that exist

- Column naming mismatch: try exact header spelling from file.
- Include `sheet` and `row` in query.
- Re-ingest source to refresh structured metadata.

### “Why did data disappear after restart?”

- If `VECTOR_BACKEND=memory`: restart clears vectors.
- If `VECTOR_BACKEND=pinecone`: vectors persist.

### “How do I update/re-index a document?”

1. Delete old vectors: `DELETE /v1/sources/{source_id}`
2. Re-upload/re-ingest updated file
3. Query with `source_id` filter when needed

## Current Limitations

- Pinecone is a managed cloud service (not an embedded local DB process).
- Registry/source stats are still in-memory process state.
- Deterministic table lookup is strongest for Excel-style row/column questions.

## Next Recommended Improvements

1. Persist source registry/counters in a database.
2. Add ingestion versioning and soft-delete + background compaction.
3. Add multi-tenant authorization-aware metadata filters.
