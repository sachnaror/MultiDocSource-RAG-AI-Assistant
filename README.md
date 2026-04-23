# MultiDocSource-RAG-AI-Assistant

Multi-source RAG assistant for **PDF + Excel + API** data with:
- FastAPI backend
- PyQt desktop client
- In-memory vector search
- Deterministic table lookup for precise row/column questions

## What This App Does

1. Ingests documents and API payloads.
2. Parses content into normalized records.
3. Generates embeddings for each record/chunk.
4. Stores vectors in an in-memory registry.
5. Answers questions using:
   - deterministic table lookup first (for table-style queries)
   - then retrieval + constrained generation fallback.

## Architecture Overview

### Vector Storage

- This app does **not** use Pinecone/FAISS/pgvector right now.
- Vector data is stored in-process (RAM) via `InMemoryRegistry`.
- On backend restart, indexed vectors are lost and must be re-ingested.

Relevant files:
- [embeddings.py](backend/app/services/embeddings.py)
- [vector_store.py](backend/app/services/vector_store.py)
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

## Project Structure

```text
MultiDocSource-RAG-AI-Assistant/
├── backend/
│   └── app/
│       ├── api/
│       │   └── routes.py
│       ├── core/
│       │   └── config.py
│       ├── models/
│       │   └── schemas.py
│       ├── services/
│       │   ├── chunking.py
│       │   ├── embeddings.py
│       │   ├── parsers.py
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
```

Optional:

- `EMBEDDING_MODEL_NAME` (default `text-embedding-3-large`)
- `EMBEDDING_REQUEST_TIMEOUT_SEC` (default `20`)
- `REASONING_EFFORT` (default `high`)
- `STRICT_LOOKUP_FAIL_MESSAGE` (custom strict-mode not-found text)

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
- `GET /v1/dashboard` - summary metrics

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

- Vector storage is in memory currently; restart clears indexed vectors.
- Re-ingest sources after each backend restart.

## Current Limitations

- No persistent vector DB yet (no Pinecone/FAISS/pgvector integration).
- In-memory index only.
- Deterministic table lookup is strongest for Excel-style row/column questions.

## Next Recommended Improvements

1. Add persistent vector store (e.g., pgvector or Pinecone).
2. Add explicit query mode selector (`strict_lookup` vs `rag_generate`).
3. Add ingestion versioning + automatic stale-index warning.
