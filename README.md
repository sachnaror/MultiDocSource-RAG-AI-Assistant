# MultiDocSource-RAG-AI-Assistant

Multi-Format AI Knowledge Assistant that ingests **PDF + Excel + API** sources, normalizes them into unified chunks, builds embeddings, and supports cross-source question answering with production-style observability.

## Features

- Multi-source ingestion (`pdf`, `excel`, `api`)
- Async background jobs for ingestion lifecycle
- Hybrid RAG-style retrieval across structured + unstructured data
- Source attribution (`document`, `page/row`, `chunk_id`, `similarity`)
- Retrieval quality metrics (Top-K, score list, average similarity)
- Token usage stats (input, output, total)
- Embedding insights (model, vector dimension, avg embedding time)
- Chunk distribution metrics (avg/min/max chunk size + overlap)
- Query performance metrics (retrieval, llm, total latency)
- Debug panel (raw prompt + retrieved context)
- Error/fallback counters (failed retrievals, empty responses)
- Desktop UI layout with right-side answer + analytics panels

## Project Structure

```text
MultiDocSource-RAG-AI-Assistant/
├── backend/
│   └── app/
│       ├── api/
│       ├── core/
│       ├── models/
│       ├── services/
│       ├── workers/
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

## Run Backend

```bash
./scripts/run_backend.sh
```

Backend URL: `http://127.0.0.1:8000`

## Run Desktop App

Open another terminal in the repo and run:

```bash
source .venv/bin/activate
./scripts/run_desktop.sh
```

## API Endpoints

- `POST /v1/ingest/file` - Upload PDF/Excel
- `POST /v1/ingest/api` - Ingest API JSON source
- `GET /v1/jobs/{job_id}` - Check async ingestion status
- `POST /v1/query` - Ask cross-source question
- `GET /v1/sources` - List indexed sources
- `GET /v1/dashboard` - Aggregated system stats

## Notes

- Embeddings use OpenAI by default (`EMBEDDING_MODEL_NAME`, default `text-embedding-3-large`) when `OPENAI_API_KEY` is set.
- If OpenAI embedding calls are unavailable, the backend automatically falls back to a local hash embedding implementation to stay operational.
- Answer generation supports follow-up context (`chat_history`) and produces grounded, refined explanations instead of raw chunk dumps.
- Optional: set `GENERATION_MODEL` (default: `gpt-5.4-mini`).
- Optional: set `CONFIDENCE_MODE` to `strict` (default), `high`, or `normal`.
- If your custom model alias fails, the app automatically falls back to `gpt-5.4`, `gpt-5.4-mini`, `gpt-4.1-mini`, and `gpt-4o-mini`.
- After embedding/chunking/config changes, restart backend and re-ingest documents for best retrieval quality.
- Excel ingestion is table-aware (sheet/row/column metadata) and row/column questions are answered directly when possible.
