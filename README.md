# MultiDocSource RAG AI Assistant

A cheerful little chaos-tamer for PDFs, Excels, and APIs.

Think of it as:
- one part librarian
- one part detective
- one part intern who never sleeps
- and one part "please just tell me the exact answer"

---

## What This App Does

This app lets you ingest multiple data sources and ask questions in plain English.

It can handle:
- PDF content (including table-like and field-like data)
- Excel rows/columns
- API JSON payloads

It aims to return:
- direct factual answers when possible
- grounded concise answers for broader questions
- reliable lookups for field/table queries

---

## How It Works (High-Level)

1. You upload data (PDF / Excel / API).
2. Parser converts documents into records.
3. Records are chunked (for non-tabular text).
4. Embeddings are generated.
5. Vectors are stored in either memory or Pinecone.
6. On query:
   - deterministic lookup tries first (field/table/entity)
   - then retrieval + multi-agent generation runs
   - anti-vague guard catches weak answers and falls back to deterministic extraction

In short: try exact first, generate second.

---

## Multi-Agent Flow (LangGraph)

Generation path is orchestrated with agents:

- `retrieval_agent` -> fetches candidate context
- `reasoning_agent` -> drafts answer from context
- `critic_agent` -> rejects weak/noisy outputs
- `formatter_agent` -> returns clean final response

Flow:

`retrieval -> reasoning -> critic -> formatter`

---

## Directory Structure

```text
MultiDocSource-RAG-AI-Assistant/
├── backend/
│   └── app/
│       ├── agents/                 # LangGraph orchestration layer
│       │   ├── graph.py            # workflow wiring
│       │   ├── state.py            # shared graph state
│       │   ├── executor.py         # run_agents entrypoint
│       │   ├── nodes/              # retrieval/reasoning/critic/formatter nodes
│       │   ├── tools/              # vector/parser helper tools
│       │   └── prompts/            # agent prompt templates
│       ├── api/
│       │   └── routes.py           # REST endpoints
│       ├── core/
│       │   ├── config.py           # env + app config
│       │   └── guardrails.py       # query mode + response style rules
│       ├── models/
│       │   └── schemas.py          # request/response models
│       ├── services/
│       │   ├── chunking.py         # chunk creation strategy
│       │   ├── embeddings.py       # OpenAI/hash embedding service
│       │   ├── parsers.py          # PDF/Excel/API parsing + structured extraction
│       │   ├── pinecone_store.py   # Pinecone vector backend
│       │   ├── rag.py              # main retrieval + answer orchestration
│       │   ├── source_registry.py  # in-memory source/chunk registry
│       │   └── vector_store.py     # local hybrid retrieval backend
│       ├── workers/
│       │   └── jobs.py             # async ingestion jobs
│       ├── state.py                # runtime wiring (embedder/store/manager)
│       └── main.py                 # FastAPI app entrypoint
├── desktop/
│   ├── api_client.py               # desktop-to-backend client
│   └── app.py                      # PyQt UI
├── scripts/
│   ├── run_backend.sh
│   ├── run_desktop.sh
│   └── run_all.sh
├── data/uploads/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Vector Backends

This app supports:

- `memory` mode: quick local tests, data disappears on restart
- `pinecone` mode: persistent vectors, scalable retrieval

Recommended Pinecone pattern:
- single index (`rag-docs`)
- filter by `source_id`
- optional namespace strategy

---

## Query Behavior (Practical)

The query pipeline prefers precision:

1. Entity/field extraction (e.g. `notary name`, `phone number`, `address`)
2. Structured row-data lookup
3. Source-wide deterministic field search
4. Table lookup
5. Multi-agent generative answer
6. Final anti-vague relevance check

This helps avoid random chunk blurbs when user asks a simple factual question.

---

## Setup

```bash
cd /Users/homesachin/Desktop/zoneone/practice/MultiDocSource-RAG-AI-Assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Environment

Use `.env` (do not hardcode secrets).

Minimum local setup:

```env
OPENAI_API_KEY=
VECTOR_BACKEND=memory
```

Pinecone mode:

```env
VECTOR_BACKEND=pinecone
PINECONE_API_KEY=
PINECONE_INDEX_NAME=rag-docs
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE_MODE=single
PINECONE_NAMESPACE=default
```

Note: embedding dimension must match your Pinecone index dimension.

---

## Run

Backend:

```bash
source .venv/bin/activate
./scripts/run_backend.sh
```

Desktop:

```bash
source .venv/bin/activate
./scripts/run_desktop.sh
```

All-in-one:

```bash
source .venv/bin/activate
./scripts/run_all.sh
```

---

## API Endpoints

- `POST /v1/ingest/file`
- `POST /v1/ingest/api`
- `GET /v1/jobs/{job_id}`
- `POST /v1/query`
- `GET /v1/sources`
- `DELETE /v1/sources/{source_id}`
- `GET /v1/dashboard`

---

## Golden Rule: Re-Ingest After Parser/Retrieval Changes

If parsing/chunking/retrieval logic changes, old vectors are stale.

Do this:
1. clear old indexed data
2. restart backend
3. re-ingest documents
4. retest

If you skip this, you will absolutely get old behavior and then question your life choices.

---

## Troubleshooting

### Port 8000 already in use

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill -9 <PID>
```

### `No module named 'websockets.typing'`

```bash
source .venv/bin/activate
python -m pip install "websockets==13.1"
```

### Still getting old/wrong answers

- clear Pinecone/indexed data
- re-ingest with latest code
- confirm latest `source_id` is used

---

