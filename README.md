# RAG Bot with Pinecone, OpenAI Embeddings, and Groq

This project is a Retrieval-Augmented Generation (RAG) assistant focused on Python documentation-style question answering. It combines:

- OpenAI embeddings for semantic search
- Pinecone for vector storage
- BM25 for keyword retrieval
- Groq-hosted LLMs for answer generation
- Optional Groq-judged evaluation pipelines
- FastAPI for serving the application
- A lightweight browser frontend for interacting with the bot

The current implementation is tuned around answering Python documentation questions with grounded, code-oriented responses.

## What This Project Does

The application ingests local documents from the [`data/`](/Users/nishantgupta/rag-bot-pinecone/data) directory, chunks them, embeds them with OpenAI, stores vectors in Pinecone, and persists chunk objects for keyword retrieval. At query time, it:

1. Accepts a question through the API or frontend.
2. Optionally rewrites the query into alternate retrieval-friendly variants.
3. Retrieves relevant chunks using both:
   - Pinecone vector similarity
   - BM25 keyword scoring
4. Merges results using Reciprocal Rank Fusion (RRF).
5. Builds a prompt from the retrieved context.
6. Sends the prompt to a Groq chat model.
7. Returns a structured answer with explanation, code, and notes.

## Core Features

- Hybrid retrieval with vector search plus BM25
- Groq-powered answer generation
- OpenAI embeddings via `langchain-openai`
- Pinecone-backed vector store
- Query rewriting support for recall improvement
- S3-based chunk persistence for retrieval bootstrap
- FastAPI API with health and query endpoints
- Browser-based UI
- Deterministic evaluation with measurable retrieval and answer-quality metrics

## Architecture Overview

### Ingestion flow

Documents are loaded from the local [`data/`](/Users/nishantgupta/rag-bot-pinecone/data) directory, either via [`scripts/ingest.py`](/Users/nishantgupta/rag-bot-pinecone/scripts/ingest.py) (CLI) or the `POST /ingest` endpoint. The ingestion pipeline:

- loads `.txt` and `.pdf` files
- hashes each source document's content and compares it to the previously ingested corpus:
  - **unchanged** sources are skipped entirely - no re-chunking, no re-embedding, no upsert
  - **new or changed** sources are chunked with `RecursiveCharacterTextSplitter` and (re-)embedded
  - **changed or removed** sources have their old chunks deleted from Pinecone by id first, so edits and deletions don't leave stale vectors behind
- assigns deterministic chunk IDs
- computes a dataset hash for traceability
- upserts new/changed chunks into Pinecone
- saves the full chunk corpus (reused + new) to `storage/chunks.pkl`
- uploads the full chunk corpus to S3

This logic lives primarily in:

- [`src/ragbot/services/ingestion_service.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/services/ingestion_service.py) (document loading + orchestration, shared by the CLI and the API)
- [`src/ragbot/pipeline/ingestion_pipeline.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/pipeline/ingestion_pipeline.py) (change-detection, chunking, upsert/delete)
- [`src/ragbot/vectorstore/pinecone_store.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/vectorstore/pinecone_store.py)
- [`src/ragbot/ingestion/s3_storage.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/ingestion/s3_storage.py)

### Query flow

At runtime, [`src/ragbot/pipeline/rag_pipeline.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/pipeline/rag_pipeline.py) initializes:

- a Pinecone vector store backed by OpenAI embeddings
- a document list restored from S3
- a hybrid retriever using vector search and BM25
- a Groq chat model
- an optional query rewriter

The pipeline then:

- retrieves documents
- formats them into a context block
- fills the RAG prompt template
- generates an answer through Groq
- lightly normalizes the answer formatting for markdown output

### Retrieval strategy

Hybrid retrieval is implemented in [`src/ragbot/retrievers/hybrid_retriever.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/retrievers/hybrid_retriever.py). It combines:

- vector similarity from Pinecone
- BM25 scoring across persisted chunk texts
- Reciprocal Rank Fusion to merge both ranked lists

This gives the project both semantic recall and exact-term sensitivity.

### Prompting strategy

The answer prompt is defined in [`src/ragbot/prompts/rag_prompt.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/prompts/rag_prompt.py). It instructs the model to:

- stay grounded in the provided context
- return an explicit fallback if context is insufficient
- format answers into:
  - Explanation
  - Code Example(s)
  - Notes

The project is therefore optimized for implementable, documentation-style answers rather than open-ended chat.

## Tech Stack

- Python 3.11+
- FastAPI
- LangChain
- Groq
- OpenAI Embeddings
- Pinecone
- BM25 via `rank-bm25`
- Boto3 for S3 chunk storage

## Project Structure

```text
rag-bot-pinecone/
├── data/                         # Source documents for ingestion
├── evaluation/                   # Query sets and evaluation outputs
├── frontend/                     # Static HTML frontend
├── scripts/                      # Ingestion, deletion, and evaluation scripts
├── src/ragbot/
│   ├── api/                      # FastAPI routes
│   ├── config/                   # Environment-based settings
│   ├── ingestion/                # S3 storage helpers
│   ├── pipeline/                 # Ingestion and runtime RAG pipelines
│   ├── prompts/                  # Prompt templates
│   ├── retrievers/               # Hybrid retrieval and query rewriting
│   ├── services/                 # Ingestion service backing scripts/ingest.py and /ingest, /index
│   ├── utils/                    # Logging and formatting helpers
│   └── vectorstore/              # Pinecone + embeddings wrapper
├── storage/                      # Local chunk persistence
├── requirements.txt
├── runtime.txt
└── setup.py
```

## Requirements

### Python version

The project is configured for Python 3.11 or higher:

- [`setup.py`](/Users/nishantgupta/rag-bot-pinecone/setup.py) sets `python_requires=">=3.11"`
- [`runtime.txt`](/Users/nishantgupta/rag-bot-pinecone/runtime.txt) specifies `3.11.9`

Use Python 3.11 to avoid package compatibility problems.

### External services

You will need accounts and credentials for:

- OpenAI
- Groq
- Pinecone
- AWS S3

## Environment Variables

The app reads configuration from environment variables through [`src/ragbot/config/settings.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/config/settings.py).

Set the following in your `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key

PINECONE_INDEX_NAME=ragbot-index
PINECONE_ENVIRONMENT=your_pinecone_environment

AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=your_aws_region
S3_BUCKET=your_s3_bucket
S3_CHUNKS_KEY=path/to/chunks.pkl
```

Optional, for Opik tracing (see [Tracing (Opik)](#tracing-opik)):

```env
OPIK_API_KEY=your_opik_api_key
OPIK_WORKSPACE=your_opik_workspace
OPIK_PROJECT_NAME=ragbot
RAGBOT_ENABLE_OPIK=true
```

### Runtime defaults from settings

The project currently defaults to:

- embedding model: `text-embedding-3-small`
- LLM model: `llama-3.1-8b-instant`
- temperature: `0.0`
- max tokens: `512`
- chunk size: `800`
- chunk overlap: `150`
- top-k retrieval: `5`

These are defined in [`src/ragbot/config/settings.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/config/settings.py).

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-bot-pinecone
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The editable install is useful so `ragbot` imports work cleanly during development.

## Running the Application

### Start the API server

```bash
uvicorn ragbot.main:app --reload
```

By default this will start the FastAPI app locally on port `8000`.

### API endpoints

The routes are defined in [`src/ragbot/api/routes.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/api/routes.py).

Available endpoints:

- `GET /`
  - returns a simple welcome message
- `GET /health`
  - returns service health status
- `POST /query`
  - accepts a user question and returns the generated answer
- `POST /ingest`
  - loads documents from the local `data/` directory and upserts only the new/changed ones into Pinecone (unchanged documents are skipped; changed or removed documents have their stale vectors deleted); reloads the running retriever afterward so new data is queryable without a restart
- `DELETE /index`
  - deletes all vectors from the configured Pinecone index

### Example query request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"How does async await work in Python?"}'
```

### Example response

```json
{
  "answer": "## Answer\n...\n## Evidence\n...\n## Code Example(s)\n...\n## Notes\n...",
  "sources": [
    {"source": "data/library/asyncio-task.txt", "url": "https://docs.python.org/3/library/asyncio-task.html"}
  ],
  "metrics": { "...": "retrieval/generation latency, scores, etc. - see QueryTelemetry" }
}
```

`sources` lists the distinct documents the answer was grounded in. Since the ingested corpus mirrors docs.python.org's own directory layout, each source maps mechanically back to its canonical page (`ragbot.utils.citations.source_to_docs_url`); `url` is `null` for any source that doesn't fit that shape.

## Frontend

The project includes a static frontend in [`frontend/index.html`](/Users/nishantgupta/rag-bot-pinecone/frontend/index.html).

It provides:

- a styled question input
- a response rendering area
- markdown rendering via `marked`
- a Python-documentation-assistant themed UI

The frontend is best used while the FastAPI backend is running locally.

If you want to serve it quickly during development, you can use a simple static server:

```bash
python -m http.server 3000
```

Then open the page in your browser and point requests to your local API if needed.

## Data Ingestion

### Supported sources

The ingestion script currently loads:

- `.txt` files
- `.pdf` files

from the [`data/`](/Users/nishantgupta/rag-bot-pinecone/data) folder.

### Run ingestion

```bash
python scripts/ingest.py
```

### What ingestion does

During ingestion:

1. documents are loaded from `data/`
2. documents are split into overlapping chunks
3. each chunk receives a deterministic SHA-256-based `chunk_id`
4. a dataset fingerprint is computed
5. chunks are embedded using OpenAI embeddings
6. vectors are upserted to Pinecone
7. chunks are serialized locally to `storage/chunks.pkl`
8. chunks are uploaded to S3 for later retrieval bootstrap

### Why local and S3 chunk persistence exist

This project uses BM25 in addition to vector search. BM25 needs access to the raw chunk texts, not just vectors. To support this, chunk objects are persisted locally and also uploaded to S3 so the runtime pipeline can rebuild the keyword retriever without re-running ingestion.

## Query Rewriting

Query rewriting is implemented in [`src/ragbot/retrievers/query_rewriter.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/retrievers/query_rewriter.py).

When enabled, the pipeline:

- keeps the original user query
- asks the LLM for 3 alternate rewrites
- retrieves against each variant
- merges the result sets with RRF

This is designed to improve recall when the original query is underspecified or phrased differently than the underlying documents.

In [`src/ragbot/pipeline/rag_pipeline.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/pipeline/rag_pipeline.py), query rewriting is controlled by:

```python
self.use_query_rewriting = False
```

The evaluation scripts compare baseline retrieval against retrieval with rewriting enabled.

## Evaluation

The repo currently contains two evaluation approaches.

### 1. Lightweight evaluation runner

[`scripts/evaluate.py`](/Users/nishantgupta/rag-bot-pinecone/scripts/evaluate.py) computes deterministic metrics for:

- retrieval hit rate at k
- mean reciprocal rank at k
- context precision at k
- answer keyword recall
- fallback rate
- latency summaries

It writes outputs to:

- [`evaluation/results_no_rewrite.json`](/Users/nishantgupta/rag-bot-pinecone/evaluation/results_no_rewrite.json)
- [`evaluation/results_with_rewrite.json`](/Users/nishantgupta/rag-bot-pinecone/evaluation/results_with_rewrite.json)
- [`evaluation/comparison.json`](/Users/nishantgupta/rag-bot-pinecone/evaluation/comparison.json)

Run it with:

```bash
python scripts/evaluate.py
```

### Evaluation query set

The evaluation flow relies on:

- [`evaluation/queries.json`](/Users/nishantgupta/rag-bot-pinecone/evaluation/queries.json)

Edit that file to customize the benchmark set for your own domain or retrieval tests.

## Maintenance Scripts

### Clear the Pinecone index

If you need to remove all vectors from the configured Pinecone index:

```bash
python scripts/delete.py
```

This script calls [`scripts/delete.py`](/Users/nishantgupta/rag-bot-pinecone/scripts/delete.py), which delegates to the `PineconeStore.delete_all()` method.

Use this carefully because it clears the entire configured index.

## Development Notes

### Logging

Logging is handled through [`src/ragbot/utils/logger.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/utils/logger.py). The current logger:

- logs to stdout
- uses `INFO` level by default
- includes timestamps, logger names, and message levels

### Output formatting

[`src/ragbot/utils/formatter.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/utils/formatter.py) lightly normalizes model responses into markdown-style section headings.

### Packaging

The package metadata is defined in [`setup.py`](/Users/nishantgupta/rag-bot-pinecone/setup.py). The package name is `ragbot`.

### Testing

Automated tests live under [`tests/`](tests/) and run with `pytest`. They're hermetic `unittest` suites — no network calls, no API keys beyond what importing `ragbot.config.settings` already needs from `.env` — achieved by bypassing `__init__` on classes that build real Pinecone/LLM/S3 clients (`Class.__new__(Class)`) and injecting fakes for just the collaborators each test needs, or faking a module out via `sys.modules` before importing something that instantiates one at import time (see `tests/test_api_routes.py`).

```bash
pytest tests/
```

Covers: hybrid retrieval reranking, recursive document loading, evaluation metrics, Opik tracing's fail-open behavior, observability aggregation, answer/section Markdown normalization, source-to-docs-URL citation mapping, the ingestion pipeline's new/changed/unchanged/removed source diffing, the RAG pipeline's result merging and citation building, and the `/query`/`/health` API contract.

### Tracing (Opik)

[Opik](https://www.comet.com/docs/opik/) tracing is wired through [`src/ragbot/observability/opik_tracing.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/observability/opik_tracing.py), which exposes:

- `track` - a drop-in replacement for `@opik.track` that becomes a true no-op when `RAGBOT_ENABLE_OPIK=false`
- `get_langchain_tracer` - builds an `OpikTracer` LangChain callback (returns `None` when tracing is disabled)
- `update_trace_metadata` - safely attaches metadata/tags to the current trace; never raises, even with no active trace

Instrumented today:

- `IngestionPipeline.ingest` (root trace, tagged `ingestion`, annotated with the same `IngestionMetrics` that get logged locally)
- `PineconeStore.add_documents` / `similarity_search`, and the S3 `upload_chunks` / `download_chunks` helpers, as nested tool spans
- `RAGPipeline.run` (root trace, tagged `rag_query`, annotated with `QueryTelemetry` and the app's own `trace_id`) and `_retrieve_documents`
- `HybridRetriever.retrieve` / `vector_search` / `keyword_search`, and `QueryRewriter.rewrite`, as nested tool/llm spans
- both LLM calls (main generation and query rewriting) via an `OpikTracer` LangChain callback, giving token/cost/latency detail per call

Tracing is designed to fail open: with `OPIK_API_KEY` unset, Opik logs a warning and traces simply don't go anywhere - it will not raise or add meaningful latency to a request. Set `RAGBOT_ENABLE_OPIK=false` to disable instrumentation entirely (e.g. in tests or before Opik credentials are configured).

## Deployment

Deploys to Cloud Run (`ragbot`, project `ragbag-508901`, region `us-central1`) run through [`cloudbuild.yaml`](cloudbuild.yaml) via a Cloud Build trigger on push to `main` — no more manual `gcloud run deploy --source .`.

- The build steps: build the Docker image, push it to Artifact Registry, then `gcloud run deploy` with that image.
- Credentials (OpenAI, Pinecone, Groq, AWS, Opik keys) are pulled from **Secret Manager** at deploy time via `--update-secrets`, not committed anywhere — this repo is public, so nothing in `cloudbuild.yaml` is sensitive. Non-secret runtime config (bucket names, region, index name, etc.) is set directly via `--set-env-vars` in the same file.
- To rotate a credential: `gcloud secrets versions add <SECRET_NAME> --project=ragbag-508901 --data-file=-`, then push (or manually re-trigger the build) so Cloud Run picks up `:latest`.
- One-time setup already done: the six secrets exist in Secret Manager with the Cloud Run runtime service account (`521740466585-compute@developer.gserviceaccount.com`) granted `roles/secretmanager.secretAccessor` on each.

## Example End-to-End Workflow

### Step 1. Add your source documents

Place `.txt` or `.pdf` files into [`data/`](/Users/nishantgupta/rag-bot-pinecone/data).

### Step 2. Configure environment variables

Create and populate your `.env` file with OpenAI, Groq, Pinecone, and AWS credentials.

### Step 3. Ingest the corpus

```bash
python scripts/ingest.py
```

### Step 4. Start the API

```bash
uvicorn ragbot.main:app --reload
```

### Step 5. Query the bot

Use either:

- `curl`
- Postman
- the browser frontend

### Step 6. Evaluate retrieval and answer quality

```bash
python scripts/evaluate.py
```

## Troubleshooting

### `Chunks not found in S3`

Cause:

- ingestion has not been run yet
- the S3 bucket or key is incorrect
- AWS credentials are missing or invalid

Fix:

- verify your AWS environment variables
- verify `S3_BUCKET` and `S3_CHUNKS_KEY`
- run ingestion again

### `Chunks not found. Run ingestion first`

Cause:

- local chunk persistence was not created

Fix:

- run `python scripts/ingest.py`

### Pinecone returns no useful results

Possible causes:

- wrong index name
- wrong API key
- embeddings were written to a different index
- the dataset was not ingested successfully

Fix:

- confirm Pinecone credentials and index settings
- inspect Pinecone index stats after ingestion
- re-ingest documents if needed

### API starts but answers are weak or generic

Possible causes:

- poor source data quality
- insufficient chunk coverage
- retrieval misses relevant sections
- prompt context does not contain enough evidence

Fix:

- improve document quality
- adjust chunk size and chunk overlap
- inspect retrieved chunks
- compare no-rewrite and rewrite evaluation results

### Import errors during development

Fix:

- make sure you activated a Python 3.11 virtual environment
- run `pip install -r requirements.txt`
- run `pip install -e .`

## Current Design Assumptions

This repository is currently oriented around:

- Python documentation or Python-adjacent technical content
- code-centric answers
- chunk-based retrieval
- a single Pinecone index
- runtime chunk restoration from S3

If you want to adapt it to another domain, the main places to update are:

- the prompt template
- evaluation queries
- source documents
- query rewriting prompt
- frontend copy

## Suggested Next Improvements

- add metadata filters for retrieval
- add Makefile or task runner commands
- split configuration by environment
- add structured observability for retrieval metrics
- use Redis to cache the built BM25 index (not just raw chunks) to reduce cold-start latency
- improve answer formatting — drop the "Evidence" section from generated answers
- set Cloud Run `min-instances >= 1` to eliminate scale-to-zero cold starts
- switch `/query` to `async def` with LangChain's `.ainvoke()` so one instance isn't capped at ~40 concurrently-executing requests by FastAPI's sync threadpool
- add retry/backoff around Groq/OpenAI/Pinecone calls so provider rate-limit errors (429s) degrade gracefully instead of failing the request
- check actual Groq/OpenAI/Pinecone account rate limits (RPM/TPM) - likely the real ceiling on concurrent throughput, ahead of any Cloud Run scaling setting

## License

No license file is currently present in the repository. Add one if you plan to distribute or open-source the project.

## Author

Nishant Gupta
