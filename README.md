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
- Two evaluation workflows:
  - a lightweight Groq-judge evaluator
  - a simplified TruLens-style evaluator

## Architecture Overview

### Ingestion flow

Documents are loaded from the local [`data/`](/Users/nishantgupta/rag-bot-pinecone/data) directory by [`scripts/ingest.py`](/Users/nishantgupta/rag-bot-pinecone/scripts/ingest.py). The ingestion pipeline:

- loads `.txt` and `.pdf` files
- chunks them with `RecursiveCharacterTextSplitter`
- assigns deterministic chunk IDs
- computes a dataset hash for traceability
- upserts chunks into Pinecone
- saves chunks to `storage/chunks.pkl`
- uploads the chunk bundle to S3

This logic lives primarily in:

- [`scripts/ingest.py`](/Users/nishantgupta/rag-bot-pinecone/scripts/ingest.py)
- [`src/ragbot/pipeline/ingestion_pipeline.py`](/Users/nishantgupta/rag-bot-pinecone/src/ragbot/pipeline/ingestion_pipeline.py)
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
- TruLens package for simplified evaluation support

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

### Example query request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"How does async await work in Python?"}'
```

### Example response

```json
{
  "answer": "## Explanation\n...\n## Code\n...\n## Notes\n..."
}
```

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

[`scripts/evaluate.py`](/Users/nishantgupta/rag-bot-pinecone/scripts/evaluate.py) uses a Groq model as a judge for:

- retrieval relevance
- context coverage
- answer groundedness

It writes outputs to:

- [`evaluation/results_no_rewrite.json`](/Users/nishantgupta/rag-bot-pinecone/evaluation/results_no_rewrite.json)
- [`evaluation/results_with_rewrite.json`](/Users/nishantgupta/rag-bot-pinecone/evaluation/results_with_rewrite.json)
- [`evaluation/comparison.json`](/Users/nishantgupta/rag-bot-pinecone/evaluation/comparison.json)

Run it with:

```bash
python scripts/evaluate.py
```

### 2. Simplified TruLens-style evaluator

[`scripts/tru_lens_simple_eval.py`](/Users/nishantgupta/rag-bot-pinecone/scripts/tru_lens_simple_eval.py) performs post-hoc scoring for:

- answer relevance
- context relevance
- groundedness

It also compares:

- baseline retrieval
- retrieval with query rewriting

Outputs are written to:

- [`evaluation/simple_trulens_no_rewrite.json`](/Users/nishantgupta/rag-bot-pinecone/evaluation/simple_trulens_no_rewrite.json)
- [`evaluation/simple_trulens_with_rewrite.json`](/Users/nishantgupta/rag-bot-pinecone/evaluation/simple_trulens_with_rewrite.json)
- [`evaluation/simple_trulens_comparison.json`](/Users/nishantgupta/rag-bot-pinecone/evaluation/simple_trulens_comparison.json)

Run it with:

```bash
python scripts/tru_lens_simple_eval.py
```

### Evaluation query set

Both evaluation flows rely on:

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
python scripts/tru_lens_simple_eval.py
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

- add proper automated tests with assertions instead of one-off scripts
- return retrieved sources in the API response
- add source citations in generated answers
- add metadata filters for retrieval
- improve frontend integration and deployment setup
- add Docker support
- add Makefile or task runner commands
- split configuration by environment
- add structured observability for retrieval metrics

## License

No license file is currently present in the repository. Add one if you plan to distribute or open-source the project.

## Author

Nishant Gupta
