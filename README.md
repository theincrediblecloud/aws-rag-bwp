# RAG-Tutorials

**Goal:** Run a zero‑cloud Retrieval‑Augmented Generation (RAG) POC on your laptop. Ingest PDFs/DOCX/TXT/MD, build a local vector index (NumPy by default; FAISS optional), and answer questions via FastAPI with citations. The same repo also supports an AWS Bedrock mode for production.

---

## ragbot — local RAG proof‑of‑concept

Lightweight RAG for local use *and* a Bedrock‑backed production path. Ingest local documents (PDF/DOCX/TXT/MD), build a local vector store and metadata, then run an HTTP API that answers questions with short, cited passages.

### Project metadata

- **Name:** aws‑rag‑bwp (aka “ragbot”)  
- **Version:** 0.1.0  
- **Python:** 3.10+

### Repository layout (important files)

```
repo-root/
├─ data/                    # raw sources (pdfs, word, text_files) and collections/
├─ golden/                  # golden sets and evaluation runner
├─ src/rag/                 # python package (application code)
│  ├─ api/app.py            # Lambda + FastAPI handlers (/health, /chat)
│  ├─ core/config.py        # configuration and env parsing
│  ├─ ingest/pipeline.py    # ingest pipeline to build vectors/meta
│  └─ adapters/             # embedding + vector store adapters
│     ├─ vs_numpy.py        # default vector store: vectors.npy + meta.jsonl
│     ├─ embeddings_local.py
│     └─ embeddings_bedrock.py
├─ store*/                  # local index artifacts (vectors.npy, meta.jsonl)
├─ infra/sam/template.yaml  # AWS SAM infra for PROD
├─ tests/smoke_test.sh      # quick smoke against local API
├─ Makefile                 # common tasks (ingest/run/smoke/deploy)
├─ src/requirements.txt         # runtime deps for local mode
└─ src/requirements-dev.txt # dev deps (fastapi, loaders, sbert, etc.)
```

---

## Quickstart (local)

### 1) Prepare a data collection

> Use symlinks so `collections/kt` points only at the sources you want indexed. PDFs with heavy diagrams are supported; converting DOCX→MD often yields cleaner text for retrieval.

```bash
mkdir -p data/collections/kt
ln -sf ../../word/AWS-GenAI.md                             data/collections/kt/AWS-GenAI.md
ln -sf ../../word/AWS-GenAI.docx                           data/collections/kt/AWS-GenAI.docx
ln -sf ../../pdfs/bedrock-or-sagemaker.pdf                 data/collections/kt/Bedrock-or-SageMaker.pdf
ln -sf ../../pdfs/next-generation-sagemaker-ug.pdf         data/collections/kt/Next-Gen-SM-UG.pdf
ln -sf ../../pdfs/prompt-engineering-guidelines.pdf        data/collections/kt/Prompt-Eng-Guidelines.pdf
ln -sf ../../text_files/RAG.txt                            data/collections/kt/RAG.txt
```

### 2) Create env & install locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
export PYTHONPATH=$PWD/src
# (Optional) export values from .env into your shell
export $(grep -v '^#' .env | xargs) 2>/dev/null || true
```

### 3) Ingest & run locally

```bash
python -m rag.ingest.pipeline --data-dir data/collections/kt --fresh
uvicorn rag.api.app:app --reload --port 8000 --app-dir src
```

### 4) Smoke test

```bash
curl -s http://127.0.0.1:8000/health | jq .
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_msg":"What is Generative AI?","session_id":"dev"}' | jq .
```

---

## Evaluation (golden set)

The `golden/` folder contains evaluation helpers and JSONL golden data tailored to example collections. Run the evaluator against a running API:

```bash
python golden/eval_golden.py --api http://127.0.0.1:8000/chat golden/golden_set.jsonl -v
```

**Targets (project‑defined):** Hit@5: 80–90% | Citation present: 95–100%

---

## Embeddings & vector store

- **Local embeddings (offline default):** `sentence-transformers/all-MiniLM-L6-v2` (384‑dim)  
- **Offline 1024‑dim (optional):** `intfloat/e5-large` or `BAAI/bge-large-en-v1.5`  
- **Vector store:** NumPy store (`vectors.npy`) + `meta.jsonl` (Lambda‑friendly, no native deps).

> ⚠️ **Do not mix** indices and query embedders from different models/dimensions. If the index is 1024‑dim (e.g., Titan/e5/bge), the runtime must use the same model & dim.

---

## Features (Phase‑1 baseline)

- 🔎 **Citation‑first** answers from your corpus  
- 🧠 **Follow‑up aware**: “more / examples / elaborate / deep‑dive” expands the previous turn (same `session_id`)  
- ⚡ **Provisioned Concurrency** for Lambda (keeps N warm; reduces cold starts)  
- 🧰 **Two‑tier answer cache**  
  - Tier‑1: in‑memory LRU inside the Lambda process  
  - Tier‑2: DynamoDB item cache (TTL)  
- 📈 **CloudWatch telemetry**: retrieval scores, cache path, per‑request timings

> **Phase‑1 baseline (Oct 2025):** Follow‑up awareness · Provisioned Concurrency · Two‑tier caching (LRU + DynamoDB) · CloudWatch telemetry (scores, cache path, timings)

---

## Environment knobs (example `.env`)

```ini
# Mode
APP_ENV=local                 # local | bedrock
USE_S3_INDEX=false            # local uses files; bedrock downloads from S3
EMBED_PROVIDER=local

# Index
INDEX_DIR=store
INDEX_PATH=${INDEX_DIR}/vectors.npy
META_PATH=${INDEX_DIR}/meta.jsonl

# Embeddings (local)
EMBED_PROVIDER=local
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM=384

# Retrieval
CHUNK_SIZE=600
CHUNK_OVERLAP=80
RETRIEVE_K=24
CONTEXT_K=8
FALLBACK_MIN_SCORE=0.32
TOP_K=8
ANSWER_FALLBACK=allow         # allow | deny

# Bedrock (prod)
# EMBED_PROVIDER=bedrock
# AWS_REGION=us-east-1
# BEDROCK_EMBEDDINGS_ID=amazon.titan-embed-text-v2:0  # 1024‑dim
# LLM_PROVIDER=bedrock
# LLM_MODEL_ID=arn:aws:bedrock:us-east-1:<acct>:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0
# ARTIFACTS_BUCKET=slack-rag-artifacts-73918652
# INDEX_PREFIX=rag/index

# Caching (Lambda)
CACHE_TIER1=true
CACHE_TTL_SEC=1800
CACHE_MAX_ITEMS=1000

# Tier‑2 (DynamoDB) answer cache
CACHE_ANSWERS=true
DDB_CACHE_TABLE=<your-ddb-table-name>
CACHE_TTL_SECONDS=86400
INDEX_VERSION=v2
```

### ⚠️ Embedding dimensions

- Local default: all‑MiniLM‑L6‑v2 → **384‑dim**  
- Prod Bedrock (Titan v2): **1024‑dim**  
- **Index dim must match** query embedder dim. If you switch models, rebuild the index.

### Tips for working locally

- Ensure `PYTHONPATH=$PWD/src` so `import rag` works.  
- Keep `store*/` out of version control; it contains generated index files.  
- If PDFs are image‑heavy, consider alternate text sources (MD/DOCX) alongside.

---

## Slack integration (optional)

There is an example Lambda handler to forward Slack `/ask` to the RAG API. See `infra/sam/template.yaml` for deployment.

- Create Slack app with a Slash Command `/ask` and a Signing Secret.  
- Deploy Lambda behind API Gateway that forwards to this repo’s `/chat` endpoint.

---

## Deployment (AWS SAM)

`infra/sam/template.yaml` deploys:

- **RAGApi** (Python/arm64) with Provisioned Concurrency and env‑driven knobs  
- **SlackEventsFunction** that forwards Slack events to `/chat`  
- **DynamoDB** table for the answer cache (TTL enabled)  
- **HttpApi** (API Gateway v2) routes: `/chat`, `/health`, `/slack/events`

**Prod env excerpt:**

- `USE_S3_INDEX=true`, `ARTIFACTS_BUCKET`, `INDEX_PREFIX`  
- `EMBED_PROVIDER=bedrock` and `BEDROCK_EMBEDDINGS_ID`  
- `LLM_PROVIDER=bedrock` and `LLM_MODEL_ID`

---

## Troubleshooting

- **`dim mismatch: index_dim=1024 query_dim=384`** → embeddings model ≠ index model; rebuild or switch env.  
- **`RAG back-end not initialized`** → local env points at S3/Bedrock; set `USE_S3_INDEX=false` + local `INDEX_*`.  
- **Evaluator missing file** → point to the correct golden JSONL or create a tiny placeholder.  
- **Control character parse errors** → avoid prefixing JSON with extra text before piping to `jq`.

---

## Contributing & tests

- Add new collections under `data/collections/` and corresponding golden sets under `golden/`.  
- Keep ingest & evaluation deterministic; when chunking/model changes, update golden sets.  
- Run `make smoke` for a quick E2E check; `make ingest-*` to build indices.

**Quick checks:**

```bash
make ingest-kt     # example helper, if present
make run-local     # uvicorn
make smoke         # basic /health + /chat
pytest -q          # unit tests
```

---

## License

GPL‑3.0

---

## API samples

### `/health`

```bash
curl -s https://<api>/Prod/health | jq .
# { ok: true, rag_ready: true, ... }
```

**Sample response:**

```json
{ "ok": true, "rag_ready": true }
```

### `/chat`

```bash
curl -s -X POST https://<api>/Prod/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_msg":"what is generative AI?"}' | jq .
```

**Sample response:**

```json
{
  "answer": "High-level: Retrieved relevant passages.\n- ...",
  "citations": [
    {"idx": 1, "title": "bedrock-or-sagemaker.pdf", "source_path": "data/collections/kt/bedrock-or-sagemaker.pdf", "page": 12, "score": 0.92, "chunk_text": "..."}
  ],
  "session_id": "test",
  "domain": null
}
```

---

## Vector store (today & Phase‑2)

- **Today (Phase‑1):** NumPy (`vectors.npy`) + `meta.jsonl` (Lambda‑friendly). Loads from S3 in prod (`USE_S3_INDEX=true`) or local paths in dev.  
- **Phase‑2 (planned):** Amazon OpenSearch Serverless (k‑NN) or Aurora PostgreSQL (pgvector); hybrid search (BM25 + vector), ACL‑aware filters, streaming/online ingest, and latency/recall/cost benchmarks vs NumPy.
