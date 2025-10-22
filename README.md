# RAG-Tutorials
**Goal:** Run a zero-cloud Retrieval-Augmented Generation POC on your laptop. Ingest PDFs/DOCX/TXT/MD, build a FAISS index, and answer questions via FastAPI with citations.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add a few files under data/sample/
python -m rag.ingest.pipeline --data-dir data/collections/fam --fresh
uvicorn rag.api.app:app --reload --port 8000 --app-dir src
```

### Test
```bash
## ragbot — local RAG proof-of-concept

Lightweight Retrieval-Augmented-Generation (RAG) proof-of-concept for local use. Ingest local documents (PDF/DOCX/TXT/MD), build a local vector store and metadata, then run an HTTP API that answers questions with short, cited passages.

Project metadata
- Name: ragbot
- Version: 0.1.0
- Python: 3.10+

Repository layout (important files)

```
repo-root/
├─ data/                    # raw sources (pdfs, word, text_files) and collections/
├─ golden/                  # golden sets and evaluation runner
├─ src/rag/                 # python package (application code)
│  ├─ api/app.py            # Lambda + FastAPI handlers (/health, /chat)
│  ├─ core/config.py        # configuration and env parsing
│  ├─ ingest/pipeline.py    # ingest pipeline to build vectors/meta
│  └─ adapters/             # embedding + vector store adapters
├─ store/                   # local index artifacts (faiss.index, meta.jsonl, vectors.npy)
├─ pyproject.toml           # project metadata
└─ requirements.txt         # (optional) pinned deps for local venv
```

Quickstart (local, minimal)

1. Create and activate a virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare env and PYTHONPATH

```bash
cp .env.example .env
export $(grep -v '^#' .env | xargs) 2>/dev/null || true
export PYTHONPATH=$PWD/src
```

3. Add or symlink a collection of documents to `data/collections/<name>` (example: `fam`) and ingest

```bash
python -m rag.ingest.pipeline --data-dir data/collections/fam --fresh
```

4. Run the API (development)

```bash
uvicorn rag.api.app:app --reload --port 8000 --app-dir src
```

5. Smoke test

```bash
curl -s http://127.0.0.1:8000/health | jq .
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_msg":"What is the primary purpose of the FAM Moderation system?"}' | jq .
```

Evaluation (golden set)

The `golden/` folder contains evaluation helpers and JSONL golden data tailored to example collections. Run the evaluator against a running API:

```bash
python golden/eval_golden.py --api http://127.0.0.1:8000/chat golden/fam.jsonl -v
```

Typical targets (project-defined)
- Hit@5: 80–90%
- Citation present: 95–100%

Embedding & vector-store choices

- Local embeddings: `sentence-transformers/all-MiniLM-L6-v2` is the default (384 dim).
- Vector store: FAISS (native) or numpy-backed store (used for simplified Lambda deployment). See `src/rag/adapters/` for implementations: `embeddings_local.py`, `vs_numpy.py`.

Environment knobs (example `.env`)

APP_ENV=local
INDEX_DIR=store
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM=384
CHUNK_SIZE=600
CHUNK_OVERLAP=80
TOP_K=8

Tips for working locally

- Ensure `PYTHONPATH=$PWD/src` so `import rag` works when running modules directly.
- If you prefer, install the package in editable mode: `pip install -e .` then run modules without `--app-dir` hacks.
- Keep `store/` out of version control; it contains generated index files.

Slack integration (optional)

There is an example Lambda handler to forward Slack /ask commands to the running RAG API. See `infra/lambda_slack_handler.py` and `infra/sam/template.yaml` for deployment hints. Typical flow:

- Create Slack app with a Slash Command `/ask` and a Signing Secret.
- Deploy Lambda behind a Function URL or API Gateway that forwards the command to this repo's `/chat` endpoint.

Deployment (AWS SAM)

The repo contains SAM resources in `infra/sam/template.yaml` and helper scripts. The Lambda entrypoint is `rag.api.app.handler`. Secrets should be provided via Secrets Manager with least-privilege IAM policies.

Troubleshooting

- "ModuleNotFoundError: rag": make sure `PYTHONPATH=./src` or install the package editable.
- If you see `AttributeError: 'list' object has no attribute 'astype'` from `vs_numpy`, ensure embeddings are passed as numeric arrays or lists are converted; `vs_numpy.search` accepts list inputs.
- If the evaluator complains about missing `golden_set.jsonl`, point it to the correct golden file under `golden/` or generate a small placeholder to test.

Contributing and tests

- Add new collections under `data/collections/` and corresponding golden sets under `golden/`.
- Keep ingest and evaluation deterministic where possible; changes to chunking or embedding models should be reflected in the golden set.

License

See `LICENSE` at the repo root.

Acknowledgements

This project is a compact demo to explore local RAG, citation-first behavior, and simple evaluation workflows.

---
Generated from repository metadata and existing documentation.

Sample response (what /chat returns)

```json
{
  "answer": "High-level: Retrieved relevant passages.\n- ...",
  "citations": [
    {"idx": 1, "title": "FAM-Moderation-System-Design.pdf", "source_path": "data/collections/fam/FAM-Moderation-System-Design.pdf", "page": 12, "score": 0.92, "chunk_text": "...extracted snippet..."}
  ],
  "session_id": "test",
  "domain": null
}
```

Try it

Start the API and run the smoke test:

```bash
uvicorn rag.api.app:app --reload --port 8000 --app-dir src
./tests/smoke_test.sh
```

