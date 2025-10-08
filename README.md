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
curl -s http://localhost:8000/health | jq
curl -s -X POST http://localhost:8000/chat \
 -H 'Content-Type: application/json' \
 -d '{"user_msg":"What does the document say about 3DS challenges?","session_id":"dev"}' | jq
```

## Demo
```bash
python golden/eval_golden.py --api http://127.0.0.1:8000/chat golden/fam.jsonl
```

## Notes


---

### Secure RAG — FAM MVP (Local)

A tiny, production-feeling RAG for FAM docs with:
Local embeddings + FAISS (no public LLM)
Clear citations (cite-or-silence)
Golden set evaluation (Hit@k & citation presence)
Optional Slack /ask command

## Prereqs

Python 3.10–3.12 (3.13 works if your wheels installed)
macOS/Linux
(Optional) jq, pandoc for convenience
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
## Project layout
repo-root/
├─ data/
│  ├─ pdfs/                # raw PDFs
│  ├─ word/                # raw DOCX/MD/TXT
│  ├─ text_files/
│  └─ collections/
│     └─ fam/              # ✅ symlinks to only FAM sources
├─ golden/
│  ├─ fam.jsonl            # ✅ FAM golden set
│  └─ eval_golden.py       # eval runner
├─ src/
│  └─ rag/
│     ├─ api/app.py        # FastAPI /chat, /health
│     ├─ core/config.py    # AppConfig (env/paths)
│     ├─ ingest/pipeline.py
│     ├─ adapters/
│     │  ├─ vs_faiss.py    # FAISS store (search, upsert)
│     │  └─ embeddings_local.py
│     └─ ingest/loaders.py # file loaders
└─ store/                  # ✅ local index dir (faiss.index, meta.jsonl)


Tip: add store/ to .gitignore.

## Environment knobs

Create .env (or export in shell):

APP_ENV=local
INDEX_DIR=store

# retrieval
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM=384
CHUNK_SIZE=600
CHUNK_OVERLAP=80
TOP_K=8

# (optional) behavior toggles you may already support
ANSWER_MODE=simple        # simple | smart


Load env in your shell:
    ```bash
    export $(grep -v '^#' .env | xargs) 2>/dev/null || true
    export PYTHONPATH=$PWD/src
    ```
## Prepare the FAM collection

Symlink only the FAM sources you want included:
```bash
mkdir -p data/collections/fam

ln -sf ../../pdfs/FAM\ Moderation\ System\ Design.pdf              data/collections/fam/FAM-Moderation-System-Design.pdf
ln -sf ../../pdfs/FAM\ Moderation\ Solution\ Discussion.pdf        data/collections/fam/FAM-Moderation-Solution-Discussion.pdf
ln -sf "../../word/FAM Team_ Monitoring & Moderation PRFAQ.docx"   data/collections/fam/FAM-PRFAQ.docx
ln -sf ../../pdfs/\[SantosDesign\]\ Santos\ Catalog\ Moderation.pdf data/collections/fam/Santos-Catalog-Moderation.pdf
```
If DOCX retrieval seems weak, convert PRFAQ to Markdown or TXT and re-link:
```bash
pandoc "data/word/FAM Team_ Monitoring & Moderation PRFAQ.docx" -o "data/word/FAM-PRFAQ.md"
ln -sf ../../word/FAM-PRFAQ.md data/collections/fam/FAM-PRFAQ.md
```
## Build index (clean) for FAM
# wipe current index then ingest only FAM
```bash
rm -rf "$INDEX_DIR"/*
python -m rag.ingest.pipeline --data-dir data/collections/fam --fresh
ls -l store/   # expect: faiss.index, meta.jsonl
```
## Run API + smoke tests
```bash
uvicorn rag.api.app:app --reload --port 8000 --app-dir src
```
# new terminal
```bash
curl -s http://127.0.0.1:8000/health | jq .
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_msg":"What is the primary purpose of the FAM Moderation system?"}' | jq .
```

You should see answers with 1–3 citations (title, file, page).

## Evaluate with golden set

Golden set lives at golden/fam.jsonl (already tailored to your FAM files).
```bash
python golden/eval_golden.py --api http://127.0.0.1:8000/chat golden/fam.jsonl -v
```
Targets (MVP):
    Hit@5 ≥ 80–90%
    Citation present ≥ 95–100%

## Slack (optional MVP)

Fastest path is a Lambda Function URL hitting your API then posting to Slack via response_url.
Create the Slack app (slash command /ask) and note Signing Secret.
Deploy infra/lambda_slack_handler.py to AWS Lambda (Python 3.11).
    Env vars:
        RAG_API=https://<your-api>/chat
        SLACK_SIGNING_SECRET=<from Slack>

Create a Function URL (auth: NONE) and paste that URL into the Slash Command → Request URL.
Test in Slack:
/ask Summarize the FAM moderation architecture

If your /chat is slower than ~2–3s, keep the handler’s “delayed response” pattern (ACK fast, post later via response_url).

## Common commands (cheat-sheet)
# activate venv
source .venv/bin/activate

# export env
```bash
export $(grep -v '^#' .env | xargs) 2>/dev/null || true
export PYTHONPATH=$PWD/src

# rebuild FAM index from scratch
rm -rf "$INDEX_DIR"/*
python -m rag.ingest.pipeline --data-dir data/collections/fam --fresh

# run API
uvicorn rag.api.app:app --reload --port 8000 --app-dir src

# health & sample question
curl -s http://127.0.0.1:8000/health | jq .
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"user_msg":"Which external services does Santos Catalog Moderation depend on?"}' | jq .

# run evaluation
python golden/eval_golden.py --api http://127.0.0.1:8000/chat golden/fam.jsonl
```
## Troubleshooting

Connection refused
    Start API: uvicorn …
    Use 127.0.0.1 not localhost if you see socket issues.
Wrong/duplicate citations
    Ensure both ingest and API print:
    "[INFO] Ingest writing index to: …/store" and "[INFO] API using index dir: …/store"
    ```bash
    Clean rebuild: rm -rf store/* && python -m … --fresh
    ```
PRFAQ never retrieved
    Convert DOCX → MD/TXT, re-link in data/collections/fam/, re-ingest.
    Keep golden entries that accept either FAM-PRFAQ.md or FAM-PRFAQ.docx.
FAISS path TypeError
    Ensure vs_faiss.py uses os.fspath(path) for read_index/write_index and open().

## Quality policy (MVP)
Citations are required. If no grounding found, answer should hedge/refuse.
Golden set gates: no change promoted unless Hit@5 and citation presence meet targets.

## Demo script (2–3 minutes)
“This is a local RAG over FAM docs—no public LLM.”
Show /health and point out env, index_size, model.
Ask 2–3 questions via /chat (or Slack /ask) and highlight citations.
Run the golden eval: python golden/eval_golden.py … → call out Hit@5 and cite %.
Explain refresh: python -m rag.ingest.pipeline --data-dir data/collections/fam --fresh.

## make file
make fresh       # wipe + ingest FAM
make run         # start API
make health      # check /health
make chat        # sample query
make eval        # run golden set


## Notes
- store/ is your single source of truth (index + metadata). Don’t commit it.
- To add Onboarding later, create data/collections/onboarding/, ingest that directory, and use a separate golden set.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, 384‑dim)
- Vector store: FAISS (L2/cosine). Index + metadata saved under `store/`.
- Chunking: word-based approx (size 900, overlap 60). Tweak via `.env`.
- Citations: each answer includes top‑k chunk sources with doc title + page/section when available.
- No LLM generation (to keep costs $0). The `/chat` endpoint composes a grounded answer by stitching top chunks (simple heuristic). Swap with your LLM later if desired.