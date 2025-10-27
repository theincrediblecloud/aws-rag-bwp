**aws-rag-bwp v0.1.0 — “Baseline RAG” (2025-10-27)**

**TL;DR**

First public baseline of the repo with local and Bedrock modes, a FastAPI chat endpoint, FAISS retrieval, ingestion pipeline, golden-set evaluation, and refreshed engineer-ready README + system design diagrams.

**Highlights**

  Two run modes: local (sentence-transformers) and AWS Bedrock (Titan embeddings + Claude/Llama)
  
  FastAPI service with /chat and /health
  
  Ingestion pipeline (PDF/DOCX/MD/TXT → chunk → embed → FAISS)
  
  Deterministic citations (“cite-or-silence”)
  
  Golden-set evaluator with Hit@K + citation-presence metrics
  
  Interview-ready artifacts: README revamp + Mermaid diagrams (arch + ingestion)
  
  Optional Slack /ask Lambda shim

**✨ New**

  RAG service (FastAPI) with /chat (augmented generation) and /health
  
  Pluggable embeddings + LLM adapters (Local/Bedrock)
  
  FAISS vector store + meta.jsonl metadata
  
  Ingestion CLI: python -m rag.ingest.pipeline --data-dir ... --fresh
  
  Golden set + evaluator: golden/fam.jsonl, golden/eval_golden.py
  
  System design Mermaid diagrams for TPM/SDI discussions
  
  New README with end-to-end local/Bedrock setup, eval, troubleshooting

**🔧 Improvements**

  Sensible defaults for chunking (size/overlap) and TOP_K
  
  Clear .env.example with APP_ENV, BEDROCK_MODEL_ID, etc.
  
  Repo structure for data/collections/<topic> using symlinks

**🧩 Optional Integrations**

  Minimal Slack Lambda handler: forwards /ask to the API and returns answer

**🛠️ Breaking Changes**

  None. This is the first tagged baseline.

**⬆️ Upgrade Notes**

  N/A for this release. For future changes, expect semantic versioning:

  MAJOR: breaking changes

  MINOR: features (backward-compatible)

  PATCH: fixes/docs

**🔍 Verification**

Health: curl -s http://127.0.0.1:8000/health | jq .

Chat smoke test:

  curl -s -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"user_msg":"What does the doc say about 3DS?","session_id":"dev"}' | jq .


**Ingestion:**

  python -m rag.ingest.pipeline --data-dir data/collections/fam --fresh
  ls -l store/   # expect faiss.index, meta.jsonl


**Golden-set eval:**

  python golden/eval_golden.py --api http://127.0.0.1:8000/chat golden/fam.jsonl -v

**📦 Artifacts (suggested to attach in GitHub Release)**

  README.pdf (optional export of the README)
  
  diagrams/arch-high-level.svg
  
  diagrams/ingestion-sequence.svg
  
  golden/fam.jsonl
  
  (If you containerize) Dockerfile + image tag reference

**📚 Docs**

  Updated README with Quickstart (local + Bedrock), eval, troubleshooting, and performance tips
  

**⚠️ Known Issues**

  Very large PDFs may benefit from pre-conversion to Markdown for cleaner chunking.
  
  Ensure chosen Bedrock models are enabled in your region and permitted by IAM.

**🙌 Contributors**

  **Venkat (@theincrediblecloud)**
