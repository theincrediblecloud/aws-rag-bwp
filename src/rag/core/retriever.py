# src/rag/core/retriever.py
import os
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "none").lower()
LocalReranker = None
if RERANKER_PROVIDER == "local":
    try:
        from rag.adapters.reranker_local import LocalReranker
    except Exception:
        LocalReranker = None
