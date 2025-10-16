# src/rag/api/app.py (near the top)
import os, traceback
from fastapi import FastAPI, Request, HTTPException, Query
from pydantic import BaseModel

from rag.core.config import AppConfig
from rag.core.secrets import get_secret
from rag.core import retriever
from rag.adapters.vs_faiss import FaissStore

cfg = AppConfig()
app = FastAPI(title="RAG API (Cloud)")

def _make_embedder():
    provider = os.getenv("EMBEDDER_PROVIDER", "bedrock").lower()
    if provider == "bedrock":
        from rag.adapters.embeddings_bedrock import BedrockEmbedder
        return BedrockEmbedder(
            model_id=os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2"),
            region=os.getenv("AWS_REGION", "us-east-1"),
        )
    else:
        from rag.adapters.embeddings_local import LocalEmbedder
        return LocalEmbedder(cfg.model_name, local_dir=cfg.model_local_dir)

# initialize RAG backend safely
rag_ready = False
rag_error = None
try:
    embedder = _make_embedder()               # import happens here, based on env
    vector = FaissStore(cfg.index_path, cfg.meta_path, embedder.dim)
    vector.ensure(bucket=cfg.s3_bucket, faiss_key=cfg.faiss_key, meta_key=cfg.meta_key)
    rag_ready = True
except Exception as e:
    rag_ready = False
    rag_error = f"{e.__class__.__name__}: {e}\n{traceback.format_exc()}"

@app.get("/health")
def health():
    size = int(vector.index.ntotal) if rag_ready and vector.index is not None else 0
    return {"ok": True, "rag_ready": rag_ready, "error": rag_error, "index_size": size}
