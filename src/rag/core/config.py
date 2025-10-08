# src/rag/core/config.py
from pathlib import Path
import os

class AppConfig:
    def __init__(self):
        # environment label used by /health and logs
        self.app_env = os.getenv("APP_ENV", "local")

        # One place to control where faiss.index + meta.jsonl live
        self.index_dir   = Path(os.getenv("INDEX_DIR", "store")).resolve()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path  = self.index_dir / "faiss.index"
        self.meta_path   = self.index_dir / "meta.jsonl"

        # Retrieval/ingest knobs (env-overridable)
        self.model_name    = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embed_dim     = int(os.getenv("EMBED_DIM", "384"))
        self.chunk_size    = int(os.getenv("CHUNK_SIZE", "600"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "80"))
        self.top_k         = int(os.getenv("TOP_K", "8"))
