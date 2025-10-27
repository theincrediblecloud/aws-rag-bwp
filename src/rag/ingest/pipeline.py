# src/rag/ingest/pipeline.py
import argparse
from typing import List, Dict
from pathlib import Path
import os

from rag.core.config import AppConfig
from rag.core.chunker import chunk_text
from rag.adapters.vs_numpy import NumpyStore

def _make_embedder(provider: str, model_id: str):
    provider = (provider or os.getenv("EMBED_PROVIDER", "local")).lower()
    if provider == "bedrock":
        from rag.adapters.embeddings_bedrock import BedrockEmbedder
        return BedrockEmbedder(
            model_id=model_id or os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2"),
            region=os.getenv("AWS_REGION", "us-east-1"),
        )
    else:
        from rag.adapters.embeddings_local import LocalEmbedder
        return LocalEmbedder(model_id or os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))

def ingest_dirs(
    data_dirs: List[str],
    fresh: bool = False,
    provider: str = "local",
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    cfg = AppConfig()

    # Coerce to Paths defensively (AppConfig may return str)
    index_dir  = Path(getattr(cfg, "index_dir", "/tmp/index"))
    index_path = Path(getattr(cfg, "index_path", index_dir / "vectors.npy"))
    meta_path  = Path(getattr(cfg, "meta_path", index_dir / "meta.jsonl"))

    print(f"[INFO] Ingest writing index to: {index_dir}")

    # optional fresh rebuild
    if fresh:
        if index_path.exists():
            index_path.unlink()
            print(f"[INFO] Deleted {index_path}")
        if meta_path.exists():
            meta_path.unlink()
            print(f"[INFO] Deleted {meta_path}")

    # embedder
    emb = _make_embedder(provider, model_id)

    # vector store
    vs = NumpyStore(index_path, meta_path)

    # Respect AppConfig.use_s3_index for loading/writing existing artifacts
    if getattr(cfg, "use_s3_index", False):
        if not cfg.s3_bucket:
            raise ValueError("USE_S3_INDEX is true but ARTIFACTS_BUCKET (s3_bucket) is not set for ingest.")
        vs.ensure(bucket=cfg.s3_bucket, prefix=cfg.index_prefix)
    else:
        # local mode: ensure local files if present, don’t hit S3
        vs.ensure(bucket="", prefix=getattr(cfg, "index_prefix", "rag/index"))

    # walk + load docs
    total_chunks = 0
    records: List[Dict] = []

    def _iter_dirs(items: List[str]):
        for p in items:
            p = p.strip()
            if not p:
                continue
            yield p

    from rag.ingest.loaders import walk_files, load_any
    for root in _iter_dirs(data_dirs):
        print(f"[INFO] Scanning: {root}")
        for path in walk_files([root]):
            for doc in load_any(path):
                chunks = chunk_text(doc["text"], cfg.chunk_size, cfg.chunk_overlap)
                vecs = emb.embed(chunks)
                total_chunks += len(chunks)
                for ch, v in zip(chunks, vecs):
                    records.append({
                        "title": doc["title"],
                        "chunk_text": ch,
                        "vector": v,
                        "source_path": doc["source_path"],
                        "page": doc.get("page")
                    })

    # write index
    if records:
        vs.upsert(records)
        size = len(vs.X) if getattr(vs, "X", None) is not None else "n/a"
        print(f"[INFO] Ingested {len(records)} chunks from {len(data_dirs)} roots. Index size now: {size}")
    else:
        print("[WARN] No records found to ingest.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest one or more folders into the vector store")
    ap.add_argument("--data-dir", action="append", required=True,
                    help="Folder with docs. Use multiple --data-dir flags, or comma-separated.")
    ap.add_argument("--fresh", action="store_true", help="Delete existing index before ingest")
    ap.add_argument("--provider", default=os.getenv("EMBED_PROVIDER", "local"),
                    choices=["local", "bedrock"], help="Embedding provider")
    ap.add_argument("--model", dest="model_id", default=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                    help="Embedding model id (local or Bedrock)")
    args = ap.parse_args()

    # support comma-separated in a single flag
    dirs: List[str] = []
    for d in args.data_dir:
        dirs.extend([s.strip() for s in d.split(",") if s.strip()])

    ingest_dirs(dirs, fresh=args.fresh, provider=args.provider, model_id=args.model_id)
