# src/rag/ingest/pipeline.py
import argparse
from typing import List, Dict
from pathlib import Path
from rag.core.config import AppConfig
from rag.core.chunker import chunk_text
#from rag.adapters.embeddings_local import LocalEmbedder
from rag.adapters.vs_numpy import NumpyStore
import os

def _make_embedder():
    provider = os.getenv("EMBEDDER_PROVIDER", "local").lower()
    if provider == "bedrock":
        from rag.adapters.embeddings_bedrock import BedrockEmbedder
        return BedrockEmbedder(
            model_id=os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2"),
            region=os.getenv("AWS_REGION", "us-east-1"),
        )
    else:
        from rag.adapters.embeddings_local import LocalEmbedder
        return LocalEmbedder(os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))

def ingest_dirs(data_dirs: List[str], fresh: bool = False) -> None:
    cfg = AppConfig()
    print(f"[INFO] Ingest writing index to: {cfg.index_dir}")

    # optional fresh rebuild
    if fresh:
        if cfg.index_path.exists():
            cfg.index_path.unlink()
            print(f"[INFO] Deleted {cfg.index_path}")
        if cfg.meta_path.exists():
            cfg.meta_path.unlink()
            print(f"[INFO] Deleted {cfg.meta_path}")

    #emb = LocalEmbedder(cfg.model_name)
    emb = _make_embedder()
    # set up vector store
    vs = NumpyStore(cfg.index_path, cfg.meta_path)
    # Respect AppConfig.use_s3_index: download from S3 when desired, otherwise load/write local
    if cfg.use_s3_index:
        if not cfg.s3_bucket:
            raise ValueError("USE_S3_INDEX is true but ARTIFACTS_BUCKET (s3_bucket) is not set for ingest.")
        vs.ensure(bucket=cfg.s3_bucket, prefix=cfg.index_prefix)
    else:
        vs.ensure(bucket="", prefix=cfg.index_prefix)

    total_chunks = 0
    records: List[Dict] = []
    for path in data_dirs:
        print(f"[INFO] Scanning: {path}")
    from rag.ingest.loaders import walk_files, load_any
    for path in walk_files(data_dirs):
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

    if records:
        vs.upsert(records)
        size = getattr(vs, "X", []) and len(vs.X) or "n/a"
        print(f"[INFO] Ingested {len(records)} chunks from {len(data_dirs)} roots. Index size now: {size}")
    else:
        print("[WARN] No records found to ingest.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest one or more folders into local store")
    ap.add_argument("--data-dir", action="append", required=True,
                    help="Folder with docs. Use multiple --data-dir flags, or comma-separated.")
    ap.add_argument("--fresh", action="store_true", help="Delete existing index before ingest")
    args = ap.parse_args()

    # support comma-separated values in a single flag
    dirs: List[str] = []
    for d in args.data_dir:
        dirs.extend([s.strip() for s in d.split(",") if s.strip()])

    ingest_dirs(dirs, fresh=args.fresh)
