# src/rag/ingest/pipeline.py
from __future__ import annotations
import numpy as np
import os, sys, json, argparse, pathlib, traceback
from pathlib import Path
from typing import Iterable, List, Dict, Any
from rag.ingest.loaders import load_file_to_chunks

# Optional: load .env for local runs
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---- Config / adapters -------------------------------------------------------

# We keep ingest independent of runtime store internals.
# It just writes two artifacts that runtime can read:
#   - vectors.npy (float32, L2-normalized rows)
#   - meta.jsonl  (one JSON per chunk; must include "chunk_text", "title"/"source_path"/"page" if available)

# Embeddings adapters (same ones used by app.py)
def _make_embedder(provider: str | None, model_id: str | None):
    provider = (provider or os.getenv("EMBED_PROVIDER") or "local").strip().lower()
    if provider == "bedrock":
        from rag.adapters.embeddings_bedrock import BedrockEmbedder
        model = model_id or os.getenv("BEDROCK_EMBEDDINGS_ID") or os.getenv("BEDROCK_MODEL") \
                or "amazon.titan-embed-text-v2:0"
        region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        print(f"[embed] provider=bedrock model={model} region={region}")
        return BedrockEmbedder(model_id=model, region=region)
    else:
        from rag.adapters.embeddings_local import LocalEmbedder
        model = model_id or os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
        print(f"[embed] provider=local  model={model}")
        return LocalEmbedder(model)


# ---- File discovery & loading -----------------------------------------------

# We try to use your existing loaders if present; otherwise fall back to simple .txt/.md reader.
def _yield_chunks_from_file(fp: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield dicts: { 'text': str, 'title': Optional[str], 'source_path': str, 'page': Optional[int] }
    """
    # Prefer project loaders if available
    try:
        from rag.ingest.loaders import load_file_to_chunks  # your repo-specific helper (if exists)
        for ch in load_file_to_chunks(str(fp)):
            # normalize keys
            yield {
                "text": ch.get("text") or ch.get("chunk_text") or "",
                "title": ch.get("title"),
                "source_path": ch.get("source") or ch.get("source_path") or str(fp),
                "page": ch.get("page"),
            }
        return
    except Exception:
        pass

    # Fallback minimal loader for text-like files
    try:
        suffix = fp.suffix.lower()
        if suffix in {".txt", ".md"}:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            # naive chunking to keep things moving; your real loaders do better
            chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
            overlap    = int(os.getenv("CHUNK_OVERLAP", "80"))
            start = 0
            n = len(txt)
            while start < n:
                end = min(n, start + chunk_size)
                yield {
                    "text": txt[start:end],
                    "title": fp.stem,
                    "source_path": str(fp),
                    "page": None,
                }
                if end == n:
                    break
                start = max(end - overlap, start + 1)
            return
        else:
            print(f"[warn] No loader for {fp.name}; skipping (install loaders for PDFs/DOCX).")
    except Exception as e:
        print(f"[warn] Error reading {fp}: {e}")


def _discover_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        # index common doc types; your loaders will handle most
        if p.suffix.lower() in {".pdf", ".docx", ".md", ".txt"}:
            out.append(p)
    return out


# ---- Write artifacts ---------------------------------------------------------

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    # L2 row-normalize; adds numerical safety
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _write_artifacts(index_path: Path, meta_path: Path, vectors: np.ndarray, metas: List[Dict[str, Any]]):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if vectors.size == 0:
        # write empty but valid artifacts
        np.save(index_path, np.zeros((0, 1), dtype=np.float32))
        with meta_path.open("w", encoding="utf-8") as f:
            pass
        print(f"[ingest] wrote empty vectors -> {index_path}")
        print(f"[ingest] wrote empty meta    -> {meta_path}")
        return

    vectors = _normalize_rows(vectors)
    np.save(index_path, vectors)
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metas:
            # enforce required fields for runtime
            rec = {
                "chunk_text": m.get("chunk_text") or m.get("text") or "",
                "title": m.get("title"),
                "source_path": m.get("source_path") or m.get("source") or "",
                "page": m.get("page"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[ingest] wrote vectors -> {index_path}  shape={vectors.shape}")
    print(f"[ingest] wrote meta    -> {meta_path}   lines={len(metas)}")


# ---- Optional upload to S3 ---------------------------------------------------

def _maybe_upload_to_s3(upload: bool, index_path: Path, meta_path: Path):
    if not upload:
        return
    bucket = os.getenv("ARTIFACTS_BUCKET", "").strip()
    prefix = os.getenv("INDEX_PREFIX", "rag/index").strip().rstrip("/")
    if not bucket:
        raise ValueError("--upload-s3 was set but ARTIFACTS_BUCKET env var is empty.")

    import boto3
    s3 = boto3.client("s3")
    key_vec = f"{prefix}/vectors.npy"
    key_meta = f"{prefix}/meta.jsonl"
    print(f"[s3] uploading {index_path} -> s3://{bucket}/{key_vec}")
    s3.upload_file(str(index_path), bucket, key_vec)
    print(f"[s3] uploading {meta_path}  -> s3://{bucket}/{key_meta}")
    s3.upload_file(str(meta_path),  bucket, key_meta)
    print(f"[s3] upload complete.")


# ---- Main ingest routine -----------------------------------------------------

def ingest_dirs(
    dirs: List[Path],
    fresh: bool,
    provider: str | None = None,
    model_id: str | None = None,
    upload_s3: bool = False,
):
    # Resolve output paths (default to ./store)
    index_dir  = Path(os.getenv("INDEX_DIR", "store"))
    index_path = Path(os.getenv("INDEX_PATH", str(index_dir / "vectors.npy")))
    meta_path  = Path(os.getenv("META_PATH",  str(index_dir / "meta.jsonl")))

    # Make local build independent of USE_S3_INDEX (that’s a runtime concern)
    print(f"[INFO] Ingest writing index to: {index_dir}")

    embedder = _make_embedder(provider, model_id)

    # Collect chunks and metadata
    all_texts: List[str] = []
    all_meta:  List[Dict[str, Any]] = []

    for d in dirs:
        d = d.resolve()
        if not d.exists():
            print(f"[warn] path not found: {d}")
            continue
        print(f"[INFO] Scanning: {d}")
        files = _discover_files(d)
        if not files:
            print(f"[warn] no ingestible files under {d}")
        for fp in files:
            try:
                for ch in _yield_chunks_from_file(fp):
                    text = (ch.get("text") or "").strip()
                    if not text:
                        continue
                    all_texts.append(text)
                    all_meta.append({
                        "chunk_text": text,
                        "title": ch.get("title"),
                        "source_path": ch.get("source_path") or str(fp),
                        "page": ch.get("page"),
                    })
            except Exception as e:
                print(f"[warn] loader failed for {fp}: {e}")
    
    print(f"[dbg] total files scanned: {sum(1 for _ in _discover_files(dirs[0])) if dirs else 0}")
    print(f"[dbg] total chunks: {len(all_texts)}  total metas: {len(all_meta)}")
    if all_texts[:3]:
        print(f"[dbg] sample chunk[0..2]: {[t[:80] for t in all_texts[:3]]}")

        # Embed in batches (to control memory & API costs)
    if not all_texts:
        print("[warn] no chunks produced; writing empty artifacts.")
        _write_artifacts(index_path, meta_path, np.zeros((0, 1), dtype=np.float32), [])
        _maybe_upload_to_s3(upload_s3, index_path, meta_path)
        return

    try:
        batch = int(os.getenv("EMBED_BATCH", "64"))
        vec_batches: List[np.ndarray] = []
        for i in range(0, len(all_texts), batch):
            slc = all_texts[i : i + batch]
            vecs = embedder.embed(slc)                  # -> List[List[float]] or np.ndarray
            vecs = np.asarray(vecs, dtype=np.float32)   # uses *module-level* np
            vec_batches.append(vecs)
            print(f"[embed] {i+len(slc)}/{len(all_texts)}")
        vectors = np.vstack(vec_batches) if vec_batches else np.zeros((0, 1), dtype=np.float32)
    except Exception as e:
        print(f"[error] embedding failed: {e}")
        import traceback as _tb                         # ← only traceback here
        _tb.print_exc()
        vectors = np.zeros((0, 1), dtype=np.float32)

    # Sanity: vector rows == meta rows
    if vectors.shape[0] != len(all_meta):
        raise ValueError(f"mismatch texts({vectors.shape[0]}) != meta({len(all_meta)})")

    # Always write artifacts, then (optionally) upload
    _write_artifacts(index_path, meta_path, vectors, all_meta)
    _maybe_upload_to_s3(upload_s3, index_path, meta_path)
    print(f"[done] ingest complete → vectors={index_path}  meta={meta_path}")

def main(argv=None):
    """
    Minimal CLI entrypoint so `python -m rag.ingest.pipeline ...` works.
    Adjust the internals to call your module's ingest function(s).
    """
    import argparse
    from pathlib import Path
    import sys, traceback, os

    p = argparse.ArgumentParser(description="Build NumPy/FAISS index from one or more data dirs.")
    p.add_argument("--data-dir", action="append", required=True,
                   help="Directory to scan (can be given multiple times).")
    p.add_argument("--fresh", action="store_true",
                   help="Rebuild from scratch.")
    p.add_argument("--provider", choices=["local","bedrock"], default=None,
                   help="Embedding provider override.")
    p.add_argument("--model-id", default=None,
                   help="Embedding model id override.")
    # Optional: if your pipeline supports upload; harmless if ignored.
    p.add_argument("--upload-s3", action="store_true",
                   help="After building locally, upload artifacts to S3.")
    args = p.parse_args(argv)

    try:
        # Prefer your module's top-level ingest routine if present:
        if 'ingest_dirs' in globals():
            dirs = [Path(d) for d in args.data_dir]
            # Call your existing ingest function signature:
            # ingest_dirs(dirs, fresh: bool, provider: str|None, model_id: str|None, upload_s3: bool|None)
            try:
                # Try modern signature (with upload_s3)
                globals()['ingest_dirs'](dirs, args.fresh, args.provider, args.model_id, getattr(args, "upload_s3", False))
            except TypeError:
                # Fallback to legacy signature
                globals()['ingest_dirs'](dirs, args.fresh, args.provider, args.model_id)
        else:
            raise RuntimeError("No function named ingest_dirs() found in pipeline module.")
    except Exception as e:
        print("[ingest] FAILED:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()