#!/usr/bin/env python3
"""Create a tiny sample vector index (vectors.npy + meta.jsonl) for local testing.

Usage:
    python tools/make_sample_index.py --out-dir store --dim 384 --n 3

The script writes <out_dir>/vectors.npy and <out_dir>/meta.jsonl
"""
import argparse
import json
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.getenv("INDEX_DIR", "store"), help="Output directory")
    ap.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    ap.add_argument("--n", type=int, default=3, help="Number of sample vectors")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vec_path = os.path.join(args.out_dir, "vectors.npy")
    meta_path = os.path.join(args.out_dir, "meta.jsonl")

    # Create random vectors and normalize them to unit length
    rng = np.random.default_rng(42)
    X = rng.normal(size=(args.n, args.dim)).astype("float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    np.save(vec_path, X)

    # Create simple meta lines
    metas = []
    for i in range(args.n):
        metas.append({
            "title": f"sample-doc-{i+1}",
            "source_path": f"data/collections/sample/doc{i+1}.txt",
            "page": None,
            "chunk_text": f"This is a short sample chunk {i+1}.",
        })

    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Wrote {vec_path} ({X.shape}) and {meta_path} ({len(metas)} rows)")


if __name__ == "__main__":
    main()
