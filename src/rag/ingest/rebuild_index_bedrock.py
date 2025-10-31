# src/rag/ingest/rebuild_index_bedrock.py
import os, argparse
import boto3
import json
import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    raise RuntimeError("faiss is required locally to build the index (pip install faiss-cpu)")

from rag.adapters.embeddings_bedrock import BedrockEmbedder  # uses amazon.titan-embed-text-v2

def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[warn] skipping line {i}: {e}")
    return rows

def _pick_text(rec: dict, preferred: str) -> str | None:
    for key in (preferred, "text", "chunk_text", "content", "body"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--meta_local", required=True, help="input JSONL (one object per line)")
    ap.add_argument("--text_field", default="chunk_text", help="primary field to use for text")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    args = ap.parse_args()

    # Load chunks
    raw = _load_jsonl(args.meta_local)
    print(f"[ingest] loaded {len(raw)} meta records from {args.meta_local}")

    # Extract texts + keep slim metadata
    texts, metas = [], []
    for rec in raw:
        t = _pick_text(rec, args.text_field)
        if not t:
            continue
        texts.append(t)
        # carry a small, useful subset of metadata; keep original keys if you prefer
        metas.append({
            "title": rec.get("title"),
            "source_path": rec.get("source_path"),
            "page": rec.get("page"),
        })
    if not texts:
        raise RuntimeError("No texts found. Check --text_field value and your meta file schema.")

    print(f"[ingest] will embed {len(texts)} texts (batch={args.batch})")

    # Embed with Bedrock (Titan embed)
    emb = BedrockEmbedder(region=args.region)
    vecs = []
    for i in range(0, len(texts), args.batch):
        batch = texts[i:i+args.batch]
        v = emb.embed(batch)
        vecs.append(np.array(v, dtype="float32"))
        print(f"[ingest] embedded {min(i+args.batch, len(texts))}/{len(texts)}")
    X = np.concatenate(vecs, axis=0)
    # normalize for cosine similarity with IndexFlatIP
    faiss.normalize_L2(X)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    print(f"[ingest] FAISS index built: dim={d}, ntotal={index.ntotal}")

    # Write local artifacts
    faiss_path = "faiss.index"
    meta_json_path = "meta.json"   # convert JSONL -> JSON array for serving
    faiss.write_index(index, faiss_path)
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)

    # Upload to S3
    s3 = boto3.client("s3", region_name=args.region)
    faiss_key = f"{args.prefix.rstrip('/')}/faiss.index"
    meta_key = f"{args.prefix.rstrip('/')}/meta.json"

    s3.upload_file(faiss_path, args.bucket, faiss_key)
    s3.upload_file(meta_json_path, args.bucket, meta_key)

    print("[ingest] uploaded:")
    print(f"  s3://{args.bucket}/{faiss_key}")
    print(f"  s3://{args.bucket}/{meta_key}")
    print("[ingest] done.")

if __name__ == "__main__":
    main()
