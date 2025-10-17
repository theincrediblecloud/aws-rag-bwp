# src/rag/ingest/build_vectors_numpy.py
import argparse, json, os
import numpy as np
import boto3

from rag.adapters.embeddings_bedrock import BedrockEmbedder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="rag/index")
    ap.add_argument("--meta_local", required=True)   # e.g., store/meta.jsonl
    ap.add_argument("--vectors_out", default="store/vectors.npy")
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    ap.add_argument("--model_id", default=os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"))
    args = ap.parse_args()

    # load meta.jsonl
    meta = []
    with open(args.meta_local, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                meta.append(json.loads(line))

    # texts to embed (match your chunk field)
    texts = [(m.get("chunk_text") or m.get("text") or "").strip() for m in meta]

    emb = BedrockEmbedder(model_id=args.model_id, region=args.region)

    vecs = []
    B = 64
    for i in range(0, len(texts), B):
        batch = [t for t in texts[i:i+B]]
        vecs_batch = emb.embed(batch)  # -> List[List[float]]
        vecs.append(np.array(vecs_batch, dtype="float32"))
    X = np.vstack(vecs)

    os.makedirs(os.path.dirname(args.vectors_out), exist_ok=True)
    np.save(args.vectors_out, X)

    # upload vectors.npy and meta.jsonl to S3 under the same prefix the API expects
    s3 = boto3.client("s3", region_name=args.region)
    with open(args.vectors_out, "rb") as f:
        s3.upload_fileobj(f, args.bucket, f"{args.prefix}/vectors.npy")
    with open(args.meta_local, "rb") as f:
        s3.upload_fileobj(f, args.bucket, f"{args.prefix}/meta.jsonl")

    print(f"Uploaded: s3://{args.bucket}/{args.prefix}/vectors.npy  and  meta.jsonl  (shape={X.shape})")

if __name__ == "__main__":
    main()
