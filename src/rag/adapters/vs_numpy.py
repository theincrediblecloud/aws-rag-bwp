# src/rag/adapters/vs_numpy.py
import os, json, io, boto3
import numpy as np

class NumpyStore:
    def __init__(self, local_vec_path: str, local_meta_path: str):
        self.local_vec_path = local_vec_path
        self.local_meta_path = local_meta_path
        self.vecs = None   # np.ndarray [N, D]
        self.meta = []     # list[dict]

    def ensure(self, bucket: str, prefix: str = "rag/index"):
        """
        Expect these S3 keys:
          s3://{bucket}/{prefix}/vectors.npy
          s3://{bucket}/{prefix}/meta.jsonl
        Download to /tmp and load.
        """
        os.makedirs(os.path.dirname(self.local_vec_path), exist_ok=True)

        # If no bucket provided, attempt to load from local paths instead of S3.
        if not bucket:
            if os.path.exists(self.local_vec_path) and os.path.exists(self.local_meta_path):
                self.vecs = np.load(self.local_vec_path).astype("float32")
                with open(self.local_meta_path, "r", encoding="utf-8") as f:
                    self.meta = [json.loads(line) for line in f if line.strip()]
            else:
                raise ValueError(
                    f"No S3 bucket configured and local index files not found at: {self.local_vec_path}, {self.local_meta_path}. "
                    "Run the ingest pipeline to generate a local index or set ARTIFACTS_BUCKET to a valid S3 bucket."
                )
            # continue to validation below
        else:
            s3 = boto3.client("s3")
            vec_key = f"{prefix.rstrip('/')}/vectors.npy"
            meta_key = f"{prefix.rstrip('/')}/meta.jsonl"

            # download (let boto3 raise its native exceptions so calling code / CloudWatch shows full details)
            s3.download_file(bucket, vec_key, self.local_vec_path)
            s3.download_file(bucket, meta_key, self.local_meta_path)

            # load
            self.vecs = np.load(self.local_vec_path).astype("float32")
            with open(self.local_meta_path, "r", encoding="utf-8") as f:
                self.meta = [json.loads(line) for line in f if line.strip()]

        if self.vecs.ndim != 2:
            raise ValueError(f"vectors.npy must be 2-D, got shape {self.vecs.shape}")
        if len(self.meta) != self.vecs.shape[0]:
            raise ValueError(f"meta rows ({len(self.meta)}) != vectors rows ({self.vecs.shape[0]})")

    def search(self, q_vec, k: int = 10):
        q = np.asarray(q_vec, dtype=np.float32).reshape(-1)
        if q.size == 0:
            return []
        qn = q / (np.linalg.norm(q) + 1e-8)

        X = self.vecs
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        sims = Xn @ qn
        k = max(1, min(int(k), sims.size))
        top = np.argpartition(-sims, k-1)[:k]
        top = top[np.argsort(-sims[top])]

        hits = []
        for i in top:
            m = self.meta[int(i)] if 0 <= int(i) < len(self.meta) else {}
            hits.append({
                "title": m.get("title"),
                "source_path": m.get("source_path") or m.get("source") or m.get("url"),
                "page": m.get("page"),
                "score": float(sims[int(i)]),
                "chunk_text": m.get("chunk_text") or m.get("text") or "",
            })
        return hits
