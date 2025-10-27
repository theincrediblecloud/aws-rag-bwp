# src/rag/adapters/vs_numpy.py
from pathlib import Path
import numpy as np
import json

class NumpyStore:
    def __init__(self, local_vec_path, local_meta_path):
        self.local_vec_path = Path(local_vec_path)
        self.local_meta_path = Path(local_meta_path)
        self.X = None          # np.ndarray of shape (N, D) or None
        self.metadatas = []    # list[dict]

    def ensure(self, bucket: str = "", prefix: str = ""):
        """
        Load vectors+meta from local disk or S3 if present; otherwise initialize empty.
        This MUST NOT raise just because files don't exist yet (ingest will create them).
        """
        vec_p = self.local_vec_path
        meta_p = self.local_meta_path
        vec_p.parent.mkdir(parents=True, exist_ok=True)

        # 1) If loading from local disk
        import json
        if not bucket:
            if vec_p.exists() and meta_p.exists():
                self.X = np.load(vec_p)
                with open(meta_p, "r", encoding="utf-8") as f:
                    self.metadatas = [json.loads(line) for line in f]
            else:
                # initialize empty; upsert() will populate
                self.X = np.zeros((0, 0), dtype=np.float32)
                self.metadatas = []
            return

        # 2) If loading from S3
        import boto3, botocore
        import json
        s3 = boto3.client("s3")
        vec_key = f"{prefix.rstrip('/')}/vectors.npy"
        meta_key = f"{prefix.rstrip('/')}/meta.jsonl"

        def _s3_exists(key: str) -> bool:
            try:
                s3.head_object(Bucket=bucket, Key=key)
                return True
            except botocore.exceptions.ClientError as e:
                if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404 or e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                    return False
                raise

        any_found = False
        if _s3_exists(vec_key):
            s3.download_file(bucket, vec_key, str(vec_p))
            any_found = True
        if _s3_exists(meta_key):
            s3.download_file(bucket, meta_key, str(meta_p))
            any_found = True

        if any_found and vec_p.exists() and meta_p.exists():
            self.X = np.load(vec_p)
            with open(meta_p, "r", encoding="utf-8") as f:
                self.metadatas = [json.loads(line) for line in f]
        else:
            # initialize empty; upsert() will populate and you can upload later if desired
            self.X = np.zeros((0, 0), dtype=np.float32)
            self.metadatas = []

    def upsert(self, records):
        """
        records: list of { vector: list[float], ...meta... }
        Persists to local_vec_path/local_meta_path.
        """
        import json
        new_vecs = [np.asarray(r["vector"], dtype=np.float32).reshape(-1) for r in records]
        if not new_vecs:
            return

        D = new_vecs[0].shape[0]
        new_X = np.vstack(new_vecs)

        # initialize or append
        if self.X is None or self.X.size == 0:
            self.X = new_X
        else:
            # if first dim inference was (0,0), fix the dimensionality
            if self.X.shape[1] == 0:
                self.X = np.zeros((0, D), dtype=np.float32)
            if self.X.shape[1] != D:
                raise ValueError(f"dimension mismatch: existing {self.X.shape[1]} vs new {D}")
            self.X = np.vstack([self.X, new_X])

        # append metadata
        for r in records:
            m = {k: v for k, v in r.items() if k != "vector"}
            self.metadatas.append(m)

        # persist
        self.local_vec_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.local_vec_path, self.X)
        with open(self.local_meta_path, "w", encoding="utf-8") as f:
            for m in self.metadatas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def search(self, q_vec, k: int = 10):
        import numpy as np
        # handle empty index gracefully
        if self.X is None or self.X.size == 0:
            return []

        q = np.asarray(q_vec, dtype=np.float32).reshape(-1)
        print(f"q.shape: {q.shape}, X.shape: {self.X.shape}")
        if q.size != self.X.shape[1]:
            raise ValueError(f"dimension mismatch: stored {self.X.shape[1]}, query {q.size}")

        # normalize
        qn = q / (np.linalg.norm(q) + 1e-8)
        Xn = self.X / (np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-8)

        sims = Xn @ qn
        k = min(int(k), sims.size)
        if k <= 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out = []
        for i in idx:
            m = self.metadatas[int(i)] if 0 <= int(i) < len(self.metadatas) else {"index": int(i)}
            out.append({
                "index": int(i),
                "score": float(sims[int(i)]),
                **({k: v for k, v in m.items()} if isinstance(m, dict) else {"metadata": m}),
            })
        return out
