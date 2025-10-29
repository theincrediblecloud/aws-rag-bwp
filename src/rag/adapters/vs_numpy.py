# src/rag/adapters/vs_numpy.py
import os, json, io, boto3, numpy as np

class NumpyStore:
    """
    Minimal vector store backed by NumPy arrays.
    Expects:
      - vectors: npy file of shape [N, D], dtype float32 (or convertible)
      - meta:    jsonl file with N lines (one dict per vector)
    """

    def __init__(self, index_path: str, meta_path: str):
        self.index_path = index_path
        self.meta_path  = meta_path
        self.vecs: np.ndarray | None = None   # [N, D]
        self.meta: list[dict] = []
        self.dim:  int | None = None
        self._ready = False

    def begin_build(self):
        """Start a brand-new in-memory build (fresh index)."""
        self._b_vecs = []
        self._b_meta = []
        self._ready = False

    def add_batch(self, vecs, metas):
        """Append a batch of vectors + meta dicts (same length)."""
        import numpy as np
        assert len(vecs) == len(metas), "vecs/metas length mismatch"
        # ensure float32 + L2-normalize
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        self._b_vecs.append(vecs)
        self._b_meta.extend(metas)

    def finalize(self):
        """Write vectors.npy + meta.jsonl to index_path/meta_path and load them."""
        import numpy as np, json, os
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        V = np.vstack(self._b_vecs) if self._b_vecs else np.zeros((0, 1), dtype=np.float32)
        np.save(self.index_path, V)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self._b_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        # load into runtime arrays
        self._b_vecs, self._b_meta = [], []
        self._load_local(self.index_path, self.meta_path)  # reuses your existing loader

    def _load_local(self, idx_path: str, meta_path: str):
        if not os.path.isfile(idx_path):
            raise FileNotFoundError(f"vectors npy not found: {idx_path}")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"meta jsonl not found: {meta_path}")

        vecs = np.load(idx_path, mmap_mode="r")
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32, copy=False)

        # L2 normalize rows (safe)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    meta.append({})
                    continue
                try:
                    meta.append(json.loads(line))
                except Exception:
                    meta.append({})

        if vecs.shape[0] != len(meta):
            raise ValueError(f"count mismatch: vectors={vecs.shape[0]} meta_lines={len(meta)}")

        self.vecs = vecs
        self.meta = meta
        self.dim  = int(vecs.shape[1])
        self._ready = True
        print(f"[vectors] loaded local index: N={vecs.shape[0]} D={self.dim}")

    def _download_s3(self, bucket: str, prefix: str, tmp_dir: str):
        """Downloads s3://bucket/prefix/{vectors.npy,meta.jsonl} to tmp_dir."""
        s3 = boto3.client("s3")
        os.makedirs(tmp_dir, exist_ok=True)
        idx_key  = f"{prefix.rstrip('/')}/vectors.npy"
        meta_key = f"{prefix.rstrip('/')}/meta.jsonl"
        idx_dst  = os.path.join(tmp_dir, "vectors.npy")
        meta_dst = os.path.join(tmp_dir, "meta.jsonl")

        print(f"[vectors] downloading s3://{bucket}/{idx_key} -> {idx_dst}")
        s3.download_file(bucket, idx_key, idx_dst)
        print(f"[vectors] downloading s3://{bucket}/{meta_key} -> {meta_dst}")
        s3.download_file(bucket, meta_key, meta_dst)
        return idx_dst, meta_dst

    def ensure(self, bucket: str = "", prefix: str = ""):
        """
        Loads the vectors + meta into memory (or from S3), sets self.vecs/meta.
        """
        try:
            if bucket:
                tmp_dir = "/tmp/rag_index"
                idx_path, meta_path = self._download_s3(bucket, prefix, tmp_dir)
                self._load_local(idx_path, meta_path)
            else:
                # local mode
                self._load_local(self.index_path, self.meta_path)
        except Exception as e:
            # mark not ready but do not crash process; caller can check .ready()
            self._ready = False
            print(f"[vectors] ensure() failed: {e}")

    def ready(self) -> bool:
        return bool(self._ready and self.vecs is not None and self.meta)

    def size(self) -> int:
        return 0 if self.vecs is None else int(self.vecs.shape[0])

    def search(self, q_vec, k: int = 10):
        """
        Cosine similarity search on L2-normalized vectors.
        Returns a list[dict] with fields:
          - chunk_text, title, source_path, page, score
        """
        if not self.ready():
            raise RuntimeError("Vector index not loaded. Call ensure() and verify paths/bucket/prefix.")

        V = self.vecs  # [N, D]
        q = np.asarray(q_vec, dtype=np.float32)
        # normalize q
        n = np.linalg.norm(q)
        if n == 0:
            return []
        q = q / n
        if V.shape[1] != q.shape[0]:
            raise ValueError(f"dim mismatch: index_dim={V.shape[1]} query_dim={q.shape[0]}")
        sims = V @ q  # cosine similarity
        k = min(k, V.shape[0])
        # partial sort for top-k
        idx = np.argpartition(-sims, k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]

        out = []
        for i in idx:
            m = self.meta[i] if i < len(self.meta) else {}
            out.append({
                "chunk_text": m.get("chunk_text") or m.get("text") or "",
                "title": m.get("title"),
                "source_path": m.get("source_path") or m.get("source") or m.get("url"),
                "page": m.get("page"),
                "score": float(sims[i]),
                "meta": m,
            })
        return out
