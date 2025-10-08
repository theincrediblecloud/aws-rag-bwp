# src/rag/adapters/vs_faiss.py
import json, os
from pathlib import Path
import faiss
import numpy as np

class FaissStore:
    def __init__(self, index_path: Path, meta_path: Path, dim: int):
        self.index_path = Path(index_path)
        self.meta_path  = Path(meta_path)
        self.dim = dim
        self.index = None
        self.meta = []   # keep metadata rows in append order

    def ensure(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            self.index = faiss.read_index(os.fspath(self.index_path))
        else:
            # Inner-product (cosine) index. Switch to L2 if you prefer.
            self.index = faiss.IndexFlatIP(self.dim)
        # Load meta.jsonl (if present) into memory in the same insertion order
        self.meta = []
        if self.meta_path.exists():
            with open(os.fspath(self.meta_path), "r", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        self.meta.append(json.loads(ln))

    def _save_index(self):
        faiss.write_index(self.index, os.fspath(self.index_path))

    def _append_meta(self, recs):
        with open(os.fspath(self.meta_path), "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.meta.extend(recs)

    def upsert(self, records):
        # stack vectors and normalize for cosine/IP
        vecs = np.vstack([r["vector"] for r in records]).astype("float32")
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self._save_index()
        # store metadata without vectors
        self._append_meta([{k: v for k, v in r.items() if k != "vector"} for r in records])

    # >>> NEW: used by /chat
    def search(self, q_vec: np.ndarray, k: int = 8):
        """
        q_vec: 1D ndarray shape (dim,) or list
        returns: list of dicts with score + metadata (title, source_path, page, chunk_text)
        """
        q = np.asarray(q_vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        # FAISS returns distances (similarities for IP) and indices
        D, I = self.index.search(q, min(k, max(1, self.index.ntotal)))
        idxs = I[0]
        scores = D[0]
        results = []
        for rank, (idx, score) in enumerate(zip(idxs, scores), start=1):
            if idx < 0 or idx >= len(self.meta):
                continue  # safety if meta/index ever drift
            m = self.meta[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "title": m.get("title", "(untitled)"),
                "source_path": m.get("source_path", ""),
                "page": m.get("page"),
                "chunk_text": m.get("chunk_text", ""),
            })
        return results

    # >>> OPTIONAL: for /health
    def size(self) -> int:
        try:
            return int(self.index.ntotal)
        except Exception:
            return len(self.meta)
