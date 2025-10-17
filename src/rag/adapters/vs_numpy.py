# src/rag/adapters/vs_numpy.py
from __future__ import annotations

import os
import io
import json
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import boto3  # optional at runtime (only needed if ensure() pulls from S3)
except Exception:  # pragma: no cover
    boto3 = None


class NumpyStore:
    """
    Lightweight vector store backed by NumPy:
      - vecs stored as .npy (shape: [N, D], dtype float32)
      - meta stored as .jsonl or .json (list of dicts aligned with vec rows)

    Typical usage:
      store = NumpyStore(local_vec_path="store/vectors.npy", local_meta_path="store/meta.jsonl")
      store.ensure(bucket="my-bucket", prefix="rag/index")  # downloads s3://my-bucket/rag/index/{vectors.npy,meta.jsonl} if missing
      hits = store.search(q_vec, k=10)  # returns list of {title, source_path, page, score, chunk_text, index}
    """

    def __init__(self, local_vec_path: str, local_meta_path: str) -> None:
        self.local_vec_path = local_vec_path
        self.local_meta_path = local_meta_path

        self.vecs: Optional[np.ndarray] = None   # shape [N, D] float32
        self.meta: List[Dict[str, Any]] = []     # length N, aligned with vecs
        self._Xn: Optional[np.ndarray] = None    # cached row-normalized matrix

    # ------------------------
    # Loading / saving helpers
    # ------------------------
    def _load_local(self) -> None:
        if not (os.path.isfile(self.local_vec_path) and os.path.isfile(self.local_meta_path)):
            return

        self.vecs = np.load(self.local_vec_path).astype(np.float32, copy=False)

        # meta can be JSONL or a single JSON array
        self.meta = self._read_meta_file(self.local_meta_path)

        # sanity align
        if self.vecs.ndim != 2:
            raise ValueError(f"vectors file must be 2-D array, got shape {self.vecs.shape}")
        if len(self.meta) != self.vecs.shape[0]:
            raise ValueError(f"meta rows ({len(self.meta)}) != vectors rows ({self.vecs.shape[0]})")

        # cache normalized matrix
        self._Xn = self._normalize_rows(self.vecs)

    def _read_meta_file(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                return []
            # try JSON array first
            if text.startswith("["):
                data = json.loads(text)
                if not isinstance(data, list):
                    raise ValueError("meta.json must be a JSON array")
                return [x or {} for x in data]
            # else assume JSONL
            lines = text.splitlines()
            out: List[Dict[str, Any]] = []
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                out.append(json.loads(ln))
            return out

    @staticmethod
    def _normalize_rows(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return X / (norms + 1e-8)

    def _save_local(self) -> None:
        if self.vecs is None:
            return
        os.makedirs(os.path.dirname(self.local_vec_path) or ".", exist_ok=True)
        np.save(self.local_vec_path, self.vecs)
        os.makedirs(os.path.dirname(self.local_meta_path) or ".", exist_ok=True)
        # default to JSONL
        with open(self.local_meta_path, "w", encoding="utf-8") as f:
            for row in self.meta:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------
    # S3 ensure / download
    # ------------------------
    def ensure(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        vec_key: Optional[str] = None,
        meta_key: Optional[str] = None,
    ) -> None:
        """
        Make sure local files exist. If they don't, try to download from S3.

        Defaults:
          - bucket  := $ARTIFACTS_BUCKET (if None)
          - prefix  := $INDEX_PREFIX or "rag/index"
          - vec_key := f"{prefix}/vectors.npy"
          - meta_key tries, in order: f"{prefix}/meta.jsonl", f"{prefix}/meta.json"
        """
        # If already loaded, nothing to do
        if self.vecs is not None and self.meta:
            return

        # If both files exist locally, just load them
        if os.path.isfile(self.local_vec_path) and os.path.isfile(self.local_meta_path):
            self._load_local()
            return

        # Else try to fetch from S3 if available
        bucket = bucket or os.getenv("ARTIFACTS_BUCKET") or ""
        prefix = prefix or os.getenv("INDEX_PREFIX") or "rag/index"
        if not vec_key:
            vec_key = f"{prefix.rstrip('/')}/vectors.npy"
        # meta can be .jsonl or .json
        meta_candidates = [meta_key] if meta_key else [
            f"{prefix.rstrip('/')}/meta.jsonl",
            f"{prefix.rstrip('/')}/meta.json",
        ]

        if not bucket:
            # nothing else we can do; leave not-ready
            return

        if boto3 is None:
            return  # boto3 not available in this environment

        s3 = boto3.client("s3")

        # download vectors
        os.makedirs(os.path.dirname(self.local_vec_path) or ".", exist_ok=True)
        s3.download_file(bucket, vec_key, self.local_vec_path)

        # try meta.jsonl then meta.json
        got_meta = False
        os.makedirs(os.path.dirname(self.local_meta_path) or ".", exist_ok=True)
        for mk in meta_candidates:
            try:
                s3.download_file(bucket, mk, self.local_meta_path)
                got_meta = True
                break
            except Exception:
                continue

        if not got_meta:
            raise FileNotFoundError(f"Could not find meta under {meta_candidates} in s3://{bucket}")

        # finally load them into memory
        self._load_local()

    # ------------------------
    # Query
    # ------------------------
    def search(self, q_vec, k: int = 10) -> List[Dict[str, Any]]:
        """
        Cosine-similarity top-k over L2-normalized rows.
        Returns list of RAG-friendly hits:
          {title, source_path, page, score, chunk_text, index}
        """
        if self.vecs is None or self.vecs.size == 0:
            return []

        q = np.asarray(q_vec, dtype=np.float32).reshape(-1)
        if q.size == 0:
            raise ValueError("query vector is empty")

        qn = q / (np.linalg.norm(q) + 1e-8)
        Xn = self._Xn if self._Xn is not None else self._normalize_rows(self.vecs)

        if Xn.ndim != 2:
            raise ValueError(f"stored embeddings must be 2-D array, got shape {Xn.shape}")
        if Xn.shape[1] != qn.shape[0]:
            raise ValueError(f"dimension mismatch: stored dim {Xn.shape[1]} vs query dim {qn.shape[0]}")

        sims = Xn @ qn  # (N,)
        if sims.size == 0:
            return []

        k = int(max(0, min(k, sims.size)))
        if k == 0:
            return []

        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        scores = sims[top_idx]

        hits: List[Dict[str, Any]] = []
        for idx, score in zip(top_idx, scores):
            m = {}
            if self.meta and idx < len(self.meta):
                m = self.meta[int(idx)] or {}

            text = m.get("chunk_text") or m.get("text") or ""
            hits.append({
                "title": m.get("title"),
                "source_path": m.get("source_path") or m.get("url") or m.get("source"),
                "page": m.get("page"),
                "score": float(score),
                "chunk_text": (text.strip() if isinstance(text, str) else ""),
                "index": int(idx),
            })

        return hits
