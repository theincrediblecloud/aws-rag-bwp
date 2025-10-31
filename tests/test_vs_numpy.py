# tests/test_vs_numpy.py
import numpy as np
import json, os, tempfile
from rag.adapters.vs_numpy import NumpyStore

def test_search_cosine_top1():
    # tiny synthetic index: 3 vectors, 4-dim
    X = np.array([
        [1, 0, 0, 0],
        [0.9, 0.1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    metas = [
        {"title": "v1", "text": "unit v1"},
        {"title": "v2", "text": "unit v2"},
        {"title": "v3", "text": "unit v3"},
    ]

    with tempfile.TemporaryDirectory() as d:
        vecp = os.path.join(d, "vectors.npy")
        metap = os.path.join(d, "meta.jsonl")
        np.save(vecp, X, allow_pickle=False)
        with open(metap, "w") as f:
            for m in metas:
                f.write(json.dumps(m) + "\n")
        store = NumpyStore(vecp, metap)
        store.ensure(bucket="", prefix="")   # load local artifacts
        
        
    q = [1, 0, 0, 0]
    hits = store.search(q, k=2)

    assert len(hits) == 2
    assert hits[0]["title"] in ("v1","v2")
    assert hits[0]["score"] >= hits[1]["score"]