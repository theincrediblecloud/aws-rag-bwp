# tests/test_vs_numpy.py
import numpy as np

from rag.adapters.vs_numpy import NumpyStore

def test_search_cosine_top1():
    # tiny synthetic index: 3 vectors, 4-dim
    X = np.array([
        [1, 0, 0, 0],
        [0.9, 0.1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    meta = [
        {"title": "v1", "text": "unit v1"},
        {"title": "v2", "text": "unit v2"},
        {"title": "v3", "text": "unit v3"},
    ]

    store = NumpyStore("/tmp/vec.npy", "/tmp/meta.jsonl")
    # force-load in-memory
    store.vecs = X
    store.metadatas = meta

    q = [1, 0, 0, 0]
    hits = store.search(q, k=2)
    assert len(hits) == 2
    assert hits[0]["title"] == "v1"
    assert hits[0]["score"] >= hits[1]["score"]
    assert hits[1]["title"] == "v2" 