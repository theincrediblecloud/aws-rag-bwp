# simple local cross-encoder reranker
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder

class LocalReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # small, fast model; good quality for reranking
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: List[Dict], text_key: str = "snippet", top_k: int = 8) -> List[Dict]:
        if not hits:
            return hits
        pairs: List[Tuple[str, str]] = [(query, h.get(text_key, "")) for h in hits]
        scores = self.model.predict(pairs).tolist()
        for h, s in zip(hits, scores):
            h["_rerank"] = float(s)
        hits.sort(key=lambda x: x.get("_rerank", 0.0), reverse=True)
        return hits[:top_k]
