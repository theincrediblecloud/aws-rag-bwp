from typing import List
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.tolist()