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

# from sentence_transformers import SentenceTransformer
# import numpy as np

# class LocalEmbedder:
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         self.model_name = model_name
#         self.model = SentenceTransformer(model_name)
#         # important: expose dim for guardrails
#         self.dim = int(self.model.get_sentence_embedding_dimension())

#     def embed(self, texts):
#         embs = self.model.encode(texts, normalize_embeddings=False)
#         # return plain python floats for JSON-ability
#         return [list(map(float, v)) for v in np.asarray(embs)]
