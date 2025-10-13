# src/rag/adapters/vs_faiss.py
import os, json, boto3
import faiss

class FaissStore:
    def __init__(self, index_path, meta_path, embed_dim):
        self.index_path = index_path
        self.meta_path = meta_path
        self.embed_dim = embed_dim
        self.index = None
        self.meta = None

    def ensure(self, bucket=None, faiss_key=None, meta_key=None):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        s3 = boto3.client("s3") if bucket else None

        # pull from S3 if local missing
        if bucket and faiss_key and not os.path.exists(self.index_path):
            s3.download_file(bucket, faiss_key, self.index_path)
        if bucket and meta_key and not os.path.exists(self.meta_path):
            s3.download_file(bucket, meta_key, self.meta_path)

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                self.meta = json.load(f)

    def search(self, q_vec, k=10):
        D, I = self.index.search(q_vec.reshape(1, -1), k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0: continue
            h = self.meta[str(idx)]
            h["score"] = float(score)
            results.append(h)
        return results
