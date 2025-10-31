# src/rag/adapters/embeddings_bedrock.py
import os, time
import json
import boto3

class BedrockEmbedder:
    """
    Supports Titan (amazon.titan-embed-text-v2:0) and Cohere (cohere.embed-english-v3 / cohere.embed-multilingual-v3).
    - Titan v2: one text per call -> we loop.
    - Cohere v3: accepts batch -> one call per batch.
    """
    def __init__(self, model_id: str = None, region: str = None, dim: int | None = None):
        self.model_id = model_id or os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=self.region)
        self.dim = dim  # optional; not required

    def _invoke(self, body: dict):
        #print(f"[bedrock] invoking model: {self.model_id}")
        resp = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        return json.loads(resp["body"].read().decode("utf-8"))

    def _is_cohere(self):
        return self.model_id.startswith("cohere.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._is_cohere():
            # Cohere v3: batch API
            body = {
                "texts": texts,
                "input_type": "search_document"  # reasonable default
            }
            #print(f"[bedrock] invoking model: {self.model_id}")
            out = self._invoke(body)
            # response: {"id":"...", "embeddings":[{"embedding":[...]} , ...]}
            return [e["embedding"] for e in out.get("embeddings", [])]

        else:
            # Titan v2: one text per call -> loop
            vecs: list[list[float]] = []
            for t in texts:
                body = {"inputText": t}
                out = self._invoke(body)
                # response: {"embedding":[...]}
                vecs.append(out["embedding"])
                # tiny sleep to be gentle with rate limits
                time.sleep(0.01)
            return vecs
