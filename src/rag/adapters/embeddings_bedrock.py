# src/rag/adapters/embeddings_bedrock.py
import os, json, base64
import boto3

class BedrockEmbedder:
    """
    Minimal embedder that calls Amazon Bedrock text embedding model.
    Defaults: amazon.titan-embed-text-v2:0 in us-east-1.
    """
    def __init__(self, model_id: str | None = None, region: str | None = None):
        self.model_id = model_id or os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
        region = region or os.environ.get("BEDROCK_REGION", os.environ.get("AWS_REGION", "us-east-1"))
        self.client = boto3.client("bedrock-runtime", region_name=region)

        # Titan v2: dimension is 1024 (text), we’ll expose as property
        self.dim = int(os.environ.get("EMBED_DIM", "1024"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Titan v2 accepts one input per call; batch naively
        out = []
        for t in texts:
            body = {"inputText": t}
            resp = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )
            payload = json.loads(resp["body"].read().decode("utf-8"))
            vec = payload.get("embedding") or payload.get("vector")  # model field name
            out.append(vec)
        return out
