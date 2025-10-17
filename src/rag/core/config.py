# src/rag/core/config.py
import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    app_env: str = os.getenv("APP_ENV", "prod")

    # Embeddings / model
    model_name: str = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    model_local_dir: str = os.getenv("MODEL_LOCAL_DIR", "/tmp/model")  # Lambda writable

    # Vector index (S3)
    s3_bucket: str  = os.getenv("ARTIFACTS_BUCKET", "")
    index_prefix: str = os.getenv("INDEX_PREFIX", "rag/index")
    meta_key: str     = os.getenv("META_KEY", "rag/index/meta.json")
    faiss_key: str    = os.getenv("FAISS_KEY", "rag/index/faiss.index")

    # Slack secrets (stored as Secrets Manager ARNs)
    slack_signing_secret_arn: str = os.getenv("SLACK_SIGNING_SECRET_ARN", "")
    slack_bot_token_arn: str      = os.getenv("SLACK_BOT_TOKEN_ARN", "")

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "8"))

    # Computed local paths (Lambda’s /tmp)
    index_dir: str = os.getenv("INDEX_DIR", "/tmp/index")
    index_path: str = os.path.join(index_dir, "faiss.index")
    meta_path: str  = os.path.join(index_dir, "meta.json")

    # Embedding provider and related configs
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    embed_provider: str = os.getenv("EMBED_PROVIDER", "bedrock")  # bedrock | local
    bedrock_region: str = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))
    bedrock_model: str = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1024"))
    use_s3_index: bool = os.getenv("USE_S3_INDEX", "true").lower() == "true"

    