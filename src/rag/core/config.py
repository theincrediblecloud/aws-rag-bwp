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
    #faiss_key: str    = os.getenv("FAISS_KEY", "rag/index/faiss.index")

    # Slack secrets (stored as Secrets Manager ARNs)
    slack_signing_secret_arn: str = os.getenv("SLACK_SIGNING_SECRET_ARN", "")
    slack_bot_token_arn: str      = os.getenv("SLACK_BOT_TOKEN_ARN", "")

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "24"))
    RETRIEVE_K: int = int(os.getenv("RETRIEVE_K", "24"))
    CONTEXT_K: int = int(os.getenv("CONTEXT_K", "8"))
    ANSWER_FALLBACK = "allow"
    FALLBACK_MODE = (os.getenv("ANSWER_FALLBACK", "deny") or "deny").strip().lower()
    FALLBACK_ALLOWED = FALLBACK_MODE in {"allow", "on", "true", "1"}
    FALLBACK_MIN_SCORE = float(os.getenv("FALLBACK_MIN_SCORE", "0.25"))
    FALLBACK_MESSAGE = os.getenv(
        "FALLBACK_MESSAGE",
        "I couldn’t find an in-corpus basis. Here’s a brief general answer (no citations)."
    )
    STRICT_MESSAGE = os.getenv(
        "STRICT_MESSAGE",
        "I’m sorry, I can’t provide an answer based on the available documents.")

    # Computed local paths (Lambda’s /tmp)
    index_dir: str = os.getenv("INDEX_DIR", "/tmp/index")
    index_path: str = os.path.join(index_dir, "vectors.npy")
    #index_path: str = os.path.join(index_dir, "faiss.index")
    meta_path: str  = os.path.join(index_dir, "meta.jsonl")

    # Embedding provider and related configs
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    embed_provider: str = os.getenv("EMBED_PROVIDER", "bedrock")  # bedrock | local
    bedrock_region: str = os.getenv("BEDROCK_REGION", "us-east-1")
    bedrock_model: str = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1024"))
    use_s3_index: bool = os.getenv("USE_S3_INDEX", "true").lower() == "true"
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    SNIPPET_CHARS = int(os.getenv("SNIPPET_CHARS", "800"))  # was 180
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "bedrock")  # bedrock | local
    llm_model_id: str = os.getenv("LLM_MODEL_ID", "arn:aws:bedrock:us-east-1:471112701253:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0").strip()
    llm_inference_profile_arn: str = os.getenv("LLM_INFERENCE_PROFILE_ARN", "arn:aws:bedrock:us-east-1:471112701253:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0").strip()
    bedrock_region_llm: str = os.getenv("BEDROCK_REGION_LLM", "us-east-1")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "500"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))


    