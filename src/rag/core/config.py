# src/rag/core/config.py
import os
from dataclasses import dataclass

# Only try to load .env locally; in Lambda, env vars are injected by SAM
if os.environ.get("APP_ENV", "local") == "local":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # python-dotenv not installed in Lambda package, that's fine
        pass

@dataclass
class AppConfig:
    # Environment
    app_env: str = os.environ["APP_ENV"]

    # Embeddings / model
    model_name: str = os.environ["MODEL_NAME"]
    model_local_dir: str = os.getenv("MODEL_LOCAL_DIR", "/tmp/model")  # safe local default

    # Vector index (S3)
    s3_bucket: str = os.environ["ARTIFACTS_BUCKET"]
    index_prefix: str = os.environ["INDEX_PREFIX"]
    meta_key: str = os.environ["META_KEY"]

    # Slack secrets (Secrets Manager ARNs)
    slack_signing_secret_arn: str = os.environ["SLACK_SIGNING_SECRET_ARN"]
    slack_bot_token_arn: str = os.environ["SLACK_BOT_TOKEN_ARN"]

    # Retrieval
    top_k: int = int(os.environ["TOP_K"])
    retrieve_k: int = int(os.environ["RETRIEVE_K"])
    context_k: int = int(os.environ["CONTEXT_K"])
    fallback_mode: str = os.environ["ANSWER_FALLBACK"].strip().lower()
    fallback_allowed: bool = fallback_mode in {"allow", "on", "true", "1"}
    fallback_min_score: float = float(os.environ["FALLBACK_MIN_SCORE"])
    fallback_message: str = os.environ["FALLBACK_MESSAGE"]
    strict_message: str = os.environ["STRICT_MESSAGE"]

    # Computed local paths
    index_dir: str = os.getenv("INDEX_DIR", "/tmp/index")
    index_path: str = os.path.join(index_dir, "vectors.npy")
    meta_path: str = os.path.join(index_dir, "meta.jsonl")

    # Embedding provider
    aws_region: str = os.environ["AWS_REGION"]
    embed_provider: str = os.environ["EMBED_PROVIDER"]
    bedrock_region: str = os.environ["BEDROCK_REGION"]
    bedrock_model: str = os.environ["BEDROCK_MODEL"]
    embed_dim: int = int(os.environ["EMBED_DIM"])
    use_s3_index: bool = os.environ["USE_S3_INDEX"].lower() == "true"
    chunk_size: int = int(os.environ["CHUNK_SIZE"])
    snippet_chars: int = int(os.environ["SNIPPET_CHARS"])
    chunk_overlap: int = int(os.environ["CHUNK_OVERLAP"])
    llm_provider: str = os.environ["LLM_PROVIDER"]
    llm_model_id: str = os.environ["LLM_MODEL_ID"].strip()
    llm_inference_profile_arn: str = os.environ["LLM_INFERENCE_PROFILE_ARN"].strip()
    max_tokens: int = int(os.environ["MAX_TOKENS"])
    temperature: float = float(os.environ["TEMPERATURE"])

    # Cache
    index_version: str = os.environ["INDEX_VERSION"]
    cache_tier1_enabled: bool = os.environ["CACHE_TIER1"].lower() == "true"
    cache_ttl_sec_t1: int = int(os.environ["CACHE_TTL_SEC"])
    cache_max_items_t1: int = int(os.environ["CACHE_MAX_ITEMS"])
    cache_tier2_ddb_enabled: bool = os.environ["CACHE_ANSWERS"].lower() == "true"
    ddb_cache_table: str = os.environ["DDB_CACHE_TABLE"]
    cache_ttl_seconds_t2: int = int(os.environ["CACHE_TTL_SECONDS"])
