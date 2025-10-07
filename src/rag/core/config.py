import os
from dataclasses import dataclass

# Make dotenv optional; if it's not installed, just proceed.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class AppConfig:
    env: str = os.getenv("APP_ENV", "local")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")

    # bedrock
    bedrock_region: str = os.getenv("BEDROCK_REGION", "us-east-1")
    bedrock_embed_model: str = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2")
    bedrock_llm_model: str = os.getenv("BEDROCK_LLM_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0")

    # opensearch
    os_endpoint: str = os.getenv("OPENSEARCH_ENDPOINT", "")
    os_index: str = os.getenv("OPENSEARCH_INDEX", "kb-v1")
