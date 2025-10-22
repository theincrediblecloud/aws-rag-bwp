# src/rag/api/app.py
import  base64
import rag.core.config as config
import boto3, os, logging, json, traceback
# --- env/config ---
config = config.AppConfig()
APP_ENV          = config.app_env.lower()
AWS_REGION       = config.aws_region
TOP_K            = config.top_k 
EMBED_PROVIDER   = config.embed_provider
ARTIFACTS_BUCKET = config.s3_bucket
INDEX_PREFIX     = config.index_prefix
INDEX_PATH       = config.index_path
META_PATH        = config.meta_path
SNIPPET_CHARS    = config.SNIPPET_CHARS
LLM_PROVIDER = config.llm_provider.lower()
#LLM_MODEL_ID = config.llm_model_id
MAX_TOKENS   = config.max_tokens
TEMPERATURE  = config.temperature
BEDROCK_REGION = config.bedrock_region
LLM_INFERENCE_PROFILE_ARN = config.llm_inference_profile_arn
# ------------------

# internal flags
_rag_ready   = False
_rag_error   = None
_run_chat_fn = None

logger = logging.getLogger()
logger.setLevel(logging.INFO) 
    
_LOGGED_IDENTITY = False
def _log_identity_once():
    global _LOGGED_IDENTITY
    if _LOGGED_IDENTITY:
        return
    try:
        sts = boto3.client("sts", region_name=os.getenv("AWS_REGION", "us-east-1"))
        ident = sts.get_caller_identity()
        msg = f"[diag] caller_identity arn={ident.get('Arn')} account={ident.get('Account')} region={os.getenv('AWS_REGION','us-east-1')}"
        logger.info(msg)   # visible now
        print(msg)         # belt-and-suspenders: print always shows
        _LOGGED_IDENTITY = True
    except Exception as e:
        logger.exception("[diag] sts error: %s", e)

def _init_rag():
    """
    Build a minimal RAG pipeline, but never crash Lambda init.
    If anything fails, stash the error for /health and /chat.
    """
    global _rag_ready, _rag_error, _run_chat_fn
  
    if _run_chat_fn is not None or _rag_ready:
        return

    llm_client = None
    if LLM_PROVIDER == "bedrock" and LLM_INFERENCE_PROFILE_ARN: #LLM_MODEL_ID
        print
        llm_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    
    try:
        from rag.core.config import AppConfig
        from rag.core import retriever

        cfg = AppConfig()  # uses env for bucket/prefix/paths

        # ---- Embeddings (default: Bedrock in Lambda) ----
        if EMBED_PROVIDER == "bedrock":
            from rag.adapters.embeddings_bedrock import BedrockEmbedder as Embedder
            embedder = Embedder(
                model_id=config.bedrock_model,
                region=AWS_REGION,
            )
        else:
            from rag.adapters.embeddings_local import LocalEmbedder as Embedder
            embedder = Embedder(cfg.model_name, local_dir=cfg.model_local_dir)

        # ---- Vector store: NumPy (no FAISS native deps) ----
        # NumpyStore.ensure() should pull /tmp files from S3 (prefix=INDEX_PREFIX)
        print(f"[vectors] bucket={cfg.s3_bucket} prefix={cfg.index_prefix}")
        print(f"[vectors] expecting: s3://{cfg.s3_bucket}/{cfg.index_prefix}/vectors.npy and meta.jsonl")

        from rag.adapters.vs_numpy import NumpyStore
        vector = NumpyStore(cfg.index_path, cfg.meta_path)
        vector.ensure(bucket=cfg.s3_bucket, prefix=cfg.index_prefix)

        def _normalize(v):
            # safe L2 normalize -> list[float]
            s = sum(x * x for x in v) ** 0.5
            if s < 1e-8:
                return [0.0 for _ in v]
            return [float(x / s) for x in v]
        
        def _llm_complete(system_prompt: str, user_prompt: str) -> str:
            # Only Bedrock Anthropic shown here
            if not llm_client or not LLM_INFERENCE_PROFILE_ARN :#LLM_MODEL_ID:
                # Fallback: just return the user_prompt or a stub
                return None
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": [{"type":"text","text": user_prompt}]}
                ],
            }
            #LLM_TARGET_ID = LLM_INFERENCE_PROFILE_ARN or LLM_MODEL_ID
            print(f"[llm] invoking model/profile: {LLM_INFERENCE_PROFILE_ARN}") #LLM_TARGET_ID
            resp = llm_client.invoke_model(
                modelId=LLM_INFERENCE_PROFILE_ARN,                 
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            out = json.loads(resp["body"].read())
            print
            # Anthropic format: choices-like -> content list -> {text}
            parts = out.get("content", [])
            txt = ""
            for p in parts:
                if p.get("type") == "text":
                    txt += p.get("text","")
            return txt.strip() or None
        
        def run_chat(user_msg: str, session_id: str = "cloud", domain: str | None = None):
            q_user = (user_msg or "").strip()
            if not q_user:
                return {
                    "answer": "Ask me something (e.g., “summarize FAM moderation solution?”).",
                    "citations": [], "session_id": session_id, "domain": domain
                }

            # (optionally) condense query; keep simple for now
            # q_text = retriever.condense_query([], q_user) or q_user
            q_text = q_user
            q_vec = embedder.embed([q_text])[0]     # list[float]
            q_vec = _normalize(q_vec)

            hits = vector.search(q_vec, k=24)       # list[dict]
            if not hits:
                return {
                    "answer": "No strong matches yet. Try more specific terms (e.g., Oberon, FRE, phases).",
                    "citations": [], "session_id": session_id, "domain": domain
                }
            # Build a compact context from top-k hits (keep it short to avoid token bloat)
            context_snippets = []
            for h in hits[:TOP_K]:
                t = (h.get("chunk_text") or h.get("text") or "").strip()
                if len(t) > SNIPPET_CHARS:  # trim long chunks
                    t = t[:SNIPPET_CHARS].rstrip() + "…"
                context_snippets.append(t)
            system_prompt = (
                "You are a precise assistant. Answer using the provided context only. "
                "If the context is insufficient, say so briefly. Keep answers concise and structured."
            )
            user_prompt = f"Question: {q_user}\n\nContext:\n- " + "\n- ".join(context_snippets)
            _log_identity_once()
            final = _llm_complete(system_prompt, user_prompt)
            if not final:
                # fallback to previous bullet-style summary if LLM call is unavailable
                top = hits[:3]
                bullets = []
                for i, h in enumerate(top, 1):
                    txt = (h.get("chunk_text") or h.get("text") or h.get("title") or "").replace("\n", " ").strip()
                    if len(txt) > SNIPPET_CHARS: txt = txt[:SNIPPET_CHARS].rstrip() + "…"
                    bullets.append(f"- {txt} [^{i}]")
                answer = "**High-level:** Retrieved relevant passages.\n\n" + "\n".join(bullets)
            else:
                answer = final
            citations = []
            for i, h in enumerate(hits[:8]):
                citations.append({
                    "idx": i + 1,
                    "title": h.get("title"),
                    "source_path": h.get("source_path") or h.get("source") or h.get("url"),
                    "page": h.get("page"),
                    "score": h.get("score"),
                    "chunk_text": h.get("chunk_text") or h.get("text"),
                })
            return {"answer": answer, "citations": citations, "session_id": session_id, "domain": domain}

        _run_chat_fn = run_chat
        _rag_ready = True
    except Exception as e:
        _rag_error = f"{e.__class__.__name__}: {e}"
        _rag_ready = False


def _json(status: int, body: dict, headers: dict | None = None):
    h = {"Content-Type": "application/json"}
    if headers: h.update(headers)
    return {"statusCode": status, "headers": h, "body": json.dumps(body)}


def handler(event, context):
    """
    API Gateway HTTP API (v2) event router.
      GET  /health
      POST /chat   (JSON: {"user_msg": "...", "session_id":"...", "domain": null})
    """
    _log_identity_once()
    try:
        path = event.get("rawPath") or event.get("path") or "/"
        method = (event.get("requestContext", {}).get("http", {}).get("method")
                  or event.get("httpMethod") or "GET").upper()

        # Parse body if present
        body_str = event.get("body") or ""
        if event.get("isBase64Encoded"):
            try:
                body_str = base64.b64decode(body_str).decode("utf-8", "ignore")
            except Exception:
                body_str = ""
        try:
            payload = json.loads(body_str) if body_str else {}
        except Exception:
            payload = {}

        # /health
        if method == "GET" and path.endswith("/health"):
            _init_rag()
            resp = {"ok": True, "rag_ready": _rag_ready}
            if not _rag_ready and _rag_error:
                resp["error"] = f"RAG back-end not initialized: {_rag_error}"
            return _json(200, resp)

        # /chat
        if method == "POST" and path.endswith("/chat"):
            _init_rag()
            if not _rag_ready or _run_chat_fn is None:
                return _json(200, {
                    "answer": f"(stub) RAG back-end not initialized: {_rag_error or 'unknown'}",
                    "citations": [], "session_id": "cloud", "domain": None
                })
            user_msg  = payload.get("user_msg") or payload.get("text") or ""
            session_id= payload.get("session_id", "cloud")
            domain    = payload.get("domain")
            out = _run_chat_fn(user_msg, session_id=session_id, domain=domain)
            return _json(200, out)

        # 404
        return _json(404, {"detail": "Not Found"})

    except Exception:
        # keep client tidy, log server-side details
        import logging
        logging.getLogger("rag.api").error("[api] unhandled error\n%s", traceback.format_exc())
        return _json(500, {"message": "Internal Server Error"})