# src/rag/api/app.py
import os, json, base64, traceback
import numpy as np

# --- env/config (cloud defaults) ---
APP_ENV         = os.getenv("APP_ENV", "prod")
EMBED_PROVIDER  = os.getenv("EMBEDDER_PROVIDER", "bedrock").lower()  # keep name consistent
AWS_REGION      = os.getenv("AWS_REGION", "us-east-1")
ARTIFACTS_BUCKET= os.getenv("ARTIFACTS_BUCKET", "")
INDEX_PREFIX    = os.getenv("INDEX_PREFIX", "rag/index")

# --- RAG bootstrap (lazy so we never crash on import) ---
_rag_ready   = False
_rag_error   = None
_run_chat_fn = None

def _init_rag():
    """Build the run_chat callable. On failure, capture a friendly error for /health and /chat."""
    global _rag_ready, _rag_error, _run_chat_fn
    if _run_chat_fn is not None or _rag_ready:
        return
    try:
        from rag.core.config import AppConfig
        from rag.core import retriever
        from rag.adapters.vs_numpy import NumpyStore  # <- NumPy store only

        cfg = AppConfig()

        # Embeddings: Bedrock in Lambda; allow optional local only if explicitly requested
        if EMBED_PROVIDER == "bedrock":
            from rag.adapters.embeddings_bedrock import BedrockEmbedder as Embedder
            embedder = Embedder(
                model_id=os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0"),
                region=AWS_REGION,
            )
        else:
            # LOCAL ONLY (won’t be used in Lambda unless you set EMBEDDER_PROVIDER=local)
            from rag.adapters.embeddings_local import LocalEmbedder as Embedder
            embedder = Embedder(cfg.model_name, local_dir=cfg.model_local_dir)

        # Vector store: NumPy (vectors.npy + meta.jsonl in s3://bucket/prefix/)
        vec = NumpyStore(local_vec_path="/tmp/vectors.npy",
                           local_meta_path="/tmp/meta.jsonl")
        vec.ensure(bucket=ARTIFACTS_BUCKET or cfg.s3_bucket, prefix=INDEX_PREFIX or cfg.index_prefix)

        def run_chat(user_msg: str, session_id: str = "cloud", domain: str | None = None):
            q_user = (user_msg or "").strip()
            if not q_user:
                return {
                    "answer": "Hi! Try asking a question (e.g., “summarize FAM moderation solution”).",
                    "citations": [], "session_id": session_id, "domain": domain
                }

            #q_text = retriever.condense_query([], q_user) or q_user
            q_vec = embedder.embed([q_user])[0]               # -> list[float]
            hits  = vec.search(q_vec, k=24)                  # -> list[dict]

            if not hits:
                return {
                    "answer": "I couldn’t find supporting passages yet. Try terms like “Oberon”, “FRE”, or broaden the query.",
                    "citations": [], "session_id": session_id, "domain": domain
                }

            top = hits[:3]
            bullets = []
            for i, h in enumerate(top, 1):
                txt = (h.get("chunk_text") or h.get("title") or "").strip().replace("\n", " ")
                if len(txt) > 180: txt = txt[:180].rstrip() + "…"
                bullets.append(f"- {txt} [^{i}]")

            answer = "**High-level:** Retrieved relevant passages.\n\n" + "\n".join(bullets)
            citations = [{
                "idx": i + 1,
                "title": h.get("title"),
                "source_path": h.get("source_path"),
                "page": h.get("page"),
                "score": h.get("score"),
                "chunk_text": h.get("chunk_text"),
            } for i, h in enumerate(top)]

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
    API Gateway HTTP API (v2) router—no FastAPI/Starlette.
      GET  /health
      POST /chat   {"user_msg": "...", "session_id":"...", "domain": null}
    """
    try:
        path = event.get("rawPath") or event.get("path") or "/"
        method = (event.get("requestContext", {}).get("http", {}).get("method")
                  or event.get("httpMethod") or "GET").upper()

        # Parse body safely
        body_str = event.get("body") or ""
        if event.get("isBase64Encoded"):
            body_str = base64.b64decode(body_str).decode("utf-8", "ignore")
        payload = {}
        if body_str:
            try:
                payload = json.loads(body_str)
            except Exception:
                payload = {}

        # /health
        if method == "GET" and path.endswith("/health"):
            _init_rag()
            msg = {"ok": True, "rag_ready": _rag_ready}
            if not _rag_ready and _rag_error:
                msg["error"] = f"RAG back-end not initialized: {_rag_error}"
            return _json(200, msg)

        # /chat
        if method == "POST" and path.endswith("/chat"):
            _init_rag()
            if not _rag_ready or _run_chat_fn is None:
                return _json(200, {
                    "answer": f"(stub) RAG back-end not initialized: {_rag_error or 'unknown'}",
                    "citations": [], "session_id": "cloud", "domain": None
                })
            out = _run_chat_fn(
                (payload.get("user_msg") or payload.get("text") or ""),
                session_id=payload.get("session_id", "cloud"),
                domain=payload.get("domain"),
            )
            return _json(200, out)

        # 404
        return _json(404, {"detail": "Not Found"})

    except Exception:
        import logging, traceback as tb
        logging.getLogger("rag.api").error("[api] unhandled error\n%s", tb.format_exc())
        return _json(500, {"message": "Internal Server Error"})
