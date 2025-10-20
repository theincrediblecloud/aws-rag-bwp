# src/rag/api/app.py
import os, json, base64, traceback

# --- env/config (cloud defaults) ---
APP_ENV          = os.getenv("APP_ENV", "prod")
# accept either var; default to bedrock in Lambda
EMBED_PROVIDER   = (os.getenv("EMBEDDER_PROVIDER") or os.getenv("EMBED_PROVIDER") or "bedrock").lower()
AWS_REGION       = os.getenv("AWS_REGION", "us-east-1")
SNIPPET_CHARS     = int(os.getenv("SNIPPET_CHARS", "400"))  # was 180

# --- RAG bootstrap (lazy so we never crash on import) ---
_rag_ready   = False
_rag_error   = None
_run_chat_fn = None

def _init_rag():
    """
    Build the run_chat callable. On any failure, record _rag_error and keep the API up.
    """
    global _rag_ready, _rag_error, _run_chat_fn
    if _run_chat_fn is not None or _rag_ready:
        return

    try:
        from rag.core.config import AppConfig
        from rag.core import retriever
        cfg = AppConfig()

        # Choose embedder
        if os.getenv("EMBEDDER_PROVIDER", "bedrock").lower() == "bedrock":
            from rag.adapters.embeddings_bedrock import BedrockEmbedder as Embedder
            embedder = Embedder(
                model_id=os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0"),
                region=os.getenv("AWS_REGION", "us-east-1"),
            )
        else:
            from rag.adapters.embeddings_local import LocalEmbedder as Embedder
            embedder = Embedder(cfg.model_name, local_dir=cfg.model_local_dir)

        # Vector store (NumPy; via Lambda Layer)
        try:
            from rag.adapters.vs_numpy import NumpyStore as VectorStore
        except ImportError as ie:
            VectorStore = None
            raise RuntimeError("NumpyStore not available") from ie
        
        vector = VectorStore(cfg.index_path, cfg.meta_path)
        vector.ensure(bucket=cfg.s3_bucket, prefix=getattr(cfg, "index_prefix", None))
        _vs_mode = "numpy"

        def run_chat(user_msg: str, session_id: str = "cloud", domain: str | None = None):
            q_user = (user_msg or "").strip()
            if not q_user:
                return {"answer": "Ask me something, e.g., “summarize FAM moderation solution?”.",
                        "citations": [], "session_id": session_id, "domain": domain}

            # (Keep it NumPy-free here)
            q_vec = embedder.embed([q_user])[0]
            norm = (sum(x*x for x in q_vec) ** 0.5) or 1.0
            q_norm = [float(x / norm) for x in q_vec]

            hits = vector.search(q_norm, k=24)
            if not hits:
                return {"answer": "I couldn’t find relevant passages yet.",
                        "citations": [], "session_id": session_id, "domain": domain}

            top = hits[:3]
            bullets = []
            for i, h in enumerate(top, 1):
                txt = (h.get("chunk_text") or h.get("text") or h.get("title") or "").replace("\n", " ").strip()
                if len(txt) > SNIPPET_CHARS: txt = txt[:SNIPPET_CHARS].rstrip() + "…"
                bullets.append(f"- {txt} [^{i}]")
            answer = (
                "*High-level:* I found relevant passages.\n\n"
                + "\n\n".join(bullets)
                + "\n\n_Reply `more` for additional excerpts or `sources` to list all citations._"
            )
            citations = []
            for i, h in enumerate(top):
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
    API Gateway HTTP API (v2) router
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
    
