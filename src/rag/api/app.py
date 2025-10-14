# src/rag/api/app.py
from fastapi import FastAPI, Request, HTTPException, Query
from pydantic import BaseModel
import hmac, hashlib, time, json

from rag.core.config import AppConfig
from rag.core.secrets import get_secret
from rag.core import retriever
from rag.adapters.embeddings_local import LocalEmbedder
from rag.adapters.vs_faiss import FaissStore
from rag.core.constants import (
    GOOD, GOOD_SUMMARY, BAD_START, TOO_META, BAD_META, BAD_PRONOUN_LEAD,
    KEY_DECISION, HAS_OBERON, HAS_FRE, HAS_PHASE, HAS_OUTPUTS, HAS_INPUTS,
    BAD_LEAD, MODE_DEF, MODE_FLOW, MODE_PROS, MODE_FLOW_HINT, MODE_PROS_HINT,
    APP_REGEX, TABLEY_LINE, ALLCAPS_RUN, SENT_SPLIT,
    EXCERPT_MAX, EXCERPT_MAX_APPENDIX, PREFER_PAGES_BELOW, BAD_PREFIX
)

try:
    from mangum import Mangum
    _MANGUM = True
except Exception:
    _MANGUM = False

cfg = AppConfig()
app = FastAPI(title="RAG API (Cloud)")

embedder = LocalEmbedder(cfg.model_name, local_dir=cfg.model_local_dir)
vector = FaissStore(cfg.index_path, cfg.meta_path, embedder.dim)
vector.ensure(bucket=cfg.s3_bucket, faiss_key=cfg.faiss_key, meta_key=cfg.meta_key)

class ChatReq(BaseModel):
    session_id: str = "cloud"
    user_msg: str
    domain: str | None = None

@app.get("/health")
def health():
    size = int(vector.index.ntotal) if vector.index is not None else 0
    return {"ok": True, "env": cfg.app_env, "index_size": size, "model": cfg.model_name, "top_k": cfg.top_k}

# --- your run_chat(...) function here (unchanged behavior) ---

@app.post("/chat")
def chat(req: ChatReq):
    return run_chat(req.user_msg, req.session_id)

# ---- Slack slash command (POST x-www-form-urlencoded) ----
def _verify_slack_signature(req: Request, body: bytes, signing_secret: str):
    ts = req.headers.get("X-Slack-Request-Timestamp", "")
    sig = req.headers.get("X-Slack-Signature", "")
    if not ts or not sig:
        raise HTTPException(status_code=401, detail="Missing Slack headers")

    # replay protection (5 minutes)
    if abs(time.time() - int(ts)) > 300:
        raise HTTPException(status_code=401, detail="Stale Slack request")

    base = f"v0:{ts}:{body.decode('utf-8')}"
    mac = hmac.new(signing_secret.encode("utf-8"), base.encode("utf-8"), hashlib.sha256)
    expected = f"v0={mac.hexdigest()}"
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=401, detail="Bad Slack signature")

@app.post("/slack/command")
async def slack_command(req: Request):
    body = await req.body()
    signing_secret = get_secret(cfg.slack_signing_secret_arn)
    _verify_slack_signature(req, body, signing_secret)

    # Slack sends form-encoded
    form = dict([kv.split("=", 1) for kv in body.decode().split("&") if "=" in kv])
    text = form.get("text", "").replace("+", " ")
    out = run_chat(text, session_id="slack")
    # Respond with a simple text (ephemeral by default if not using response_url)
    return {"response_type": "ephemeral", "text": out["answer"]}

# Lambda entry
handler = Mangum(app) if _MANGUM else None
