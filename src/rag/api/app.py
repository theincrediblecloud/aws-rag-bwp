# src/rag/api/app.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import hmac, hashlib, time, json
from urllib.parse import parse_qs

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

# Mangum (ASGI -> Lambda) — ensure 'mangum' is in infra/requirements.txt
try:
    from mangum import Mangum
    _MANGUM = True
except Exception:
    _MANGUM = False

cfg = AppConfig()
app = FastAPI(title="RAG API (Cloud)")

embedder = LocalEmbedder(cfg.model_name, local_dir=cfg.model_local_dir)
vector = FaissStore(cfg.index_path, cfg.meta_path, embedder.dim)
# If your FaissStore.ensure supports S3 population, keep this:
vector.ensure(bucket=cfg.s3_bucket, faiss_key=cfg.faiss_key, meta_key=cfg.meta_key)

vector = None
init_err = None
try:
    embedder = LocalEmbedder(cfg.model_name, local_dir=cfg.model_local_dir)
    from rag.adapters.vs_faiss import FaissStore
    vector = FaissStore(cfg.index_path, cfg.meta_path, embedder.dim)
    vector.ensure(bucket=cfg.s3_bucket, faiss_key=cfg.faiss_key, meta_key=cfg.meta_key)
except Exception as e:
    init_err = e
    print("[rag] init warning:", repr(e))
    
# ---------- tiny cleaners / helpers (use your constants) ----------
def _clean(s: str) -> str:
    import re
    s = (s or "").replace("\u00ad", "").replace("\n", " ")
    s = re.sub(r'\s+', ' ', s).strip()
    s = APP_REGEX.sub("", s)                              # drop literal "APPENDIX"
    s = TABLEY_LINE.sub("", s)                            # strip table-y lines / dot leaders
    s = ALLCAPS_RUN.sub("", s)                            # drop ALLCAPS headings
    s = BAD_META.sub("", s)                               # remove meta prose like "we only listed..."
    s = re.sub(r'[•●◦▪‣·]+', ' ', s)                      # bullet glyphs
    s = re.sub(r'(^|\s)[0-9]+\.\s+', r'\1', s)            # numbered lists
    s = re.sub(r'(^|\s)[\[\(][0-9]+[\]\)]\s*', r'\1', s)  # [1] / (1)
    s = re.sub(r'(\b\w+\b)(\s+\1){1,}', r'\1', s, flags=re.I)  # dedupe word repeats
    return s

def _score_for_mode(text: str, mode: str) -> int:
    score = 0
    if mode == "flow" and MODE_FLOW_HINT.search(text or ""): score += 3
    if mode == "pros" and MODE_PROS_HINT.search(text or ""): score += 3
    if GOOD.search(text or ""): score += 2
    if BAD_START.search((text or "").strip()): score -= 1
    if TOO_META.search(text or ""): score -= 2
    if BAD_PRONOUN_LEAD.search((text or "").strip()): score -= 1
    return score

def _simple_excerpt(chunk_text: str, page: int | None) -> str:
    txt = _clean(chunk_text)
    sents = [s.strip() for s in SENT_SPLIT.split(txt) if len(s.strip()) >= 40]
    if not sents:
        cap = EXCERPT_MAX_APPENDIX if (page is not None and page >= PREFER_PAGES_BELOW) else EXCERPT_MAX
        return txt[:cap].rstrip() + ("…" if len(txt) > cap else "")
    pick = None
    for s in sents:
        if GOOD.search(s):
            pick = s; break
    if not pick:
        pick = sents[0]
    cap = EXCERPT_MAX_APPENDIX if (page is not None and page >= PREFER_PAGES_BELOW) else EXCERPT_MAX
    return pick[:cap].rstrip() + ("…" if len(pick) > cap else "")

def _compose_summary_from_hits(hits: list[dict], mode: str = "generic") -> str:
    sig = {"oberon": False, "fre": False, "phases": False, "outputs": False, "inputs": False}
    for h in hits[:6]:
        t = _clean(h.get("chunk_text") or "")
        sig["oberon"]  |= bool(HAS_OBERON.search(t))
        sig["fre"]     |= bool(HAS_FRE.search(t))
        sig["phases"]  |= bool(HAS_PHASE.search(t))
        sig["outputs"] |= bool(HAS_OUTPUTS.search(t))
        sig["inputs"]  |= bool(HAS_INPUTS.search(t))
    parts = ["Build a FAM moderation service"]
    if sig["inputs"]:  parts.append("that ingests catalog/text/images")
    if sig["oberon"]:  parts.append("evaluated via Oberon policies/models")
    if sig["fre"]:     parts.append("and orchestrated by FRE")
    if sig["outputs"]: parts.append("emitting APPROVE/REJECT/IN_MANUAL_REVIEW to the data lake & Paragon")
    if sig["phases"]:  parts.append("with a phased rollout (POC/manual first, then real-time streaming)")
    one_liner = " ".join(parts) + "."
    if mode == "def":  return one_liner
    if mode == "flow": return one_liner.replace("Build a FAM moderation service", "Workflow at a glance: FAM moderation")
    if mode == "pros": return one_liner.replace("Build a FAM moderation service", "Trade-off summary: leverage a FAM moderation service")
    return one_liner

def _detect_mode(q: str) -> str:
    if MODE_FLOW.search(q or ""): return "flow"
    if MODE_PROS.search(q or ""): return "pros"
    if MODE_DEF.search(q or ""):  return "def"
    return "generic"

def _choose_bullets(hits: list[dict], n=3, mode: str = "generic") -> list[dict]:
    seen, scored = set(), []
    for h in hits:
        key = (h.get("title"), h.get("page"))
        if key in seen: continue
        seen.add(key)
        txt = _clean(h.get("chunk_text") or h.get("title") or "")
        base = _score_for_mode(txt, mode)
        scored.append((base, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:n]]

# ---------- API types ----------
class ChatReq(BaseModel):
    session_id: str = "cloud"
    user_msg: str
    domain: str | None = None

@app.get("/health")
def health():
    size = int(vector.index.ntotal) if vector.index is not None else 0
    return {"ok": True, "env": cfg.app_env, "index_size": size, "model": cfg.model_name, "top_k": cfg.top_k}

# ---------- core chat ----------
def run_chat(user_msg: str, session_id: str = "cloud", domain: str | None = None):
    if init_err or vector is None:
        return {
        "answer": "Search index isn’t available in this environment yet. "
                  "I’m up, but I can’t query the knowledge base. "
                  "Once the FAISS index is packaged (or a layer is attached), I’ll summarize docs.",
        "citations": [], "session_id": session_id, "domain": domain
    }
    q_user = (user_msg or "").strip()
    if not q_user:
        return {"answer": "Hi! Try a question like: “summarize FAM moderation solution?”", "citations": [], "session_id": session_id, "domain": domain}

    q_text = retriever.condense_query([], q_user) or q_user
    q_vec  = embedder.embed([q_text])[0]
    
    hits = vector.search(q_vec, k=max(cfg.top_k * 3, 24))
    if not hits:
        return {
            "answer": "I couldn’t find supporting passages yet. Try adding product names like “Oberon” or “FRE”.",
            "citations": [], "session_id": session_id, "domain": domain
        }

    mode = _detect_mode(q_user)
    top = _choose_bullets(hits, n=3, mode=mode)

    bullets = []
    for i, h in enumerate(top, 1):
        excerpt = _simple_excerpt(h.get("chunk_text") or h.get("title") or "", h.get("page"))
        bullets.append(f"- {excerpt} [^{i}]")

    summary = _compose_summary_from_hits(hits, mode=mode).replace(" images", " images,")
    answer = f"**High-level:** {summary}\n\nHere’s what the documents say about “{user_msg}”:\n\n" + "\n".join(bullets)

    return {
        "answer": answer,
        "citations": [
            {
                "idx": i + 1,
                "title": h.get("title"),
                "source_path": h.get("source_path"),
                "page": h.get("page"),
                "score": h.get("score"),
                "chunk_text": h.get("chunk_text"),
            }
            for i, h in enumerate(top)
        ],
        "session_id": session_id,
        "domain": domain,
    }

@app.post("/chat")
def chat(req: ChatReq):
    return run_chat(req.user_msg, req.session_id, req.domain)

# ---------- Slack slash command (/ask) ----------
def _verify_slack_signature(req: Request, body: bytes, signing_secret: str):
    ts = req.headers.get("X-Slack-Request-Timestamp", "")
    sig = req.headers.get("X-Slack-Signature", "")
    if not ts or not sig:
        raise HTTPException(status_code=401, detail="Missing Slack headers")
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

    # Slack sends application/x-www-form-urlencoded
    form = {k: v[0] for k, v in parse_qs(body.decode()).items()}
    text = (form.get("text") or "").strip().replace("+", " ")
    out = run_chat(text, session_id="slack")
    # Simple ephemeral text response
    return {"response_type": "ephemeral", "text": out["answer"]}

# Lambda entry point
handler = Mangum(app) if _MANGUM else None
