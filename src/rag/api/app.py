from fastapi import FastAPI, Query
from pydantic import BaseModel
from collections import defaultdict
import re

from rag.core.config import AppConfig
from rag.core import retriever
from rag.adapters.embeddings_local import LocalEmbedder
from rag.adapters.vs_faiss import FaissStore  # you said you're on FAISS

cfg = AppConfig()
app = FastAPI(title="Local RAG POC — stable baseline")
print(f"[INFO] API using index dir: {cfg.index_dir}")

embedder = LocalEmbedder(cfg.model_name)
vector = FaissStore(cfg.index_path, cfg.meta_path, cfg.embed_dim)
vector.ensure()

# ---------- helpers ----------
def _compress(s: str, n: int = 320) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _apply_domain_filter(hits, domain: str | None):
    if not domain:
        return hits
    d = domain.lower()
    return [h for h in hits if d in (h.get("source_path","").lower()) or d in _normalize(h.get("title",""))]

def _distinct_by_doc(hits, max_per_doc=1, top_n=8):
    seen = defaultdict(int)
    out = []
    for h in hits:
        doc = h.get("source_path", h.get("title","")) or ""
        if seen[doc] < max_per_doc:
            out.append(h); seen[doc] += 1
        if len(out) >= top_n:
            break
    return out

def _compose_answer(user_query: str, hits):
    if not hits:
        return "No relevant passages found. Try rephrasing."
    bullets = [f"- {_compress(h.get('snippet',''))} [^{i+1}]" for i, h in enumerate(hits[:3])]
    header = f"Here's what the documents say about “{user_query}”:"
    return header + "\n\n" + "\n".join(bullets)

# ---------- API ----------
class ChatReq(BaseModel):
    session_id: str = "local"
    user_msg: str
    domain: str | None = None  # e.g., "bwp" or "toyota"

@app.get("/health")
def health():
    size = 0
    if vector.index is not None:
        size = int(vector.index.ntotal)
    return {"ok": True, "env": cfg.app_env, "index_size": size, "model": cfg.model_name, "top_k": cfg.top_k}

def run_chat(user_msg: str, session_id: str = "local", domain: str | None = None):
    q_user = (user_msg or "").strip()
    q_text = retriever.condense_query([], q_user) or q_user
    q_vec  = embedder.embed([q_text])[0]

    # FAISS cosine only — keep it simple and predictable
    hits = vector.search(q_vec, k=max(cfg.top_k * 3, 24))

    # Optional domain scope (highly recommended if your folders contain ‘bwp’ or ‘toyota’)
    hits = _apply_domain_filter(hits, domain)

    # One chunk per document for diversity
    hits = _distinct_by_doc(hits, max_per_doc=1, top_n=cfg.top_k)

    answer = _compose_answer(q_user, hits)
    citations = [{
        "idx": i+1,
        "title": h.get("title","(untitled)"),
        "source_path": h.get("source_path",""),
        "page": h.get("page"),
        "score": h.get("score")
    } for i, h in enumerate(hits)]
    return {"answer": answer, "citations": citations, "session_id": session_id, "domain": domain}

@app.post("/chat")
def chat(req: ChatReq):
    return run_chat(req.user_msg, req.session_id)

# browser-friendly: http://localhost:8000/ask?msg=...
@app.get("/ask")
def ask(msg: str = Query(..., description="Your question"), session_id: str = "dev"):
    return run_chat(msg, session_id)
