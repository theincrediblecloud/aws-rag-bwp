from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from collections import defaultdict
import re

from rag.core.config import AppConfig
from rag.core import retriever
from rag.adapters.embeddings_local import LocalEmbedder
from rag.adapters.vs_faiss import FaissStore  # you said you're on FAISS
from rag.core.constants import (
    GOOD, GOOD_SUMMARY, BAD_START, TOO_META, BAD_META, BAD_PRONOUN_LEAD,
    KEY_DECISION, HAS_OBERON, HAS_FRE, HAS_PHASE, HAS_OUTPUTS, HAS_INPUTS, BAD_LEAD,
    MODE_DEF, MODE_FLOW, MODE_PROS, MODE_FLOW_HINT, MODE_PROS_HINT,
    APP_REGEX, TABLEY_LINE, ALLCAPS_RUN, SENT_SPLIT,
    EXCERPT_MAX, EXCERPT_MAX_APPENDIX, PREFER_PAGES_BELOW, BAD_PREFIX,
)

cfg = AppConfig()
app = FastAPI(title="Local RAG POC — stable baseline")
print(f"[INFO] API using index dir: {cfg.index_dir}")

embedder = LocalEmbedder(cfg.model_name)
vector = FaissStore(cfg.index_path, cfg.meta_path, cfg.embed_dim)
vector.ensure()


# sentences we don't want as bullets
_BAD_PREFIX = re.compile(
    r'^(?:refer to|see (?:also )?|for more (?:background|details))\b',
    re.IGNORECASE,
)

# ---------- helpers ----------
def _clean(s: str) -> str:
    s = (s or "").replace("\u00ad", "").replace("\n", " ")
    s = re.sub(r'\s+', ' ', s).strip()

    # already present
    s = re.sub(r'\bAPPENDIX\s*\d*\:?\s*', '', s, flags=re.I)
    s = re.sub(r'(?:\b[A-Z][A-Za-z ]{2,}\b\s+(?:Yes|No)\s*){2,}', '', s)

    # NEW: drop ALL-CAPS headings (2+ words)
    s = re.sub(r'\b(?:[A-Z]{2,}(?:\s+[A-Z]{2,}){1,})\b', '', s)

    # NEW: remove dot-leaders like "Section ........ 34"
    s = re.sub(r'\.{3,}\s*\d+\b', '', s)

    # existing light cleanup
    s = re.sub(r'(^|\s)[0-9]+\.\s+', r'\1', s)
    s = re.sub(r'(^|\s)[\[\(][0-9]+[\]\)]\s*', r'\1', s)
    s = re.sub(r'(\b\w+\b)(\s+\1){1,}', r'\1', s, flags=re.I)

    # remove common bullet glyphs and dot leaders
    s = re.sub(r'[•●◦▪‣·]+', ' ', s)          # bullet symbols
    s = re.sub(r'\.{3,}\s*\d+\b', '', s)      # dot leaders "... 34"

    # strip generic heading phrases that show up before real content
    s = re.sub(r'\b(?:here (?:is|are) (?:the )?details? of (?:the )?(?:two|three|main) workflows?)\b[:\-]?\s*',
            '', s, flags=re.I)

    # collapse leftover multiple spaces after removals
    s = re.sub(r'\s{2,}', ' ', s)

    s = re.sub(r'\S{60,}', '', s)  # drop ultra-long OCR artifacts
    return s

def _sent_len_score(s: str) -> int:
    n = len(s)
    if 90 <= n <= 200: return 2
    if 60 <= n < 90 or 200 < n <= 260: return 1
    return 0

def _is_appendix_like(h: dict) -> bool:
    page = h.get("page")
    raw  = (h.get("chunk_text") or "")  # use raw (not cleaned) to detect “APPENDIX”
    title = h.get("title") or ""

    # hard signals
    if APP_REGEX.search(raw) or APP_REGEX.search(title):
        return True

    # late pages are likelier appendix
    if page is not None and page >= 12:
        # table-ish lines, dot leaders, heavy ALLCAPS headings
        if TABLEY_LINE.search(raw) or ALLCAPS_RUN.search(raw):
            return True

    return False

def detect_mode(q: str) -> str:
    q = (q or "").strip()
    if MODE_FLOW.search(q): return "flow"
    if MODE_PROS.search(q): return "pros"
    if MODE_DEF.search(q):  return "def"
    return "generic"


def _short(s: str, n: int = EXCERPT_MAX) -> str:
    return s if len(s) <= n else s[:n].rstrip() + "…"

def _score_for_mode(text: str, page: int | None, mode: str) -> int:
    score = 0
    if mode == "flow" and MODE_FLOW_HINT.search(text): score += 3
    if mode == "pros" and MODE_PROS_HINT.search(text): score += 3
    if GOOD.search(text): score += 2

    # page weighting: strongly prefer early pages
    if page is None or page < 6:
        score += 4
    elif page < 10:
        score += 2
    elif page < 14:
        score -= 1
    else:
        score -= 3
    return score

def _simple_excerpt(chunk_text: str, page: int | None) -> str:
    txt = _clean(chunk_text)

    # extra stripping for late/appendix-ish pages
    if page is not None and page >= 12:
        txt = TABLEY_LINE.sub("", txt)
        txt = ALLCAPS_RUN.sub("", txt)

    sents = [s.strip() for s in SENT_SPLIT.split(txt) if len(s.strip()) >= 40]
    if not sents:
        cap = EXCERPT_MAX_APPENDIX if (page is not None and page >= PREFER_PAGES_BELOW) else EXCERPT_MAX
        return txt[:cap].rstrip() + ("…" if len(txt) > cap else "")

    # 1) Prefer architecture/solution sentences that are not meta
    for s in sents:
        if GOOD.search(s) and not BAD_META.search(s) and not _BAD_PREFIX.search(s):
            cap = EXCERPT_MAX_APPENDIX if (page is not None and page >= PREFER_PAGES_BELOW) else EXCERPT_MAX
            return s[:cap].rstrip() + ("…" if len(s) > cap else "")

    # 2) Otherwise, pick the best non-meta sentence by a small score
    scored = []
    for s in sents:
        if BAD_META.search(s) or _BAD_PREFIX.search(s) or BAD_PRONOUN_LEAD.search(s):
            continue
        sc = _sent_len_score(s)
        if GOOD.search(s): sc += 2
        if page is None or page < 8: sc += 1  # prefer early page sentences
        scored.append((sc, s))

    if scored:
        s = max(scored, key=lambda x: x[0])[1]
    else:
        # 3) Absolute fallback: first sentence, but at least cap properly
        s = sents[0]

    cap = EXCERPT_MAX_APPENDIX if (page is not None and page >= PREFER_PAGES_BELOW) else EXCERPT_MAX
    return s[:cap].rstrip() + ("…" if len(s) > cap else "")


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

def _compose_summary_from_hits(hits: list[dict], mode: str = "generic") -> str:
    sig = {"oberon": False, "fre": False, "phases": False, "outputs": False, "inputs": False}
    for h in hits[:6]:
        t = _clean(h.get("chunk_text") or "")
        sig["oberon"]  |= bool(HAS_OBERON.search(t))
        sig["fre"]     |= bool(HAS_FRE.search(t))
        sig["phases"]  |= bool(HAS_PHASE.search(t))
        sig["outputs"] |= bool(HAS_OUTPUTS.search(t))
        sig["inputs"]  |= bool(HAS_INPUTS.search(t))

    # Base building blocks
    parts = ["Build a FAM moderation service"]
    if sig["inputs"]:
        parts.append("that ingests catalog/text/images")
    if sig["oberon"]:
        parts.append("evaluated via Oberon policies/models")
    if sig["fre"]:
        parts.append("and orchestrated by FRE")
    if sig["outputs"]:
        parts.append("emitting APPROVE/REJECT/IN_MANUAL_REVIEW to the data lake & Paragon")
    if sig["phases"]:
        parts.append("with a phased rollout (POC/manual first, then real-time streaming)")

    one_liner = " ".join(parts) + "."

    # Light framing by mode (no hallucinations—just rephrase focus)
    if mode == "def":
        return one_liner  # keep concise definition
    if mode == "flow":
        return one_liner.replace("Build a FAM moderation service", "Workflow at a glance: FAM moderation")
    if mode == "pros":
        # keep the same facts but frame as trade-offs (still grounded)
        return one_liner.replace("Build a FAM moderation service", "Trade-off summary: leverage a FAM moderation service")
    return one_liner

def _choose_bullets(hits: list[dict], n=3, mode: str = "generic") -> list[dict]:
    # filter-out appendix-like first
    filtered = [h for h in hits if not _is_appendix_like(h)]

    # if we over-filtered, fall back to originals but still sort w/ penalties
    pool = filtered if filtered else hits

    seen, scored = set(), []
    for h in pool:
        key = (h.get("title"), h.get("page"))
        if key in seen:
            continue
        seen.add(key)
        txt  = _clean(h.get("chunk_text") or h.get("title") or "")
        page = h.get("page")
        scored.append(( _score_for_mode(txt, page, mode), h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:n]]

def run_chat(user_msg: str, session_id: str = "local", domain: str | None = None):
    q_user = (user_msg or "").strip()
    q_text = retriever.condense_query([], q_user) or q_user
    q_vec  = embedder.embed([q_text])[0]

    # ---- one retrieval only ----
    hits = vector.search(q_vec, k=max(cfg.top_k * 3, 24))

    # ---- empty-hits guard ----
    if not hits:
        fallback = (
            "I couldn’t find supporting passages for that yet. "
            "Try rephrasing (e.g., add product names like “Oberon” or “FRE”), "
            "or broaden the query."
        )
        return {
            "answer": fallback,
            "citations": [],
            "session_id": session_id or "dev",
            "domain": None,
        }

    # ---- choose mode and pick top docs ONCE ----
    mode = detect_mode(user_msg)
    top = _choose_bullets(hits, n=3, mode=mode)

    # ---- bullets built from the SAME 'top' that we’ll cite ----
    bullets = []
    for i, h in enumerate(top, 1):
        excerpt = _simple_excerpt(h.get("chunk_text") or h.get("title") or "", h.get("page"))
        bullets.append(f"- {excerpt} [^{i}]")

    # ---- summary built from full hits (richer signal), but that’s fine ----
    summary = _compose_summary_from_hits(hits, mode=mode).replace(" images", " images,")

    answer = (
        f"**High-level:** {summary}\n\n"
        f"Here’s what the documents say about “{user_msg}”:\n\n" + "\n".join(bullets)
    )

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
        "session_id": session_id or "dev",
        "domain": None,
    }

@app.post("/chat")
def chat(req: ChatReq):
    return run_chat(req.user_msg, req.session_id)
    
# browser-friendly: http://localhost:8000/ask?msg=...
@app.get("/ask")
def ask(q: str = Query(..., min_length=3), session_id: str = "dev"):
    data = run_chat(q, session_id=session_id)
    cits = data.get("citations", [])[:3]

    # if run_chat already fell back, just pass its message through
    if not cits:
        return {
            "answer": data.get("answer", "No supporting documents found."),
            "citations": [],
            "session_id": session_id,
            "domain": None,
        }

    bullets = []
    for i, c in enumerate(cits, 1):
        text = _clean(c.get("chunk_text") or c.get("title") or "")
        bullets.append(f"- {_short(text)} [^{i}]")

    return {
        "answer": f"Here's what the documents say about “{q}”:\n\n" + "\n".join(bullets),
        "citations": [
            {
                "idx": i+1,
                "title": c.get("title"),
                "source_path": c.get("source_path"),
                "page": c.get("page"),
                "score": c.get("score"),
            }
            for i, c in enumerate(cits)
        ],
        "session_id": session_id,
        "domain": None,
    }
