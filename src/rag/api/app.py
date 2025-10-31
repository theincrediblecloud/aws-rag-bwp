# src/rag/api/app.py
import base64
import boto3, botocore
import json, os, re, time, logging, hashlib, threading
from collections import OrderedDict
import rag.core.config as cfgmod

try:
    # Optional: used only for local dev
    from fastapi import FastAPI, Request  # pyright: ignore[reportMissingImports]
    from fastapi.responses import JSONResponse  # pyright: ignore[reportMissingImports]
except Exception:
    FastAPI = None   # type: ignore
    Request = None   # type: ignore
    JSONResponse = None  # type: ignore

# ------------------ Config ------------------
config = cfgmod.AppConfig()
APP_ENV                = config.app_env.lower()
AWS_REGION             = config.aws_region
TOP_K                  = config.top_k
EMBED_PROVIDER         = config.embed_provider
ARTIFACTS_BUCKET       = config.s3_bucket
INDEX_PREFIX           = config.index_prefix
INDEX_PATH             = config.index_path
META_PATH              = config.meta_path
SNIPPET_CHARS          = config.snippet_chars
LLM_PROVIDER           = config.llm_provider.lower()
LLM_MODEL_ID           = config.llm_model_id
MAX_TOKENS             = config.max_tokens
TEMPERATURE            = config.temperature
BEDROCK_REGION         = config.bedrock_region
FALLBACK_MIN_SCORE     = config.fallback_min_score
FALLBACK_ALLOWED       = config.fallback_allowed
FALLBACK_MESSAGE       = config.fallback_message
STRICT_MESSAGE         = config.strict_message
RETRIEVE_K             = config.retrieve_k
CONTEXT_K              = config.context_k

# Cache/env knobs
INDEX_VERSION          = config.index_version
CACHE_TIER1_ENABLED    = config.cache_tier1_enabled
CACHE_TTL_SEC_T1       = config.cache_ttl_sec_t1 
CACHE_MAX_ITEMS_T1     = config.cache_max_items_t1 
CACHE_TIER2_DDB_ENABLED= config.cache_tier2_ddb_enabled
DDB_CACHE_TABLE        = config.ddb_cache_table
CACHE_TTL_SECONDS_T2   = config.cache_ttl_seconds_t2

# ------------------ Globals ------------------
_rag_ready   = False
_rag_error   = None
_run_chat_fn = None
_dynamo      = None  # created lazily

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger = logging.getLogger("diag")

_LOGGED_IDENTITY = False
_FOLLOWUP_RE = re.compile(
    r'^\s*(more( details| info| examples)?|examples?\??|show (me )?examples?|expand|elaborate|deep[\s-]*dive|drill down|tell me more)\b',
    re.I
)
_MEMORY: dict[str, dict] = {}
_MEM_LRU: "OrderedDict[str, dict]" = OrderedDict()
_MEM_MAX = int(os.getenv("CACHE_MAX_ITEMS", "1000"))

def _normalize_q(s: str) -> str:
    s = (s or "")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("\u00A0", " ")
    return " ".join(s.lower().split())

def _mem_key(q: str, domain: str|None, version: str="v1") -> str:
    base = f"{(domain or 'default')}|{version}|{q.strip().lower()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def _mem_get(k: str) -> dict|None:
    v = _MEM_LRU.get(k)
    if v is not None:
        _MEM_LRU.move_to_end(k)
    return v

def _mem_put(k: str, v: dict):
    _MEM_LRU[k] = v
    _MEM_LRU.move_to_end(k)
    while len(_MEM_LRU) > _MEM_MAX:
        _MEM_LRU.popitem(last=False)

# ---- Tier-2 DynamoDB cache (optional) ----
def _get_ddb():
    global _dynamo
    if _dynamo is None:
        _dynamo = boto3.client("dynamodb", region_name=AWS_REGION)
    return _dynamo

def _cache_key(q: str, domain: str | None, version: str | None = None) -> str:
    v = version or INDEX_VERSION
    base = f"{(domain or 'default')}|{v}|{q.strip().lower()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def _cache_get(q: str, domain: str | None) -> dict | None:
    if not (CACHE_TIER2_DDB_ENABLED and DDB_CACHE_TABLE):
        return None
    try:
        pk = _cache_key(q, domain)
        r = _get_ddb().get_item(
            TableName=DDB_CACHE_TABLE,
            Key={"pk": {"S": pk}},
            ConsistentRead=False
        )
        item = r.get("Item")
        if not item:
            return None
        return {
            "answer": item.get("answer", {}).get("S"),
            "citations": json.loads(item.get("cit", {}).get("S", "[]")),
            "diag": json.loads(item.get("diag", {}).get("S", "{}")),
        }
    except Exception:
        return None

def _cache_put(q: str, domain: str | None, answer: str, citations: list, diag: dict):
    if not (CACHE_TIER2_DDB_ENABLED and DDB_CACHE_TABLE):
        return
    try:
        pk = _cache_key(q, domain)
        _get_ddb().update_item(
            TableName=DDB_CACHE_TABLE,
            Key={"pk": {"S": pk}},
            UpdateExpression="SET #a=:a, #c=:c, #d=:d, #t=:t",
            ExpressionAttributeNames={"#a": "answer", "#c": "cit", "#d": "diag", "#t": "ttl"},
            ExpressionAttributeValues={
                ":a": {"S": (answer or "")[:390000]},
                ":c": {"S": json.dumps(citations, ensure_ascii=False)[:200000]},
                ":d": {"S": json.dumps(diag, ensure_ascii=False)[:200000]},
                ":t": {"N": str(int(time.time()) + CACHE_TTL_SECONDS_T2)},
            },
        )
    except Exception:
        # best-effort cache
        pass

# ------------------ Helpers ------------------
def _log_identity_once():
    global _LOGGED_IDENTITY
    if _LOGGED_IDENTITY:
        return
    try:
        sts = boto3.client("sts", region_name=os.getenv("AWS_REGION", "us-east-1"))
        ident = sts.get_caller_identity()
        msg = f"[diag] caller_identity arn={ident.get('Arn')} account={ident.get('Account')} region={os.getenv('AWS_REGION','us-east-1')}"
        logger.info(msg)
        print(msg)
        _LOGGED_IDENTITY = True
    except Exception as e:
        logger.exception("[diag] sts error: %s", e)

def _recall(session_id: str) -> dict:
    return _MEMORY.get(session_id, {})

def _remember(session_id: str, **kwargs):
    st = _MEMORY.get(session_id, {})
    st.update(kwargs)
    _MEMORY[session_id] = st

# ------------------ RAG init ------------------
def _init_rag():
    """
    Build a minimal RAG pipeline; never crash Lambda init.
    If anything fails, stash the error for /health and /chat.
    """
    global _rag_ready, _rag_error, _run_chat_fn
    if _run_chat_fn is not None or _rag_ready:
        return

    llm_client = None
    print(f"[diag] LLM_PROVIDER={LLM_PROVIDER}, LLM_MODEL_ID={LLM_MODEL_ID}, EMBED_PROVIDER={EMBED_PROVIDER}")
    if LLM_PROVIDER == "bedrock" and LLM_MODEL_ID:
        logger.info(f"[llm] initializing Bedrock client, bedrock_region={BEDROCK_REGION}")
        llm_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    try:
        cfg = config  # use env for bucket/prefix/paths

        # Embeddings
        if EMBED_PROVIDER == "bedrock":
            from rag.adapters.embeddings_bedrock import BedrockEmbedder as Embedder
            embedder = Embedder(model_id=config.bedrock_model, region=AWS_REGION)
        else:
            from rag.adapters.embeddings_local import LocalEmbedder as Embedder
            embedder = Embedder(cfg.model_name)

        # Vector store (NumPy)
        from rag.adapters.vs_numpy import NumpyStore
        print(f"[vectors] bucket={cfg.s3_bucket} prefix={cfg.index_prefix}")
        print(f"[vectors] expecting: s3://{cfg.s3_bucket}/{cfg.index_prefix}/vectors.npy and meta.jsonl")
        vector = NumpyStore(cfg.index_path, cfg.meta_path)

        if cfg.use_s3_index:
            if not cfg.s3_bucket:
                raise ValueError(
                    "USE_S3_INDEX is true but ARTIFACTS_BUCKET (s3_bucket) is not set. "
                    "Set ARTIFACTS_BUCKET to your S3 bucket or set USE_S3_INDEX=false for local dev."
                )
            print(f"[vectors] using S3 bucket={cfg.s3_bucket} prefix={cfg.index_prefix}")
            vector.ensure(bucket=cfg.s3_bucket, prefix=cfg.index_prefix)
        else:
            print(f"[vectors] USE_S3_INDEX=false -> loading local index from: {cfg.index_path}, {cfg.meta_path}")
            vector.ensure(bucket="", prefix=cfg.index_prefix)

        dim = getattr(vector, "dim", None)
        print(f"[vectors] ready: N={vector.size()} D={dim if dim is not None else 'unknown'}")
        if not vector.ready():
            raise RuntimeError(
                f"Vector index not ready. index_path={cfg.index_path} meta_path={cfg.meta_path} "
                f"s3_bucket={cfg.s3_bucket or '-'} prefix={cfg.index_prefix}"
            )

        # ---- utilities in the closure ----
        def _normalize_vec(v):
            s = sum(x * x for x in v) ** 0.5
            if s < 1e-8:
                return [0.0 for _ in v]
            return [float(x / s) for x in v]
        
        def _as_scored_hits(raw_hits):
            out = []
            for h in raw_hits or []:
                score = h.get("score")
                if score is None and "distance" in h:
                    d = float(h["distance"])
                    score = 1.0 - 0.5 * (d ** 2)  # convert L2 on unit vecs to cosine-like
                try:
                    score = float(score)
                except Exception:
                    score = 0.0
                out.append({
                    "chunk_text": h.get("chunk_text") or h.get("text") or "",
                    "title": h.get("title"),
                    "source_path": h.get("source_path") or h.get("source") or h.get("url"),
                    "page": h.get("page"),
                    "score": score,
                    "meta": h.get("meta") or {},
                })
            return out

        def _llm_complete(system_prompt: str, user_prompt: str, _json_mod=json) -> str | None:
            if not llm_client or not LLM_MODEL_ID:
                return None
            try:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
                }
                resp = llm_client.invoke_model(
                    modelId=LLM_MODEL_ID,
                    body=_json_mod.dumps(body).encode("utf-8"),
                    contentType="application/json",
                    accept="application/json",
                )
                out = json.loads(resp["body"].read())
                txt = "".join(p.get("text", "") for p in out.get("content", []) if p.get("type") == "text").strip()
                return txt or None
            except Exception:
                logging.getLogger("rag.api").exception("[llm] invoke failed")
                return None

        def run_chat(user_msg: str, session_id: str = "cloud", domain: str | None = None, mode: str | None = None):
           try:
                q_user = (user_msg or "").strip()
                prev = _recall(session_id)
                followup = bool(_FOLLOWUP_RE.match(q_user))

                t_all = time.perf_counter()
                phase_path = "miss"  # will update to hot-mem/hot-ddb later
                pm = {}  

                # Follow-up stitching
                focus = None
                if followup:
                    focus = re.sub(_FOLLOWUP_RE, "", q_user, count=1, flags=re.I).strip(" ?.!-–—")
                if followup and prev.get("last_q"):
                    q_text = f"{prev['last_q']}" if not focus else f"{prev['last_q']} — focus on '{focus}'"
                else:
                    q_text = _normalize_q(q_user)

                if not q_text:
                    return {
                        "answer": "Ask me something (e.g., “What is Generative AI?”).",
                        "citations": [], "session_id": session_id, "domain": domain
                    }

                # ---- Cache: T1 (memory) then T2 (DDB) ----
                mem_k = _mem_key(q_text, domain)
                cached = _mem_get(mem_k)
                if cached:
                    out = {
                        **cached,
                        "diag": {
                            **cached.get("diag", {}),
                            "cache": "hot-mem",
                            "phase_ms": {"total": round((time.perf_counter() - t_all) * 1000.0, 1)}
                        },
                    }
                    phase_path = "hot-mem"
                    # IMPORTANT: remember last_q for follow-ups
                    _remember(session_id,
                            last_q=q_text,
                            last_sources=[c.get("source_path") for c in out.get("citations", []) if c.get("source_path")])
                    return out

                # ---- Tier-2 DynamoDB cache (warm) ----
                ddb_hit = _cache_get(q_text, domain)
                if ddb_hit:
                    out = {
                        "answer": ddb_hit["answer"],
                        "citations": ddb_hit.get("citations", []),
                        "session_id": session_id,
                        "domain": domain,
                        "diag": {
                            **ddb_hit.get("diag", {}),
                            "cache": "hot-ddb",
                            "phase_ms": {"total": round((time.perf_counter() - t_all) * 1000.0, 1)}
                        },
                    }
                    phase_path = "hot-ddb"
                    # remember for follow-ups
                    _remember(session_id,
                            last_q=q_text,
                            last_sources=[c.get("source_path") for c in out.get("citations", []) if c.get("source_path")])
                    # promote to tier-1 (in-memory)
                    _mem_put(mem_k, out)
                    return out

                # ---- Retrieve ----
            # q_vec = _normalize_vec(embedder.embed([q_text])[0])
                t0 = time.perf_counter()
                try:
                    q_vec = _normalize_vec(embedder.embed([q_text])[0])
                except Exception:
                    # If embedding fails unexpectedly, degrade gracefully
                    _remember(session_id, last_q=q_text)
                    out_payload = {
                        "answer": STRICT_MESSAGE,
                        "citations": [],
                        "session_id": session_id, "domain": domain,
                        "diag": {"retrieve_k": RETRIEVE_K, "context_k": CONTEXT_K,
                                "above_thresh": False, "threshold": FALLBACK_MIN_SCORE}
                    }
                    # attach timings
                    pm["embed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
                    pm["total"] = round((time.perf_counter() - t_all) * 1000.0, 1)
                    out_payload.setdefault("diag", {})["phase_ms"] = pm
                    out_payload["diag"]["cache"] = phase_path
                    _mem_put(mem_k, out_payload)
                    return out_payload
                pm["embed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
                
                # ---- Search (timed + guarded) ----
                t1 = time.perf_counter()
                try:
                    raw_hits = vector.search(q_vec, RETRIEVE_K)
                except Exception:
                    _remember(session_id, last_q=q_text)
                    out_payload = {
                        "answer": STRICT_MESSAGE,
                        "citations": [],
                        "session_id": session_id, "domain": domain,
                        "diag": {"retrieve_k": RETRIEVE_K, "context_k": CONTEXT_K,
                                "above_thresh": False, "threshold": FALLBACK_MIN_SCORE}
                    }
                    pm["search_ms"] = round((time.perf_counter() - t1) * 1000.0, 1)
                    pm["total"] = round((time.perf_counter() - t_all) * 1000.0, 1)
                    out_payload.setdefault("diag", {})["phase_ms"] = pm
                    out_payload["diag"]["cache"] = phase_path
                    _mem_put(mem_k, out_payload)
                    return out_payload
                pm["search_ms"] = round((time.perf_counter() - t1) * 1000.0, 1)
                hits = _as_scored_hits(raw_hits)

                # tiny soft re-rank for follow-ups: prefer last sources & focus
                if followup:
                    prev_sources = set(prev.get("last_sources") or [])
                    focus_l = (focus or "").lower()
                    buff = []
                    for h in hits:
                        boost = 0.0
                        if (h.get("source_path") or "") in prev_sources:
                            boost += 0.10
                        if focus_l and (focus_l in (h.get("title") or "").lower() or focus_l in (h.get("chunk_text") or "").lower()):
                            boost += 0.05
                        buff.append({**h, "score": float(h.get("score") or 0.0) + boost})
                    hits = sorted(buff, key=lambda x: x["score"], reverse=True)

                # No hits -> fallback or strict
                if not hits:
                    if FALLBACK_ALLOWED:
                        generic = _llm_complete(
                            "You are a concise, neutral explainer. Do not fabricate citations.",
                            f"Provide a short definition + 4–6 bullets.\n\nTopic: {q_text}",
                        ) or FALLBACK_MESSAGE
                        _remember(session_id, last_q=q_text)
                        out_payload = {
                            "answer": f"{generic}\n\n_Citation note: no in-corpus sources matched the query above threshold._",
                            "citations": [],
                            "session_id": session_id, "domain": domain,
                            "diag": {"retrieve_k": RETRIEVE_K, "context_k": CONTEXT_K, "above_thresh": False, "threshold": FALLBACK_MIN_SCORE}
                        }
                        _mem_put(mem_k, out_payload)
                        _cache_put(q_text, domain, out_payload["answer"], out_payload["citations"], out_payload["diag"])
                        return out_payload
                    else:
                        _remember(session_id, last_q=q_text)
                        out_payload = {
                            "answer": STRICT_MESSAGE,
                            "citations": [], "session_id": session_id, "domain": domain,
                            "diag": {"retrieve_k": RETRIEVE_K, "context_k": CONTEXT_K, "above_thresh": False, "threshold": FALLBACK_MIN_SCORE}
                        }
                        _mem_put(mem_k, out_payload)
                        return out_payload

                good = [h for h in hits if h["score"] >= FALLBACK_MIN_SCORE]
                use_hits = (good or hits)[:CONTEXT_K]

                # Build context snippets
                context_snippets = []
                for h in use_hits:
                    t = (h["chunk_text"] or "").strip()
                    if len(t) > SNIPPET_CHARS:
                        t = t[:SNIPPET_CHARS].rstrip() + "…"
                    if t:
                        context_snippets.append(t)

                # System prompt
                system = """
                    You are a senior developer advocate for this repository.
                    Primary source of truth is the provided Context. Answer the user’s question by strictly prioritizing that context.

                    When to enrich:
                    If the question is generic (e.g., concepts commonly covered in vendor documentation like AWS, Google, OpenAI) and the Context is thin or lacks definitions, you may add well-established, widely accepted public knowledge. Keep such additions brief and label them as _General background (public)_.

                    Do / Don’t
                    - DO use the most relevant passages from Context; be concise & structured.
                    - DO structure the answer using Slack-friendly formatting:
                        *Title (bold, one line)*
                        *Bold sub-section lines* followed by short bullet points.
                    - DO include a **References (corpus)** section listing only items from Context (title/path and page if available). Do not fabricate links.
                    - DO prefer Context when public knowledge conflicts.
                    - DON’T invent proprietary details or API contracts not in Context.
                    - DON’T cite public sources unless they appear in Context.

                    Fallbacks
                    - If the Context is insufficient to answer meaningfully, say so briefly and provide 2–3 next steps (docs/sections/terms to search).
                    - If the user asks for API contracts and the Context doesn’t contain them, point to the section/page where contracts live (if present); otherwise state it isn’t available in the corpus.

                    Follow-ups
                    - If the user follow-up is short (“more”, “examples”, “elaborate”), treat it as a request to expand the previous answer on the same topic, and prefer the same sources unless the user names a new focus.

                    Output format (Slack)
                    - *Title (bold, one line)*
                    - *Subsection (bold)* then 2–4 bullets (concise)
                    - Optional tiny example (≤10 lines) only if materially helpful
                    - Optional _General background (public)_ (≤3 bullets)
                    - *References (corpus)*: bullets with titles/paths/pages from Context only
                """.strip()

                wants_examples = bool(re.search(r'\b(example|examples?|code|cli|how to|snippet)\b', q_user, re.I))
                extras = "\nUser intent: The user asked for EXAMPLES; prioritize corpus examples and tiny code/CLI snippets.\n" if wants_examples else ""
                user = f"Question: {q_text}\n\n{extras}Context:\n- " + "\n- ".join(context_snippets)

                _log_identity_once()
                t2 = time.perf_counter()
                final = _llm_complete(system, user)
                pm["llm_ms"] = round((time.perf_counter() - t2) * 1000.0, 1) if final is not None else 0.0


                if not final:
                    # extraction fallback
                    bullets = []
                    for i, h in enumerate(use_hits[:3], 1):
                        txt = (h["chunk_text"] or h.get("title") or "").replace("\n", " ").strip()
                        if len(txt) > SNIPPET_CHARS:
                            txt = txt[:SNIPPET_CHARS].rstrip() + "…"
                        bullets.append(f"- {txt} [^{i}]")
                    answer = "**High-level:** Retrieved relevant passages.\n\n" + "\n".join(bullets)
                    fallback_used = True
                else:
                    answer = final
                    fallback_used = False

                # Citations & diag
                citations = []
                scores = [round(h["score"], 4) for h in use_hits]
                max_score = (scores[0] if scores else 0.0)
                above_thresh = bool(scores and max_score >= FALLBACK_MIN_SCORE)
                for i, h in enumerate(use_hits[:8], 1):
                    citations.append({
                        "idx": i,
                        "title": h.get("title") or "",
                        "source_path": h.get("source_path") or "",
                        "page": h.get("page"),
                        "score": float(h.get("score") or 0.0),
                        "chunk_text": h.get("chunk_text") or "",
                    })

                out_payload = {
                    "answer": answer,
                    "citations": citations,
                    "session_id": session_id,
                    "domain": domain,
                    "diag": {
                        "scores": scores[:5],
                        "max_score": max_score,
                        "above_thresh": above_thresh,
                        "fallback_used": fallback_used,
                        "retrieve_k": RETRIEVE_K,
                        "context_k": CONTEXT_K,
                        "threshold": FALLBACK_MIN_SCORE,
                        "embed_provider": EMBED_PROVIDER,
                        "llm_provider": LLM_PROVIDER,
                    }
                }

                # Remember + Cache
                _remember(session_id,last_q=q_text,last_sources=[c.get("source_path") for c in citations if c.get("source_path")])

                # Attach phase timings & cache path
                out_payload["diag"]["cache"] = phase_path
                pm["total"] = round((time.perf_counter() - t_all) * 1000.0, 1)
                out_payload["diag"]["phase_ms"] = pm

                # Populate caches
                _mem_put(mem_k, out_payload)
                _cache_put(q_text, domain, out_payload["answer"], citations, out_payload.get("diag", {}))

                return out_payload
           except Exception as e:
               # last-ditch guard → never 500
               logger.exception("[chat] top-level run_chat failed")
               safe = {
                   "answer": STRICT_MESSAGE,
                   "citations": [],
                   "session_id": session_id,
                   "domain": domain,
                   "diag": {
                       "error": "run_chat_top_level",
                       "reason": str(e)[:500],
                       "cache": "miss",
                       "phase_ms": {"total": round((time.perf_counter())*1000.0, 1)}  # best effort
                   }
               }
               return safe
        _run_chat_fn = run_chat
        _rag_ready = True

    except Exception:
        logging.getLogger("rag.api").exception("[api] init failed")
        _rag_error = "Initialization failed"

# ------------------ HTTP glue ------------------
def _json(status: int, body: dict, headers: dict | None = None, _json_mod=json):
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    return {"statusCode": status, "headers": h, "body": _json_mod.dumps(body, ensure_ascii=False)}

def handler(event, context):
    """
    API Gateway HTTP API (v2) event router.
      GET  /health
      POST /chat   (JSON: {"user_msg": "...", "session_id":"...", "domain": null})
    """
    _log_identity_once()
    logger.info(f"boto3={boto3.__version__}, botocore={botocore.__version__}")
    global _rag_ready, _run_chat_fn, _rag_error

    try:
        path = event.get("rawPath") or event.get("path") or "/"
        method = (event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod") or "GET").upper()

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
            try:
                resp["index_ready"] = True if _rag_ready else False
            except Exception:
                resp["index_ready"] = False
            if not _rag_ready and _rag_error:
                resp["error"] = _rag_error
            return _json(200, resp)

        # /chat
        if method == "POST" and path.endswith("/chat"):
            _init_rag()
            if not _rag_ready or _run_chat_fn is None:
                return _json(200, {
                    "answer": f"(stub) RAG back-end not initialized: {_rag_error or 'unknown'}",
                    "citations": [], "session_id": "cloud", "domain": None
                })
            try:
                user_msg   = payload.get("user_msg") or payload.get("text") or ""
                session_id = payload.get("session_id", "cloud")
                domain     = payload.get("domain")

                t0 = time.perf_counter()
                out = _run_chat_fn(user_msg, session_id=session_id, domain=domain)
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

                d = out.get("diag", {}) if isinstance(out, dict) else {}
                scores        = d.get("scores", [])
                max_score     = d.get("max_score", 0.0)
                above_thresh  = bool(d.get("above_thresh", False))
                fallback_used = bool(d.get("fallback_used", False))
                embed_p       = d.get("embed_provider", EMBED_PROVIDER)
                llm_p         = d.get("llm_provider", LLM_PROVIDER)
                cite_count    = len(out.get("citations", [])) if isinstance(out, dict) else 0
                cache_path    = d.get("cache", "miss")
                phase_ms      = d.get("phase_ms", {})  
                error_tag     = d.get("error")            
                error_reason  = d.get("reason")


                logger.info("[telemetry] %s", json.dumps({
                    "latency_ms": elapsed_ms,
                    "max_score": round(float(max_score), 3) if isinstance(max_score, (int, float)) else 0.0,
                    "scores": [round(float(s), 3) for s in scores[:5] if isinstance(s, (int, float))],
                    "above_thresh": above_thresh,
                    "fallback": fallback_used,
                    "citations": cite_count,
                    "embed": embed_p,
                    "llm": llm_p,
                    "cache": cache_path,         
                    "phase_ms": phase_ms,
                    "error_tag": error_tag,
                    "error_reason": error_reason  
                }))
                return _json(200, out)
            except Exception:
                logger.exception("[chat] unhandled error")
                return _json(500, {"message": "Internal Server Error"})

        # 404
        return _json(404, {"detail": "Not Found"})

    except Exception:
        logger.exception("[api] unhandled error")
        return _json(500, {"message": "Internal Server Error"})

# ------------------ Local FastAPI (dev) ------------------
IS_LAMBDA = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
print(f"[diag] IS_LAMBDA={IS_LAMBDA}")
app = FastAPI() if not IS_LAMBDA else None

if app:
    @app.get("/health")
    async def _health():
        global _rag_ready, _run_chat_fn, _rag_error
        _init_rag()
        resp = {"ok": True, "rag_ready": _rag_ready}
        if not _rag_ready and _rag_error:
            resp["error"] = f"RAG back-end not initialized: {_rag_error}"
        return JSONResponse(status_code=200, content=resp)

    @app.post("/chat")
    async def _chat(request: Request):  # pyright: ignore[reportMissingImports]
        global _rag_ready, _run_chat_fn, _rag_error
        _init_rag()
        if not _rag_ready or _run_chat_fn is None:
            return JSONResponse(status_code=200, content={
                "answer": f"(stub) RAG back-end not initialized: {_rag_error or 'unknown'}",
                "citations": [], "session_id": "cloud", "domain": None
            })
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        user_msg  = payload.get("user_msg") or payload.get("text") or ""
        session_id= payload.get("session_id", "cloud")
        domain    = payload.get("domain")
        out = _run_chat_fn(user_msg, session_id=session_id, domain=domain)
        return JSONResponse(status_code=200, content=out)
