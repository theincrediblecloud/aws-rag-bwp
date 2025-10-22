# handlers/slack_events.py
import os, json, hmac, hashlib, time, urllib.request
from base64 import b64decode
import boto3
from typing import List, Dict

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
secrets = boto3.client("secretsmanager", region_name=AWS_REGION)

# --- simple in-memory caches (persist only on warm containers) ---
_CACHE: Dict[str, str] = {}
_SEEN_EVENTS: Dict[str, float] = {}  # event_id -> expires_at (epoch secs)
_DEDUPE_TTL = 600  # 10 minutes

SLACK_BOT_TOKEN = None  # lazily fetched

def _now() -> float:
    return time.time()

def _prune_seen():
    if not _SEEN_EVENTS:
        return
    t = _now()
    doomed = [k for k, exp in _SEEN_EVENTS.items() if exp <= t]
    for k in doomed:
        _SEEN_EVENTS.pop(k, None)

def _mark_seen(event_id: str):
    _SEEN_EVENTS[event_id] = _now() + _DEDUPE_TTL

def _already_seen(event_id: str) -> bool:
    _prune_seen()
    return event_id in _SEEN_EVENTS

def _get_secret_by_env(arn_env: str) -> str:
    if arn_env in _CACHE:
        return _CACHE[arn_env]
    arn = os.environ.get(arn_env, "").strip()
    if not arn:
        raise RuntimeError(f"Missing env {arn_env}")
    resp = secrets.get_secret_value(SecretId=arn)
    val = resp.get("SecretString")
    if val is None and "SecretBinary" in resp:
        val = b64decode(resp["SecretBinary"]).decode("utf-8")
    _CACHE[arn_env] = val
    return val

def _verify(headers, body_bytes: bytes) -> bool:
    ts = headers.get("x-slack-request-timestamp") or headers.get("X-Slack-Request-Timestamp")
    sig = headers.get("x-slack-signature") or headers.get("X-Slack-Signature")
    if not ts or not sig:
        return False
    # reject replays older than 5 minutes
    try:
        if abs(_now() - int(ts)) > 60 * 5:
            return False
    except Exception:
        return False
    signing_secret = _get_secret_by_env("SLACK_SIGNING_SECRET_ARN")
    base = f"v0:{ts}:{body_bytes.decode()}".encode()
    my_sig = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(my_sig, sig)

def _is_slack_retry(headers) -> bool:
    # Slack sets these on retries
    return (
        headers.get("X-Slack-Retry-Num") is not None
        or headers.get("x-slack-retry-num") is not None
        or headers.get("X-Slack-Retry-Reason") is not None
        or headers.get("x-slack-retry-reason") is not None
    )

def _post_message(channel: str, text: str = None, blocks: List[Dict] = None, thread_ts: str = None):
    global SLACK_BOT_TOKEN
    if SLACK_BOT_TOKEN is None:
        SLACK_BOT_TOKEN = _get_secret_by_env("SLACK_BOT_TOKEN_ARN")

    payload: Dict = {"channel": channel}
    if text is not None:
        payload["text"] = text  # fallback text for notifications
    if blocks is not None:
        payload["blocks"] = blocks
    if thread_ts:
        payload["thread_ts"] = thread_ts

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            # keep requests crisp; Slack can be slow occasionally
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=6) as r:
        return json.loads(r.read().decode())

def _clean_text(s: str, max_len: int = 2500) -> str:
    # collapse duplicate lines/paragraphs and clamp size to avoid Slack truncation
    if not s:
        return ""
    lines = []
    seen = set()
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    out = "\n".join(lines)
    if len(out) > max_len:
        out = out[:max_len].rstrip() + "…"
    return out

def _as_blocks(answer: str, citations: List[Dict]) -> List[Dict]:
    answer = _clean_text(answer, max_len=2500)
    blocks: List[Dict] = [{
        "type": "section",
        "text": {"type": "mrkdwn", "text": answer or "No answer."}
    }]
    if citations:
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "*Top sources*"}})
        for c in citations[:8]:  # cap to keep payload small
            title = c.get("title") or "(untitled)"
            page = c.get("page")
            page_hint = f" (p. {page})" if page is not None else ""
            score = c.get("score")
            score_hint = f" — _score {score:.3f}_" if isinstance(score, (float, int)) else ""
            line = f"• *{title}{page_hint}*{score_hint}"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": line}})
    return blocks

def _call_rag_api(question: str) -> Dict:
    chat_api = os.environ.get("RagApiUrl", "").strip()
    if not chat_api:
        return {"answer": "RAG API URL not configured."}
    body = json.dumps({"user_msg": question}).encode()
    req = urllib.request.Request(
        chat_api,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())

def handler(event, context):
    headers = event.get("headers") or {}
    body = event.get("body") or ""
    body_bytes = b64decode(body) if event.get("isBase64Encoded") else body.encode()

    # Slack URL verification (challenge)
    try:
        payload = json.loads(body_bytes.decode() or "{}")
    except Exception:
        payload = {}

    if payload.get("type") == "url_verification":
        return {"statusCode": 200, "headers": {"Content-Type": "text/plain"}, "body": payload.get("challenge", "")}

    # Signature check
    if not _verify(headers, body_bytes):
        return {"statusCode": 401, "body": "bad signature"}

    # Ignore Slack retries (we’ve likely already posted for this event)
    if _is_slack_retry(headers):
        return {"statusCode": 200, "body": ""}

    # De-dup by event_id
    event_id = payload.get("event_id")
    if event_id:
        if _already_seen(event_id):
            return {"statusCode": 200, "body": ""}
        _mark_seen(event_id)

    # Handle events
    if payload.get("type") == "event_callback":
        ev = payload.get("event", {})
        if ev.get("type") == "app_mention":
            text = (ev.get("text") or "").strip()
            channel = ev.get("channel")
            thread_ts = ev.get("ts")

            # strip the @mention prefix to form the question
            # Slack usually formats as "<@UXXXXXX> rest of text"
            parts = text.split(">", 1)
            question = parts[1].strip() if len(parts) > 1 else text

            # Quick ping shortcut
            if question.lower() == "ping":
                try:
                    _post_message(channel, text="pong", thread_ts=thread_ts)
                except Exception as e:
                    print("[slack] postMessage error:", e)
                return {"statusCode": 200, "body": ""}

            # Call RAG and post ONCE
            try:
                j = _call_rag_api(question)
                ans = j.get("answer") or "No answer."
                cits = j.get("citations") or []
                blocks = _as_blocks(ans, cits)
                try:
                    _post_message(channel, text="RAG result", blocks=blocks, thread_ts=thread_ts)
                except Exception as e:
                    print("[slack] blocks post error; falling back to text:", e)
                    _post_message(channel, text=_clean_text(ans), thread_ts=thread_ts)
            except Exception as e:
                print("[rag] error calling RAG API:", e)
                _post_message(channel, text="I couldn’t reach the RAG API right now.", thread_ts=thread_ts)

        # Always ACK
        return {"statusCode": 200, "body": ""}

    # Default ACK
    return {"statusCode": 200, "body": ""}
    