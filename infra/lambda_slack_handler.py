# lambda_slack_handler.py
# Drop-in Slack → RAG bridge for AWS Lambda (API Gateway proxy)
#
# Env vars required:
#   SLACK_BOT_TOKEN
#   SLACK_SIGNING_SECRET
#   RAG_API_URL              # e.g., https://xxxx.execute-api.us-east-1.amazonaws.com/Prod/chat
#
# Optional:
#   STRIP_BOT_MENTIONS=true  # remove "<@Uxxxx>" from user_msg
#   TIMEOUT_SECS=10          # HTTP timeout to RAG API
#
# Notes:
# - Handles: url_verification, event_callback (app_mention/message), slash commands
# - Uses thread root as the canonical "question" and sends it as `last_q`
# - Stable session_id per thread: f"{channel}_{thread_ts_or_ts}"

import os, json, hmac, hashlib, time, re
import urllib.parse
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
RAG_API_URL = os.environ["RAG_API_URL"]
TIMEOUT_SECS = float(os.getenv("TIMEOUT_SECS", "10"))
STRIP_BOT_MENTIONS = os.getenv("STRIP_BOT_MENTIONS", "true").lower() == "true"

slack = WebClient(token=SLACK_BOT_TOKEN)

MENTION_RE = re.compile(r"<@([A-Z0-9]+)>")
FOLLOWUP_RE = re.compile(
    r'^\s*(more( details| info| examples)?|examples?\??|show (me )?examples?|expand|elaborate|deep[\s-]*dive|drill down|tell me more)\b',
    re.I
)

def _ok(body: dict):
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}

def _text_ok(txt: str = "OK"):
    return {"statusCode": 200, "headers": {"Content-Type": "text/plain"}, "body": txt}

def _bad_request(msg: str):
    return {"statusCode": 400, "headers": {"Content-Type": "application/json"}, "body": json.dumps({"error": msg})}

def _verify_slack_signature(headers, body: str) -> bool:
    """Verify Slack signing secret (required for Events + Slash)."""
    ts = headers.get("X-Slack-Request-Timestamp") or headers.get("x-slack-request-timestamp")
    sig = headers.get("X-Slack-Signature") or headers.get("x-slack-signature")
    if not ts or not sig:
        return False
    # Reject old requests (+/− 5 minutes)
    if abs(time.time() - float(ts)) > 60 * 5:
        return False
    basestring = f"v0:{ts}:{body}".encode("utf-8")
    my_sig = "v0=" + hmac.new(SLACK_SIGNING_SECRET.encode("utf-8"), basestring, hashlib.sha256).hexdigest()
    return hmac.compare_digest(my_sig, sig)

def normalize_slack_text(s: str) -> str:
    s = (s or "").strip()
    if STRIP_BOT_MENTIONS:
        s = MENTION_RE.sub("", s).strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def fetch_root_message_text(channel: str, root_ts: str) -> str:
    """Get the first/root message text for a thread (or the original message if not threaded)."""
    try:
        # Slack returns the root as the first item when using conversations_replies with ts=root_ts
        r = slack.conversations_replies(channel=channel, ts=root_ts, inclusive=True, limit=1)
        msgs = r.get("messages", [])
        if msgs:
            return normalize_slack_text(msgs[0].get("text", ""))
    except SlackApiError as e:
        print(f"[slack] conversations_replies failed: {e.response['error']}")
    return ""

def rag_chat(user_msg: str, session_id: str, last_q: str | None, domain: str | None = None) -> dict:
    payload = {
        "user_msg": user_msg,
        "session_id": session_id,
        "domain": domain,
    }
    # pass the canonical topic for cold-start continuity
    if last_q:
        payload["last_q"] = last_q

    print(f"[rag] POST {RAG_API_URL} session={session_id} user_msg='{user_msg[:80]}' last_q='{(last_q or '')[:80]}'")
    try:
        resp = requests.post(RAG_API_URL, json=payload, timeout=TIMEOUT_SECS)
        if resp.status_code != 200:
            print(f"[rag] non-200: {resp.status_code} {resp.text[:200]}")
            return {"answer": f"(stub) RAG error {resp.status_code}", "citations": []}
        return resp.json()
    except Exception as e:
        print(f"[rag] request failed: {e}")
        return {"answer": "(stub) RAG unavailable", "citations": []}

def respond_in_thread(channel: str, thread_ts: str, text: str):
    # Keep Slack message simple; your server already formats answer nicely if needed
    try:
        slack.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text or "…")
    except SlackApiError as e:
        print(f"[slack] chat_postMessage failed: {e.response['error']}")

def _handle_event_callback(event_body: dict):
    ev = event_body.get("event", {}) or {}
    # Slack retry (avoid double processing)
    if event_body.get("headers", {}).get("X-Slack-Retry-Num"):
        return _text_ok("retry acknowledged")

    etype = ev.get("type")
    if etype not in ("app_mention", "message"):
        return _text_ok("ignored")

    # Skip bot messages
    if ev.get("subtype") in ("bot_message", "message_changed", "message_deleted"):
        return _text_ok("ignored")

    channel = ev.get("channel")
    ts = ev.get("ts")
    thread_ts = ev.get("thread_ts") or ts
    session_id = f"{channel}_{thread_ts}"

    user_text = normalize_slack_text(ev.get("text", ""))
    root_text = fetch_root_message_text(channel, thread_ts)  # canonical "last_q"

    # If the current message looks like a follow-up ("more/expand/…") and we have a root, keep it short
    if FOLLOWUP_RE.match(user_text) and root_text:
        prompt = user_text  # server will stitch with prev['last_q']
        last_q = root_text
    else:
        # First message or new topic → use user text as root, too, if not in a thread
        last_q = root_text or user_text
        prompt = user_text

    out = rag_chat(prompt, session_id=session_id, last_q=last_q, domain=None)
    answer = out.get("answer") or "(no answer)"
    respond_in_thread(channel, thread_ts, answer)
    return _text_ok("ok")

def _handle_slash_command(params: dict, headers: dict):
    # Slash commands arrive as application/x-www-form-urlencoded
    channel_id = params.get("channel_id")
    user_text = normalize_slack_text(params.get("text", ""))
    ts = params.get("trigger_id") or str(time.time())  # use a stable handle; thread_ts not given
    # We can still create a deterministic session per channel+text hash if needed
    session_id = f"{channel_id}_{int(time.time())}"

    # For slash, consider the typed text the root
    out = rag_chat(user_text, session_id=session_id, last_q=user_text, domain=None)
    # Respond in-channel using response_url if available
    response_url = params.get("response_url")
    if response_url:
        try:
            requests.post(response_url, json={"response_type": "in_channel", "text": out.get("answer", "(no answer)")}, timeout=5)
        except Exception as e:
            print(f"[slash] response_url post failed: {e}")

    return _text_ok("")

def lambda_handler(event, context):
    # API Gateway v2 (HTTP API) typically places headers/body here:
    headers = event.get("headers") or {}
    raw_body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        # Slack sends plain text; but handle defensively
        raw_body = urllib.parse.unquote_plus(raw_body)

    # 1) Slack URL verification (Events API)
    try:
        body = json.loads(raw_body)
        if body.get("type") == "url_verification":
            return _ok({"challenge": body.get("challenge", "")})
    except Exception:
        # Not JSON → maybe slash command or form-encoded event
        pass

    # Verify signature (if we can). For slash commands, body is form-encoded as a string.
    if not _verify_slack_signature(headers, raw_body):
        print("[sec] signature verification failed")
        return _bad_request("invalid signature")

    # 2) Slash command (x-www-form-urlencoded)
    ctype = headers.get("Content-Type", headers.get("content-type", ""))
    if ctype.startswith("application/x-www-form-urlencoded"):
        params = dict(urllib.parse.parse_qsl(raw_body))
        return _handle_slash_command(params, headers)

    # 3) Events API envelope
    try:
        body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
    except Exception:
        return _bad_request("bad json")

    if body.get("type") == "event_callback":
        return _handle_event_callback({"headers": headers, "event": body.get("event", {})})

    # Fallback
    return _text_ok("ignored")
