# handlers/slack_events.py
import os, json, hmac, hashlib, time, urllib.request
from base64 import b64decode
import boto3

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
secrets = boto3.client("secretsmanager", region_name=AWS_REGION)

_CACHE = {}
SLACK_BOT_TOKEN = None  # <-- define once

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
    if abs(time.time() - int(ts)) > 60 * 5:
        return False
    signing_secret = _get_secret_by_env("SLACK_SIGNING_SECRET_ARN")
    base = f"v0:{ts}:{body_bytes.decode()}".encode()
    my_sig = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(my_sig, sig)

def _post_message(channel: str, text: str = None, blocks: list = None, thread_ts: str = None):
    global SLACK_BOT_TOKEN
    if SLACK_BOT_TOKEN is None:
        SLACK_BOT_TOKEN = _get_secret_by_env("SLACK_BOT_TOKEN_ARN")

    payload = {"channel": channel}
    if text is not None:
        payload["text"] = text  # fallback for notifications / mobile previews
    if blocks is not None:
        payload["blocks"] = blocks
    if thread_ts:
        payload["thread_ts"] = thread_ts

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    )
    with urllib.request.urlopen(req, timeout=7) as r:
        return json.loads(r.read().decode())

def _safe_trim(text: str, limit: int = 2800) -> str:
    # Slack section block text limit is ~3000 chars. Keep margin.
    if text and len(text) > limit:
        return text[:limit].rstrip() + "…"
    return text

def _as_blocks(answer: str, citations: list[dict]) -> list:
    blocks = []
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": _safe_trim(answer)}
    })
    if citations:
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "*Top sources*"}})
        for c in citations[:8]:  # don’t spam
            title = c.get("title") or "(untitled)"
            page = c.get("page")
            page_hint = f" (p. {page})" if page is not None else ""
            score = c.get("score")
            score_hint = f" — _score {score:.3f}_" if isinstance(score, (float, int)) else ""
            line = f"• *{title}{page_hint}*{score_hint}"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": _safe_trim(line, 2900)}})
    return blocks

def _call_rag_api(question: str):
    chat_api = os.environ.get("RagApiUrl", "").strip()
    if not chat_api:
        raise RuntimeError("RagApiUrl env var is empty")
    body = json.dumps({"user_msg": question}).encode()
    req = urllib.request.Request(
        chat_api,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())

def handler(event, context):
    headers = event.get("headers") or {}
    body = event.get("body") or ""
    body_bytes = b64decode(body) if event.get("isBase64Encoded") else body.encode()

    try:
        payload = json.loads(body_bytes.decode() or "{}")
    except Exception:
        payload = {}

    # URL verification
    if payload.get("type") == "url_verification":
        return {"statusCode": 200, "headers": {"Content-Type": "text/plain"}, "body": payload.get("challenge", "")}

    if not _verify(headers, body_bytes):
        return {"statusCode": 401, "body": "bad signature"}

    # Ack fast; handle only app_mention
    if payload.get("type") == "event_callback":
        ev = payload.get("event", {})
        if ev.get("type") == "app_mention":
            text = (ev.get("text") or "").strip()
            channel = ev.get("channel")
            thread_ts = ev.get("ts")  # respond in thread

            if "ping" in text.lower():
                try:
                    _post_message(channel, "pong", thread_ts=thread_ts)
                except Exception as e:
                    print("[slack] postMessage error:", e)
            else:
                try:
                    j = _call_rag_api(text)
                    ans = j.get("answer") or "No answer."
                    cits = j.get("citations", [])
                    # Prefer Blocks
                    try:
                        blocks = _as_blocks(ans, cits)
                        _post_message(channel, text="RAG result", blocks=blocks, thread_ts=thread_ts)
                    except Exception as e:
                        print("[slack] post blocks error:", e)
                        _post_message(channel, ans, thread_ts=thread_ts)  # fallback
                except Exception as e:
                    print("[rag] error calling RAG API:", e)
                    _post_message(channel, "I couldn’t reach the RAG API right now.", thread_ts=thread_ts)

        return {"statusCode": 200, "body": ""}

    return {"statusCode": 200, "body": ""}