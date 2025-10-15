# handlers/slack_events.py
import os
import json
import time
import hmac
import hashlib
import urllib.request
import urllib.error
from base64 import b64decode

import boto3


# --- config / clients ---
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
_secrets = boto3.client("secretsmanager", region_name=AWS_REGION)

# Optional: if you want to call your FastAPI RAG API for non-"ping" messages
RAG_API_URL = os.environ.get("RagApiUrl") or os.environ.get("RAG_API_URL")


# --- helpers: secrets ---

def _get_secret_value(secret_id: str) -> str:
    """Read a secret from Secrets Manager. Supports SecretString or SecretBinary."""
    resp = _secrets.get_secret_value(SecretId=secret_id)
    if "SecretString" in resp and resp["SecretString"] is not None:
        return resp["SecretString"]
    if "SecretBinary" in resp and resp["SecretBinary"] is not None:
        return b64decode(resp["SecretBinary"]).decode("utf-8")
    raise RuntimeError(f"Secret {secret_id} has no value")

def _load_secret(env_arn_key: str, env_plain_key: str) -> str:
    """
    Prefer reading the ARN from env and fetching from Secrets Manager.
    If not present, fall back to a plaintext env var (handy for local testing).
    """
    arn_or_name = (os.environ.get(env_arn_key) or "").strip()
    if arn_or_name:
        # Safe debug; prints the id (not the value)
        print(f"[secrets] using {env_arn_key} -> {repr(arn_or_name)}")
        return _get_secret_value(arn_or_name)

    plain = (os.environ.get(env_plain_key) or "").strip()
    if plain:
        # plain can be an actual xoxb- token or signing secret for local/dev
        print(f"[secrets] using plaintext env {env_plain_key}")
        return plain

    raise RuntimeError(f"Missing either {env_arn_key} (ARN) or {env_plain_key} (plaintext)")


# --- helpers: slack ---

def _hget(headers: dict, key: str):
    """Case-insensitive header get (APIGW v2 may lowercase headers)."""
    if not headers:
        return None
    return headers.get(key) or headers.get(key.lower()) or headers.get(key.title())

def _verify_slack_signature(headers: dict, body_bytes: bytes) -> bool:
    signing_secret = _load_secret("SLACK_SIGNING_SECRET_ARN", "SLACK_SIGNING_SECRET")

    ts = _hget(headers, "X-Slack-Request-Timestamp")
    sig = _hget(headers, "X-Slack-Signature")
    if not ts or not sig:
        print("[verify] missing ts/signature")
        return False

    # replay guard (5 minutes)
    try:
        if abs(time.time() - int(ts)) > 60 * 5:
            print("[verify] timestamp too old")
            return False
    except Exception:
        print("[verify] bad timestamp")
        return False

    base = f"v0:{ts}:{body_bytes.decode('utf-8')}".encode("utf-8")
    expected = "v0=" + hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()

    ok = hmac.compare_digest(expected, sig)
    if not ok:
        print("[verify] signature mismatch")
    return ok

def _slack_post_message(channel: str, text: str) -> dict:
    token = _load_secret("SLACK_BOT_TOKEN_ARN", "SLACK_BOT_TOKEN")
    data = json.dumps({"channel": channel, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            # 'charset' is optional; Slack warns if it's missing but still works
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            print(f"[slack] chat.postMessage -> {payload}")
            return payload
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        print(f"[slack] HTTPError {e.code}: {err_body}")
        return {"ok": False, "error": f"http_{e.code}"}
    except Exception as e:
        print(f"[slack] error: {e!r}")
        return {"ok": False, "error": "exception"}

def _strip_bot_mention(text: str, bot_user_id: str | None) -> str:
    if not text:
        return ""
    t = text
    # strip "<@Uxxxxxx>" in both cases
    t = t.replace(f"<@{bot_user_id}>", "") if bot_user_id else t
    # lowercased variants sometimes appear in logs; be permissive
    t = t.replace(f"<@{(bot_user_id or '').lower()}>", "")
    return t.strip()


# --- (optional) call your RAG API ---

def _call_rag_api(user_msg: str) -> str:
    """POST { user_msg: ... } to your FastAPI /chat and return answer text or a fallback."""
    if not RAG_API_URL:
        return "I’m connected, but no RAG API URL is configured."
    try:
        body = json.dumps({"user_msg": user_msg}).encode("utf-8")
        req = urllib.request.Request(
            RAG_API_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            # Adjust if your API shape differs
            return (out.get("answer") or "").strip() or "I didn’t get an answer back."
    except Exception as e:
        print(f"[rag] error calling RAG API: {e!r}")
        return "I couldn’t reach the RAG API right now."


# --- lambda handler ---

def handler(event, context):
    """
    Handles Slack Events API via API Gateway HTTP API (v2).
    - Responds to url_verification with the challenge.
    - Verifies Slack signature.
    - On app_mention: replies 'pong' to '@RAG ping' or forwards text to RAG API if configured.
    """
    headers = event.get("headers") or {}
    body = event.get("body") or ""
    is_b64 = bool(event.get("isBase64Encoded"))
    body_bytes = b64decode(body) if is_b64 else body.encode("utf-8")

    # Slack url_verification challenge
    try:
        payload = json.loads(body_bytes.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge", "")
        print("[recv] url_verification")
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "text/plain"},
            "body": challenge,
        }

    # Verify signature for all other requests
    try:
        if not _verify_slack_signature(headers, body_bytes):
            return {"statusCode": 401, "body": "bad signature"}
    except Exception as e:
        print(f"[verify] exception: {e!r}")
        return {"statusCode": 401, "body": "bad signature"}

    # Handle events
    if payload.get("type") == "event_callback":
        ev = payload.get("event", {}) or {}
        et = ev.get("type")
        print(f"[recv] type={et} ts={ev.get('event_ts')}")
        if et == "app_mention":
            channel = ev.get("channel")
            text = (ev.get("text") or "").strip()
            bot_user = payload.get("authorizations", [{}])[0].get("user_id") or None
            clean = _strip_bot_mention(text, bot_user)
            print(f"[mention] channel={channel} text={clean}")

            # Fast path: @RAG ping
            if "ping" in clean.lower():
                _slack_post_message(channel, "pong")
                return {"statusCode": 200, "body": ""}

            # Optional: forward to your RAG API
            if clean and RAG_API_URL:
                ans = _call_rag_api(clean)
                _slack_post_message(channel, ans[:3500])  # keep safely under Slack limit
                return {"statusCode": 200, "body": ""}

            # Default fallback
            _slack_post_message(channel, "Hi! Try `@RAG ping` or ask me a question.")
            return {"statusCode": 200, "body": ""}

        # Unhandled event types still get 200
        return {"statusCode": 200, "body": ""}

    # Non-event callbacks (e.g., interactivity) — ACK for now
    return {"statusCode": 200, "body": ""}
