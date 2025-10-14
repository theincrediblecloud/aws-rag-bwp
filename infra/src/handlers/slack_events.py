# handlers/slack_events.py
import os, json, hmac, hashlib, time, urllib.request
from base64 import b64decode
import boto3

secrets = boto3.client("secretsmanager")
_cache = {}

def _get_secret(secret_id_env: str) -> str:
    sid = os.environ.get(secret_id_env, "")
    sid = (sid or "").strip()                  # <-- important
    if not sid:
        raise RuntimeError(f"Missing env {secret_id_env}")

    if sid in _cache:
        return _cache[sid]

    # Safe debug (won't print the secret, only the id repr)
    print(f"[debug] fetching secret id from env {secret_id_env}: {repr(sid)}")

    resp = secrets.get_secret_value(SecretId=sid)
    val = resp.get("SecretString")
    if val is None and "SecretBinary" in resp:
        val = b64decode(resp["SecretBinary"]).decode("utf-8")
    _cache[sid] = val
    return val

SLACK_SIGNING_SECRET = None  # lazy load
SLACK_BOT_TOKEN = None       # lazy load

# Optional: accept raw env overrides to simplify debugging
# def _load_secret(maybe_arn_env: str, raw_env: str) -> str:
#     raw = (os.environ.get(raw_env) or "").strip()
#     if raw and not raw.lower().startswith("arn:aws:secretsmanager:"):
#         return raw
#     return _get_secret(maybe_arn_env)

def _verify(req_headers, body: bytes):
    global SLACK_SIGNING_SECRET
    if SLACK_SIGNING_SECRET is None:
        SLACK_SIGNING_SECRET = _get_secret(os.environ["SLACK_SIGNING_SECRET_ARN"])
    ts = req_headers.get("x-slack-request-timestamp") or req_headers.get("X-Slack-Request-Timestamp")
    sig = req_headers.get("x-slack-signature") or req_headers.get("X-Slack-Signature")
    if not ts or not sig: return False
    # replay guard (5 min)
    if abs(time.time() - int(ts)) > 60*5: return False
    base = f"v0:{ts}:{body.decode()}".encode()
    my_sig = "v0=" + hmac.new(SLACK_SIGNING_SECRET.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(my_sig, sig)

def _post_message(channel: str, text: str):
    global SLACK_BOT_TOKEN
    if SLACK_BOT_TOKEN is None:
        SLACK_BOT_TOKEN = _get_secret(os.environ["SLACK_BOT_TOKEN_ARN"])
    data = json.dumps({"channel": channel, "text": text}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())

def handler(event, context):
    # API Gateway v2 (HTTP API) passthrough
    # SLACK_SIGNING_SECRET = _load_secret("SLACK_SIGNING_SECRET_ARN", "SLACK_SIGNING_SECRET")
    # SLACK_BOT_TOKEN     = _load_secret("SLACK_BOT_TOKEN_ARN", "SLACK_BOT_TOKEN")
    headers = event.get("headers") or {}
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body_bytes = b64decode(body)
    else:
        body_bytes = body.encode()

    # Slack URL verification
    try:
        payload = json.loads(body_bytes.decode() or "{}")
    except:
        payload = {}

    if payload.get("type") == "url_verification":
        return {"statusCode": 200, "headers": {"Content-Type": "text/plain"}, "body": payload.get("challenge","")}

    # Verify signature (only for non-verification events)
    if not _verify(headers, body_bytes):
        return {"statusCode": 401, "body": "bad signature"}

    # Immediately ACK to Slack
    # (post back to Slack asynchronously after)
    if payload.get("type") == "event_callback":
        ev = payload.get("event", {})
        if ev.get("type") == "app_mention":
            channel = ev.get("channel")
            text = ev.get("text","")
            # Simple ping/pong
            if "ping" in text.lower():
                try:
                    _post_message(channel, "pong")
                except Exception as e:
                    # log but still ACK
                    print("postMessage error:", e)
        return {"statusCode": 200, "body": ""}

    return {"statusCode": 200, "body": ""}
