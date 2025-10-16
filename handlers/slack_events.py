# handlers/slack_events.py
import os, json, hmac, hashlib, time, urllib.request
from base64 import b64decode
import boto3

secrets = boto3.client("secretsmanager", region_name=os.environ.get("AWS_REGION", "us-east-1"))

_CACHE = {}

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

def _post_message(channel: str, text: str):
    bot_token = _get_secret_by_env("SLACK_BOT_TOKEN_ARN")
    data = json.dumps({"channel": channel, "text": text}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {bot_token}"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:
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

    # Ack fast; do simple behavior
    if payload.get("type") == "event_callback":
        ev = payload.get("event", {})
        if ev.get("type") == "app_mention":
            text = (ev.get("text") or "").lower()
            channel = ev.get("channel")

            if "ping" in text:
                try:
                    _post_message(channel, "pong")
                except Exception as e:
                    print("[slack] postMessage error:", e)

            else:
                # call RAG API
                rag_url = os.environ.get("RagApiUrl", "").strip()
                if rag_url:
                    req = urllib.request.Request(
                        rag_url,
                        data=json.dumps({"user_msg": ev.get("text", "")}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    try:
                        with urllib.request.urlopen(req, timeout=5) as r:
                            j = json.loads(r.read().decode())
                            ans = j.get("answer") or "No answer."
                    except Exception as e:
                        print("[rag] error calling RAG API:", e)
                        ans = "I couldn’t reach the RAG API right now."
                else:
                    ans = "Hi! Try @RAG ping or ask me a question."

                try:
                    _post_message(channel, ans)
                except Exception as e:
                    print("[slack] postMessage error:", e)

        return {"statusCode": 200, "body": ""}

    return {"statusCode": 200, "body": ""}
