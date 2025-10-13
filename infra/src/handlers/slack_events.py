# infra/src/handlers/slack_events.py
import json
import base64

def _load_json(body_raw, is_b64):
    if body_raw is None:
        return {}
    if is_b64:
        try:
            body_raw = base64.b64decode(body_raw).decode("utf-8", "ignore")
        except Exception:
            return {}
    try:
        return json.loads(body_raw)
    except Exception:
        return {}

def handler(event, context):
    # Basic proxy integration safety
    body_raw = event.get("body")
    is_b64   = bool(event.get("isBase64Encoded"))
    body     = _load_json(body_raw, is_b64)

    # Slack URL verification
    if body.get("type") == "url_verification" and "challenge" in body:
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "text/plain"},
            "body": body["challenge"],
        }

    # Quick OK for event callbacks (signature verification can be added later)
    if body.get("type") == "event_callback":
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"ok": True}),
        }

    # Default OK to keep Slack happy
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"ok": True}),
    }
