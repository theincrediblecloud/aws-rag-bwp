# lambda_slack_handler.py
import os, time, hmac, hashlib, json, urllib.parse, urllib.request
import re

RAG_API = os.environ["RAG_API"].strip()          # e.g., https://api.example.com/chat
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]

def _verify(headers, body_bytes):
    ts = headers.get("X-Slack-Request-Timestamp","")
    sig = headers.get("X-Slack-Signature","")
    bases = f"v0:{ts}:{body_bytes.decode()}"
    digest = "v0=" + hmac.new(SLACK_SIGNING_SECRET.encode(), bases.encode(), hashlib.sha256).hexdigest()
    ok = hmac.compare_digest(digest, sig)
    # replay window (5 min)
    fresh = abs(time.time() - int(ts or "0")) < 60*5
    return ok and fresh

def _post_json(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())

def normalize_slack_text(text: str) -> str:
    t = (text or "").strip()
    # remove bot mentions <@UXXXX> and @rag
    t = re.sub(r"<@[^>]+>", " ", t)
    t = re.sub(r"(^|\\s)@rag(\\b)", " ", t, flags=re.IGNORECASE)
    # drop verbs that add noise
    t = re.sub(r"^(explain|define|what\\s+is)\\s+", "", t, flags=re.IGNORECASE)
    # normalize spaces/quotes
    t = t.replace("“","\"").replace("”","\"").replace("’","'")
    return re.sub(r"\\s+", " ", t).strip() or text

def handler(event, context):
    # Slack sends form-encoded body
    body_bytes = event.get("body","").encode()
    headers = {k: v for k, v in event.get("headers",{}).items()}
    if not _verify(headers, body_bytes):
        return {"statusCode": 401, "body":"invalid signature"}

    form = urllib.parse.parse_qs(body_bytes.decode())
    text = (form.get("text", [""])[0] or "").strip()
    response_url = form.get("response_url", [""])[0]
    user = form.get("user_id", ["unknown"])[0]
    channel = form.get("channel_id", [""])[0]

    # immediate ACK (shows ephemeral message)
    ack = {
      "response_type": "ephemeral",
      "text": f"Working on it, <@{user}>…",
    }
    # For API Gateway with Lambda proxy integration:
    # return early BUT ALSO kick async work via a second lambda or EventBridge.
    # Simpler: do synchronous call here if you’re confident <3s.
    # Here we’ll do the work inline for clarity; if it’s slow, move to async.

    topic = normalize_slack_text(event.get("text", ""))    
    payload = {"user_msg": topic, "session_id": channel}

    # call RAG API
    try:
        print(f"rag_api: calling {RAG_API} for channel {channel} user {user}")
        rag_resp = _post_json(RAG_API, payload)
        answer = rag_resp.get("answer","No answer.")
        cites = rag_resp.get("citations", [])
        lines = [answer, "\n*Citations:*"]
        for c in cites[:3]:
            title = c.get("title","(untitled)")
            src = c.get("source_path","")
            pg  = c.get("page")
            lines.append(f"• {title}{(' p.'+str(pg)) if pg else ''} — `{src}`")
        final = {"response_type": "in_channel", "text": "\n".join(lines)}
        # post to response_url (delayed response)
        _post_json(response_url, final)
        # return ack immediately to Slack
        return {"statusCode": 200, "body": json.dumps(ack)}
    except Exception as e:
        _post_json(response_url, {"response_type":"ephemeral","text": f"Error: {e}"})
        return {"statusCode": 200, "body": json.dumps(ack)}
