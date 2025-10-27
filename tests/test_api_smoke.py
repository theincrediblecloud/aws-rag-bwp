# tests/test_api_smoke.py
import json
import subprocess
import time
import urllib.request

BASE = "http://127.0.0.1:8000"

def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()

def _post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read().decode()

def test_local_api_health_and_chat():
    # spin up dev server
    proc = subprocess.Popen(
        ["uvicorn", "rag.api.app:app", "--app-dir", "src", "--port", "8000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        time.sleep(1.5)
        h = json.loads(_get(f"{BASE}/health"))
        assert "ok" in h

        c = json.loads(_post(f"{BASE}/chat", {"user_msg": "ping"}))
        assert "answer" in c
    finally:
        proc.terminate()
        proc.wait(timeout=5)    
        