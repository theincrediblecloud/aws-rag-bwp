import os, subprocess, time, json, urllib.request, tempfile, shutil
import numpy as np
from fastapi.testclient import TestClient
from rag.api.app import app

BASE = "http://127.0.0.1:8000"


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8")

def _wait_health(retries=15, sleep=0.2):
    for _ in range(retries):
        try:
            h = json.loads(_get(f"{BASE}/health", timeout=1))
            if h.get("ok"): return h
        except Exception:
            pass
        time.sleep(sleep)
    raise RuntimeError("API /health not ready")

def test_local_api_health_and_chat(tmp_path):
    # Prepare a tiny local index (384-dim to match MiniLM)
    dim = 384
    n = 3
    V = np.zeros((n, dim), dtype=np.float32)
    V[0,0] = V[1,1] = V[2,2] = 1.0
    index_dir = tmp_path / "store"
    index_dir.mkdir()
    np.save(index_dir / "vectors.npy", V, allow_pickle=False)
    with open(index_dir / "meta.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"chunk_text": f"unit {i}", "title": f"v{i}", "source_path": f"unit{i}.txt"})+"\n")

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.path.abspath("src"),
        "USE_S3_INDEX": "false",
        "EMBED_PROVIDER": "local",
        "MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        "LLM_PROVIDER": "none",
        "INDEX_DIR": str(index_dir),
        "INDEX_PATH": str(index_dir / "vectors.npy"),
        "META_PATH": str(index_dir / "meta.jsonl"),
    })

    proc = subprocess.Popen(
        ["uvicorn", "rag.api.app:app", "--app-dir", "src", "--port", "8000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    try:
        _wait_health()  # retries instead of fixed sleep
        h = json.loads(_get(f"{BASE}/health"))
        assert h.get("ok") is True

        req = json.dumps({"user_msg": "hello"}).encode("utf-8")
        with urllib.request.urlopen(urllib.request.Request(f"{BASE}/chat", data=req, headers={"Content-Type":"application/json"}), timeout=5) as r:
            out = json.loads(r.read().decode("utf-8"))
        assert "answer" in out
        assert isinstance(out.get("citations"), list)
    finally:
        proc.kill()
        proc.wait(timeout=2)


def test_followup_uses_previous_topic(tmp_path):
    import requests, time

    # minimal index so the app can start
    dim, n = 384, 1
    V = np.zeros((n, dim), dtype=np.float32)
    V[0,0] = 1.0
    index_dir = tmp_path / "store"
    index_dir.mkdir()
    np.save(index_dir / "vectors.npy", V, allow_pickle=False)
    with open(index_dir / "meta.jsonl", "w") as f:
        f.write(json.dumps({"chunk_text": "unit 0", "title": "v0", "source_path": "unit0.txt"})+"\n")

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.path.abspath("src"),
        "USE_S3_INDEX": "false",
        "EMBED_PROVIDER": "local",
        "MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        "LLM_PROVIDER": "none",
        "INDEX_DIR": str(index_dir),
        "INDEX_PATH": str(index_dir / "vectors.npy"),
        "META_PATH": str(index_dir / "meta.jsonl"),
    })

    proc = subprocess.Popen(
        ["uvicorn", "rag.api.app:app", "--app-dir", "src", "--port", "8000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    try:
        _wait_health()
        base = "http://127.0.0.1:8000"
        sess = f"t_{int(time.time())}"
        r1 = requests.post(f"{base}/chat", json={"user_msg":"What are the benefits of generative AI?", "session_id":sess})
        assert r1.status_code == 200
        r2 = requests.post(f"{base}/chat", json={"user_msg":"more examples?", "session_id":sess})
        assert r2.status_code == 200
        j = r2.json()
        assert "answer" in j and len(j["answer"]) > 0
    finally:
        proc.kill()
        proc.wait(timeout=2)
