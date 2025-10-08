# golden/eval_golden.py
import argparse, json, os, sys, requests

def citation_present(cites, gold):
    import os
    def norm(s): return (s or "").strip().lower()
    def base(s): return os.path.basename(norm(s))
    gold_names = { base(g.get("doc","")) for g in gold if g.get("doc") }
    if not gold_names:
        return False
    for c in cites:
        title = norm(c.get("title",""))
        src   = norm(c.get("source_path",""))
        if any(gn and (gn in title or gn in base(src) or gn in src) for gn in gold_names):
            return True
    return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("gold_path", nargs="?", default=os.path.join(os.path.dirname(__file__), "golden_set.jsonl"))
    p.add_argument("--api", default=os.environ.get("RAG_API", "http://127.0.0.1:8000/chat"))
    p.add_argument("--k", type=int, default=int(os.environ.get("HIT_K", "5")))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    # sanity: ping health
    health_url = args.api.replace("/chat", "/health")
    try:
        h = requests.get(health_url, timeout=5)
        h.raise_for_status()
        print(f"[ok] API health @ {health_url}: {h.json()}")
    except Exception as e:
        print(f"[err] Could not reach API health @ {health_url}\n{e}")
        return 2

    total = hit = hascite = 0
    with open(args.gold_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            question = q["question"]
            resp = requests.post(args.api, json={"user_msg": question}, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            cites = data.get("citations", []) or []

            if args.verbose:
                print(f"\n[q] {question}")
                if not cites:
                    print("  [no citations]")
                else:
                    for i, c in enumerate(cites[:args.k], 1):
                        print(f"  [{i}] title={c.get('title')!r} page={c.get('page')} src={c.get('source_path')!r}")

            if citation_present(cites[:args.k], q.get("gold_sources", [])):
                hit += 1
            if cites:
                hascite += 1
            total += 1

    if total == 0:
        print("[warn] No questions found in golden set.")
        return 4

    print(f"\nHit@{args.k}: {hit}/{total} = {hit/total:.1%}")
    print(f"Citation present: {hascite}/{total} = {hascite/total:.1%}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
