# #!/usr/bin/env bash
# # Simple smoke test for local ragbot API
set -euo pipefail

BASE_URL=${1:-http://127.0.0.1:8000}

echo "Checking health at ${BASE_URL}/health"
HTTP=$(curl -s -o /dev/stderr -w "%{http_code}" ${BASE_URL}/health) || true
if [ "$HTTP" != "200" ]; then
  echo "Health check failed (status $HTTP)" >&2
  exit 2
fi

echo "Posting a sample chat request"
RESP=$(curl -s -X POST ${BASE_URL}/chat -H "Content-Type: application/json" -d '{"user_msg":"What is Generative AI?","session_id":"test"}')

# basic JSON sanity checks
# Validate pure JSON first
printf '%s' "$RESP" | python -m json.tool >/dev/null

# Assert fields exist
printf '%s' "$RESP" | jq -e '.answer | length > 0' >/dev/null
printf '%s' "$RESP" | jq -e '.citations | type=="array"' >/dev/null

# Now print nicely
echo "Response:"
printf '%s' "$RESP" | jq .

echo "Smoke test OK"
