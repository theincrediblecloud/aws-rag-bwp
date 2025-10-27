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

# echo "Posting a sample chat request"
RESP=$(curl -s -X POST ${BASE_URL}/chat -H "Content-Type: application/json" -d '{"user_msg":"What is the primary purpose of the FAM Moderation system?","session_id":"test"}')
echo "Response: $RESP"

# basic JSON sanity checks
echo "$RESP" | jq -e '.answer and .citations' >/dev/null || { echo "Invalid response structure" >&2; exit 3; }

echo "Smoke test OK"
