#!/usr/bin/env bash
set -euo pipefail

STACK_NAME="${1:-slack-rag-prod}"
REGION="${2:-us-east-1}"

BASE_URL="$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ApiBaseUrl'].OutputValue" \
  --output text --region "$REGION")"

if [[ -z "$BASE_URL" || "$BASE_URL" == "None" ]]; then
  echo "Failed to resolve ApiBaseUrl output"; exit 1
fi

echo "[smoke] GET /health -> $BASE_URL/health"
HEALTH_JSON="$(curl -fsS "$BASE_URL/health")"
echo "$HEALTH_JSON" | jq .

OK="$(echo "$HEALTH_JSON" | jq -r '.ok')"
if [[ "$OK" != "true" ]]; then
  echo "Health not OK"; exit 2
fi

# We allow rag_ready=false during first cold start; just report it
READY="$(echo "$HEALTH_JSON" | jq -r '.rag_ready // false')"
echo "[smoke] rag_ready=$READY"

echo "[smoke] POST /chat"
CHAT_JSON="$(curl -fsS -X POST "$BASE_URL/chat" -H "Content-Type: application/json" \
  -d '{"user_msg":"Summarize FAM moderation solution?"}')"
echo "$CHAT_JSON" | jq .

ANSWER="$(echo "$CHAT_JSON" | jq -r '.answer // ""')"
if [[ -z "$ANSWER" ]]; then
  echo "Empty answer"; exit 3
fi
if echo "$ANSWER" | grep -q "(stub) RAG back-end not initialized"; then
  echo "Stub answer indicates RAG not ready"; exit 4
fi

echo "[smoke] PASS"
