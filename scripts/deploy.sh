#!/usr/bin/env bash
set -euo pipefail

STACK=slack-rag
REGION=${1:-us-east-1}
RAG_API_URL=${2:-https://your-fastapi.example.com/chat}

# Package & deploy
sam build -t infra/sam/template.yaml
sam deploy \
  --stack-name "$STACK" \
  --resolve-s3 \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides RagApiUrl="$RAG_API_URL" \
  --region "$REGION"
