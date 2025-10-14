set -euo pipefail

AWS_REGION=${AWS_REGION:-us-east-1}
RAG_API_URL=${RAG_API_URL:-"https://your-fastapi.example.com/chat"}

# REQUIRED: set these before running or hardcode them here
: "${ARTIFACTS_BUCKET:?Need ARTIFACTS_BUCKET env var}"
: "${SLACK_BOT_TOKEN_ARN:?Need SLACK_BOT_TOKEN_ARN env var}"
: "${SLACK_SIGNING_SECRET_ARN:?Need SLACK_SIGNING_SECRET_ARN env var}"

# sam build
# sam deploy \
#   --stack-name slack-rag \
#   --region "$AWS_REGION" \
#   --capabilities CAPABILITY_IAM \
#   --parameter-overrides \
#     RagApiUrl="$RAG_API_URL" \
#     ArtifactsBucket="$ARTIFACTS_BUCKET" \
#     SlackBotTokenArn="$SLACK_BOT_TOKEN_ARN" \
#     SlackSigningSecretArn="$SLACK_SIGNING_SECRET_ARN"

sam build -t infra/sam/template.yaml
sam deploy -t infra/sam/template.yaml \
  --stack-name slack-rag \
  --region us-east-1 \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --parameter-overrides \
    RagApiUrl=$RAG_API_URL \
    ArtifactsBucket=$ARTIFACTS_BUCKET \
    SlackBotTokenArn=$SLACK_BOT_TOKEN_ARN \
    SlackSigningSecretArn=$SLACK_SIGNING_SECRET_ARN
