#!/usr/bin/env bash
# Build the arm64 Lambda image and deploy with AWS SAM.
# Needs: AWS CLI credentials, SAM CLI, Docker (with arm64 emulation if building on x86:
#   docker run --privileged --rm tonistiigi/binfmt --install arm64).
# Optional: BEDROCK_MODEL_ID=<model or inference profile id> ./deploy/deploy.sh
set -euo pipefail
cd "$(dirname "$0")"
STACK=${STACK:-inspectagent}
REGION=${AWS_REGION:-us-east-1}
sam build --template-file template.yaml
sam deploy --stack-name "$STACK" --region "$REGION" --resolve-image-repos \
  --capabilities CAPABILITY_IAM --no-confirm-changeset --no-fail-on-empty-changeset \
  --parameter-overrides BedrockModelId="${BEDROCK_MODEL_ID:-}" DefaultPlanner="${PLANNER:-rules}"
URL=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='DemoUrl'].OutputValue" --output text)
echo "Demo URL: $URL"
curl -s "${URL}api/health" && echo
