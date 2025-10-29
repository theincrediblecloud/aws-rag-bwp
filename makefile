# ---- Makefile (drop-in) ----
PY ?= python
PIP ?= pip
VENV := .venv
SRC  := src
PYTHONPATH ?= $(PWD)/$(SRC)

export PYTHONPATH

.PHONY: venv install lint test smoke run ingest-local-384 ingest-local-1024 ingest-bedrock deploy delete logs help

venv:
	$(PY) -m venv $(VENV)

install: venv
	. $(VENV)/bin/activate; \
	$(PY) -m pip install --upgrade pip; \
	# Prefer root requirements.txt; fallback to dev if you keep it under src/
	if [ -f requirements.txt ]; then pip install -r requirements.txt; \
	elif [ -f $(SRC)/requirements-dev.txt ]; then pip install -r $(SRC)/requirements-dev.txt; fi; \
	pip install ruff mypy pytest uvicorn

lint:
	. $(VENV)/bin/activate; ruff check $(SRC) tests || true

test:
	. $(VENV)/bin/activate; pytest -q

# ---- Ingest targets ----
# Local, offline MiniLM (384-dim)
ingest-local-384:
	. $(VENV)/bin/activate; \
	export EMBED_PROVIDER=local; \
	export MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2; \
	export INDEX_DIR=$(PWD)/store_local; \
	export INDEX_PATH=$$INDEX_DIR/vectors.npy; \
	export META_PATH=$$INDEX_DIR/meta.jsonl; \
	mkdir -p $$INDEX_DIR; \
	$(PY) -m rag.ingest.pipeline --data-dir data/collections/kt --fresh

# Local, offline 1024-dim (e5-large or bge-large); choose ONE model and keep consistent
ingest-local-1024:
	. $(VENV)/bin/activate; \
	export EMBED_PROVIDER=local; \
	export MODEL_NAME=intfloat/e5-large; \
	export INDEX_DIR=$(PWD)/store_local1024; \
	export INDEX_PATH=$$INDEX_DIR/vectors.npy; \
	export META_PATH=$$INDEX_DIR/meta.jsonl; \
	mkdir -p $$INDEX_DIR; \
	$(PY) -m rag.ingest.pipeline --data-dir data/collections/kt --fresh

# Bedrock Titan 1024-dim (build locally then upload to S3 for PROD)
# Requires: AWS creds, ARTIFACTS_BUCKET, INDEX_PREFIX
ingest-bedrock:
	. $(VENV)/bin/activate; \
	export EMBED_PROVIDER=bedrock; \
	export AWS_REGION=${AWS_REGION}; \
	export BEDROCK_EMBEDDINGS_ID=amazon.titan-embed-text-v2:0; \
	export INDEX_DIR=$(PWD)/store_bedrock; \
	export INDEX_PATH=$$INDEX_DIR/vectors.npy; \
	export META_PATH=$$INDEX_DIR/meta.jsonl; \
	mkdir -p $$INDEX_DIR; \
	$(PY) -m rag.ingest.pipeline --data-dir data/collections/kt --fresh; \
	test -n "${ARTIFACTS_BUCKET}" ; \
	test -n "${INDEX_PREFIX}" ; \
	aws s3 cp $$INDEX_PATH s3://${ARTIFACTS_BUCKET}/${INDEX_PREFIX}/vectors.npy; \
	aws s3 cp $$META_PATH  s3://${ARTIFACTS_BUCKET}/${INDEX_PREFIX}/meta.jsonl

# ---- Local run & smoke ----
run:
	. $(VENV)/bin/activate; \
	export USE_S3_INDEX=false; \
	export EMBED_PROVIDER=local; \
	export MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2; \
	export INDEX_DIR=$(PWD)/store_local; \
	export INDEX_PATH=$$INDEX_DIR/vectors.npy; \
	export META_PATH=$$INDEX_DIR/meta.jsonl; \
	uvicorn rag.api.app:app --reload --port 8000 --app-dir $(SRC)

smoke:
	lsof -ti :8000 | xargs kill -9 || true
	. $(VENV)/bin/activate && \
	USE_S3_INDEX=false \
	EMBED_PROVIDER=local \
	MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
	INDEX_DIR=$(PWD)/store_local \
	INDEX_PATH=$(PWD)/store_local/vectors.npy \
	META_PATH=$(PWD)/store_local/meta.jsonl \
	PYTHONPATH=$(PYTHONPATH) \
	uvicorn rag.api.app:app --app-dir $(SRC) --port 8000 & \
	SRV=$$!; \
	sleep 2; \
	./tests/smoke_test.sh || { kill $$SRV; exit 1; }; \
	kill $$SRV

# ---- Deploy & logs ----
deploy:
	test -n "${STACK_NAME}"; \
	test -n "${AWS_REGION}"; \
	test -n "${ARTIFACTS_BUCKET}"; \
	test -n "${INDEX_PREFIX}"; \
	sam build -t infra/sam/template.yaml; \
	sam deploy \
	  --stack-name ${STACK_NAME} \
	  --region ${AWS_REGION} \
	  --capabilities CAPABILITY_IAM \
	  --resolve-s3 \
	  --parameter-overrides \
	    ArtifactsBucket=${ARTIFACTS_BUCKET} \
	    INDEX_PREFIX=${INDEX_PREFIX} \
	    USE_S3_INDEX=true \
	    RagApiUrl="https://example.invalid/local" \
	    SlackBotTokenArn=${SLACK_BOT_TOKEN_ARN} \
	    SlackSigningSecretArn=${SLACK_SIGNING_SECRET_ARN}

delete:
	test -n "${STACK_NAME}"; \
	test -n "${AWS_REGION}"; \
	sam delete --stack-name ${STACK_NAME} --region ${AWS_REGION} --no-prompts

logs:
	test -n "${STACK_NAME}"; \
	test -n "${AWS_REGION}"; \
	aws logs tail "/aws/lambda/${STACK_NAME}-RAGApi" --region ${AWS_REGION} --follow --since 1h

help:
	@echo "Targets:"
	@echo "  ingest-local-384     Build local 384-d index (MiniLM) into store_local/"
	@echo "  ingest-local-1024    Build local 1024-d index (e5-large) into store_local1024/"
	@echo "  ingest-bedrock       Build Bedrock 1024-d index and upload to S3 (needs ARTIFACTS_BUCKET, INDEX_PREFIX)"
	@echo "  run                  Run API locally against store_local/"
	@echo "  smoke                Start API, run tests/smoke_test.sh, stop API"
	@echo "  deploy               SAM build & deploy (needs STACK_NAME, AWS_REGION, ARTIFACTS_BUCKET, INDEX_PREFIX)"
	@echo "  logs                 Tail lambda logs for the stack (needs STACK_NAME, AWS_REGION)"
	@echo "  delete               Delete stack (needs STACK_NAME, AWS_REGION)"
# ---- end ----
