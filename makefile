# ---- Makefile (drop-in) ----
PY ?= python
PIP ?= pip
VENV := .venv
SRC  := src
PYTHONPATH ?= $(PWD)/$(SRC)

export PYTHONPATH
SHELL := /bin/bash

.PHONY: venv install lint test smoke run ingest-local-384 ingest-local-1024 ingest-bedrock deploy delete logs help

venv:
	$(PY) -m venv $(VENV)	

install: venv
	. $(VENV)/bin/activate; \
	$(PY) -m pip install --upgrade pip; \
	if [ -f requirements.txt ]; then pip install -r requirements.txt; \
	elif [ -f $(SRC)/requirements-dev.txt ]; then pip install -r $(SRC)/requirements-dev.txt; fi; \
	pip install ruff mypy pytest uvicorn python-dotenv	

lint:
	. $(VENV)/bin/activate; ruff check $(SRC) tests || true	

test:
	. $(VENV)/bin/activate; pytest -q	

# ---- Ingest targets ----
ingest-local-384:
	. $(VENV)/bin/activate; \
	set -a; . .env; set +a; \
	export EMBED_PROVIDER=local; \
	export MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2; \
	export INDEX_DIR=$(PWD)/store_local; \
	export INDEX_PATH=$$INDEX_DIR/vectors.npy; \
	export META_PATH=$$INDEX_DIR/meta.jsonl; \
	mkdir -p $$INDEX_DIR; \
	$(PY) -m rag.ingest.pipeline --data-dir data/collections/kt --fresh	

ingest-local-1024:
	. $(VENV)/bin/activate; \
	set -a; . .env; set +a; \
	export EMBED_PROVIDER=local; \
	export MODEL_NAME=intfloat/e5-large; \
	export INDEX_DIR=$(PWD)/store_local1024; \
	export INDEX_PATH=$$INDEX_DIR/vectors.npy; \
	export META_PATH=$$INDEX_DIR/meta.jsonl; \
	mkdir -p $$INDEX_DIR; \
	$(PY) -m rag.ingest.pipeline --data-dir data/collections/kt --fresh	

ingest-bedrock:
	. $(VENV)/bin/activate; \
	set -a; . .env; set +a; \
	export EMBED_PROVIDER=bedrock; \
	export INDEX_DIR=$(PWD)/store_bedrock; \
	export INDEX_PATH=$$INDEX_DIR/vectors.npy; \
	export META_PATH=$$INDEX_DIR/meta.jsonl; \
	mkdir -p $$INDEX_DIR; \
	$(PY) -m rag.ingest.pipeline --data-dir data/collections/kt --fresh; \
	test -n "$$ARTIFACTS_BUCKET" ; \
	test -n "$$INDEX_PREFIX" ; \
	aws s3 cp $$INDEX_PATH s3://$$ARTIFACTS_BUCKET/$$INDEX_PREFIX/vectors.npy; \
	aws s3 cp $$META_PATH  s3://$$ARTIFACTS_BUCKET/$$INDEX_PREFIX/meta.jsonl	

# ---- Local run & smoke ----

run:
	. $(VENV)/bin/activate; \
	set -a; . .env; set +a; \
	uvicorn rag.api.app:app --reload --port 8000 --app-dir $(SRC)	

smoke:
	lsof -ti :8000 | xargs kill -9 || true
	. $(VENV)/bin/activate && \
	set -a; . .env; set +a; \
	PYTHONPATH=$(PYTHONPATH) \
	uvicorn rag.api.app:app --app-dir $(SRC) --port 8000 & \
	SRV=$$!; \
	sleep 2; \
	./tests/smoke_test.sh || { kill $$SRV; exit 1; }; \
	kill -TERM $$SRV; wait $$SRV	

# ---- Deploy & logs ----

deploy:
	. $(VENV)/bin/activate; \
	set -a; . .env; set +a; \
	test -n "$$STACK_NAME"; \
	test -n "$$AWS_REGION"; \
	test -n "$$ARTIFACTS_BUCKET"; \
	test -n "$$INDEX_PREFIX"; \
	sam build -t infra/sam/template.yaml; \
	sam deploy \
	  --stack-name $$STACK_NAME \
	  --region $$AWS_REGION \
	  --capabilities CAPABILITY_IAM \
	  --resolve-s3 \
	  --parameter-overrides \
	    ArtifactsBucket=$$ARTIFACTS_BUCKET \
	    INDEX_PREFIX=$$INDEX_PREFIX \
	    USE_S3_INDEX=true \
	    RagApiUrl="https://example.invalid/local" \
	    SlackBotTokenArn=$$SLACK_BOT_TOKEN_ARN \
	    SlackSigningSecretArn=$$SLACK_SIGNING_SECRET_ARN	

delete:
	. $(VENV)/bin/activate; \
	set -a; . .env; set +a; \
	test -n "$$STACK_NAME"; \
	test -n "$$AWS_REGION"; \
	sam delete --stack-name $$STACK_NAME --region $$AWS_REGION --no-prompts	

logs:
	. $(VENV)/bin/activate; \
	set -a; . .env; set +a; \
	test -n "$$STACK_NAME"; \
	test -n "$$AWS_REGION"; \
	aws logs tail "/aws/lambda/$$STACK_NAME-RAGApi" --region $$AWS_REGION --follow --since 1h	

help:
	@echo "Targets:"
	@echo "  ingest-local-384     Build local 384-d index (MiniLM) into store_local/"
	@echo "  ingest-local-1024    Build local 1024-d index (e5-large) into store_local1024/"
	@echo "  ingest-bedrock       Build Bedrock 1024-d index and upload to S3 (needs ARTIFACTS_BUCKET, INDEX_PREFIX)"
	@echo "  run                  Run API locally using .env values"
	@echo "  smoke                Start API, run tests/smoke_test.sh, stop API"
	@echo "  deploy               SAM build & deploy (uses .env values)"
	@echo "  logs                 Tail lambda logs for the stack (uses .env values)"	