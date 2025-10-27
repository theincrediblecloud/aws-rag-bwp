# Makefile
PY?=python
PIP?=pip
VENV=.venv
SRC=src
PYTHONPATH?=$(PWD)/$(SRC)

export PYTHONPATH

.PHONY: venv install lint test smoke vectors deploy delete logs

venv:
	$(PY) -m venv $(VENV)

install: venv
	. $(VENV)/bin/activate; \
	$(PY) -m pip install --upgrade pip; \
	pip install -r $(SRC)/requirements-dev.txt; \
	pip install pytest ruff uvicorn

lint:
	. $(VENV)/bin/activate; ruff check $(SRC) tests

test:
	. $(VENV)/bin/activate; pytest -q

smoke:
	lsof -ti :8000 | xargs kill -9 || true
	. $(VENV)/bin/activate && \
	USE_S3_INDEX=false \
	EMBED_PROVIDER=local \
	LLM_PROVIDER=none \
	MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
	PYTHONPATH=$(PYTHONPATH) \
	uvicorn rag.api.app:app --app-dir $(SRC) --port 8000 & \
	SRV=$$!; \
	sleep 2; \
	./tests/smoke_test.sh; \
	kill $$SRV
# 	curl -fsS http://127.0.0.1:8000/health | jq .; \
# 	curl -fsS -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"user_msg":"ping"}' | jq .; \
# 	kill $$SRV

vectors:
	. $(VENV)/bin/activate; \
	mkdir -p store; \
	$(PY) -m rag.ingest.build_vectors_numpy \
	  --collection data/collections/fam \
	  --bucket $${ARTIFACTS_BUCKET:?ARTIFACTS_BUCKET missing} \
	  --prefix rag/index \
	  --meta_local store/meta.jsonl \
	  --vectors_out store/vectors.npy

deploy:
	sam build --use-container -t infra/sam/template.yaml; \
	sam deploy \
	  --stack-name $${STACK_NAME:?STACK_NAME missing} \
	  --region $${AWS_REGION:?AWS_REGION missing} \
	  --capabilities CAPABILITY_IAM \
	  --resolve-s3 \
	  --parameter-overrides \
	    ArtifactsBucket=$${ARTIFACTS_BUCKET:?} \
	    RagApiUrl="https://example.invalid/local" \
	    SlackBotTokenArn=$${SLACK_BOT_TOKEN_ARN:?} \
	    SlackSigningSecretArn=$${SLACK_SIGNING_SECRET_ARN:?}

delete:
	sam delete --stack-name $${STACK_NAME:?STACK_NAME missing} --region $${AWS_REGION:?AWS_REGION missing} --no-prompts

logs:
	aws logs tail "/aws/lambda/$${STACK_NAME:?}-RAGApi" --follow --since 1h
run:
	. $(VENV)/bin/activate; \
	export USE_S3_INDEX=false; \
	export PYTHONPATH=$(PYTHONPATH); \
	uvicorn rag.api.app:app --reload --port 8000 --app-dir $(SRC)	
	
