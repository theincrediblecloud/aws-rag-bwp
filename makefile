# Makefile
PYTHON ?= python
API_URL ?= http://127.0.0.1:8000/chat
INDEX_DIR ?= store
DATA_DIR ?= data/collections/fam
GOLD ?= golden/fam.jsonl

export PYTHONPATH := $(PWD)/src

.PHONY: venv install ingest fresh run health chat eval clean

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt

ingest:
	@echo "=> Ingesting from $(DATA_DIR) into $(INDEX_DIR)"
	INDEX_DIR=$(INDEX_DIR) $(PYTHON) -m rag.ingest.pipeline --data-dir $(DATA_DIR)

fresh:
	@echo "=> Fresh rebuild into $(INDEX_DIR)"
	rm -rf $(INDEX_DIR)/*
	INDEX_DIR=$(INDEX_DIR) $(PYTHON) -m rag.ingest.pipeline --data-dir $(DATA_DIR) --fresh

run:
	uvicorn rag.api.app:app --reload --port 8000 --app-dir src

health:
	curl -s http://127.0.0.1:8000/health | jq .

chat:
	curl -s -X POST $(API_URL) -H "Content-Type: application/json" \
	  -d '{"user_msg":"What is the primary purpose of the FAM Moderation system?"}' | jq .

eval:
	RAG_API=$(API_URL) $(PYTHON) golden/eval_golden.py $(GOLD) -v

clean:
	rm -rf $(INDEX_DIR)/*
