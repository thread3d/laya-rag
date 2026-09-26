# Developer tasks for the Laya RAG engine.
#
# `setup` assumes the Laya checkout has already been provisioned once with
# `PYTHON=python3 vendor/laya/setup_laya.sh`, which downloads the checkpoints.

PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
PIP_CACHE := $(CURDIR)/vendor/laya/.pip-cache
CORPUS ?= vendor/laya/docs
PORT ?= 8077

.PHONY: help setup doctor index serve query gui test test-real test-gui lint clean

help:
	@echo "setup        create .venv and install the engine in editable mode"
	@echo "doctor       check that the Laya checkpoint loads"
	@echo "index        (re)index CORPUS (default: $(CORPUS))"
	@echo "serve        run the HTTP service and GUI on PORT (default: $(PORT))"
	@echo "gui          serve, then open the GUI in a browser"
	@echo "query Q=...  ask the CLI for evidence"
	@echo "test         offline Python suite (no checkpoints needed)"
	@echo "test-real    Python suite against real Laya weights"
	@echo "test-gui     drive the running service's GUI in headless Chrome"
	@echo "lint         byte-compile the package and syntax-check the JavaScript"
	@echo "clean        remove the index and caches"

setup:
	@test -x $(PY) || $(PYTHON) -m venv $(VENV)
	PIP_CACHE_DIR="$(PIP_CACHE)" $(PY) -m pip install --quiet --upgrade pip
	PIP_CACHE_DIR="$(PIP_CACHE)" $(PY) -m pip install --quiet \
		"numpy<2" "torch==2.2.2" "transformers==4.57.6" \
		"safetensors>=0.4.0" "huggingface_hub>=0.34.0,<1.0"
	PIP_CACHE_DIR="$(PIP_CACHE)" $(PY) -m pip install --quiet -e vendor/laya --no-deps
	PIP_CACHE_DIR="$(PIP_CACHE)" $(PY) -m pip install --quiet \
		"fastapi>=0.110.0" "uvicorn>=0.27.0" "pydantic>=2.0" \
		"pytest>=8.0" "httpx>=0.27"
	PIP_CACHE_DIR="$(PIP_CACHE)" $(PY) -m pip install --quiet -e . --no-deps
	@echo "ready: $(PY)"

doctor:
	$(PY) -m laya_rag.cli doctor

index:
	$(PY) -m laya_rag.cli index $(CORPUS) --rebuild

serve:
	OMP_NUM_THREADS=16 $(PY) -m laya_rag.api

# `serve` already prints the GUI URL; this just saves the round trip on macOS.
gui:
	@(sleep 2; command -v open >/dev/null && open "http://127.0.0.1:$(PORT)/") &
	OMP_NUM_THREADS=16 $(PY) -m laya_rag.api

query:
	@test -n "$(Q)" || (echo 'usage: make query Q="your question"' >&2; exit 2)
	$(PY) -m laya_rag.cli query "$(Q)"

test:
	$(PY) -m pytest tests/ -q -m "not real_weights"

test-real:
	$(PY) -m pytest tests/test_real_weights.py -q -m real_weights

# Skips (exit 0) rather than failing when Chrome, Node's WebSocket, or a running
# engine is absent, so run `make serve` first to actually exercise the page.
test-gui:
	node tools/gui-smoke.mjs

lint:
	$(PY) -m compileall -q src/ tests/
	node --check src/laya_rag/web/app.js
	@echo "ok"

clean:
	rm -rf indexes .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
