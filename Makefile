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
EVAL_ARGS ?=
BENCH_ARGS ?=
BEIR_ARGS ?=
BEIR_ABLATE_ARGS ?=
MULTIHOP_ARGS ?=
COVERAGE_ARGS ?=
GATE_ARGS ?=
RECALIBRATE_ARGS ?=
ROUTING_ARGS ?=
FINETUNE_ARGS ?=
ONDISK_ARGS ?=
ANSWERABILITY_ARGS ?=
DIAGNOSTICS_ARGS ?=
SIGNIFICANCE_ARGS ?=
CE_CACHE_ARGS ?=
RERANK_EXPERIMENT_ARGS ?=
CHARTS_DIR ?= docs/charts
DIAGRAM ?= docs/architecture.png

.PHONY: help setup doctor index serve query gui test test-real test-gui eval bench bench-index-parallel bench-biencoder-parallel beir beir-ablate multihop multihop-gate multihop-coverage multihop-diagnostics multihop-significance multihop-ce-cache multihop-rerank-experiment multihop-recalibrate multihop-answerability beir-routing finetune ondisk charts lint clean

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
	@echo "eval         measure retrieval and calibration on labelled queries"
	@echo "bench        probe the engine's limits (evidence, batch, scale, ...)"
	@echo "bench-index-parallel  serial vs threads vs processes on the Laya encoder"
	@echo "bench-biencoder-parallel  the same for a retrieval-trained bi-encoder"
	@echo "beir         standard IR metrics on BEIR (nDCG@10, MRR@10, ...)"
	@echo "beir-ablate  sweep fusion weights and rerank depth on BEIR"
	@echo "multihop     MultiHop-RAG under the benchmark's own protocol"
	@echo "multihop-gate  answerability rejection on the null_query records"
	@echo "multihop-coverage  does a coverage objective beat per-document rank on MAP@10?"
	@echo "multihop-diagnostics  depth-stratified ceiling and second-hop retention"
	@echo "multihop-significance  permutation tests + FDR over a saved comparison grid"
	@echo "multihop-ce-cache    cache cross-encoder scores for the MultiHop subset"
	@echo "multihop-rerank-experiment  sweep second-stage selection rules over that cache"
	@echo "multihop-recalibrate  does a threshold fitted on one split transfer?"
	@echo "multihop-answerability  fit and score the feature-based answerability gate"
	@echo "beir-routing per-corpus channel selection, measured out of sample"
	@echo "finetune     in-domain domain adaptation (adapted, not zero-shot)"
	@echo "ondisk       on-disk postings and streaming dense index"
	@echo "charts       cross-system tables, error bars and the README charts"
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

# Measurement rather than pass/fail, and both need the real checkpoints. Reports are
# written under eval/ (gitignored). `make eval EVAL_ARGS="--corpus docs"` switches
# corpora, and `EVAL_ARGS="--baseline eval/planted.json"` diffs against a saved run.
eval:
	OMP_NUM_THREADS=16 $(PY) tools/eval.py $(EVAL_ARGS)

# `make bench BENCH_ARGS="--probes evidence,batch"` for a single probe.
bench:
	OMP_NUM_THREADS=16 $(PY) tools/bench.py $(BENCH_ARGS)

# `make bench-index-parallel BENCH_ARGS="--n 512"`; writes eval/index-parallel.{json,md}.
bench-index-parallel:
	$(PY) tools/bench_index_parallel.py $(BENCH_ARGS)

# `make bench-biencoder-parallel BENCH_ARGS="--model BAAI/bge-small-en-v1.5"`.
bench-biencoder-parallel:
	$(PY) tools/bench_biencoder_parallel.py $(BENCH_ARGS)

# Standard-benchmark retrieval quality. Downloads the datasets into eval/beir/data and
# needs network on the first run plus the Laya checkpoints for the dense rows.
# `make beir BEIR_ARGS="--datasets scifact --retrievers bm25"` narrows it.
beir:
	OMP_NUM_THREADS=16 $(PY) tools/beir.py $(BEIR_ARGS)

# The ablation that chose the defaults (`make beir-ablate BEIR_ABLATE_ARGS="--datasets scifact"`).
beir-ablate:
	OMP_NUM_THREADS=16 $(PY) tools/beir_ablate.py $(BEIR_ABLATE_ARGS)

# MultiHop-RAG under the benchmark's own protocol. `MULTIHOP_ARGS="--models bge-small,bge-large
# --rerank"` runs the published comparison row. Cached embeddings make the second run cheap.
multihop:
	OMP_NUM_THREADS=16 $(PY) tools/multihop_official.py $(MULTIHOP_ARGS)

# Negative rejection on the 301 null_query records (`GATE_ARGS="--null-limit 50"` to sample).
multihop-gate:
	OMP_NUM_THREADS=16 $(PY) tools/multihop_gate.py $(GATE_ARGS)

# Does a coverage objective beat per-document rank on MAP@10?
multihop-coverage:
	OMP_NUM_THREADS=16 $(PY) tools/multihop_coverage.py $(COVERAGE_ARGS)

# Depth-stratified pool ceiling and second-hop retention (RESEARCH.md §3.1).
# `make multihop-diagnostics DIAGNOSTICS_ARGS="--limit-queries 200"` for a smoke run.
multihop-diagnostics:
	OMP_NUM_THREADS=16 $(PY) tools/multihop_diagnostics.py $(DIAGNOSTICS_ARGS)

# Permutation tests + Benjamini-Hochberg FDR over a saved report's comparison grid.
# Defaults to eval/benchmarks/multihop-rag/official.json; `SIGNIFICANCE_ARGS="--report ..."`.
multihop-significance:
	$(PY) tools/eval_significance.py $(SIGNIFICANCE_ARGS)

# Cache cross-encoder scores for the MultiHop subset, then sweep second-stage rules.
# The first is the expensive model run; the second is pure NumPy over the cache.
multihop-ce-cache:
	OMP_NUM_THREADS=16 $(PY) tools/multihop_ce_cache.py $(CE_CACHE_ARGS)

multihop-rerank-experiment:
	$(PY) tools/multihop_rerank_experiment.py $(RERANK_EXPERIMENT_ARGS)

# Does an answerability threshold fitted on one split transfer to the other?
multihop-recalibrate:
	$(PY) tools/multihop_recalibrate.py $(RECALIBRATE_ARGS)

# Fit the feature gate on the gate run's recorded signals and score it out of sample.
multihop-answerability:
	$(PY) tools/answerability_train.py $(ANSWERABILITY_ARGS)

# Per-corpus channel selection, measured out of sample.
beir-routing:
	$(PY) tools/beir_routing.py $(ROUTING_ARGS)

# Domain-adapted fine-tuning; results are labelled adapted, never zero-shot.
finetune:
	OMP_NUM_THREADS=16 $(PY) tools/finetune.py $(FINETUNE_ARGS)

# On-disk postings + streaming dense index, the route to the large BEIR corpora.
ondisk:
	OMP_NUM_THREADS=16 $(PY) tools/ondisk.py $(ONDISK_ARGS)

# Cross-system tables, bootstrap error bars, and the charts embedded in the README.
# It reads the reports `make beir` and the multihop tools already wrote, so run those first.
charts:
	OMP_NUM_THREADS=16 $(PY) tools/compare.py --charts $(CHARTS_DIR)
	$(PY) tools/diagram.py --out $(DIAGRAM)

lint:
	$(PY) -m compileall -q src/ tests/
	node --check src/laya_rag/web/app.js
	@echo "ok"

clean:
	rm -rf indexes .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
