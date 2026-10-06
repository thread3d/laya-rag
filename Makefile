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
MCP_PORT ?= 8078
EVAL_ARGS ?=
BENCH_ARGS ?=
BEIR_ARGS ?=
BEIR_ABLATE_ARGS ?=
BEIR_ATOM_ARGS ?=
BEIR_COVERAGE_ARGS ?=
CONTEXT_PREFIX_ARGS ?=
BRIGHT_ARGS ?=
MULTIHOP_ARGS ?=
COVERAGE_ARGS ?=
GATE_ARGS ?=
RECALIBRATE_ARGS ?=
ROUTING_ARGS ?=
FINETUNE_ARGS ?=
DISTILL_EVAL_ARGS ?=
ONDISK_ARGS ?=
ANSWERABILITY_ARGS ?=
DIAGNOSTICS_ARGS ?=
SIGNIFICANCE_ARGS ?=
CE_CACHE_ARGS ?=
RERANK_EXPERIMENT_ARGS ?=
ATOM_ARGS ?=
LEXICAL_BENCH_ARGS ?=
ADVERSARIAL_ARGS ?=
ANSWERABILITY_FEATURES_ARGS ?=
GATE_CALIBRATION_ARGS ?=
ENGINE_BEIR_ARGS ?=
ENGINE_MULTIHOP_ARGS ?=
ENGINE_REPORT ?= eval/benchmarks/multihop-rag/engine-hybrid-400.json
ENGINE_VS_HARNESS_ARGS ?=
ENGINE_MUSIQUE_ARGS ?=
ENGINE_MUSIQUE_REPORT ?= eval/benchmarks/musique/engine-hybrid-200
ENGINE_MUSIQUE_COMPARE_ARGS ?=
ORDERING_ARGS ?=
STANCE_ARGS ?=
TRUST_ARGS ?=
ORDER_TRAIN_ARGS ?=
ORDER_EVAL_ARGS ?=
ORDER_CE_BASE ?= cross-encoder/ms-marco-MiniLM-L-6-v2
ORDER_CE_ADAPTED ?= eval/order-ce-musique
ORDER_ZERO_SHOT_REPORT ?= eval/benchmarks/musique/order-zero-shot
ORDER_ADAPTED_REPORT ?= eval/benchmarks/musique/order-adapted
ENDTOEND_ARGS ?=
COST_ARGS ?=
VERIFY_ARGS ?=
CITATION_ARGS ?=
ANNBENCH_ARGS ?=
ONDISK_SCALE_ARGS ?=
DISTILL_VERIFY_ARGS ?=
FRAMES_ARGS ?=
LEXICAL_CONTROL_ARGS ?=
ANSWERABILITY_GROUP_ARGS ?=
CHARTS_DIR ?= docs/charts
DIAGRAM ?= docs/architecture.png
JOB_STATUS_ARGS ?=
CONSISTENCY_ARGS ?=

.PHONY: help setup doctor index serve query gui mcp mcp-http test test-real test-gui test-mcp eval bench bench-index-parallel bench-biencoder-parallel beir beir-ablate beir-atom-channel beir-coverage context-prefix bright multihop multihop-gate multihop-coverage multihop-diagnostics multihop-significance multihop-ce-cache multihop-rerank-experiment multihop-atom-index multihop-recalibrate multihop-answerability answerability-features multihop-answerability-grouped beir-routing finetune distill-eval gate-calibration engine-beir engine-multihop engine-multihop-compare engine-musique engine-musique-compare multihop-ordering arguana-stance trust-report order-train order-eval-zero-shot order-eval-adapted order-compare endtoend cost-report verify-precision citation-report ann-bench ondisk-scale distill-verify frames-eval lexical-control ondisk bench-lexical adversarial-audit charts claim-check consistency-check job-status lint clean

help:
	@echo "setup        create .venv and install the engine in editable mode"
	@echo "doctor       check that the Laya checkpoint loads"
	@echo "index        (re)index CORPUS (default: $(CORPUS))"
	@echo "serve        run the HTTP service and GUI on PORT (default: $(PORT))"
	@echo "gui          serve, then open the GUI in a browser"
	@echo "mcp          run the MCP server over stdio (needs: pip install -e '.[mcp]')"
	@echo "mcp-http     run the MCP streamable-HTTP transport on MCP_PORT (default: $(MCP_PORT))"
	@echo "query Q=...  ask the CLI for evidence"
	@echo "test         offline Python suite (no checkpoints needed)"
	@echo "test-real    Python suite against real Laya weights"
	@echo "test-gui     drive the running service's GUI in headless Chrome"
	@echo "test-mcp     MCP server suite (skips without the [mcp] extra)"
	@echo "eval         measure retrieval and calibration on labelled queries"
	@echo "bench        probe the engine's limits (evidence, batch, scale, ...)"
	@echo "bench-index-parallel  serial vs threads vs processes on the Laya encoder"
	@echo "bench-biencoder-parallel  the same for a retrieval-trained bi-encoder"
	@echo "beir         standard IR metrics on BEIR (nDCG@10, MRR@10, ...)"
	@echo "beir-ablate  sweep fusion weights and rerank depth on BEIR"
	@echo "beir-atom-channel  does an atomic lexical channel help BEIR fusion?"
	@echo "beir-coverage  does the coverage selection help BEIR, or only multi-fact corpora?"
	@echo "context-prefix  lexical effect of the structural context prefix"
	@echo "bright       BRIGHT reasoning-intensive retrieval, retrieval only"
	@echo "multihop     MultiHop-RAG under the benchmark's own protocol"
	@echo "multihop-gate  answerability rejection on the null_query records"
	@echo "multihop-coverage  does a coverage objective beat per-document rank on MAP@10?"
	@echo "multihop-diagnostics  depth-stratified ceiling and second-hop retention"
	@echo "multihop-significance  permutation tests + FDR over a saved comparison grid"
	@echo "multihop-ce-cache    cache cross-encoder scores for the MultiHop subset"
	@echo "multihop-rerank-experiment  sweep second-stage selection rules over that cache"
	@echo "multihop-atom-index  atomic-unit lexical retrieval with parent-window return"
	@echo "multihop-recalibrate  does a threshold fitted on one split transfer?"
	@echo "multihop-answerability  fit and score the feature-based answerability gate"
	@echo "beir-routing per-corpus channel selection, measured out of sample"
	@echo "finetune     in-domain domain adaptation (adapted, not zero-shot)"
	@echo "distill-eval  distilled student vs zero-shot vs teacher ensemble"
	@echo "gate-calibration  learned threshold / temperature vs fixed 0.5 and raw"
	@echo "engine-beir    the engine itself on BEIR (not the tools replica)"
	@echo "engine-multihop  the engine itself on MultiHop-RAG under its own chunking (resumable)"
	@echo "engine-multihop-compare  the engine vs the 256-token harness, paired over queries"
	@echo "engine-musique  the engine itself on MuSiQue under its own chunking (resumable)"
	@echo "engine-musique-compare  the engine vs the MuSiQue harness, paired MAP@10"
	@echo "multihop-ordering  late interaction as an ordering head over the cached pool"
	@echo "arguana-stance  NLI contradiction-entailment reranking on ArguAna"
	@echo "trust-report  one deterministic report: poisoning, verification, abstention"
	@echo "order-train  adapt a cross-encoder on MuSiQue's own relevance conventions (2.2)"
	@echo "order-eval-zero-shot / order-eval-adapted / order-compare  the adapted-vs-zero-shot test"
	@echo "endtoend     engine + local generator on MultiHop-RAG (labelled; ~36s/query)"
	@echo "cost-report  latency, throughput, peak RSS and index bytes per query"
	@echo "verify-precision  precision and recall of the advisory verifier checks (--check gates)"
	@echo "citation-report  citation mechanics over the pilot's composed answers (engine-side)"
	@echo "ann-bench    approximate vs exhaustive dense search, including IVF-PQ rescore"
	@echo "ondisk-scale peak build RSS against corpus size and vocabulary (Phase 4.1 evidence)"
	@echo "distill-verify  is the exported distilled encoder faithful to its checkpoint?"
	@echo "frames-eval  FRAMES retrieval: recall@k and MRR against the linked articles"
	@echo "lexical-control  gaps against published BM25, and the counts the prose is pinned to"
	@echo "ondisk       on-disk postings and streaming dense index"
	@echo "bench-lexical  BM25 vs eager BM25: parity, index size, query latency"
	@echo "adversarial-audit  poison/trust/verification regression checks"
	@echo "answerability-features  raw gate features vs engineered features"
	@echo "multihop-answerability-grouped  leave-one-question-type-out transfer"
	@echo "charts       cross-system tables, error bars and the README charts"
	@echo "claim-check  do the numbers quoted in prose resolve against the reports?"
	@echo "job-status   every resumable job's manifest: running, complete, or half-finished"
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

# The MCP server is a separate process and an optional extra; stdio is the default
# transport, so this blocks while a host (Claude Desktop, Cursor, ...) is connected.
mcp:
	$(PY) -m laya_rag.mcp_server

# Streamable HTTP on MCP_PORT. Beyond loopback the server requires a bearer token
# (LAYA_RAG_API_KEYS or LAYA_RAG_API_KEY) before it will start.
mcp-http:
	$(PY) -m laya_rag.mcp_server --transport streamable-http --port $(MCP_PORT)

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

# Runs only when the [mcp] extra is installed; the module skips itself otherwise.
test-mcp:
	$(PY) -m pytest tests/test_mcp_server.py -q

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

# Does an atomic lexical channel help BEIR fusion? (uses cached document embeddings)
beir-atom-channel:
	OMP_NUM_THREADS=16 $(PY) tools/beir_atom_channel.py $(BEIR_ATOM_ARGS)

# Does the coverage selection help BEIR, or only multi-fact corpora?
beir-coverage:
	OMP_NUM_THREADS=16 $(PY) tools/beir_coverage.py $(BEIR_COVERAGE_ARGS)

# Lexical effect of the structural context prefix (CONTEXT_PREFIX) on a local corpus.
context-prefix:
	$(PY) tools/context_prefix_eval.py $(CONTEXT_PREFIX_ARGS)

# BRIGHT reasoning-intensive retrieval (retrieval only; downloads on first run).
bright:
	OMP_NUM_THREADS=16 $(PY) tools/bright.py $(BRIGHT_ARGS)

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

# Bridge-conditioned Laya scores over the cached pool: relevance asked as s(q, b, c)
# instead of s(q, c), with a shuffled-bridge control arm (RESEARCH.md 08 section 2.1).
# Expensive model run once; then pass the files to multihop-rerank-experiment --tri-cache.
multihop-tri-cache:
	OMP_NUM_THREADS=16 $(PY) tools/multihop_tri_cache.py $(TRI_ARGS)

# Is a late-interaction (MaxSim) channel worth building? Token states come from the
# shipped checkpoint, so this answers DISCo's blocking dependency without a download
# (RESEARCH.md 08 section 3.1). Reports MAP@10 with a paired interval and the index cost.
laya-maxsim-probe:
	OMP_NUM_THREADS=16 $(PY) tools/laya_maxsim_probe.py $(MAXSIM_ARGS)

# Does the true bridge discriminate gold facts better than the shuffled control? AUC, which
# is invariant to the ordering weakness that hides the signal in a ranking metric.
multihop-tri-discriminate:
	$(PY) tools/multihop_tri_discriminate.py $(TRI_DISCRIMINATE_ARGS)

# Does duplication in the cached pool cost top-10 slots? Pure NumPy over the CE cache.
multihop-pool-dedup:
	$(PY) tools/multihop_pool_dedup.py $(POOL_DEDUP_ARGS)

# Does a *fitted* combination of the cached channels beat the best hand-built rule? Half the
# queries train, half are scored; supervised on benchmark labels, so see the caveat in the report.
multihop-ltr:
	$(PY) tools/multihop_ltr.py $(LTR_ARGS)

# Signal or algorithm? The oracle restricted to the cross-encoder's own top-N separates what a
# better *scorer* could reach from what a better *selector* could reach on the list it already has.
multihop-ceiling-decomp:
	$(PY) tools/multihop_ceiling_decomp.py $(CEILING_DECOMP_ARGS)

# Per-question-type routing of the selection rule under a two-direction nested protocol.
multihop-routing:
	$(PY) tools/multihop_routing.py $(ROUTING_ARGS)

# Atomic-unit lexical retrieval with parent-window return (RESEARCH.md Tier 1.5).
multihop-atom-index:
	OMP_NUM_THREADS=16 $(PY) tools/multihop_atom_index.py $(ATOM_ARGS)

# Does an answerability threshold fitted on one split transfer to the other?
multihop-recalibrate:
	$(PY) tools/multihop_recalibrate.py $(RECALIBRATE_ARGS)

# Fit the feature gate on the gate run's recorded signals and score it out of sample.
multihop-answerability:
	$(PY) tools/answerability_train.py $(ANSWERABILITY_ARGS)

# Does feature engineering improve the gate? Offline comparison on the same splits.
answerability-features:
	$(PY) tools/answerability_features.py $(ANSWERABILITY_FEATURES_ARGS)

# The engine itself on a BEIR dataset, not the tools/beir.py replica.
engine-beir:
	$(PY) tools/engine_beir.py $(ENGINE_BEIR_ARGS)

# The engine itself on MultiHop-RAG under its own corpus-aware chunking (the stated largest
# evidence gap). Resumable: re-running the identical command continues a killed run.
# `ENGINE_MULTIHOP_ARGS="--limit 25 --embed-model BAAI/bge-small-en-v1.5"` is the smoke.
engine-multihop:
	OMP_NUM_THREADS=16 $(PY) tools/engine_multihop.py $(ENGINE_MULTIHOP_ARGS)

# Pair the engine report against a harness arm on the same subset. Requires both to have been
# run with the same --limit/--sample-seed; the tool refuses rather than compare different queries.
engine-multihop-compare:
	$(PY) tools/engine_vs_harness.py --engine $(ENGINE_REPORT) $(ENGINE_VS_HARNESS_ARGS)

# The engine itself on MuSiQue under its own chunking. The corpus is always the full split;
# `--limit` narrows only the queries, so a subset stays comparable to the harness's rows.
engine-musique:
	OMP_NUM_THREADS=16 $(PY) tools/engine_musique.py $(ENGINE_MUSIQUE_ARGS)

# Pair the finished MuSiQue report's MAP@10 against the harness's per-query rows.
engine-musique-compare:
	$(PY) tools/engine_musique.py --compare --out $(ENGINE_MUSIQUE_REPORT) $(ENGINE_MUSIQUE_COMPARE_ARGS)

# ROADMAP 2.1: a ColBERT MaxSim ordering head over the cached MultiHop pool, scored against the
# fused, cross-encoder and coverage+quota arms from the same pool. Resumable per query.
multihop-ordering:
	OMP_NUM_THREADS=16 $(PY) tools/multihop_ordering.py $(ORDERING_ARGS)

# ROADMAP 2.3: relational (stance) reranking for ArguAna, where a pointwise ranker collapses.
# Both arms reorder the same dense pool, so a delta is ordering alone. Resumable per query.
arguana-stance:
	OMP_NUM_THREADS=16 $(PY) tools/arguana_stance.py $(STANCE_ARGS)

# Phase 5.1: the deterministic trust surfaces in one report. `TRUST_ARGS="--check"` gates a change.
trust-report:
	OMP_NUM_THREADS=16 $(PY) tools/trust_report.py $(TRUST_ARGS)

# ROADMAP 2.2: mine pairs from MuSiQue's train split and adapt a cross-encoder on them.
order-train:
	OMP_NUM_THREADS=16 $(PY) tools/order_train.py $(ORDER_TRAIN_ARGS)

# The same architecture zero-shot and adapted, over the same MuSiQue pool, then paired.
order-eval-zero-shot:
	OMP_NUM_THREADS=16 $(PY) tools/order_eval.py --cross-encoder $(ORDER_CE_BASE) \
		--out $(ORDER_ZERO_SHOT_REPORT) $(ORDER_EVAL_ARGS)

order-eval-adapted:
	OMP_NUM_THREADS=16 $(PY) tools/order_eval.py --cross-encoder $(ORDER_CE_ADAPTED) \
		--out $(ORDER_ADAPTED_REPORT) $(ORDER_EVAL_ARGS)

order-compare:
	$(PY) tools/order_eval.py --compare-against $(ORDER_ZERO_SHOT_REPORT).json \
		--out $(ORDER_ADAPTED_REPORT)

# ROADMAP 3.2/3.3: the engine at shipped defaults plus a labelled local generator. Reuses the
# Phase 1.2 index, so the retrieval half is the system those measurements describe.
endtoend:
	OMP_NUM_THREADS=16 $(PY) tools/endtoend_multihop.py $(ENDTOEND_ARGS)

# Phase 3.4: the cost axis, against an existing index so nothing is rebuilt.
cost-report:
	OMP_NUM_THREADS=16 $(PY) tools/cost_report.py $(COST_ARGS)

# Phase 5.3: does the advisory verifier meet its precision bar? Constructed labelled set.
verify-precision:
	$(PY) tools/verify_precision.py $(VERIFY_ARGS)

# Phase 3.3: citation mechanics over composed answers -- the engine's half of an end-to-end row.
citation-report:
	OMP_NUM_THREADS=16 $(PY) tools/citation_report.py $(CITATION_ARGS)

# Phase 4.5: approximate dense search against the exhaustive scan, at scale.
ann-bench:
	OMP_NUM_THREADS=16 $(PY) tools/ann_bench.py $(ANNBENCH_ARGS)

# Phase 4.1 evidence: peak build RSS against the two axes, one subprocess per point.
ondisk-scale:
	OMP_NUM_THREADS=16 $(PY) tools/ondisk_scale.py $(ONDISK_SCALE_ARGS)

# Phase 4.3: is the exported distilled encoder faithful to the checkpoint it came from?
distill-verify:
	$(PY) tools/distill_verify.py $(DISTILL_VERIFY_ARGS)

# Phase 3.1: FRAMES retrieval, document-level gold from the dataset's Wikipedia links.
frames-eval:
	OMP_NUM_THREADS=16 $(PY) tools/frames_eval.py $(FRAMES_ARGS)

# The lexical-control summary, generated so the prose describing it can be pinned by claim-check.
lexical-control:
	$(PY) tools/lexical_control.py $(LEXICAL_CONTROL_ARGS)

# Does a learned gate threshold or a fitted temperature beat the fixed 0.5 and raw score?
gate-calibration:
	$(PY) tools/gate_calibration.py $(GATE_CALIBRATION_ARGS)

# Leave-one-question-type-out answerability transfer: the strongest split available here.
multihop-answerability-grouped:
	$(PY) tools/answerability_train.py \
		--features eval/benchmarks/multihop-rag/gate-features.json \
		--group-by question_type \
		--out eval/benchmarks/multihop-rag/answerability-grouped.json \
		$(ANSWERABILITY_GROUP_ARGS)

# Per-corpus channel selection, measured out of sample.
beir-routing:
	$(PY) tools/beir_routing.py $(ROUTING_ARGS)

# Domain-adapted fine-tuning; results are labelled adapted, never zero-shot.
finetune:
	OMP_NUM_THREADS=16 $(PY) tools/finetune.py $(FINETUNE_ARGS)

# Evaluate a distilled student against its zero-shot self and the teacher ensemble.
distill-eval:
	$(PY) tools/distill_eval.py $(DISTILL_EVAL_ARGS)

# On-disk postings + streaming dense index, the route to the large BEIR corpora.
ondisk:
	OMP_NUM_THREADS=16 $(PY) tools/ondisk.py $(ONDISK_ARGS)

# Ranking parity, index size and query latency for the two BM25 backends.
bench-lexical:
	OMP_NUM_THREADS=16 $(PY) tools/bench_lexical.py $(LEXICAL_BENCH_ARGS)

# Deterministic adversarial retrieval regression suite; `--check` gates a change.
adversarial-audit:
	$(PY) tools/adversarial_audit.py $(ADVERSARIAL_ARGS)

# Cross-system tables, bootstrap error bars, and the charts embedded in the README.
# It reads the reports `make beir` and the multihop tools already wrote, so run those first.
charts:
	OMP_NUM_THREADS=16 $(PY) tools/compare.py --charts $(CHARTS_DIR)
	$(PY) tools/compare_more.py --charts $(CHARTS_DIR)
	$(PY) tools/diagram.py --out $(DIAGRAM)

# Do the numbers quoted in prose still resolve against the reports that produced them?
# Skips (exit 0) when the report is absent: eval/ is gitignored and regenerable, so a fresh
# checkout has nothing to check against until `make multihop-diagnostics` has run.
claim-check:
	$(PY) tools/claim_check.py

# Do two documents disagree about the same quantity? `claim-check` pins prose to a report;
# this pins the documents to each other, holds the BEIR coverage count computed from disk
# rather than hand-typed, and refuses to read a superseded report as current.
consistency-check:
	$(PY) tools/consistency_check.py $(CONSISTENCY_ARGS)

# What resumable work is in flight? Reads the `<stem>.manifest.json` sidecars jobkit writes.
job-status:
	$(PY) tools/job_status.py $(JOB_STATUS_ARGS)

lint:
	$(PY) -m compileall -q src/ tests/
	node --check src/laya_rag/web/app.js
	@echo "ok"

clean:
	rm -rf indexes .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
