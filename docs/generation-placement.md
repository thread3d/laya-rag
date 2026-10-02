# Generation and placement: preparation, not a submission

This document prepares the generation/placement workstream. It records, per external
target, **what the target scores, the exact interface or artifact it requires, what in
this repo already maps to it, what is missing, and a runnable command sketch** — so the
runs can be executed later without re-deriving any of this.

It is a *preparation* document. Nothing here was submitted, no benchmark was run, and no
model was downloaded. Where a target needs network, NIST access, or a generator, that is
marked explicitly. The scaffold it describes is `tools/generation_eval.py`; its offline
tests are `tests/test_generation_eval.py`.

Provenance for the verdicts: [`BENCHMARKS.md`](../BENCHMARKS.md) (§["The capability
boundary"](../BENCHMARKS.md) lines 11–33, §["End-to-end RAG"](../BENCHMARKS.md) lines
185–196, §["Faithfulness and hallucination"](../BENCHMARKS.md) lines 223–228). This
document expands those verdicts into the concrete preparation the runs need.

---

## 1. The capability boundary

The engine is a **retrieval and evidence system**, not a generator. That single fact
decides most of the placement questions, so it is worth writing down as three classes of
number rather than one.

| what a benchmark scores | whose number is it? | what produces it | can it be submitted today? |
| --- | --- | --- | --- |
| Retrieval quality (nDCG@10, Hits@k, MAP) | **the engine's** | chunking + hybrid recall + Laya's relevance decision | yes — run it locally (`tools/beir.py`) |
| Answerability / abstention | **the engine's** | Laya's calibrated answerability gate | yes locally; does not transfer off-domain ([`BENCHMARKS.md`](../BENCHMARKS.md) lines 134–178) |
| Citation support (does the prose cite real spans) | **the engine's** | `src/laya_rag/verify.py`, deterministic | the mechanics run today; the TREC support *assessment* needs NIST access |
| Answer correctness / faithfulness / answer relevance | **the attached LLM's** | the compose endpoint | only with a generator, and the number is the model's |
| Hallucination detection | **neither** — the engine is not a detector | n/a | no honest leaderboard row |

Two things follow, and the scaffold enforces the second in code.

**1. Composition is opt-in and clearly labelled.** `src/laya_rag/compose.py` is the only
module that can invent text. It resolves an OpenAI-compatible or Ollama endpoint from
`LAYA_RAG_COMPOSE_MODEL` / `LAYA_RAG_COMPOSE_BASE_URL` (`compose.py:115`
`resolve_target`), builds a numbered-evidence prompt that demands a citation on every
factual sentence (`compose.py:52` `SYSTEM_PROMPT`, `compose.py:146` `build_messages`),
and streams prose (`compose.py:214` `Composer.stream`). The API route
`POST /v1/compose` (`src/laya_rag/api.py:366`) is the only route that returns generated
text.

**2. Every generation-side number is labelled `engine + <model>`.** A faithfulness,
answer-accuracy or answer-relevance number is a statement about the model that wrote the
answer, not about the retrieval and evidence engine around it. The label is mandatory in
the scaffold: `generation_side_metric(generator=..., metric=..., value=...)` returns a
record that always carries `generator` (and the `engine + <model>` label), and raises
`ValueError` on an empty generator. A number that loses its model is not a weaker result;
it is a different, unearned claim.

The citation check is the exception that proves the boundary. `verify_citations`
(`verify.py:138`) decides only whether each claim cites in-range evidence and whether each
quoted span really appears in the passages; its module docstring says plainly that it
makes no claim about truth (`verify.py:1–17`), and its `clean` flag means "cites in-range
evidence and quotes were found", not "correct" (`verify.py:228`). That is why it is the
engine's number, and why `citation_report()` in the scaffold returns it with
`generator=None` and `truth_claim=False`.

---

## 2. The four targets

Each subsection follows the same shape: **what it scores → interface/artifact required →
what is already in the repo → what is missing → command sketch**. `blocked_by` values
below are the same strings the scaffold carries in `TARGETS`.

### 2.1 MTEB retrieval — submit a *model*, not the engine

**What it scores.** Retrieval quality (nDCG@10) of a submitted encoder across the MTEB
retrieval tasks. The leaderboard ranks **models**, not engines around models. Its
retrieval subset is largely BEIR's, which is why the local BEIR numbers are evidence of
quality but not a leaderboard position ([`BENCHMARKS.md`](../BENCHMARKS.md) line 209).

**Interface/artifact required.**
- A single, loadable `SentenceTransformer`-compatible encoder on the Hub (weights +
  pooling config + model card).
- A fixed embedding dimension and pooling/normalisation policy, so the submitted model
  reproduces the measured scores.
- The `mteb` submission harness and its task data.
- Network access to fetch tasks and push the model.

**Already in the repo.**
- `tools/beir.py` — the same retrieval metrics MTEB's retrieval tasks report (nDCG@10,
  nDCG@100, MRR@10, Recall@100, MAP@100) over official qrels, with `bm25` / `dense` /
  `hybrid` / `biencoder` rows (`tools/beir.py:1–24`, `tools/beir.py:120`). It also writes
  the document-embedding caches the projection head consumes.
- `tools/projection_head.py` — distils a **linear projection** onto Laya's frozen document
  embeddings by ridge regression to a retrieval teacher (`BAAI/bge-small-en-v1.5` by
  default), then scores `raw` / `projected` / `teacher_docs_only` three ways. The map is
  fit and saved as a plain JSON affine map (`LinearProjection.save`/`load`), and the
  module stays pure-NumPy until `main()` loads the Laya **query** encoder, so it imports
  and unit-tests offline with no weights (`tools/projection_head.py:1–32`,
  `evaluate_projected` at line 344, `main` at line 504).
- The three-model dense ensemble and its measured BEIR numbers
  ([`BENCHMARKS.md`](../BENCHMARKS.md) lines 37–68).

**Missing pieces.**
- **A packaged single encoder.** [`BENCHMARKS.md`](../BENCHMARKS.md) lines 274–276 list
  "a distillation of the three-model ensemble back into one encoder" as open work. The
  projection head produces an in-repo affine map, not a loadable Hub model, so MTEB still
  wants an artifact the repo does not package.
- **A serving wrapper that turns the projection into a `SentenceTransformer`-compatible
  encoder** (weights + pooling config + model card). The JSON map has no tokenizer,
  pooling layer, or Hub entry.
- A model card and Hub repository (network).

**Command sketch** (local distillation exists; packaging and submission do not):

```bash
# local — retrieval quality over BEIR/MTEB-retrieval datasets, and the caches the
# projection head reads (Laya doc embeddings + teacher doc embeddings)
.venv/bin/python tools/beir.py --datasets scidocs --retrievers dense
.venv/bin/python tools/beir.py --datasets scidocs --retrievers biencoder \
    --biencoder-model BAAI/bge-small-en-v1.5

# local — fit and score the projection (loads the Laya query encoder; no network)
.venv/bin/python tools/projection_head.py --dataset scidocs \
    --teacher BAAI/bge-small-en-v1.5 --out eval/projection-scidocs

# planned, does NOT exist yet: wrap the JSON projection in an encoder, then run mteb
# .venv/bin/python tools/package_encoder.py --projection eval/projection-scidocs.json \
#     --out eval/distilled-encoder/              # [model]
# .venv/bin/python -m mteb run -m eval/distilled-encoder -t <retrieval tasks> \
#     --output eval/mteb/                        # [network] [model]
```

**`blocked_by`.** model-submission only: the projected encoder is an in-repo JSON affine
map, not a packaged SentenceTransformer/Hub model, so there is no MTEB model submission
yet.

---

### 2.2 KILT — entity linker + retriever + seq2seq generator

**What it scores.** A single leaderboard score for a full stack: entity linking +
retrieval + seq2seq generation. Because generation is part of the score, the downstream
number belongs to the generator, not to the retriever
([`BENCHMARKS.md`](../BENCHMARKS.md) line 196).

**Interface/artifact required.**
- An **entity linker** (BLINK or equivalent) over the KILT Wikipedia snapshot, emitting
  the KILT entity ids the benchmark expects.
- A **retriever** over that snapshot.
- A **seq2seq generator**, i.e. exactly the compose endpoint the repo already has.
- The KILT download, its task formats, and the leaderboard submission format.

**Already in the repo.**
- The generator half: `src/laya_rag/compose.py` is an OpenAI-compatible/Ollama chat
  client (`ComposeTarget`, `resolve_target` at `compose.py:66`/`compose.py:115`;
  `Composer.stream` at `compose.py:214`), reachable over `POST /v1/compose`
  (`api.py:366`).
- The retriever half: `RagEngine.query` (`engine.py:464`) and `adjudicate_claims`
  (`engine.py:679`).
- Numbered, document-grouped evidence so `[n]` is stable across compose, the API and the
  citation check (`evidence.py:45`, `verify.py:138`).

**Missing pieces.**
- An entity linker. Nothing in the repo links mentions to KILT entity ids.
- A KILT-format candidate set and submission writer.
- The KILT Wikipedia snapshot and network access to it.
- A model card describing the compose model, since the number is `engine + <model>`.

**Command sketch** (planned adapters only):

```bash
# planned, does NOT exist yet
# .venv/bin/python tools/kilt.py --tasks <kilt tasks> --linker blinker \
#     --compose-model "$LAYA_RAG_COMPOSE_MODEL"     # [network] [model] [full stack]
```

**`blocked_by`.** no entity linker in the repo, and the leaderboard expects a full stack;
the score would be the generator's, not the engine's.

---

### 2.3 TREC RAG track — AutoNuggetizer + citation support

**What it scores.** Nugget **recall/precision** via the official AutoNuggetizer assessor,
plus **citation support** assessment ([`BENCHMARKS.md`](../BENCHMARKS.md) line 189).

**Interface/artifact required.**
- Access to the TREC RAG **corpus** (NIST), and **gold nuggets** produced by NIST
  assessors.
- The **official AutoNuggetizer** scorer.
- A retrieval run over the corpus and a generator that composes cited answers (the
  support half is about the prose).
- The track's submission format.

**Already in the repo.**
- `src/laya_rag/verify.py` is the **structural match to the support half**: given an
  answer and the passages, it checks that each claim cites in-range evidence and that
  quoted spans are verbatim (`verify_citations` at `verify.py:138`; `extract_citations`
  at `verify.py:79`; `split_claims` at `verify.py:107`). The API exposes it at
  `POST /v1/verify` (`api.py:394`), while `POST /v1/adjudicate` (`api.py:415`) asks Laya
  whether the *corpus* supports each claim via `adjudicate_claims` (`engine.py:679`,
  return at `engine.py:930`).
- The generator half: `compose.py` (see §2.2).

**Missing pieces.**
- **NIST access** to the corpus and gold nuggets — the hard blocker.
- The official assessor (AutoNuggetizer), which is the scoring authority; `verify.py`
  cannot stand in for it.
- A nugget-recall scorer and a submission writer.

**Command sketch** (nothing runnable without access):

```bash
# blocked — requires NIST corpus + gold nuggets + the official AutoNuggetizer
# .venv/bin/python tools/trec_rag.py --retrieval eval/trec/run.jsonl \
#     --answers eval/trec/answers.jsonl --autonuggetizer <path> \
#     --out eval/trec/score.json                      # [access] [network] [model]
```

**`blocked_by`.** NIST access to the corpus and gold nuggets is required, and scoring
needs the official assessor; there is nothing to submit it to without that access.

---

### 2.4 Vectara HHEM — not a system ranking

**What it scores.** HHEM ranks **hallucination-detection models** by agreement with human
labels. It does not score a RAG system, and the engine is not a hallucination detector
([`BENCHMARKS.md`](../BENCHMARKS.md) line 228).

**Interface/artifact required.**
- A hallucination-detection **model** to submit.
- The HHEM data and evaluation harness.
- Claim/premise pairs to score.

**Already in the repo.** Nothing that is a hallucination detector. `verify.py` checks
citation *mechanics* — do cited spans exist — which is a different and narrower question;
the module says so itself (`verify.py:1–17`), and
[`BENCHMARKS.md`](../BENCHMARKS.md) line 227 makes the same distinction for RAGTruth.

**Missing pieces.** A detection model, which the workstream is not building. The
**honest use** of what exists is an **optional faithfulness filter over composed prose**:
run the deterministic citation check on a composed answer and surface uncited claims or
fabricated quotes. That is a product feature, not an HHEM leaderboard row.

**Command sketch** (a filter, explicitly not a submission):

```bash
# local, offline, no model — the honest use of verify.py over composed prose
.venv/bin/python - <<'PY'
from laya_rag.verify import verify_citations
answer = "Refunds are processed within five business days [1]."
passages = ["Refunds are processed within five business days to the original payment."]
print(verify_citations(answer, passages))   # mechanics only; makes no truth claim
PY
```

**`blocked_by`.** nothing in the engine is a hallucination detector, so there is no
honest leaderboard row; the defensible use is an optional faithfulness filter over
composed prose.

---

## 3. What can be run today with no submission

Two families of work are runnable now: the one generation-adjacent metric that is the
engine's, and the retrieval-only benchmarks that never touch a generator.

### 3.1 RAGAS / ARES-style context relevance — the engine's half

[`BENCHMARKS.md`](../BENCHMARKS.md) line 235 records the split precisely: of the RAGAS
metrics, **faithfulness** and **answer relevance** measure the generator against the
retrieved evidence, while **context relevance** — are the retrieved contexts relevant to
the query — is the only part that is this system's.

- **The engine's part (runnable).** Context relevance is a retrieval/relevance judgement.
  The engine already produces a calibrated relevance score per passage through Laya's
  reranker (`RagEngine.query`, `engine.py:464`) and can re-adjudicate prose claims against
  the whole corpus (`adjudicate_claims`, `engine.py:679`). Aggregating the per-passage
  relevance of a returned evidence set into a context-relevance number needs no generator
  and no submission.
- **The generator's part (must be labelled).** Faithfulness and answer relevance require
  composed answers, so they run only with a model and must be reported as
  `engine + <model>` via `generation_side_metric`. RAGAS/ARES are Python packages and
  would be network-installed (`[network]`).
- **Cheap, deterministic proxy available now.** `verify_citations` gives a model-free
  citation-mechanics signal over any composed answer (`verify.py:138`); it is *not* a
  faithfulness metric and must not be labelled as one.

Command sketch (context relevance is planned; the citation proxy runs today):

```bash
# today, offline, no model
.venv/bin/python -c "import tools.generation_eval as g; \
print(g.citation_report('Refunds [1].', [('p.md','Refunds are processed in five days.')]))"

# planned, needs generator and the ragas package
# .venv/bin/python tools/generation_eval.py --context-relevance --corpus <indexed corpus>
#     # [model for the generation half] [network to install ragas]
```

> The scaffold's `main()` accepts **only** `--list` and `--check`. The `--context-relevance`
> hook above is a named future entry point, not implemented in this scaffold.

### 3.2 Retrieval-only benchmarks — no generator needed

These score retrieval, which is unambiguously the engine's number. None needs a generator
or a leaderboard submission; all need their data (network) and a thin repo-side adapter.

| benchmark | what it scores | what maps in-repo | missing adapter | marker |
| --- | --- | --- | --- | --- |
| **BRIGHT** | reasoning-intensive retrieval, 12 datasets | `tools/beir.py` metrics + engine hybrid recall | dataset loader / query format; some collections are large | `[network]` |
| **LoTTE** | search/forum long-tail retrieval | same retrieval channels as BEIR | search/forum split loader; several collections large | `[network]` |
| **LegalBench-RAG** | retrieval precision/recall over legal text | exactly the engine's shape; `RagEngine.query` + `tools/beir.py` metrics | corpus ingestion and gold-span scoring | `[network]` |
| **FRAMES** | multi-hop factual QA over Wikipedia; exact-match answer accuracy | retrieval over the shipped Wikipedia links; 824 questions, ~0.5 MB | Wikipedia corpus build; **answer scoring needs a judge model** | `[network]`, `[model]` for the answer half |
| **RGB** | noise robustness, negative rejection, information integration, counterfactual | the answerability gate addresses negative rejection directly; `tools/multihop_gate.py` is the closest existing rejection harness | English-subset loader and per-task scoring | `[network]` |

Source verdicts: [`BENCHMARKS.md`](../BENCHMARKS.md) lines 191–192 (RGB, FRAMES) and
lines 210–220 (BRIGHT, LoTTE, LegalBench-RAG). Note the honest qualifier on FRAMES: its
*retrieval* half is the engine's, its *answer accuracy* half is a judge model's.

Command sketch (planned adapters; the existing BEIR harness is the template):

```bash
# today — the retrieval harness the adapters would reuse
.venv/bin/python tools/beir.py --datasets scifact --retrievers bm25,dense,hybrid

# planned, does NOT exist yet
# .venv/bin/python tools/bright.py       --out eval/bright/       # [network]
# .venv/bin/python tools/lotte.py        --out eval/lotte/        # [network]
# .venv/bin/python tools/legalbench_rag.py --out eval/legal/      # [network]
# .venv/bin/python tools/frames.py       --out eval/frames/       # [network] [model: judge]
# .venv/bin/python tools/rgb.py          --out eval/rgb/          # [network]
```

---

## 4. Run order

The order is cheapest-and-most-certain first. Every marker is explicit: `[offline]`,
`[network]`, `[model]`, `[access]` (NIST), `[blocked]`.

### Step 0 — today, offline: validate the preparation scaffold

```bash
make generation-check     # planned hook; equivalent to:
.venv/bin/python tools/generation_eval.py --check
make generation-list      # planned hook; equivalent to:
.venv/bin/python tools/generation_eval.py --list
```

Existing verified commands (no hook needed):

```bash
.venv/bin/python -m pytest tests/test_generation_eval.py -q   # [offline] 13 passed
.venv/bin/python -m pytest tests/ -q -m "not real_weights"    # make test, [offline]
```

### Step 1 — retrieval-only benchmarks (the engine's own numbers)

`[network]` for data; `[model]` only where a judge is named.

1. `make beir` / `.venv/bin/python tools/beir.py` — already implemented.
2. Add adapters `tools/bright.py`, `tools/lotte.py`, `tools/legalbench_rag.py`,
   `tools/rgb.py` (planned). Reuse `tools/beir.py`'s metric path.
3. Add `tools/frames.py` (planned) for the retrieval half; the answer-accuracy half is a
   separate `[model]` step and its number is `engine + <judge>`.

### Step 2 — RAGAS/ARES context relevance

`[offline]` for context relevance using the engine's own relevance scores; `[network]`
to install `ragas`; `[model]` for faithfulness/answer relevance, whose numbers are
`engine + <model>`.

```bash
# planned hook; the scaffold does not implement it
# make ragas-context RAGAS_ARGS="--corpus <indexed corpus>"
```

### Step 3 — KILT

`[network]` for the KILT snapshot; `[model]` for the compose generator; a new
`[entity-linker]` dependency (BLINK or equivalent). The score is the generator's.

```bash
# planned hook; requires the entity linker and KILT download
# make kilt KILT_ARGS="--tasks <tasks>"
```

### Step 4 — MTEB submission

`[model]` distillation first, then `[network]` for the Hub and the `mteb` harness.

1. Build the caches the projection head needs, then fit the map with the existing
   `tools/projection_head.py` (see the §2.1 sketch).
2. Score locally with `tools/beir.py` (raw vs projected) before packaging.
3. `[model]` wrap the JSON projection in a `SentenceTransformer`-compatible encoder.
4. `[network]` run `mteb` and push the model + card.

### Step 5 — TREC RAG track

`[access]` — **blocked-by-access** until NIST grants the corpus and gold nuggets;
`[blocked]` for the official AutoNuggetizer without that access. `verify.py` is ready as
the structural support-half check the moment the answers and passages exist.

### Step 6 — HHEM

`[blocked]` as a leaderboard row. The runnable, honest substitute is the offline
faithfulness **filter** over composed prose (§2.4), which needs no submission.

### Proposed Makefile hooks (documented, not applied)

These targets do **not** exist in `Makefile` today. Adding them is part of the
implementation this preparation unblocks; `Makefile:27` is the `.PHONY` line they would
join, beside the existing `test:88`, `eval:102`, `beir:120`, `multihop:129` and
`charts:162` targets.

```make
# --- generation/placement hooks (proposed; not yet in Makefile) ---
generation-list:
	$(PY) tools/generation_eval.py --list

generation-check:
	$(PY) tools/generation_eval.py --check

# [network] retrieval-only adapters
bright:
	$(PY) tools/bright.py $(BRIGHT_ARGS)

lotte:
	$(PY) tools/lotte.py $(LOTTE_ARGS)

legalbench-rag:
	$(PY) tools/legalbench_rag.py $(LEGALBENCH_ARGS)

frames:
	$(PY) tools/frames.py $(FRAMES_ARGS)

rgb:
	$(PY) tools/rgb.py $(RGB_ARGS)

# [model] distillation -> MTEB submission
projection-head:
	$(PY) tools/projection_head.py $(PROJECTION_ARGS)

# [model] KILT full stack
kilt:
	$(PY) tools/kilt.py $(KILT_ARGS)

# [access] TREC RAG (blocked without NIST access)
trec-rag:
	$(PY) tools/trec_rag.py $(TREC_ARGS)
```

Add the new variables (`BRIGHT_ARGS ?=`, `LOTTE_ARGS ?=`, …) and extend the `.PHONY`
line at `Makefile:27` in the same change.

---

## 5. Scaffold summary

`tools/generation_eval.py` records the above in importable form and enforces the
attribution rule:

- `GenerationResult` — query, answer, generator, evidence, `citations_ok`,
  `support`, `notes`. `citations_ok`/`support` are `None` until measured;
  `None` means *not measured*, never *false* or *zero*. `.label` renders
  `engine + <model>`.
- `BenchmarkTarget` + `TARGETS` — the four targets with their `requires`, `adapter`,
  `blocked_by` and notes. `describe_targets()` returns them as JSON-serialisable dicts.
- `generation_side_metric(*, generator, metric, value)` — cannot emit a number without
  its model; raises `ValueError` on an empty generator.
- `citation_report(answer, passages)` — groups `(path, text)` passages by document exactly
  as `build_messages` does, calls `verify_citations`, and adds `generator=None`,
  `truth_claim=False`, `mechanics_only=True`, `scored_by="engine"`.
- `main(argv)` — `--list` (default) prints the table; `--check` validates that the
  scaffold imports and every target has non-empty `requires` and `blocked_by`. Neither
  mode downloads, contacts a network, or runs a benchmark.

`tests/test_generation_eval.py` pins all of this offline: target uniqueness and
completeness, the generator rule, the citation-report keys and its no-truth-claim
provenance, JSON-serialisability, and that both CLI modes return 0 with sockets blocked.

### Assumptions and verified findings

1. **`tools/projection_head.py` exists in the working tree but is not a submission.** It
   is untracked at the time of writing (created inside this workstream) and distils a
   JSON affine map, not a packaged `SentenceTransformer`/Hub model. The MTEB section
   therefore treats the *packaged encoder* as the missing artifact, not the projection
   tool.
2. **The four targets are all blocked as submissions**, so all four carry a non-empty
   `blocked_by`; the test allows `blocked_by=None` for a genuinely runnable future target.
3. **No model was run and no network was used** to produce this document or the scaffold;
   the only model-adjacent code the scaffold touches is the model-free
   `laya_rag.verify.verify_citations`.
4. **The local generator default is `ornith-1.5:35b` via Ollama**
   ([`BENCHMARKS.md`](../BENCHMARKS.md) line 30); the scaffold names no model and requires
   the caller to supply `generator`.
