# Producing the two owed hand-labelled sets

The roadmap owes two human-labelled evaluations: **~300 claim/corpus pairs** for the three-state
adjudication's no-coverage recall (Phase 5.2) and **~200 composed answers** for the verifier's
precision (Phase 5.3). `tools/labelling_queue.py` could queue and score them but could not *produce*
them, and the ledger recorded the gap precisely: the pairs need "a run (or manual construction) that
persists `(query, claim, evidence)`", and the pilot reports store aggregate counts without per-claim
evidence. `tools/labelling_pairs.py` (`make labelling-pairs`) produces the records. It never writes
a label.

## The adjudication set — built from cache, no retrieval run

For each of 150 queries (seed 17), two records are emitted over the evidence the engine actually
retrieved for that query — the cached pool from `ce-scores-full.json` and the corpus text, so the
build takes **~4 seconds** and re-runs nothing:

- **`supported`** — the claim is one of this query's gold facts, and the evidence contains it.
- **`no_coverage`** — the claim is **another** query's gold fact, and this evidence is silent about it.

That produced **299 items (149 supported, 150 no_coverage)**. Two details in the construction are
the difference between a usable set and a silently broken one:

- **The cache is keyed by dataset index, not by position in the answerable list.** The 301 null
  records are simply absent, so the position drifts after the first null. Indexing by position
  produced 276 items with only **11** supported — which reads as "the pool rarely contains its own
  gold" rather than as a mapping bug. The fixture in `tests/test_labelling_pairs.py` puts a null
  between two answerable records and pins that the dataset index survives.
- **Whole-fact coverage is searched over the whole pool, not the shown top-5**, and the covering
  chunk is pulled into the review window even when it ranks below it. Checking only the shown
  passages marked 33 of 300 queries as having no covered fact at all, when the pool's fact recall@100
  is 0.85.

```sh
.venv/bin/python tools/labelling_pairs.py --mode adjudication --limit 150 \
    --out eval/labelling/adjudication-pairs.json          # 299 records
.venv/bin/python tools/labelling_queue.py build \
    --input eval/labelling/adjudication-pairs.json \
    --out eval/labelling/adjudication-queue.jsonl         # 299 unlabelled items
```

Each queue item carries `query`, `claim`, `evidence` (path + text), `claim_from_query` and
`label: null`. The reviewer assigns `supported` / `contradicted` / `no_coverage`; `score` then
reports the agreement with the (absent) proposal, the per-label counts, and **recall of the
`no_coverage` class**, which is the metric the roadmap owes.

## The verifier set — composed answers with the shipped generator

`--mode verifier --generate N` runs the engine over the first N answerable queries, composes an
answer from the retrieved evidence with the configured generator (`ornith-1.5:35b` at the default
Ollama endpoint) and writes each record — query, answer, evidence — to a checkpointed JSONL sidecar
as it is produced. A re-run resumes, and `--answers` reads that sidecar directly, so an interrupted
run still yields a smaller queue rather than nothing: two hours of generation must not be one crash
away from starting over, which this repository has now had to learn three times.

```sh
.venv/bin/python tools/labelling_pairs.py --mode verifier --generate 320 \
    --out eval/labelling/verifier-pairs.json              # resumable via verifier-pairs.jsonl
.venv/bin/python tools/labelling_queue.py build \
    --input eval/labelling/verifier-pairs.json \
    --out eval/labelling/verifier-queue.jsonl             # 202 unlabelled items
```

Over **320** answerable queries that produced **202 composed answers** — 118 were abstentions (the
engine declined, or the composer returned nothing), a 37% abstention rate on this convenience
sample — at roughly 15 s/query. The reviewer decides whether each answer is faithful to its
evidence; the deterministic verifier's flags are then scored against those labels, which is the
precision the Phase 5.3 item asks for.

## What this does not show

- **No labels exist.** Both queues are unlabelled by construction; `score` reports `labelled: 0`
  until a human edits the JSONL, and a model-generated label would defeat the point of a
  hand-labelled set. Nothing here may be quoted as a precision or a recall.
- **The `supported` / `no_coverage` kinds are a proposal, not ground truth.** They are constructed
  from the benchmark's gold facts and the cached evidence, and the coverage test is the benchmark's
  normalised substring rule — a paraphrase that answers the same question without repeating the gold
  sentence is filed as `no_coverage` until a reviewer says otherwise. That is the point of asking.
- **No `contradicted` arm.** The set is supported-vs-silent; genuine contradictions are not
  synthesised here, so the queue exercises two of the three states (the shipped three-state
  demonstration covers the third on constructed claims).
- **The verifier set is the local generator's prose**, not a frontier model's, and it is one model on
  one corpus; precision measured on it is a statement about that generator.
- **The first 320 answerable queries are not a random sample of the benchmark** — they are dataset
  order, which front-loads the 2023 crypto/tech records. A reviewer should read the queue as a
  convenience sample, and the roadmap's item is the *labelling*, not a benchmark claim.
- **The 118 abstentions are dropped, not labelled.** Whether the engine *should* have answered is
  the Phase 5.2/5.3 question about the gate, not the verifier; the queue is composed answers only.

## Reproduction

```sh
.venv/bin/python tools/labelling_pairs.py --check       # offline invariants, no file needed
.venv/bin/python -m pytest tests/test_labelling_pairs.py -q
```
