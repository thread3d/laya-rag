# R4 — the evidence-layer protocol, and the number it produced on the first run

`docs/best-in-the-world-2026-10.md` R4 asks for the four axes this programme's claim rests on as **one
command and one report**, so that a reader can see which axes were measured and which were not. It
asks for something else too, implicitly: that the protocol be run, because a protocol nobody runs is a
diagram. On its first run, against a dataset built the day before, it produced the strongest single
result of the programme.

## The command

```sh
make evidence-protocol
# python tools/evidence_protocol.py --answer-row <engine-extractive report> \
#     --gate-rows <labelled feature rows> --out eval/evidence-protocol
```

Four axes, each reading an artefact the programme already produces, each number naming its file and
field:

1. **Evidence quality** — MAP@10 / Hits@10 / Hits@4 / MRR@10 of the retrieval the answers were composed
   from, plus the answer's own two numbers (nugget recall, and **structural** citation validity — every citation in range and every quote found; the advisory content layer is separate).
2. **Evidence-side abstention** — AUC, risk@80 and AURC of the decision that says "this corpus cannot
   answer", scored on the benchmark's own negative class, against the shipped scalar as a baseline.
3. **Joint score** — evidence quality × (1 − false-access rate), the score that stops abstention being
   bought with blanket refusals.
4. **Cost** — wall and CPU seconds, thread count, device.

**No axis needs a generator, a judge model or an API key.** Anything that did would belong in the
`engine + <model>` column, and the report says so in its footer.

Every axis whose artefact is absent prints **not measured** with the command that would produce it —
never a zero, never a skip. That is a tested property (`tests/test_evidence_protocol.py`), because a
protocol that can be gamed by omission is worse than no protocol.

## The run

Inputs: the extractive row over 150 MultiHop-RAG queries with gates zeroed (`findings-extractive-coverage.md`),
and a gate dataset built from the benchmark itself — **601 rows: all 301 `null_query` records plus 300
sampled answerable ones** (`tools/gate_dataset.py`). The `null_query` records are the benchmark's own
negative class: 301 questions whose gold answer is literally "Insufficient information", which its
retrieval metric excludes.

```
## 1. Evidence quality                     ## 2. Evidence-side abstention
MAP@10             0.3128                 scorer                  n    AUC     risk@80  AURC
Hits@10            0.9067                 feature gate (held out) 300  0.8159  0.4042   0.2603
Hits@4             0.7400                 shipped scalar, same     300  0.5208  0.5000   0.4878
MRR@10             0.6152
answer nugget recall 0.2060               At the deployed threshold (≥ 0.5) the shipped scalar
citation clean     1.0000                 would answer 0.6933 of the unanswerable queries.
abstention rate    0.0000

## 3. Joint score
evidence MAP@10 0.3128 × (1 − false-access 0.4042) = 0.1864
```

## What this establishes, and what it does not

**The gate carries answerability signal on real negatives; the scalar in production does not.** AUC
**0.8159** on a held-out half of genuine unanswerable questions, against the scalar's **0.5208** — and
the gate's number is essentially the same as the in-domain 0.8146 measured on the earlier 300-query
protocol, so the feature set is not a corpus artefact of that protocol. The scalar's number is the
repository's own long-standing "answerability does not transfer" finding, confirmed on the class it
was always about rather than on topic-shift proxies.

**The deployed system answers two thirds of questions the corpus cannot answer.** At its shipped
threshold the scalar would answer **69.3%** of the 301 `null_query` records. On the full set, before
the half-split, that figure is **72.1%** (217 of 301, mean scalar 0.627, median 0.731). The relevance
gate does some of the work — 25 of the 301 never reached the answerability decision because the
evidence set was emptied first — and of the 276 that did reach it, **78.6%** were called answerable.
That is the most useful number this programme has produced about its own abstention, and it is the
one the evidence protocol exists to publish.

**Three limits that must travel with it.**

- These negatives are *in-corpus*: the gate is fitted on half of these rows and scored on the other
  half. That tests generalisation across queries, not across corpora. The cross-corpus arm is
  `tools/gate_transfer.py`, and its two-corpus pilot was undecidable at 19 rows — MTRAG-UN's 97
  unanswerable and 78 underspecified turns are the sample that would settle it.
- The classes are **balanced by construction** (301 nulls against 300 sampled answerable). AUC is
  base-rate independent, so it survives this; `risk@80` and the joint score are not, and a deployment's
  true negative prevalence would move both. The dataset builder samples positives rather than weighting
  them, and the report says nothing about deployment prevalence.
- The joint score multiplies evidence quality from one sample by abstention from another. It is a
  **joint report, not a product over one sample**, and the report carries that caveat verbatim rather
  than letting the arithmetic imply otherwise.

## Cost is still not measured

Axis 4 printed "not measured — `make cost-report`" because no cost report exists for the MultiHop-RAG
index. The envelope was measured on the docs corpus in R0 (6.71 s wall / 96.3 s CPU per query at 16
threads, CPU-only) and that number is in `findings-cost-envelope.md`; the axis exists so that the
omission is visible, which it now is.

## Reproducing it elsewhere — the falsifier this document set itself

R4's own falsifier was: *if an external reviewer cannot reproduce the four-axis report from a fresh clone
in under an hour, it is a tool and not a protocol.* Two files were missing from that path — the
benchmark's records and its 609-document corpus both arrived by hand — so the report was not
reproducible at all. `make multihop-fetch` closes that, and it **asserts the dataset's shape before any
measurement is derived from it**: 2,556 records, of which exactly **301 are `null_query`**, and 609
documents. A truncated or re-versioned download fails there rather than producing a smaller number that
reads as a result — the discipline `tools/bioasq.py` applies to a 26 GB dump. `--check` re-verifies an
existing copy without touching the network.

The chain, with the wall-clock each step actually took on this host at the defaults:

| step | command | measured |
| --- | --- | --- |
| 1. fetch and verify | `make multihop-fetch` | **12 s** (12 MB) |
| 2. build the engine's corpus and index | `laya-rag index eval/benchmarks/multihop-rag/engine-corpus --rebuild` | **580 s** (609 documents → 4,592 chunks) |
| 3. one extractive arm | `tools/engine_extractive.py --limit 150 --sample-seed 17` | **736 s** |
| 4. labelled gate rows | `make gate-dataset` (601 rows) | **~2,760 s** |
| 5. fit the gate | `make gate-train` | **2 s** |
| 6. the report | `make evidence-protocol` | **1 s** |
| | **total** | **~68 min** |

So the chain is **~68 minutes** at full size — ten minutes over the bar R4 set — and it lands **under an
hour** at `--limit 100` for step 4 (401 rows, ~31 min). Either way the number is stated rather than
asserted, and the protocol reports its own `n` so a reduced sample is visible in the report rather than
implied to be the full one.

What a second host would still have to supply: the Laya checkpoints (2.3 GB, `vendor/laya/setup_laya.sh`),
and the willingness to spend the hour. Nothing else in the chain is machine-specific — the corpus is
downloaded and verified, the index is rebuilt from it, and every number in the report names the artefact
it came from.

## A fifth axis, added because the first four could not say what produced them

The report's own caveat — *"quality and abstention come from different samples"* — was prose that a
reader had to notice. It is now **computed**: `configuration` reads the settings block from each input
and reports `same_query_sample`, with the warning when it is false. The first run of that axis on this
repository's own artefacts printed

```
- answer_row: limit=150, sample_seed=17, rerank_model=None, …, index_dir=eval/…/engine-extractive-rank150
- gate_rows:  limit=300, sample_seed=17, …                               index_dir=eval/…/engine-default
  the axes were measured on different samples; the joint score is a joint report, not a product
```

and in doing so caught a **provenance defect**: the answer row recorded an `index_dir` that `--out`
would have *derived*, while the run actually used the index named by `--index-dir` /
`LAYA_RAG_INDEX_DIR`. A row measured against an existing index could therefore carry a settings block
naming a directory that never existed on disk. `tools/engine_extractive.py` now records the effective
path. Four tests pin the axis, including the case where the gate dataset carries its settings at the top
level rather than under `_settings`.

**And it distinguishes a non-default arm**, which is what it exists for. The same protocol run over the
cross-encoder row shows `rerank_model=BAAI/bge-reranker-v2-m3` in the answer row's settings, so the CE arm
and the shipped arm are distinguishable at a glance rather than by filename — while `index_identity` is
identical across both, which is correct: the identity hashes the corpus and the ingest settings, not the
second stage.

This is the same class of defect as the three others this programme has found — a cache keyed by a
basename, a scalar baseline read from the wrong column, a resume fingerprint that ignored the composer's
objective — and it was found the same way: by making a number's provenance *visible* rather than
reviewable.

## Axis 4 wires itself now, and is still owed here

`--cost` defaults to the report `make cost-report` writes, so the chain's last axis populates without a
flag to remember — and an absent file leaves it reported as **not measured**, which is the behaviour the
protocol exists to have. It reads "not measured" on this repository today: the cost report for the
MultiHop index has not been run beside these two axes, so the honest state is a named gap rather than a
number from a different index.

## What is owed

- A cross-corpus negative set (MTRAG-UN), which turns "generalises across queries" into "travels".
- A deployment-prevalence weighting, so `risk@80` and the joint score can be read at a base rate.
- The gate **shipped**: the numbers above are an argument for making the twelve-feature gate the
  default answerability path when a fitted artefact is present, rather than an opt-in.
