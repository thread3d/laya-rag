# R1 — refreshing the retrieval components: the harness first, then the encoder

`docs/best-in-the-world-2026-10.md` R1 asks whether a 2026 encoder changes the repository's flagship
finding — that the dense channel never earned its weight — when the measurement runs under this
repository's own protocol. This is the first half: what had to be true before such a row could be
believed, and the control that shows it now is.

## 1. The control reproduces the published rows to four decimals

Before any new encoder is scored, the harness has to reproduce a row that is already recorded, on this
host, with a different ML stack — otherwise a "gain" could be the stack. SciFact, all four retrievers,
`eval/beir/r1-scifact-control.json`:

| retriever | this run | recorded | Δ |
| --- | ---: | ---: | ---: |
| `bm25` (published settings) | **0.6539** | 0.654 | 0.000 |
| `bm25-engine` (stemmed) | **0.6762** | 0.676 | 0.000 |
| `biencoder` (`bge-small`) | **0.7200** | 0.720 | 0.000 |
| `hybrid-biencoder` | **0.7320** | 0.732 | 0.000 |

All four match at the precision the project prints, and the NFCorpus control reproduces its rows too
— 0.3042 against 0.304, 0.3244 against 0.324, 0.3391 against 0.339, and a hybrid of 0.3633 against
the **0.364** the README's fusion paragraph quotes. The 0.367 and 0.741 hybrids in `RANKINGS.md`'s
standing table are a later configuration, not this one: matching the README pair exactly is what
establishes that this is the historical pipeline, and it is the pair a new encoder has to beat. The stack differs from the one that produced the
originals (torch 2.15 CPU, `sentence-transformers` 4.1, numpy 2.5 against a pinned 2.2.2 / numpy 1.x
environment), the corpus was downloaded fresh, and the numbers did not move — which is the property
that makes the next row comparable rather than merely higher. It took 183 s to embed 5,183 documents
with the 33M encoder, i.e. **28 documents/s on CPU**; that rate is the baseline every 2026 candidate is
priced against.

## 2. Two silent mismatches had to be fixed first

The harness could not have measured a 2026 decoder-class encoder *correctly*, and neither defect
raises: both produce a number that reads as a fact about the model.

**Pooling.** `tools/beir.py` pooled `cls` or `mean` only. Qwen3-Embedding (and the gte and e5-mistral
family) are trained on the **last token**, so a Qwen3 run would have been embedded under a pooling it
was never trained for. The engine had already learned this — `src/laya_rag/model.py` carries a family
table whose entries were each verified by loading the model — but the harness carried its own "CLS if
the name contains `bge`, else mean" heuristic, and the two disagreed. The harness now has a `last`
mode (which takes the last token *with attention 1*, not position `-1`, because batches pad on the
right) and delegates the name heuristic to `laya_rag.model.family_pooling`:

```
BAAI/bge-small-en-v1.5   -> cls
Qwen/Qwen3-Embedding-0.6B -> last
intfloat/e5-base-v2       -> mean
unknown/model             -> mean      (the historical default, not a guess)
```

**Query prefix.** `--query-instruction` applied one hardcoded string — bge's *"Represent this sentence
for searching relevant passages: "* — to every family. Qwen3 wants an `Instruct: … Query: ` pair and e5
a bare `query: ` token; applying one family's line to another model is the same class of silent
mismatch as the wrong pooling, and it is the third on this project's list after the pooling and the
passage prefix. The harness now asks `laya_rag.model.family_query_prefix`, and **bge falls back to its
historical string**, so no previously recorded row moves — which is the property the test pins.

Both are covered by `tests/test_beir_passages.py` against the real functions (`pool_hidden`, the
`BiEncoder` constructor) rather than a re-implementation of them.

## 3. A hazard worth recording: a fresh report breaks the consistency gate

The first control run was written to `eval/beir/`, the canonical tree. `make consistency-check` then
failed on every "of 18" claim in the repository — correctly, because its authority is the reports on
disk and one report is one dataset. The gate was doing its job; the run was in the wrong place.

Ad-hoc measurement therefore writes to its own root (`eval/r1/…`) and leaves `eval/beir/` to the
canonical programme. The general rule this establishes: **anything that adds a report to the canonical
tree has to regenerate the documents that quote it in the same commit.** That is exactly what the gate
exists to enforce, and it is worth knowing before the first 2026 row lands.

## 4. The first dataset says the 2026 encoder does not pay — on dense, it loses

Qwen3-Embedding-0.6B (Apache-2.0, 596M, `last` pooling via the new mode, 300 SciFact queries, same
corpus, same protocol, same harness). The instruction is a variable in its own right — three wordings
spread the relevant/irrelevant separation 7× on this repository's own measurement — so it was swept:

| encoder and instruction | dense nDCG@10 | hybrid (dense + BM25, RRF) | index time for 5,183 docs |
| --- | ---: | ---: | ---: |
| `bge-small` (2023, **33M**), CLS | **0.7200** | 0.7320 | **183 s** |
| Qwen3-0.6B, no instruction | 0.6825 | 0.7316 | 3,810 s |
| Qwen3-0.6B, generic web-search instruction | 0.7023 | 0.7352 | cached |
| Qwen3-0.6B, SciFact claim/evidence instruction | 0.7012 | **0.7389** | cached |

Three things follow, and the first is the one that matters.

1. **On dense retrieval the 2026 model loses to the 2023 one, by 0.018 to 0.038.** It is not a small
   loss and it is not a fluke of one instruction: the best of the three wordings is still 0.018 below a
   33M model from 2023 that this repository already had. On the corpus where `bge-small` is strongest
   and the 2026 model's 15-task average is not, the average does not carry.
2. **The fused row is where it wins, barely**: +0.0069 with the SciFact instruction, +0.0032 generic,
   +0.0001 with none. SciFact carries 300 queries, so the repository's own minimum-detectable-effect
   guidance (≈0.008–0.011 at 2,255 queries, and larger on a smaller set) puts all three inside noise.
3. **It costs 21× the indexing**: 3,810 s against 183 s for 5,183 documents — 1.36 docs/s against 28.
   At BEIR's larger corpora that is not a CPU-hour problem, it is a CPU-week problem: 5.4M documents
   is roughly 46 days at this rate.

So **R1's prediction (+0.03 on the five-dataset mean) is not supported on the first dataset measured**,
and the honest statement is that swapping the encoder alone did not pay here. What that does *not* yet
establish is that it never pays: SciFact is one corpus, the repository's rule is that a claim about the
system needs several with paired intervals, and this model's documented strength is across a
heterogeneous 15-task set rather than a single scientific-claim corpus. NFCorpus is running now, and a
negative there too would move "refresh the encoder" from the top of the programme to the do-not-retry
ledger — which is why it is worth two datasets and not one.

A harness bug found by the sweep, worth recording because of its shape: the first
`--query-instruction-text` run printed a row **byte-identical** to the no-instruction baseline, because
the flag set the text but did not enable the instruction. An arm that matches its control exactly is a
bug signal, not a result — this repository recorded the same lesson when a harness read JSON passages
with `getattr` and both arms came out equal to four decimals. The flag now implies enabling it, and a
test pins the implication.

## 5. What would settle it

Two more datasets (NFCorpus is running, ArguAna next), paired bootstrap over queries rather than a
point-estimate comparison, and then — because a replica is not the system — the same encoder through
`tools/engine_beir.py`, which measures the engine's own chunking rather than the harness's. The
falsifier stands as written: fused nDCG@10 ≥ +0.03 over the 0.452 single-config mean, and the dense
channel re-earning a non-zero weight on at least three of five datasets.

## What this does not claim

- It does not claim a 2026 encoder wins here. The control shows the harness is now a valid instrument
  and that a decoder-class model can be measured under its own protocol; it says nothing about the
  score.
- It does not claim the pooling and prefix fixes *raise* any recorded number: bge is outside both
  family tables, so its rows are unchanged by construction, and the control above is the evidence.
- The 28 documents/s figure is one host, one model, one corpus, warm cache, `OMP_NUM_THREADS=32` — the
  same caveat that applies to every timing in this repository.
