# Findings: BEIR breadth — quora, and how much the mean depends on its members

ROADMAP Phase 4.2. The headline BEIR claim covers **five of eighteen** datasets
(scifact, nfcorpus, arguana, scidocs, fiqa), so it is not comparable to the field's own
18-dataset averages. This adds a sixth, under exactly the retrievers the headline table uses.

Reproduce:

```sh
.venv/bin/python tools/beir.py --datasets quora \
    --retrievers bm25-engine,biencoder,hybrid-biencoder --out eval/beir/quora
```

## Result — nDCG@10, shipped configuration

| dataset | `bm25-engine` | `bge-small` dense | fused (hybrid) |
| --- | --- | --- | --- |
| scifact | 0.6762 | 0.7200 | 0.7413 |
| nfcorpus | 0.3244 | 0.3385 | 0.3673 |
| arguana | 0.4169 | 0.5956 | 0.5679 |
| scidocs | 0.1497 | 0.1973 | 0.1992 |
| fiqa | 0.2386 | 0.3848 | 0.3817 |
| **quora** | **0.7689** | **0.8876** | **0.8629** |
| **5-set mean** | 0.3612 | 0.4472 | **0.4515** |
| **6-set mean (with quora)** | **0.4291** | **0.5206** | **0.5201** |

## Two readings

**1. The mean is composition-sensitive, and by more than the thing being compared.** Adding one
easy dataset moves the fused mean from **0.4515 to 0.5201 — +0.069**. For scale, the project's
whole per-parameter claim over published BGE-large is a 0.04 margin, and the gap between the
published 4-set matrix (0.491) and its own 5-set row is comparable. So "the BEIR average" is only
meaningful with its membership named, and the README's existing caveat that the 0.495 ensemble row
is *"a single configuration rather than a per-dataset choice"* extends to the mean itself: it is a
statement about a set, and a different set is a different claim.

**2. Fusion is not a free win, again.** On quora the **fused** row (0.8629) is **0.0247 below the
dense row alone** (0.8876). That is the fourth corpus where a "helpful" instrument reverses
(MultiHop-RAG's dense channel, MuSiQue's dense channel, the cross-encoder on five BEIR sets, and
now RRF fusion on quora), and it is the same conclusion each time: pick per corpus, do not
configure globally.

## What this does not show

- **Not the 18-dataset average.** Six of eighteen. Robust04 and TREC-NEWS return stubs on this
  mirror, `specter` and `signal1m` are not served, and the largest (BioASQ 14.9M, FEVER 5.4M,
  NQ 2.7M) are past the ingestion work in Phase 4.1.
- **Not an ensemble row.** The three-model dense ensemble is not extended to quora; only the
  shipped single-encoder configuration is.
- **No BM25 control check.** The harness verifies its `bm25` control against a pinned published
  baseline and reports any drift, but no published anchor is configured for quora, so that check
  did not run for it — ours is 0.7689 and is not validated against an external number here.
- **Not a claim about quora as a task.** Its documents are short questions, which is why the
  embedding pass is cheap and the dense arm is strong; it is an easy dataset, and that is exactly
  why it moves the mean so much.

## Method notes

- **Record the membership with the mean.** The tool writes `datasets` into the report, and the
  table above names every member, because a mean without its set is not a claim anyone can check.
- **A new dataset costs one embedding pass, and nothing else.** Five of six datasets were already
  cached; adding quora was a 15 MB download, 799 s of embedding for 522,931 documents and 199 s of
  scoring, with no harness change.
