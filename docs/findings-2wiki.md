# 2WikiMultiHopQA: recall is saturated, and the coverage objective does not travel

2WikiMultiHopQA is the programme's third external multi-hop set. It was built to answer one question:
the coverage objective is the largest single retrieval win on MultiHop-RAG and it **falsified on
MuSiQue** (−0.0109 MAP@10) — is that failure idiosyncratic to MuSiQue, or general? This is the full
dev split: **12,576 questions** over the **56,651-paragraph distractor pool** (the union of the
context paragraphs the records carry, deduplicated by `(title, text)`).

## Channels: the gold is found, the order is the bottleneck

| arm | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| bm25-engine | 0.9927 | 0.9848 | 0.5037 | 0.9293 |
| biencoder (`bge-small`) | 0.9986 | 0.9959 | 0.5223 | 0.9505 |
| **hybrid-biencoder** (RRF, k=5) | **0.9997** | **0.9986** | **0.5337** | **0.9564** |
| coverage-0.7 | 0.9997 | 0.9984 | 0.5291 | 0.9484 |
| covquota-3-0.7 | 0.9997 | 0.9979 | 0.5282 | 0.9454 |

| paired comparison | delta | 95% CI | p |
| --- | --- | --- | --- |
| biencoder − bm25, MAP@10 | **+0.0186** | [+0.0163, +0.0208] | 0.0000 |
| hybrid − bm25, MAP@10 | **+0.0300** | [+0.0285, +0.0315] | 0.0000 |
| **coverage-0.7 − hybrid**, MAP@10 | **−0.0046** | [−0.0056, −0.0036] | **0.0000** |
| **covquota-3-0.7 − hybrid**, MAP@10 | **−0.0056** | [−0.0061, −0.0050] | **0.0000** |
| coverage-0.7 − hybrid, Hits@4 | −0.0002 | [−0.0006, +0.0001] | 0.2488 (ns) |
| covquota-3-0.7 − hybrid, Hits@4 | −0.0008 | [−0.0014, −0.0003] | 0.0014 |

**Hits@10 is 0.9927 while MAP@10 is 0.5037** for BM25: the pool almost always contains the supporting
paragraphs and the score is lost in the order. A retrieval-trained bi-encoder beats the lexical arm
(**+0.0186 MAP@10**) and RRF fusion adds again (**+0.0300**) — so the shipped dense channel's BEIR
failure is the *encoder's* training, not dense retrieval as such.

## The replication: coverage loses here too

| corpus | coverage rule vs the order it was given |
| --- | --- |
| MultiHop-RAG | **won** — the project's headline retrieval result |
| MuSiQue | **falsified**, −0.0109 MAP@10 [−0.0181, −0.0038] |
| **2WikiMultiHopQA** | **falsified**, coverage **−0.0046** and covquota **−0.0056** MAP@10 (both p<0.0001) |

The MuSiQue failure is **not idiosyncratic**. On two external multi-hop corpora the set-coverage rule
that wins on MultiHop-RAG is *significantly worse* than the fused order it was handed. The mechanism
is visible in the metrics: coverage preserves **Hits@4** (−0.0002, ns) while losing **MAP@10**, so it
trades ordering for coverage — and on 2Wiki, where Hits@10 is already 0.9997, there is almost nothing
left for coverage to buy and only the ordering cost remains. That is the same trade the coverage
objective always makes, on a corpus where the premise (missing gold) does not hold.

**The relevance term differs, and it matters for how to read this.** The MultiHop and MuSiQue
protocols feed the coverage rule a cross-encoder logit and compare it against the cross-encoder; this
harness is deliberately retrieval-only, so the rule is fed the fused RRF score and compared against
the fused order. The comparison is therefore "coverage versus the order it was given", not
"coverage+CE versus CE" — a different, narrower test. The direction is unambiguous and the intervals
exclude zero, but the magnitude is not directly comparable to MuSiQue's.

## What this does not show

- **Distractor setting, not open corpus.** The corpus is the union of the dev records' context
  paragraphs, so retrieval is measured inside the provided pool.
- **No generation.** No EM or answer-accuracy figure; that needs a generator and a judge, and any such
  number would be a statement about them.
- **Not Laya's own dense channel.** The bi-encoder row is `bge-small`.
- **One relevance term for the coverage arms** (fused RRF, above); a cross-encoder-fed coverage arm on
  this corpus is untried.

## Reproduction

```sh
.venv/bin/python tools/twowiki.py --split dev --out eval/benchmarks/2wiki/retrieval
```

The corpus embedding pass took **389 s** for 56,651 paragraphs and is checkpointed every 20,000
documents; the run holds the heavy-job lock. The coverage arms add `--coverage-lambda` (0.7) and
`--min-docs` (3).
