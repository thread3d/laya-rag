# Findings: chunking costs ~0.013, but not through short documents

Round 29's engine-level run put the engine 0.0144 below the whole-document replica. SciFact's
median document is **1,426 characters** against a 1,200-character chunk size, so most
abstracts were split into two heavily overlapping chunks — a plausible cause. This tests two
responses to it at the engine level, same 300 queries, `bge-small`, top-10.

| configuration | chunks | nDCG@10 | vs base |
| --- | --- | --- | --- |
| default chunking, 1200/400 | 9,986 | 0.7264 | — |
| `CHUNK_MERGE_SHORT` — documents ≤ 1600 chars kept whole | 8,470 | 0.7236 | −0.0028 [−0.0130, +0.0070], p=0.59 |
| whole-document — one chunk per abstract | 5,183 | **0.7396** | **+0.0132 [−0.0002, +0.0272]**, p=0.058 |

Reproduce:

```sh
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag merge --chunk-merge-short"
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag whole --chunk-chars 100000 --chunk-overlap 0"
```

## The interesting comparison

Whole-document beat the merge rule **significantly**: **+0.0160 [+0.0025, +0.0304]**, p=0.016.
So the short-document split is not what costs the retrieval — merging those documents back
(62% of the corpus) left the score marginally *worse*, while making *every* document whole
moved it up.

The loss therefore sits in the documents **longer than the merge window** (1,600+ characters),
or in candidate diversity: with chunking, a fixed `RETRIEVE_K=20` returns ~14–19 distinct
documents because long documents contribute several chunks each, so the top-10 can spend
slots on one document and miss a relevant one — a mechanism the whole-document run removes
entirely by making every candidate a different document.

## What shipped

`chunk_text` / `chunk_documents` / `RagEngine` gained a `merge_short` knob
(`CHUNK_MERGE_SHORT`, default **false**) and `engine_beir --chunk-merge-short` can measure it.
It is documented as a **negative**: it does not recover the gap, and the default does not
change.

## Honest limits and the next experiment

- One dataset, and the whole-document delta is borderline (p=0.058); the *comparison* against
  the merge rule is what is significant.
- The two candidates — chunk length for long documents, and candidate depth against a fixed
  `RETRIEVE_K` — are not separated here. `RETRIEVE_K=40` under chunking should distinguish
  them: if the score moves to ~0.7396, the mechanism is document coverage, not chunk size.
  That is the next measurement, and it is cheap to state and bounded to run.
- Raising the default chunk size is not proposed: it would change behaviour for every corpus
  on the strength of one short-document dataset.
