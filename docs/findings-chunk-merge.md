# Findings: the chunking loss is the chunked representation, not its selection

Round 29's engine-level run put the engine 0.0144 below the whole-document replica. SciFact's
median document is **1,426 characters** against a 1,200-character chunk size, so most
abstracts were split into two heavily overlapping chunks — a plausible cause. This tests four
responses to it at the engine level, same 300 queries, `bge-small`, top-10. **None of the
three targeted fixes recovers the whole-document score**; the conclusion is that the loss is
in the chunked representation itself.

| configuration | chunks | nDCG@10 | vs base |
| --- | --- | --- | --- |
| default chunking, 1200/400 | 9,986 | 0.7264 | — |
| `CHUNK_MERGE_SHORT` — documents ≤ 1600 chars kept whole | 8,470 | 0.7236 | −0.0028 [−0.0130, +0.0070], p=0.59 |
| `RETRIEVE_K=40` — double the candidate pool | 9,986 | 0.7273 | +0.0009 [−0.0031, +0.0048], p=0.66 |
| `CONTEXT_PREFIX` — breadcrumb (path + title line) on every chunk | 9,986 | 0.7243 | −0.0021 [−0.0178, +0.0128], p=0.79 |
| whole-document — one chunk per abstract | 5,183 | **0.7396** | **+0.0132 [−0.0002, +0.0272]**, p=0.058 |

Reproduce:

```sh
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag merge --chunk-merge-short"
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag rk40 --retrieve-k 40"
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag ctxpre --context-prefix"
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag whole --chunk-chars 100000 --chunk-overlap 0"
```

## Three plausible fixes, none of which works

Whole-document beat the merge rule **significantly**: **+0.0160 [+0.0025, +0.0304]**, p=0.016.
Each remaining explanation was then tested directly, and all three were rejected:

| hypothesis | test | result |
| --- | --- | --- |
| merging short documents back recovers it | `CHUNK_MERGE_SHORT` (62% of the corpus) | −0.0028, p=0.59 — no |
| the pool covers too few **documents** under chunking | `RETRIEVE_K=40` (20 → 40 candidates) | +0.0009, p=0.66 — no |
| later chunks lose the document's **title/context** | `CONTEXT_PREFIX` on every chunk | −0.0021, p=0.79 — no |

So the loss is not pool depth, not the short-document split, and not missing context. It is the
**chunked representation itself**: for SciFact's abstracts a single embedding over the whole
document matches the claim better than the embedding of any of its chunks, and the
document-level ranking takes a document's best chunk as its score, so deeper pools cannot
repair it. That points at chunk *size* rather than chunk *selection* — and at a corpus-shape
decision the engine cannot make for an unknown corpus.

Note the bisection that remains: `merge_short` covered 1,200–1,600 characters and changed
nothing, so the gain must sit in the documents **longer than 1,600**. `--chunk-chars 2000`
would locate the band; it is left as the next measurement rather than run to exhaustion here.

## What shipped

`chunk_text` / `chunk_documents` / `RagEngine` gained a `merge_short` knob
(`CHUNK_MERGE_SHORT`, default **false**), and `engine_beir` gained `--chunk-merge-short`,
`--retrieve-k` and `--context-prefix` so each hypothesis above is one command. The knob is
documented as a **negative**; no default changed.

## Honest limits and the next experiment

- One dataset, and the whole-document delta itself is borderline (p=0.058); the significant
  comparison is whole-document against the merge rule (+0.0160, p=0.016).
- The bisection is unfinished. `merge_short` covered 1,200–1,600 characters and changed
  nothing, so the gain must sit in documents **longer than 1,600**; `--chunk-chars 2000`
  would locate the band. If it does not recover the gain either, the mechanism is not a length
  threshold and a document-level score aggregation (rather than first-occurrence selection)
  becomes the next candidate.
- Raising the default chunk size is **not** proposed: it would change behaviour for every
  corpus on the strength of one short-document dataset, and the engine cannot know a corpus's
  document-length distribution before it is indexed.
- A corpus-aware default — chunk size derived from the measured document-length distribution
  at index time — is the shape a real fix would take, and it is a design change, not a
  parameter.
