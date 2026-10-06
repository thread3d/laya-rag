# Quantization on the encoder this project actually ships

The external brief asked for RaBitQ + OPQ + mmap and for binary/int8 quantization, and the audit's
tension 14 records that the question was **never re-measured here** — where the encoder is
`bge-small`, which is **not MRL-trained**, so the brief's "~96% retained" figure is itself
model-dependent (74.77–96.45% by its own footnote). `tools/quantization_probe.py` answers it on the
**real** vectors rather than synthetic ones.

## Method

Self-retrieval on a cached BEIR corpus embedding matrix (L2-normalised, so inner product is cosine):
sample query vectors from the corpus, take the exact `IndexFlatIP` top-10 as ground truth, and
measure what each compressed IVF index keeps. `nlist=256`, `nprobe=32`, recall@10, **no rescoring**.
Four codecs over the same coarse quantiser, so the comparison isolates the per-vector codec.

## The result

| codec | scidocs recall@10 | scidocs B/vec | fiqa recall@10 | fiqa B/vec |
| --- | --- | --- | --- | --- |
| flat-exact | 1.0000 | 1536 | 1.0000 | 1536 |
| ivf-flat | 0.9942 | 1559 | 0.9874 | 1551 |
| ivf-fp16 | **0.9942** | **791** | **0.9872** | **783** |
| ivf-sq8 (int8) | **0.9876** | **408** | **0.9738** | **399** |
| ivf-pq (m=48, 8-bit) | **0.6792** | **87** | **0.5750** | **70** |

Three readings, on the encoder that actually ships:

1. **fp16 is free.** It halves the index (1536 → ~790 B/vector) at a recall change of **0.0000**
   (scidocs) and **−0.0002** (fiqa). For a CPU/self-hosted engine this is the obvious default.
2. **int8 costs little and is not free.** ~**399–408 B/vector** (3.8× smaller) for **−0.66**
   (scidocs) and **−1.36** (fiqa) recall@10 — consistent with the brief's ~96% *retained* figure
   (ours is 97.4–98.8%), and the loss is larger on the larger corpus.
3. **PQ at this setting is far worse than the brief implies.** `m=48`, 8-bit, **69–87 B/vector**
   (22–32× smaller) loses **31–41 recall points** (0.6792 / 0.5750). Whatever the "~96% retained"
   figure describes, it is not this codec on these vectors.

## What this does not show

- **No rescoring.** These are raw codec recalls. The engine's shipped ANN path is **IVF-PQ with
  `rescore 500`**, measured at **recall@10 0.998** (`findings-ann-scale.md`) — the exact vectors
  re-score the candidates, which recovers almost all of what the raw PQ row loses. This probe says
  what the *compression* costs, not what the *engine* loses.
- **Self-retrieval is easier than real queries.** Querying with corpus vectors makes the nearest
  neighbour the vector itself; real queries are harder, so these recalls are optimistic in level
  (not necessarily in ranking of the codecs against each other).
- **One encoder, two corpora, untuned parameters.** `bge-small`, scidocs and fiqa, `nlist`/`nprobe`
  and PQ `m`/`nbits` fixed; a different encoder (especially an MRL-trained one) or a tuned PQ can
  move the middle of the table.
- **No binary (1-bit) codec.** That is faiss's Hamming API (`IndexBinaryIVF`), a different metric; it
  is not measured here.

## Reproduction

```sh
.venv/bin/python tools/quantization_probe.py \
    --cache eval/beir/cache/scidocs.bge-small-en-v1.5.cls.npy --out eval/quantization-probe-scidocs
```

No model and no network: the cache is already on disk, and the whole probe runs in seconds.
