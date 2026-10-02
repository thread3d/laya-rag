# Findings: the engine's fusion constant was not the measured one

The engine fused its dense and lexical channels with `reciprocal_rank_fusion`'s historical
default `k=60`, while **every MultiHop-RAG number this repository documents was produced with
`k=5`** (`tools/multihop_official.py`, `tools/multihop_coverage.py`, `tools/multihop_diagnostics.py`).
So the engine did not reproduce its own measured baseline. This measures the constant and
aligns the engine with the harness.

Reproduce:

```sh
# MultiHop, full 2,255 queries, depth 20 (the engine's first stage)
HF_HUB_OFFLINE=1 .venv/bin/python tools/multihop_diagnostics.py --max-depth 20 --depths 10 --rrf-k 5
HF_HUB_OFFLINE=1 .venv/bin/python tools/multihop_diagnostics.py --max-depth 20 --depths 10 --rrf-k 60

# BEIR, cached embeddings
HF_HUB_OFFLINE=1 .venv/bin/python tools/beir.py --datasets scifact,nfcorpus,arguana \
    --retrievers hybrid-biencoder --rrf-k 5
```

## MultiHop-RAG (full protocol, paired over 2,255 queries)

| k | MAP@10 | Hits@10 | Hits@4 |
| --- | --- | --- | --- |
| **5** | **0.2552** | 0.8120 | **0.6860** |
| 20 | 0.2514 | 0.8173 | 0.6776 |
| 60 | 0.2512 | 0.8173 | 0.6776 |

`k=5` minus `k=60`, paired:

| metric | delta | 95% CI | p |
| --- | --- | --- | --- |
| MAP@10 | **+0.0040** | [+0.0012, +0.0069] | 0.0078 |
| Hits@4 | +0.0084 | [−0.0031, +0.0200] | 0.17 |
| MRR@10 | +0.0050 | [−0.0014, +0.0114] | 0.13 |
| Hits@10 | −0.0053 | [−0.0111, +0.0004] | 0.10 |

## BEIR (cached `bge-small`, `hybrid-biencoder`)

| dataset | k=5 | k=60 | delta | paired p |
| --- | --- | --- | --- | --- |
| scifact | **0.7408** | 0.7320 | +0.0088 | 0.12 |
| nfcorpus | **0.3673** | 0.3635 | +0.0038 | 0.26 |
| arguana | **0.5645** | 0.5602 | +0.0043 | 0.13 |

## Verdict

**`FUSION_K` now defaults to 5.** It is significantly better on the headline MultiHop MAP@10,
positive but not individually significant on all three BEIR datasets, and no metric is
significantly worse (the MultiHop Hits@10 cost is −0.0053, p=0.10). The change costs nothing —
it is the same fusion, one constant — and it removes an inconsistency where the shipped
engine would not have reproduced the numbers in its own documentation.

## Honest limits

- BEIR's per-dataset deltas are positive but each interval crosses zero at n=300–1,406; the
  case rests on the MultiHop significance plus three consistent BEIR signs, not on a BEIR
  significance claim.
- `k=20` and `k=60` are indistinguishable, so the effect is specifically the very low constant
  weighting each channel's top hit, not a smooth trend.
- The blend constant used when mixing the first stage with a cross-encoder is a separate
  parameter and was measured separately (`RERANK_BLEND`); it is unchanged at 60.
