# The learned selector, at full scale and against a shuffled-bridge control

Bet 2 is "train the set-aware selector on one corpus and score it on the other". The prototype
(`tools/multihop_ltr.py`) fits a logistic scorer per candidate — the cross-encoder logit, the fused
pool position, and optionally the bridge-conditioned score — on half the queries and scores the other
half. It had been run once on a 400-query subset (`eval/benchmarks/multihop-rag/ltr.json`, 200
held-out queries); this is the full-scale run plus the control that was missing.

## The full protocol — 2,255 queries, 1,127 held out

| ranker | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| fused (first stage) | 0.8023 | 0.6658 | 0.2473 | 0.5340 |
| cross-encoder | 0.8644 | 0.7234 | 0.2881 | 0.5997 |
| **LTR** (ce + fused_rank) | 0.8608 | 0.7305 | 0.2939 | 0.6144 |
| blend-0.75 | 0.8652 | 0.7402 | **0.2975** | 0.6157 |
| **covquota-3-0.7** (hand rule) | **0.8918** | **0.7598** | **0.3147** | 0.6100 |

| paired comparison (held-out, 10,000 samples) | delta | 95% CI | p |
| --- | --- | --- | --- |
| LTR − cross-encoder, MAP@10 | +0.0058 | [−0.0011, +0.0128] | 0.1034 (ns) |
| LTR − blend-0.75, MAP@10 | −0.0036 | [−0.0078, +0.0008] | 0.0961 (ns) |
| **LTR − covquota-3-0.7**, MAP@10 | **−0.0208** | [−0.0287, −0.0129] | **0.0001** |
| LTR − fused, MAP@10 | **+0.0466** | [+0.0370, +0.0560] | 0.0001 |

**The null survives the full protocol.** The fitted scorer beats the first stage by a wide margin and
beats the cross-encoder by a margin that is *not* significant, and it is **significantly worse than
the hand-set coverage-plus-quota rule** — which the subset run only hinted at (0.3102 vs 0.3155,
untested). At 1,127 held-out queries the interval on LTR − covquota is tight enough to call:
**−0.0208 [−0.0287, −0.0129]**.

## The control: does the bridge feature do anything?

The 400-query subset carries two cached bridge-score sets, one from the real first-hop bridge and one
from a shuffled bridge — the same procedure with the conditioning removed. Both runs use the identical
train/test split (seed 17), so they are directly pairable.

| ranker | real bridge | shuffled bridge | paired delta (shuffled − real) | p |
| --- | --- | --- | --- | --- |
| bridge score alone | 0.1288 | 0.0834 | **−0.0454** [−0.0711, −0.0205] | **0.0006** |
| **LTR** (ce + fused_rank + tri) | 0.3102 | 0.3085 | −0.0018 [−0.0072, +0.0033] | 0.5104 (ns) |
| cross-encoder | 0.2962 | 0.2962 | 0.0000 | 1.0000 |

The fitted weight on the bridge feature is **5× larger when the bridge is real** (+0.075 vs +0.014),
and the bridge score by itself carries real signal (−0.0454, p=0.0006) — **but the selector's held-out
MAP@10 does not move** (−0.0018, p=0.51). The bridge signal is real and **redundant with the
cross-encoder** for ranking: the model weights it, and the ranking is unchanged.

## What this says, and what it does not

- **The learned selector is not the lever.** Five model-free or fitted optimisers over the cached
  channels are now null — LTR (here, at full scale), IDF atom weights, beam search, per-question-type
  routing, and the hand blend — and the hand-set set-aware rule remains the best ranking rule measured.
- **The bridge feature is real but redundant** in this feature set. That is a sharper statement than
  "the bridge did not help": the score separates (p=0.0006) and the *selector* gains nothing (p=0.51).
- **This does not test the generator-free-iteration headroom.** These runs rank the cross-encoder's
  depth-100 pool. The Stage-1 experiment measured **+0.0248 ceiling MAP@10** on a *probe-expanded
  lexical* pool; realizing *that* would need the selector scored over the expanded pool, and the
  probe-added candidates have no cross-encoder scores cached — it needs new inference, not a
  re-ranking of what exists. Recording the distinction is the point.
- **Supervised on the benchmark's own labels** — "recalibrate on the target domain", the same caveat
  as the answerability gate. It is evidence about headroom, not about transfer.

## Reproduction

```sh
# full protocol, no bridge feature
.venv/bin/python tools/multihop_ltr.py \
    --ce-cache eval/benchmarks/multihop-rag/ce-scores-full.json --samples 10000 \
    --out eval/benchmarks/multihop-rag/ltr-full
# the control pair, on the subset that has bridge caches
for arm in real shuffled; do
  .venv/bin/python tools/multihop_ltr.py --ce-cache eval/benchmarks/multihop-rag/ce-scores.json \
      --tri-cache eval/benchmarks/multihop-rag/tri-scores-$arm-bridge.json --samples 10000 \
      --out eval/benchmarks/multihop-rag/ltr-bridge-$arm
done
```

Each run takes seconds; the paired real-vs-shuffled comparison is a `evalkit.paired_bootstrap` over
the two runs' `per_query` LTR values.
