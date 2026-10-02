# Findings: distilling the dense ensemble into one small encoder

This is the strategic Tier 3.1 item: the project's best retrieval comes from a **three-model
ensemble** (bge-small + bge-base + bge-large, 478M parameters), which is expensive to serve
and not an artefact the project owns. This distils it into the single 33M `bge-small`.

Reproduce (the training run is ~6 minutes on this host):

```sh
HF_HUB_OFFLINE=1 .venv/bin/python tools/finetune.py --dataset nfcorpus \
    --teacher-models BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5 \
    --teacher-positives 3 --epochs 1 --out eval/distill-nfcorpus \
    --save-model eval/distill-nfcorpus-model

HF_HUB_OFFLINE=1 .venv/bin/python tools/distill_eval.py --dataset nfcorpus \
    --adapted eval/distill-nfcorpus-model
```

## Design

NFCorpus is the only BEIR dataset shipped here with a train split (110,575 gold pairs). The
teachers' **document** embeddings are already cached, so mining costs one query-embedding
pass per teacher and no corpus re-embed:

1. each teacher ranks the 2,590 train queries over the 3,633-document corpus;
2. the three rankings are fused by RRF (k=5) and the top **3** documents per query become
   the student's positives — teacher supervision, not the qrels;
3. the student trains with `MultipleNegativesRankingLoss` for **1 epoch** on **7,770** pairs.

## Result (NFCorpus test, 323 queries, nDCG@10)

| model | parameters | nDCG@10 | paired vs zero-shot |
| --- | --- | --- | --- |
| zero-shot `bge-small` | 33M | 0.3385 | — |
| **distilled `bge-small`** | **33M** | **0.3505** | **+0.0120 [+0.0022, +0.0216]** (significant) |
| teacher ensemble | 478M | 0.3798 | +0.0413 [+0.0298, +0.0541] (significant) |

The distilled student recovers **~29% of the ensemble's advantage** (0.0120 of 0.0413) while
keeping a single 33M encoder at serve time — one fourteenth of the parameters and one model
to load instead of three. It also beats the earlier gold-pair fine-tune (+0.0085 on 20k
pairs) with 7,770 pairs and one epoch.

**This is an adapted number, not a zero-shot one.** The report labels it that way and it must
never be compared as if it were the zero-shot row.

## Honest limits

- **One dataset**, and only because NFCorpus ships a train split; BEIR's other buildable
  sets here do not, so this does not generalize to a claim about BEIR as a whole.
- The student learns the **teachers'** notion of relevance, so a teacher error is inherited
  and self-training can reinforce it. The evaluation is against gold test qrels, which is
  what makes the +0.0120 meaningful rather than circular.
- One epoch, three positives and one teacher configuration; none was swept, so this is a
  floor, not a tuned result.
- It produces a **local model directory**, not a packaged, versioned artefact on a Hub — the
  packaging step in `RESEARCH.md` Tier 3.1 is still outstanding, and the engine does not yet
  pick up a distilled encoder by default.
