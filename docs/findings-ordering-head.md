# Findings: late interaction does not capture the ordering headroom

ROADMAP Phase 2.1. `ceiling-decomp.md` measures the headroom as mostly *ordering* — **+0.1254**
to reorder the cross-encoder's own top-10 — and every model-free optimiser over the cached
features is already null (LTR, IDF weights, beam search, per-query-type routing). The hypothesis
was that a **late-interaction** head supplies the new information, because different query tokens
can match different evidence tokens, which is the mechanism a single pooled vector cannot express.

This tests it with `answerdotai/answerai-colbert-small-v1` (33M, 96-dim token output) over the
same 400-query seed-17 subset, the same pools, and the same metric function as the project's own
second-stage experiments.

Reproduce:

```sh
make multihop-ordering                     # 400 queries, ~4 minutes on MPS
.venv/bin/python tools/multihop_ordering.py --limit 5 --out /tmp/ordering-smoke
```

## Result

| arm | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| fused first stage | 0.7975 | 0.6625 | 0.2522 | 0.5342 |
| cross-encoder order | 0.8600 | 0.7225 | 0.2986 | 0.6010 |
| **coverage + 3-doc quota** | **0.8850** | **0.7550** | **0.3280** | **0.6115** |
| ColBERT MaxSim over the pool | 0.7350 | 0.5625 | 0.2092 | 0.4705 |
| ColBERT MaxSim, reordering the CE top-10 | 0.8600 | 0.6800 | 0.2730 | 0.5594 |

Paired over the 400 queries:

| comparison | MAP@10 | Hits@4 | Hits@10 |
| --- | --- | --- | --- |
| colbert vs **fused** | **−0.0430 [−0.0632, −0.0229]** | −0.1000 [−0.1525, −0.0475] | −0.0625 [−0.1125, −0.0150] |
| colbert vs **ce** | **−0.0894 [−0.1109, −0.0679]** | −0.1600 [−0.2125, −0.1050] | −0.1250 [−0.1725, −0.0800] |
| colbert vs **covquota** | **−0.1016 [−0.1229, −0.0804]** | −0.1800 [−0.2325, −0.1275] | −0.1500 [−0.1950, −0.1050] |
| colbert-ce10 vs **ce** (ordering only) | **−0.0257 [−0.0434, −0.0079]** | −0.0425 [−0.0825, −0.0025] | +0.0000 [0, 0] |

Every interval excludes zero. The finding is a **negative**, and the third row is the sharpest
part of it.

Three readings:

1. **MaxSim over the pool is worse than the unordered first stage.** −0.0430 MAP@10 against
   `fused`, which selects no ordering at all. A general-purpose late-interaction reranker is not
   merely unable to beat the second stage; it is below the baseline it was meant to reorder.
2. **The ordering-headroom test is cleanly falsified.** `colbert-ce10` reorders the
   cross-encoder's *own* top-10 by MaxSim, so recall is held fixed and the only variable is
   order — Hits@10 is +0.0000 by construction. MAP@10 still falls by **0.0257**, with Hits@4 and
   MRR falling too. The **+0.1254** the decomposition measures as available from reordering that
   set is therefore not reachable by this instrument: it is headroom a *better* scorer would
   have to reach, not headroom any reranker can collect.
3. **The likely reason is the corpus, not the mechanism.** `answerai-colbert-small-v1` is trained
   for topical passage retrieval, and MultiHop-RAG's difficulty is the second hop, which is
   low-topical-relevance *by construction* (the measured second-hop tax). A topical late-interaction
   scorer inherits exactly the limitation that defeated the pointwise cross-encoder on ArguAna,
   and unlike the cross-encoder it was not trained on this benchmark's relevance conventions. That
   is consistent with the project's standing prior: the failures are almost all "trust a stronger
   pointwise model more", and late interaction over a topically-trained model is still topical.

## What this does not show

- **Not that late interaction is wrong in general.** It is one 33M model, one corpus, one
  chunking. This is the first *trained* MaxSim head tested here — the earlier probe ran MaxSim
  over Laya's untrained token states and failed for a different reason
  (`make laya-maxsim-probe`), and that distinction still stands.
- **Not the +0.1254 ceiling itself.** The decomposition's ceiling is unchanged; what is falsified
  is that an off-the-shelf reranker reaches it. Phase 2.2 (a cross-encoder trained on bridge
  conventions) remains untried.
- **Not a normalisation finding.** The head is L2-normalised MaxSim. Against the model card's own
  example the absolute scale differs (its `similarity` normalises differently) but the *ordering*
  is identical, and a test pins that.

## Method notes

- **A quota is per document, not per chunk.** The first full run passed chunk ids where the
  project passes `document_of(chunk_id)`, so every chunk was its own article and the document
  quota never fired: covquota scored 0.3108 instead of **0.3280**. The corrected harness
  reproduces `fused` 0.2522, `ce` 0.2986 and `covquota-3-0.7` **0.3280** exactly — the project's
  own subset bars — which is what makes the other rows comparable. A test pins the grouping.
- **The checkpoint stores scores, not embeddings.** One JSON line per query of MaxSim scores,
  aligned to the pool. Changing an *arm's* logic therefore needs no re-encoding: the corrected
  re-run reused all 400 checkpoints and only recomputed the report.
- **Regenerate before comparing.** The baselines are recomputed from the cached pool rather than
  quoted from another report, so this table cannot inherit a stale number the way Phase 0 did.
