# Findings: a relational stance reranker does not fix ArguAna

ROADMAP Phase 2.3. ArguAna is the corpus where a pointwise ranker collapses: its gold is the
**counter-argument**, so an *agreeing* passage is topically near-identical and a pointwise
MS MARCO cross-encoder scores it high. The measured collapse is severe — `bge-reranker-base` over
the fused top-100 took ArguAna from 0.568 to **0.250**. `RESEARCH.md` §5.2 diagnoses the cause as
the absence of supervision for the *relation* between two texts, and prescribes scoring by
**contradiction minus entailment** with an NLI cross-encoder.

This tests that prescription. Both arms reorder the *same* dense pool, so any difference is
ordering alone.

Reproduce:

```sh
make arguana-stance          # 1406 queries, ~20 minutes on MPS
.venv/bin/python tools/arguana_stance.py --limit 40 --out /tmp/stance-smoke
```

## Result

Text-only `bge-small` dense pool of 20 per query, reordered by
`cross-encoder/nli-deberta-v3-base` scored as `P(contradiction) − P(entailment)`; nDCG@10 with
BEIR's `ignore_identical_ids`:

| arm | nDCG@10 | MRR@10 |
| --- | --- | --- |
| dense order (baseline) | **0.6467** | **0.5722** |
| stance order (same pool) | 0.3767 | 0.2874 |

Paired over the 1,406 queries:

| metric | delta | 95% interval | p |
| --- | --- | --- | --- |
| nDCG@10 | **−0.2700** | [−0.2921, −0.2474] | < 0.0001 |
| MRR@10 | **−0.2849** | [−0.3087, −0.2602] | < 0.0001 |

**The roadmap's own falsification criterion fires.** It reads: *if contradiction-only ordering
loses to BM25 there, the hypothesis is wrong*. BM25 on ArguAna is ~0.417; the stance order is
**0.3767**. So this is not a near miss — a relational NLI score is a worse ordering signal than
the dense similarity it was meant to correct, and worse than the lexical baseline the pointwise
cross-encoder was meant to beat.

## Why it may have failed

Recorded as hypotheses, not findings:

- **Truncation.** The NLI model takes 512 tokens; ArguAna arguments are 300–500 *words*, so the
  stance-bearing conclusion is often cut off. The roadmap flagged this risk before the run.
- **The relation is not sentence-level NLI.** "This argues against that" is a discourse relation
  over whole arguments, and a sentence-pair NLI classifier is a proxy for it at best.
- **Dense is already strong here.** 0.6467 is a high bar, and the stance score discards the
  topical signal entirely rather than combining with it. The measured fix may have to be a
  *blend* of dense and stance, not a replacement — which this experiment did not test.

## What this does not show

- **Not a like-for-like archival comparison.** The dense arm is recomputed here under this tool's
  settings and reads **0.6467**, against the archived `dense-text-only-fixed` row of 0.596. The
  two may differ in pooling or in how identical ids are dropped. The *paired* comparison — the
  falsification — is same-pool and same-metric, so the conclusion does not depend on that gap; the
  absolute column does.
- **Not a test of premise/hypothesis order, or of chunked-stance aggregation**, both of which
  could matter. Neither was tried.
- **Not a general claim about NLI for retrieval.** One model (`nli-deberta-v3-base`), one corpus,
  one pool depth (20). The engine's own `NLIAnswerability` hook remains a separate, gate-side use.
- **Not a recall claim.** Both arms see the same 20 candidates; only the order differs.

## Method notes

- **Restrict the qrels to the queries you ranked.** `evalkit.evaluate_rankings` averages over
  every query in the qrels, so a `--limit 8` smoke against the full 1,406-query qrels diluted the
  eight ranked queries to **nDCG@10 0.0045** — which reads like a broken retriever and was a
  broken harness. The tool now scores against the subset it actually ranked, and the same trap
  would silently corrupt any limited run in this repository.
- **Read the label columns from the checkpoint.** `cross-encoder/nli-deberta-v3-base` is
  contradiction/entailment/neutral, but a swapped pair inverts every score; the indices come from
  `config.id2label` and a test pins it against a permuted mapping.
- **Phase 2's second clean negative.** With Phase 2.1 this makes two falsified ordering heads
  (topical late interaction, relational NLI) against one strong baseline rule. The +0.1254
  ordering headroom is proving hard to reach from either direction, which strengthens the case
  that it needs a scorer trained on this task's own relevance conventions (Phase 2.2) rather than
  another off-the-shelf model.
