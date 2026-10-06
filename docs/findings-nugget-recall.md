# An LLM-free nugget / claim-recall protocol

The evaluation brief (`docs/research/07` §5) asks for a nugget protocol with **no API and no
generator**: nuggets derived programmatically from the benchmark's gold support sentences and
measured against the retrieved evidence. It fills a real hole. The benchmark's headline MAP@10 is a
*single-gold-set* metric — it does not say how much of each gold fact a ranking actually contains —
and the repository's own coverage objective raises Hits@4 but not MAP@10. Claim recall separates
those.

`tools/nugget_recall.py` (`make nugget-recall`) implements it over what is already on disk: the
cached fused pool and cross-encoder scores (`ce-scores-full.json`), re-chunking `corpus.json` with
the harness's own 256-token window so chunk ids line up exactly (`17,259` chunks, matching the
report's own `corpus_chunks`).

## Nuggets and coverage

A **nugget** is extracted from a gold fact without a model:

- **numbers** — digit runs; **dates** — month names;
- **entities** — capitalised word runs of one to three tokens;
- **key terms** — content words of four or more characters that are not stopwords.

A nugget is **covered** when it appears in a retrieved chunk — numbers, dates and entities as the
benchmark's own whitespace-normalised substring match, key terms as a token. A query's claim recall
is the fraction of its gold facts' nuggets covered anywhere in the arm's top-k; the reported value
is the macro mean over queries.

## The measurement

| depth | fused claim-recall | cross-encoder claim-recall |
| --- | --- | --- |
| @10 | **0.7438** | **0.7803** |
| @100 | 0.9676 | 0.9676 |

Two readings:

1. **The cross-encoder's ordering gain is +0.0365 claim recall at 10** on the same pool — it is not
   only a MAP effect. Reordering the same 100 candidates puts measurably more of each gold fact into
   the top 10.
2. **The pool covers 96.8% of nuggets at depth 100**, and the two arms are identical there (as they
   must be — the CE only reorders). So the top-10 shortfall is **ordering, not recall**, which
   independently reproduces the diagnostics conclusion with a metric that is not single-gold.

## What this does not show

- **Only two arms.** Per-query rankings are not persisted for the coverage/quota rules, so this
  cannot yet score `covquota-3-0.7` or the document-quota arms; the comparison isolates the second
  stage only.
- **Depth 100, no R@1000.** The cached pool is 100 deep.
- **Nugget presence, not entailment.** A covered term is present, not necessarily used correctly;
  this is a recall proxy with no judge, which is the point, but it is a proxy.
- **The extraction thresholds are choices** (entity run length, term length, stopword list) and were
  not tuned; they are stated here so a different choice is a different number.
