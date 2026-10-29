# R3 — can the answerability gate travel? The protocol, the pilot, and why the pilot cannot answer

`docs/best-in-the-world-2026-10.md` R3. The abstention claim is the one this architecture should own —
SemEval-2026 Task 8 reports that *"abstention is a differentiating capability not correlated with
overall generation quality"* — and it is the one with the weakest evidence in this repository. The
gate reads held-out AUC **0.8146** in domain (0.8290 with the twelve-feature arm) and, out of domain,
the shipped scalar is at chance: **0.5087** AUC, risk@80 **0.4927**.

No benchmark in the 2026 landscape grades a *retrieval-only* answerability predictor with a held-out
AUC, so the transfer question has to be instrumented here or not asked at all. This delivers the
instrument and its first two-corpus run, and the run's honest verdict is that the sample is too small
to answer the question — a fact the protocol itself produced rather than a caveat added afterwards.

## What landed

`tools/gate_transfer.py`: fit the gate on one corpus's feature rows, score another's, and report three
arms — in-domain, transfer, and a **shuffled-label control** — with `risk@80` (the false-accept rate
when the most confident 80% are accepted) beside every AUC. Rows are any run that records
`extract_features` output, so the tool does no retrieval and no model calls and is cheap to re-run as
rows accumulate. Two feature subsets: `shape` (max passage coverage, coverage spread, pairwise
redundancy — the three that need **no decision-model call**, so negatives can be mined at corpus
scale) and `all` (the thirteen the shipped gate uses).

Seven offline tests pin the protocol's own properties: a gate *can* transfer, a shifted corpus *can
fail to*, the shuffled control sits at chance on synthetic data, a single-class training set is
refused, and a missing feature is named in the error. A gate that could only succeed would be a
checker nobody could trust.

## The pilot

Two corpora deliberately unlike each other: **A** is this repository's own documentation (78 findings
and design files, 345 chunks), **B** is the Laya project's user documentation (40 files, 346 chunks).
Queries came from both directions, so each corpus carries positives and negatives:

| corpus | positive queries | negative queries |
| --- | --- | --- |
| A | 10 questions written from A's document titles | 9 Laya documentation questions |
| B | 7 labelled Laya documentation questions (`evalkit.docs_queries()`) | 10 of A's questions |

**What those negatives are, and are not.** They are *topic-shift* negatives: the question is about a
domain the corpus does not cover. They are **not** MultiHop-RAG's hard negatives — questions the
corpus discusses but cannot answer — which are the ones that reach the answerability gate in
production, and which the repository's own diagnostics show are the only negatives that ever do
(`README.md`, "Known limits"). A transfer number measured on topic-shift negatives is the easiest
version of the question, and the pilot below is labelled accordingly.

## The result: not decidable at this sample, and the null says why

| arm | features | transfer AUC | null median | null p95 | p |
| --- | --- | ---: | ---: | ---: | ---: |
| A → B | shape | 0.5143 | 0.4857 | 0.8150 | 0.44 |
| B → A | shape | 0.8111 | 0.5556 | 0.8667 | 0.16 |
| A → B | all 13 | 0.5571 | 0.4857 | 0.8157 | 0.37 |
| B → A | all 13 | 0.8111 | 0.5000 | 0.7444 | 0.015 |

The four cells are a family, so the repository's own rule applies — Benjamini–Hochberg over four
tests turns the best p of 0.015 into roughly 0.06, which is not a result. But the more useful number
is the **null p95**: at 19 training rows, permuting the training labels produces an apparent
cross-corpus AUC of **0.82** one time in twenty. The observed 0.811 sits *under* that. It is
consistent with noise, and so is 0.51.

The shuffled control makes the same point from the other side, and it is the diagnostic worth
keeping: the B → A arm read **0.811** with its own shuffled control at **0.822** — the treatment and
its control identical to two decimal places. That is the shape this repository has already learned to
distrust (two arms equal to four decimals once meant a harness was reading the wrong field). Here it
means the statistic, not the gate, is doing the work.

A plain confidence interval agrees: with 7 positives and 12 negatives, AUC 0.811 has a standard error
of 0.112 and a 95% interval of **[0.59, 1.00]**; the A → B arm's 0.514 has **[0.23, 0.80]**. The two
intervals overlap almost completely.

**What the pilot does establish:**

1. The protocol works end to end on real feature rows produced by the real engine and the real
   checkpoint, and it refuses to flatter itself — every arm carries its control and its null.
2. At ~18 rows per corpus, this design cannot answer the question, and no reading of the four cells
   above should be quoted as a transfer result in either direction. That is a fact about the sample.
3. The three **shape** features are usable without any decision-model call, which is the
   precondition for mining negatives at the scale the next attempt needs.

**A method note with a false start in it.** A first attempt at a power curve resampled the 19 rows
with replacement to n = 40…320 and re-ran the null. It is not informative and is not quoted: with 19
distinct rows, resampling produces duplicates, the model fits the duplicates, and the null stays
0.72–0.87 wide at every n — measuring the copying, not the sample size. A power statement needs
independent rows, which means running the corpora, not resampling them.

## What would settle it

The labelled sample has to come from the corpora that carry the claim, not from two documentation
sets:

- **MultiHop-RAG's 301 `null_query` records**, which the benchmark's own retrieval metric excludes and
  whose gold answer is literally "Insufficient information" — a real negative class, on a corpus the
  repository already measures.
- **MTRAG-UN's** 97 unanswerable and 78 underspecified turns.
- Mined cross-corpus negatives as an *additional* training signal (a question from corpus A asked of
  corpus B), never as the only negative class.

That is ~300 labelled negatives rather than 9, an order of magnitude more than the pilot, and it is a
download plus a run rather than a design problem. The prediction and falsifier in R3 stand unchanged;
what changes is that the instrument they will be read through now exists, and has been shown to
catch a false positive on its first outing.

## What this does not claim

- It does not claim the gate does or does not transfer. Neither arm is decidable at this n, and the
  null distribution says so explicitly.
- It does not claim anything about hard negatives. The pilot's negatives are topic shifts.
- It does not claim the shape features are sufficient; it claims only that they are *free*, which is
  what makes a corpus-scale mining run possible.
