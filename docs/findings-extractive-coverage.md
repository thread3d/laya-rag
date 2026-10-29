# R2 — the extractive answer gets an objective

`docs/best-in-the-world-2026-10.md` R2. The measured row (`findings-extractive-row.md`, 2,255
MultiHop-RAG queries) has two halves. The first is perfect: **citation clean 1.0000**, 0 of 9,806
sentences uncited, mechanically checked. The second is the cost: **answer nugget recall 0.1906**
against a pool carrying **0.7438** of the same gold content. Three quarters of the facts in the
evidence never reach the answer.

That gap was not a model problem. `compose_extractive` selected sentences by **their parent
document's retrieval score**, deduplicated by text, and stopped at five. It had no objective of its
own, so the sentence it kept was the first one in the first passage.

## What changed

`compose_extractive(..., coverage=True)` and `LAYA_RAG_EXTRACT_COVERAGE=1` select by a submodular
max-coverage objective over the same candidate sentences, under the same guarantees:

| term | what it rewards | why |
| --- | --- | --- |
| **new facts** | numbers, dates, ordinals, percentages, acronyms, proper nouns not yet used | a gold nugget *is* one of those; a sentence about the right subject with no number in it cannot answer a question about a number |
| **new content** | content terms not yet covered, at a quarter weight | so five sentences of one passage cannot fill a five-sentence budget |
| **parent relevance** | the retrieval score the sentence inherits, at half weight | a fact-dense sentence from an irrelevant document must not outrank a fact-dense relevant one |
| **redundancy guard** | rejects a candidate whose content vocabulary overlaps an already-selected sentence by Jaccard ≥ 0.7 | see below |

Query terms are deliberately **not** the coverage unit. Every candidate comes from a passage that was
retrieved for the query, so all of them contain its terms and covering them again measures nothing.
An earlier probe of this file's selection used query terms as the unit and showed the shipped order
covering 0–5 where greedy coverage covered 2–7 (`.scratch/extractive-probe.json`); that was a symptom
of the missing objective, not the target.

The guarantees are untouched: every emitted sentence is still a verbatim span of a cited passage, the
marker is still in range and still points at the document it quotes, and an empty evidence set still
abstains. `tests/test_extractive.py` pins each of those under `coverage=True`.

**One deliberate visible difference:** markers can appear out of numerical order, because a
fact-dense sentence from the second document may be selected first. The numbers are *not* renumbered —
`[n]` has to keep meaning the same document here, in the composed prompt and in `verify_citations`,
and renumbering would make the answer agree with itself and disagree with the evidence it is checked
against.

## A defect the objective found in itself

The first version had no redundancy guard, and a test built to prove near-duplicates could not fill
the budget failed:

```
[1] The figure was 42 percent again. [1] The figure was 42 percent.
```

Both sentences were selected, because the novelty term is permissive: the second adds one token
("again") and therefore scores above zero. Five such sentences carry one fact between them — which is
exactly the failure the objective exists to remove, reproduced by the objective. The Jaccard guard
fixes it, and the case is now a test rather than a comment.

## Measured on the real corpus, real weights

The docs corpus, the multilingual checkpoint, `RETRIEVE_K=20`, five-sentence budget. "Facts" is the
count of distinct fact-bearing tokens in the emitted answer; "docs" is how many documents the answer
draws on:

| query | rank order: facts, docs | coverage: facts, docs |
| --- | --- | --- |
| SciFact nDCG@10 | 4, **1** | **14, 3** |
| how the answerability gate abstains | 1, 1 | **7, 2** |
| which chunk sizes are chosen | 8, 1 | 8, 1 |

Two of three queries go from one document and a handful of facts to two or three documents and 3.5× to
7× the facts. The third is unchanged in density — the rank order already happened to pick the
fact-dense sentences there — but the *selection* differs, so it is not a no-op.

**This is a mechanism, not an effect size.** Fact-token count is not nugget recall: the published
0.1906 is measured against gold nuggets derived from MultiHop-RAG's support sentences, and only that
row can say whether the objective converts. What the probe establishes is the thing that was
previously unknown — that the shipped selection is blind to fact content, and that an objective costs
nothing to give it.

## The measured row: **0.2060 → 0.2610**, significant, and still short of the prediction

MultiHop-RAG downloaded from HuggingFace (2,556 records, 609 documents), the engine's own corpus built
from it — **4,592 chunks**, which is the figure the repository documents for `engine-default`, so the
index is the one every other row in this repository was measured on. 150 answerable queries,
`--sample-seed 17`, gates zeroed so the row is comparable with the published retrieval rows:

| arm | answer nugget recall | sentences per answer | citation clean |
| --- | ---: | ---: | ---: |
| rank order (shipped) | **0.2060** | 5.00 | **1.0000** |
| coverage objective | **0.2610** | 5.00 | **1.0000** |
| paired delta, 150 queries | **+0.0549** [+0.0240, +0.0841], p=0.0005 | | |

Each row now records `extract_coverage` in its settings block, so which arm produced it is readable from
the artefact rather than from its filename — a gap the rows above were written before closing, and one of
the five provenance failures catalogued in
[`findings-measurement-failures.md`](findings-measurement-failures.md).

Per query: **107 improved, 37 worse, 6 unchanged**. Citation validity is unchanged at 1.0000 — the
guarantees held, which is the property that had to survive a change of objective.

**The volume confound is checked and absent.** Both arms averaged exactly 5.00 sentences per answer
(the cap), so the gain cannot be "it emitted more words". The claim counts differ (653 against 727)
because the citation checker splits differently, which is what made the first read look like a volume
effect; the sentence counts are the ones that matter.

## The prediction was too optimistic, and the falsifier fired

R2 recorded, before the run: **answer nugget recall 0.1906 → 0.35–0.50**, falsified if the rise were
**below 0.10** at unchanged citation validity, which would mean selection is not the binding
constraint.

The observed rise is **+0.055**, which is half the bar. So:

- the objective is **real**: significant at p=0.0005 with a paired interval excluding zero, on the
  engine's own corpus, with the guarantees intact;
- the **prediction was wrong** by roughly a factor of two, and the falsifier as written fires: the
  selection is *a* binding constraint, not *the* binding constraint.

Where the rest of the gap lives is now a sharper question than it was. The answer carries
**0.2610** of the gold nuggets while the *pool* carries **0.7438**, and a five-sentence extract cannot
hold a three-fact answer however well it selects: the budget, not the ranking, is the obvious next
constraint. That is a testable claim — raise `max_sentences` and see whether coverage keeps its
advantage or both arms converge — and it is the next experiment, not an assumption.

## A harness bug that produced two identical arms, twice

The first attempt at this comparison ran both arms with `EXTRACT_COVERAGE=1`, which does nothing: the
repository's environment variables carry the `LAYA_RAG_` prefix, so the correct spelling is
`LAYA_RAG_EXTRACT_COVERAGE`. Both arms therefore composed identically — 0.2060, 653 claims, both — and
the identical numbers were the only signal. That is the third time this session that an arm matching
its control exactly has meant a bug rather than a null.

Two things came out of it. The run was redone with the prefixed name. And `engine_extractive.py` now
puts `LAYA_RAG_EXTRACT_COVERAGE` into its **resume fingerprint**: the composer's objective is a
configuration of this row, so a resumed run that changed only that variable used to skip every
completed row and silently report the other arm's numbers.

## Checked against real weights, through the shipped verifier

The API test for the per-request objective runs against the stub model, which proves the plumbing and says
nothing about whether the objective changes what a *real* model emits. This does, on the documentation
index with the multilingual checkpoint, both arms per query, with every answer passed through
`verify_citations`:

| query | sentences | selection differs | rank arm clean | coverage arm clean |
| --- | ---: | --- | --- | --- |
| SciFact nDCG@10 | 5 / 5 | **yes** | yes | yes |
| how the answerability gate abstains | 5 / 5 | **yes** | yes | yes |
| which chunk sizes are chosen | 5 / 5 | **yes** | yes | yes |

Three of three selections differ, and **both arms are citation-clean on every query** according to the
same verifier the end-to-end row uses. That is the property that had to hold: a different selection is
only acceptable if the guarantees survive it, and they do under the real checkpoint rather than only under
the stub.

## Registered before the runs: the two composer measurements still owed

The lock is held by the full-set retrieval row, so these are registered now and measured when it frees. A
prediction written after the run is a description; these are written while the answer is unknown.

### (a) The full-set composer row — does +0.055 hold at 2,255 queries?

**What it is.** The `0.2060 → 0.2610` delta above comes from a **150-query** subset. The full answerable set
is 2,255 queries, and the full-set *shipped-default* row is already running.

**Prediction.** The paired delta lands between **+0.03 and +0.08** — that is, within ±0.025 of the 150-query
estimate — with both arms' cite-clean rates at 1.0000.

**Falsifier.** If the delta falls **below +0.02**, the 150-query subset was unrepresentative and every number
in the table above has to be re-stated as a subset result with its own interval. If it **exceeds +0.09**,
the subset understated it and the same re-statement is owed in the other direction. Either way the
finding's headline changes shape rather than merely its digits.

**Cost.** ~12 min of index reuse plus one pass over 2,255 queries; the coverage arm is the same order.

### (b) The sentence budget — is five sentences the next constraint?

**What it is.** R2's falsifier already established that selection is **a** binding constraint rather than
*the* one: changing the objective recovers 0.055 of the 0.538 gap to the pool's 0.7438. The obvious
remaining constraint is the budget itself — `max_sentences` defaults to **5**.

**Prediction.** Sweeping 5 → 8 → 10 raises answer nugget recall monotonically but with sharply diminishing
returns: **8 sentences buys under +0.04 over 5**, and by 10 the marginal gain is under +0.01 per sentence.
Cite-clean stays at 1.0000 throughout, because every sentence is still a verbatim span.

**Falsifier.** If 8 sentences buys **more than +0.08**, the budget is *the* dominant constraint and R2's
conclusion — that selection is one of several — is wrong; the composer should ship a larger budget and the
finding must say the objective was the smaller lever. If cite-clean drops below 1.0000 at any budget, the
guarantee that makes this engine quotable would be budget-dependent, which would be the more serious
result of the two.

**Cost.** Three arms × 150 queries ≈ 40 min at the measured 736 s/150.

Both are registered with the measurement that would overturn them, and neither has been run.

## Still owed

- The 400-query and full 2,255-query rows: 150 queries is a subset, and the published row's 0.1906 is
  the full set. The direction and significance are established; the absolute levels are not the
  published ones.
- The `max_sentences` sweep, which is where the remaining 0.2610 → 0.7438 gap most plausibly lives.
- The 12-feature answerability gate as a `min_score` source: a sentence's parent relevance is the
  retrieval score today, and a calibrated *answerability* term would be a different and possibly
  better weight. Not measured, and not assumed.
