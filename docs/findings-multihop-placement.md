# R4's placement half — the MultiHop-RAG retrieval row, and a refusal that came first

`docs/best-in-the-world-2026-10.md` R4 asked for two things: the evidence-layer protocol (landed, five
axes, reproducible) and **generator-free leaderboard placements**. This is the second half, as a
command — plus the number it will show once the run that fills it exists.

## The command

```sh
python tools/engine_extractive.py --zero-thresholds \
  --index-dir eval/benchmarks/multihop-rag/index/engine-default \
  --out eval/benchmarks/multihop-rag/engine-extractive-full
make multihop-row MULTIHOP_ROW_REPORT=eval/benchmarks/multihop-rag/engine-extractive-full.json
```

`tools/engine_extractive.py` already reports exactly the four columns the benchmark's table carries
(`Hits@10`, `Hits@4`, `MAP@10`, `MRR@10`). `tools/multihop_row.py` places them beside the published
anchors, and refuses to place a subset.

## The refusal is the point

A report over 150 of the benchmark's 2,255 answerable queries looks exactly like a report over all of
them. The tool checks the query count and, unless `--allow-subset` is given for internal reading, prints:

```
**not placeable** — 150 queries, not the 2255 answerable records the table scores — this is a subset,
and a subset presented as the member is the failure class 11 records
```

with the command that would make it placeable. That is the eleventh class in
[`findings-measurement-failures.md`](findings-measurement-failures.md) — *a subset read as the member* —
turned into a gate rather than a footnote.

## What the subset shows anyway, marked as a subset

Because the numbers are worth seeing while the full run is pending:

| system | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| voyage-02 + rerank (published) | 0.747 | 0.663 | 0.480 | 0.586 |
| bge-large + rerank (published) | 0.718 | 0.636 | 0.476 | 0.563 |
| bge-large, no rerank (published) | 0.672 | 0.522 | 0.342 | 0.430 |
| **ours: engine + `bge-reranker-v2-m3`, 150-query subset** | **0.940** | **0.847** | 0.371 | **0.688** |

On three of four columns the engine exceeds the best published row and on `MAP@10` it trails by 0.11.
That is this repository's long-standing shape — **the gold is found and the order is where the score
goes** — now with the 2026 reranker having closed most of the `MRR@10` gap (0.688 against 0.586) while
`MAP@10` stays behind.

**None of it is placeable**, and the tool says so before printing it. The full-set run is ~3 hours at the
measured 736 s per 150 queries; it is the one remaining measurement R4 needs, and it is a scheduling
problem rather than a design one.

## A mid-flight check, because a two-hour run should not be trusted only at the end

The full-set row writes a resumable shard, so its first **135** rows can be summarised while it
runs. That is worth doing: a measurement that turns out to have used the wrong index, or to be emitting
empty answers, should be caught at 6% rather than at 100%.

| metric | 150-query subset (recorded) | first 135 of the full set |
| --- | ---: | ---: |
| answer nugget recall | 0.2060 | **0.1981** |
| citation-clean, all queries | 1.0000 | **1.0000** |
| abstention rate | 0.0000 | **0.0000** |
| Hits@10 | 0.9067 | 0.9333 |
| Hits@4 | 0.7400 | 0.8000 |
| MAP@10 | 0.3128 | 0.3278 |
| MRR@10 | 0.6152 | 0.6436 |

The run is healthy on the properties that would betray a misconfiguration — no empty answers, every
answer citation-clean at 5.00 sentences, no abstentions on an answerable-only set — and its quality tracks
the subset it was previewed on, a little better on every retrieval column.

**This is not a result.** It is 6% of the set, read before the run finished, on the arm whose number was
never in question. The registered prediction in
[`findings-extractive-coverage.md`](findings-extractive-coverage.md) is about the **delta between the two
composer objectives** on the full set, and this check says nothing about it: only one arm is running.

## A fairness problem found before the placement was made

The published table this row is meant to sit in scores **ten documents** per query. The engine returns
**chunks**, and this corpus chunks **609 documents into 4,592** — 7.5 chunks per document. So a chunk-level
row is answering a different question in the same metric's clothes: ten slots hold a fraction of one
document's text where the published rows hold ten whole ones.

The metric itself does not care — `official_metrics` matches gold facts by normalised substring against
whatever strings it is handed — which is exactly why the problem is invisible in the number. The direction of
the bias was not obvious either: chunks make each slot *smaller* (harsher for "does a gold fact appear in the
top ten"), while a chunk that contains the fact can outrank the document that contains it, because the
document's other chunks dilute it (kinder). A comparison whose bias direction cannot be signed is not a
comparison.

So the row now records **both**, and the pair is named in the artefact:

| field | question it answers |
| --- | --- |
| `Hits@10`, `MAP@10`, … | the system's own question: does a gold fact appear in the top ten **chunks** |
| `doc_Hits@10`, `doc_MAP@10`, … | the published table's question: does it appear in the top ten **documents** |
| `doc_passages` | how many distinct documents those ten chunks actually covered |

The collapse reuses the harness's own `ranking_from_passages`, which maps a returned path back to a corpus id
and de-duplicates in first-appearance order, so the ranking is preserved rather than re-sorted. **The row that
gets placed is the `doc_` one;** the chunk row stays because it is the system's own measurement and because
quoting a number without its document-level counterpart is what got this repository into the situation the
placement tool was written to prevent.

### The document-level row comes from the coverage arm, and that is valid

The rank arm running now writes the old schema — it started before the field existed — so its document-level
counterpart cannot be derived after the fact (the row stores answers, not document ids). **The queued coverage
arm runs the current code and will record it.**

Using the coverage arm's retrieval numbers for the placed row is legitimate rather than a compromise, and the
reason is worth stating: **retrieval is deterministic and identical between the two composer arms**. The
objective changes which sentences are *selected*, not which passages are *retrieved* — `multihop_official`'s
own docstring says re-running reproduces the same ranking exactly. So the coverage arm's `doc_*` metrics are
the rank arm's document-level retrieval, measured once.

`make multihop-row` now prefers `doc_*` when present and, when it is absent, prints that the row is **indicative
rather than placeable** instead of putting a chunk-level number in the same four columns.

**What this does not fix.** Chunking still differs — the published rows re-chunk at a 200-character overlap
and the engine derives chunk size from the corpus — so `doc_Hits@10` compares *documents retrieved by
different chunkers*, which is the closest this data allows and is stated rather than glossed.

## Why the anchors are quoted rather than merged into one sorted column

Three caveats travel with the row, and the tool prints all three:

- **Chunking differs.** The published anchors re-chunk with the official harness at a 200-character
  overlap; this engine derives chunk size from the corpus (2,000 characters here). Sorting the two groups
  into one column would imply the protocols match.
- **The paper's table carries no lexical baseline.** Every published row is a dense model, so a `Hits@10`
  advantage over `bge-large` is partly lexical retrieval the table never measured.
- **The engine's row adds two columns it owns** — citation validity and answer nugget recall — which is
  the entire argument for a retrieval table that admits a non-generative system at all.

## What is owed

- The full-set run. ~3 h, one heavy job, and the numbers above are its expectation at 15× the queries
  (they will move — a subset is not the member, which is why the tool refuses to quote one).
- A `--check` in CI once the full report exists, so the row cannot be edited into a claim.
