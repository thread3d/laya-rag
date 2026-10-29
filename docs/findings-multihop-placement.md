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
