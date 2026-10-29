# The CPU-only envelope, measured — and why the instrument needed fixing first

`docs/best-in-the-world-2026-10.md` R0. The claim this project wants to make is *fully local,
CPU-only, no API key* — and the latency it publishes is not from that envelope. The README's
*"the engine auto-selects MPS and answers a query in 2–5 s end to end"* is an accelerated number
from an Intel Mac. This is the same engine, same index, same queries, with the accelerator refused.

Two things landed, one of them an instrument and one of them a warning.

## The instrument

`cpu_seconds` now travels with every result — `retrieval_cpu_seconds`, `rerank_cpu_seconds`,
`total_cpu_seconds` in `QueryResult.timings` and in the HTTP `Timings` model — and so does
`device`, the device the decision model **actually** ran on rather than the one that was asked
for. `DEVICE=auto` is a request; a CPU-only claim needs the answer, so `LayaModel.device` reads the
resolved module back (`next(agent.model.parameters()).device`) and `BiEncoderModel.device` returns
the value it loaded with. `tools/cost_report.py` now records both clocks per query, the thread
count, and the resolved device, and says out loud when a row was taken on an accelerator.

The reason for two clocks is not completeness. Wall clock is what a caller waits and it is
contaminated by whoever else is using the host; CPU time is the work the process did. This
repository has already withdrawn three speed claims that died on `seconds` (`CHANGELOG.md`, 1.2.1),
and `tools/engine_beir.py` has recorded `cpu_seconds` since.

## The measurement

`LAYA_RAG_DEVICE=cpu`, 15 decision passes over 5 fixed queries × 3 repeats, `RETRIEVE_K=20`,
`LAYA_RAG_TORCH_THREADS=16`, the 345-chunk corpus, the multilingual checkpoint:

| quantity | value |
| --- | ---: |
| resolved device | **cpu** |
| wall per query — median / mean / max | **6.71 s / 7.32 s / 9.40 s** |
| CPU per query — median / mean / max | **96.3 s / 105.9 s / 134.0 s** |
| CPU per candidate — mean | **5.29 s** |

So the CPU-only answer to a query costs **about 7 seconds of waiting and about 96 seconds of CPU
time**, at 16 torch threads, on a 56-core host with no accelerator. The README's 2–5 s is the same
engine with MPS; the CPU-only envelope is roughly **2–3× the wall clock** and, because the decision
pass fans out across threads, an order of magnitude more CPU time than either.

**CPU time is a thread-sum, not a wall-clock equivalent.** 16 threads × 7 s ≈ 112 s of process
time, which is what `process_time()` reports by design. This is the first thing a reader must know
before comparing these numbers to anything: they are only comparable at a **fixed** thread count,
and `torch_threads` is recorded beside them for that reason. A "CPU-seconds per query" quoted
without its thread count is not a measurement.

## The warning: the falsifier fired, twice

R0 stated its own falsifier in the assessment: *if CPU-seconds per candidate still varies by more
than ±5% across repeats at fixed depth, the host is not a valid instrument; say so and defer the
cost axis rather than publish a jitter number.*

Per query, the repeat-to-repeat spread of CPU time at fixed depth was:

| query | spread (max − min) / min |
| --- | ---: |
| answerability gate | 0.5% |
| chunk sizes | 0.7% |
| SciFact nDCG | 5.8% |
| cross-encoder cost | 7.4% |
| false-accept rate | **8.1%** |

Three of five exceed the ±5% bar. **The falsifier fires: this host cannot support a published
cost number at 8% resolution, and the honest response is to publish the spread beside the median
rather than to publish the median alone.**

The second firing is more instructive, because it is the same failure the project has paid for
three times already. An earlier probe in the same session measured `RETRIEVE_K=20` at **15.83 s**
on 32 threads; this one measures **6.71 s** (median) on 16 threads. Neither configuration is
10× faster than the other — the first run happened while two research agents were fetching web
pages concurrently, and the second after they finished. Same engine, same index, same query
shapes: **a factor of 2.4 between two readings of the same work**, entirely from co-tenancy.

## What this changes

1. **The published latency claim must name its device.** "2–5 s per query" becomes "2–5 s on MPS;
   ~7 s waiting / ~96 s CPU per query at 16 threads on a 16-core-share host with no accelerator",
   and the second is the sentence that supports "CPU-only".
2. **Every cost row carries `device`, `torch_threads`, and the observed spread.** The report writes
   all three; a row missing any of them is not comparable to another row.
3. **The cost axis is not deferred — it is downgraded to medians with intervals.** The R0 falsifier
   asked whether to publish at all; the answer is that 8% is enough resolution to compare a
   `RETRIEVE_K=10` cascade against `RETRIEVE_K=40` (a 2× difference) and nothing like enough to
   compare 3% variants. That is a usable instrument for R6 and a useless one for micro-benchmarks,
   and saying which is the point.
4. **The decision pass is the whole cost.** 5.29 s of CPU per candidate against 0.03 s for
   retrieval: at 20 candidates the second stage is >99% of the work. R6's cascade has an obvious
   target and no ambiguity about where to look.

## Reproducing

```sh
LAYA_RAG_MODELS=<checkpoints> LAYA_RAG_INDEX_DIR=<index> LAYA_RAG_DEVICE=cpu \
LAYA_RAG_TORCH_THREADS=16 PYTHONPATH=src python3 .scratch/envelope_probe.py
```

The probe writes `.scratch/envelope-probe.json` (per-query rows plus the summary above) and the
`instrument_valid` flag computed from the falsifier, so a later run on a quiet host can say whether
its own reading is trustworthy instead of assuming it.

## The tail, found by accident while checking a run

Investigating what looked like a stalled full-set row turned up something this envelope should carry:
**per-query latency has a ~10x tail on that corpus**, at the same engine settings the medians above were
measured with.

| query | wall, 4 threads |
| --- | ---: |
| typical MultiHop-RAG query | ~3 s |
| `answerable[751]` — "Between the Sporting News report on NBA betting sites…" | **29.2 s** |

The row was **not** stalled — 25 rows completed in the two minutes after the probe — but the observation
stands, and it is exactly what a median hides. A capacity plan built on 6.71 s would be wrong on p95 and
worse on the tail, which is why `tools/cost_report.py` reports `p95` beside the median and why this document
quotes a spread rather than a point for the R0 falsifier.

The mechanism is visible in the number: the decision pass reads the whole evidence block, so a query whose
top passage is long costs more, and nothing in the engine bounds that ratio.

**What it does not establish.** One slow query is not a distribution. The honest statement is that the tail
exists, that its cause is state length, and that a `--limit`-free row is where p95 for this corpus would come
from if someone needs it. The check itself is worth recording as a practice: **a run that appears stuck is
usually a slow query, and the way to know is to reproduce the one query rather than watch the counter.**

## What this does not claim

- It does not claim the CPU-only envelope is *slow in general*: this host's 56 cores are shared, the
  index is small (345 chunks), and the query mix is documentation prose. A different corpus changes
  the decision-pass cost linearly in candidate length, not in corpus size.
- It does not claim the GPU/MPS comparison: no accelerator was available to measure, so the
  README's 2–5 s figure is quoted as published and not re-derived here.
- It does not claim a stable absolute number for anything. The medians are the medians of one
  session on a busy host, with the spread stated, which is the strongest claim the instrument
  supports today.
