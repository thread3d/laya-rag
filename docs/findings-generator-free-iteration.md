# Generator-free iteration, Stage 1: the mechanism holds

The audit's Bet 1 is the largest claim in the programme: CoRAG's MuSiQue EM moves **17.4 → 27.7**
from one retrieval to six with the *same fixed retriever*, and this engine does not iterate. Its
first stage is separately falsifiable, and it is the cheap gate:

> if a second retrieval pass, driven by entities the first pass found but the query did not name,
> raises the pool's gold-fact coverage, the mechanism is real; if it does not, the bet closes
> without a controller, a generator or an LLM.

`tools/generator_free_iteration.py` (`make generator-free-iteration`) runs exactly that, with **no
model and no LLM**: BM25 (the shipped default's channel) retrieves a first pool of 100; from the top
of that pool it mines **capitalised entity runs absent from the query** (most frequent first, at most
4 per round); each probe retrieves again into the pool; coverage is the benchmark's own normalised
whole-fact substring test. A **shuffled-probe control** runs the identical procedure using another
query's probes, so a gain can be attributed to the bridge rather than to "more retrieval".

## The result

2,255 answerable queries over the 17,259-chunk corpus:

| iteration | real coverage | shuffled-probe coverage | complete | mean pool |
| --- | --- | --- | --- | --- |
| 0 | 0.8326 | 0.8326 | 0.6115 | 100.0 |
| 1 | **0.8937** | 0.8428 | **0.7277** | 326.7 |
| 2 | 0.8937 | 0.8428 | 0.7277 | 326.7 |

| comparison | delta | 95% CI | p |
| --- | --- | --- | --- |
| real expansion − baseline | **+0.0611** | [+0.0555, +0.0670] | 0.0000 |
| real − shuffled at the final round | **+0.0510** | [+0.0453, +0.0568] | 0.0000 |

**The mechanism is real, and it is the bridge.** One round of entity-probe expansion raises
gold-fact coverage by **+6.11 points** and lifts the share of queries with complete evidence from
**61.2% to 72.8%**; **+5.10 of those points survive the shuffled-probe control**, so the gain is the
conditioning, not the extra retrieval. The control is not zero (+1.02 points), which is the honest
reading: generic "search more" helps a little, and the bridge is five times that.

Two further readings fall out. **One round saturates** at these settings — iteration 2 moves nothing
— so a controller should run one expansion and stop, which makes the cost bounded. And the pool
**triples** (100 → 327), so the extra coverage is bought with a larger candidate set, which the
second stage then has to order.

## What this does not show

- **Lexical-only.** The shipped default's channel; the fused (dense + lexical) version is the same
  procedure with the dense channel added, not run here.
- **No answerability controller.** The audit's Bet 1 gates expansion on the head's decision; this
  measures the mechanism with a fixed one round, which is what Stage 1 asks.
- **Coverage is not MAP@10.** The coverage objective *falsified on MuSiQue* (−0.0109 MAP@10), so the
  open question is now whether this extra pool coverage translates into ordering — the next stage,
  not established here.
- **Probe mining is capitalised entities only.** "Noun chunks" and rarer selection rules are
  untried; one round already saturates, so a better probe set is the way to more.

## Verification

Six tests: probe frequency/query-exclusion/cap/minimum-length; the coverage definition; a synthetic
corpus in which the gold fact is reachable **only** through a probe entity (real reaches 1.0 while
the shuffled control stays 0.0); and query loading with the null-query filter and a limit. The full
run reproduces from cached data in ~5 minutes. `make test`: **1397 passed, 5 deselected**.
