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

| iteration | real coverage | shuffled coverage | real ceiling MAP@10 | shuffled ceiling | complete | mean pool |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8326 | 0.8326 | 0.5955 | 0.5955 | 0.6115 | 100.0 |
| 1 | **0.8937** | 0.8428 | **0.6203** | 0.5994 | **0.7277** | 326.7 |
| 2 | 0.8937 | 0.8428 | 0.6203 | 0.5994 | 0.7277 | 326.7 |

| comparison | delta | 95% CI | p |
| --- | --- | --- | --- |
| real expansion − baseline (coverage) | **+0.0611** | [+0.0555, +0.0670] | 0.0000 |
| real − shuffled at the final round (coverage) | **+0.0510** | [+0.0453, +0.0568] | 0.0000 |
| real expansion − baseline (ceiling MAP@10) | **+0.0248** | [+0.0222, +0.0277] | 0.0000 |
| real − shuffled at the final round (ceiling MAP@10) | **+0.0210** | [+0.0183, +0.0237] | 0.0000 |

**The mechanism is real, and it is the bridge.** One round of entity-probe expansion raises
gold-fact coverage by **+6.11 points** and lifts the share of queries with complete evidence from
**61.2% to 72.8%**; **+5.10 of those points survive the shuffled-probe control**, so the gain is the
conditioning, not the extra retrieval. The control is not zero (+1.02 points), which is the honest
reading: generic "search more" helps a little, and the bridge is five times that.

**And part of it is ordering headroom, not just a bigger set.** The pool's oracle **ceiling MAP@10
rises 0.5955 → 0.6203 (+0.0248)**, of which **+0.0210 survives the control** — so about **40%** of
the coverage gain is reachable MAP@10 for a ranker that can order the larger pool, and almost all of
that headroom is the bridge. The baseline ceiling (0.5955) sits just under the diagnostics' fused
depth-100 ceiling (0.6088), as it should for the lexical-only channel.

Two further readings fall out. **One round saturates** at these settings — iteration 2 moves nothing
— so a controller should run one expansion and stop, which makes the cost bounded. And the pool
**triples** (100 → 327), so the extra coverage is bought with a larger candidate set, which the
second stage then has to order.

## Where the gain comes from

The benchmark labels each query's type, and the breakdown is the strongest evidence that the gain is
the bridge rather than generic extra retrieval:

| question type | queries | coverage 0 → final | ceiling MAP@10 0 → final |
| --- | --- | --- | --- |
| **inference** | 816 | 0.7824 → **0.8891** (+0.1067) | 0.5115 → **0.5488** (+0.0373) |
| temporal | 583 | 0.8273 → 0.8805 (+0.0532) | 0.6112 → 0.6389 (+0.0277) |
| comparison | 856 | 0.8842 → 0.9071 (+0.0229) | 0.6648 → 0.6758 (+0.0110) |

**Inference queries gain five times what comparison queries gain.** A comparison query names both
entities it compares, so the first pool already contains what a probe would search for; an inference
query asks for something the query does not name, which is exactly the bridge-hop case — and it is
also the *hardest* type (baseline ceiling 0.5115). Temporal questions sit in between. This is the
mechanism behaving as the BridgeRAG reading predicts, and it says where a controller should fire:
on the type with the largest headroom, not uniformly.

## What this does not show

- **Lexical-only.** The shipped default's channel; the fused (dense + lexical) version is the same
  procedure with the dense channel added, not run here.
- **No answerability controller.** The audit's Bet 1 gates expansion on the head's decision; this
  measures the mechanism with a fixed one round, which is what Stage 1 asks.
- **The ceiling is a bound, not an achieved MAP@10.** The **+0.0248** is what an *oracle* ordering of
  the larger pool could reach; no ranker has captured it here, and the coverage objective itself
  falsified on MuSiQue (−0.0109 MAP@10). Whether a selector realizes the headroom is the next stage.
- **Probe mining is capitalised entities only.** "Noun chunks" and rarer selection rules are
  untried; one round already saturates, so a better probe set is the way to more.

## Verification

Six tests: probe frequency/query-exclusion/cap/minimum-length; the coverage definition; a synthetic
corpus in which the gold fact is reachable **only** through a probe entity (real reaches 1.0 while
the shuffled control stays 0.0); and query loading with the null-query filter and a limit. The full
run reproduces from cached data in ~5 minutes. `make test`: **1397 passed, 5 deselected**.
