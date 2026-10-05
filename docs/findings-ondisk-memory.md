# Findings: where the on-disk lexical build's memory actually goes

Evidence for ROADMAP 4.1, and it **corrects the roadmap's target**. The plan says the residual wall
for 10M+ document corpora is *"a bounded external merge for the term dictionary"*. Measured
independently on its two axes, the term dictionary is **not** what dominates: peak build RSS is flat
in vocabulary to 400,000 terms and grows with **documents**. The item stands — an unbounded
component is real — but the fix should aim at per-document state, not the dictionary.

Reproduce:

```sh
make ondisk-scale                       # the two-axis sweep
.venv/bin/python tools/ondisk_scale.py --docs 100000 --vocab 50000 --tokens 100 \
    --block-tokens 4000000              # one point, in its own process
```

Each point runs in its **own subprocess** because `ru_maxrss` is a high-water mark: several builds in
one process would report the largest peak for every point. The vocabulary is a *parameter* of the
synthetic corpus rather than a byproduct, which is the only way to hold one axis fixed.

## The two axes, 100 tokens per document

**Documents, vocabulary fixed at 50,000:**

| docs | terms | build s | index MB | peak RSS MB |
| --- | --- | --- | --- | --- |
| 25,000 | 50,000 | 12.2 | 22.0 | **312.4** |
| 50,000 | 50,000 | 25.2 | 42.4 | **417.9** |
| 100,000 | 50,000 | 51.4 | 83.2 | **596.3** |

Roughly **+3.8 KB per document**, linear.

**Vocabulary, documents fixed at 50,000:**

| vocab | terms | build s | index MB | peak RSS MB |
| --- | --- | --- | --- | --- |
| 10,000 | 10,000 | 24.8 | 40.9 | 375.7 |
| **50,000** | 50,000 | 25.2 | 42.4 | 417.9 |
| 100,000 | 100,000 | 26.0 | 44.1 | **391.0** |
| 400,000 | 400,000 | 26.7 | 54.6 | **421.2** |

Flat, and non-monotonic — 100k terms costs *less* than 50k. **An eightfold vocabulary increase moves
the peak by 0.8%.** The dictionary is not the wall at this scale.

## The knob that already exists

`block_tokens`, at a fixed corpus (100k documents, 50k vocabulary):

| block_tokens | index MB | peak RSS MB |
| --- | --- | --- |
| 250,000 | 83.2 | **399.9** |
| 1,000,000 | 83.2 | **593.5** |
| 4,000,000 | 83.2 | **808.1** |

A 16× range in the token cap moves peak RSS by 408 MB with an identical index, so a large part of the
peak is the block and is already tunable. Whatever 4.1 does, it must be measured at a stated
`block_tokens` or it is not a comparison.

## The unbounded part is real, and it is not the dictionary

At **fixed** `block_tokens`, peak RSS still grows with the corpus (312 → 418 → 596 MB), even though
each block holds the same number of documents. And with the cap scaled to hold the whole corpus in
**one** block, it grows further:

| docs | block_tokens | peak RSS MB | index MB |
| --- | --- | --- | --- |
| 25,000 | 2,500,000 (one block) | 491.4 | 22.0 |
| 100,000 | 10,000,000 (one block) | **1,031.9** | 83.2 |

So there is a component that scales with the corpus and with neither the vocabulary nor the block
count. **I have not separated which structure it is** — the candidates are the accumulating
`ids: list[str]` and `lengths: list[int]`, the merge-phase buffers, and allocator retention across
many blocks — and the next step is to profile the build rather than to guess. That is stated as an
open question, not papered over.

## What this changes

- **The roadmap's 4.1 framing is corrected.** The target is the per-document state and the merge, not
  a term-dictionary external merge. The item stays open; only its aim changes.
- **The target is measurable.** ~3.8 KB per document is what a fix has to beat, and the sweep is
  reproducible.
- **A stated caveat.** Synthetic documents of uniform length with a uniform vocabulary are not a
  natural corpus; a Zipfian vocabulary would make the dictionary term larger than it is here, though
  the 400k-term point bounds how much.

## The profile: the peak is transient, not the accumulators

The sweep said the cost tracks documents and not vocabulary. It did not say *which* structure, so the
build was profiled with `tracemalloc` at 100,000 documents, `block_tokens=1M`:

| reading | value |
| --- | --- |
| **live Python allocations when the build finished** | **7.4 MB** |
| peak RSS | 942.8 MB (inflated by tracemalloc's own overhead) |
| largest live site | `lexical.py:72` — the tokenizer's stopword set, **2.74 MB** |
| next two | `ondisk.py:132/133` — the term dictionary and its document frequencies, **3.3 MB together** |
| the rest | under 0.5 MB each |

**The accumulators are negligible.** The vocabulary, its document frequencies, the id list and the
length list together are a few megabytes at 100,000 documents — against a peak approaching a
gigabyte. Whatever costs the memory is *transient*: held during a block or the merge and freed
before the build returns.

Combined with the sweep, the two transient contributors are:

1. **The block, which `block_tokens` already controls** — 399.9 → 808.1 MB across a 16× range of the
   cap at a fixed corpus, and 491.4 → 1,031.9 MB when the cap is scaled to hold a whole corpus in one
   block.
2. **Retention across blocks** — peak still rises with the corpus at a *fixed* cap (312 → 418 → 596 MB
   for 25k → 50k → 100k documents), where the block size is constant and the persistent state is
   ~7 MB. More blocks means more freed-but-retained arenas.

## What 4.1 should therefore do

Not an external merge for the dictionary — that would move a few megabytes. The tractable change is
to make the **per-block representation cheaper**: today a block is
`list[tuple[doc_id, dict[term, tf]]]`, i.e. one Python dict per document, ~10,000 of them per block
at these settings. A flat per-block form — doc offsets plus parallel term/tf arrays — holds the same
information without the per-document dicts, and the *runs it writes would be byte-identical*, which
is what makes it parity-testable rather than a format migration.

That is the next step, and it is a much smaller change than the plan's: no format bump, no on-disk
dictionary, and the 20/20 and 300/300 ranking-parity checks apply directly.

## The rewrite, measured

The per-block representation was changed: term ids are now assigned as documents stream, so a
document's `term -> count` dict is transient instead of one of ~10,000 alive at once. Term ids are
still assigned in first-appearance order, so the spilled runs are unchanged.

| documents | peak RSS before | **peak RSS after** | change | index MB | build s |
| --- | --- | --- | --- | --- | --- |
| 25,000 | 312.4 MB | **241.9 MB** | **−22.6%** | 22.0 | 12.4 (was 12.2) |
| 50,000 | 417.9 MB | **377.3 MB** | **−9.7%** | 42.4 | 25.1 (was 25.2) |
| 100,000 | 596.3 MB | **548.9 MB** | **−8.0%** | 83.2 | 50.9 (was 51.4) |

Under `tracemalloc` at 100k documents the peak falls further — 942.8 → **703.0 MB**, −25.5% — with the
live state unchanged at 7.4 MB.

**Ranking parity is proven, not asserted.** `tests/test_ondisk_engine.py` is 15/15, including
`test_ondisk_bm25_matches_the_in_memory_ranking`,
`test_ondisk_bm25_ranking_is_stable_across_build_blocks` and
`test_ondisk_lexical_channel_reproduces_in_memory_ids`; index bytes and build times are unchanged.

**This is a partial fix, and the numbers say so.** The saving is 8–23% and it *shrinks* as the corpus
grows — from −22.6% at 25k documents to −8.0% at 100k — because the per-document dicts were a
constant per-block cost while the rest of the peak scales with the corpus. That remainder is the run
and merge arrays plus allocator retention, which the profile could not separate and this change does
not touch. The 10M-document wall is therefore **reduced but not removed**, and the next lever is the
merge, not the block.

## The merge is not the lever — and the block is a tradeoff, not a win

The previous round's write-up said *"the next lever is the merge, not the block"*. That was wrong,
and varying the run count at a fixed corpus shows it:

| block_tokens | runs | peak RSS | index MB | build s |
| --- | --- | --- | --- | --- |
| 100,000 | ~101 | **340.7 MB** | 83.2 | 55.0 |
| 1,000,000 | ~11 | 641.4 MB | 83.2 | 53.9 |
| 10,000,000 | ~2 | **911.8 MB** | 83.2 | 53.3 |

**More runs means *less* memory.** The 101-run build is the cheapest configuration measured and the
2-run build the most expensive, so the peak is the **block**, and the k-way merge over 101 runs is
cheap — the opposite of what I recorded. The `block_tokens` knob I already described is the lever.

It is a **tradeoff, not a free win**, which is why the default was *not* changed on this evidence.
A lower cap bounds memory but multiplies runs: at 10M documents a 100k-token cap would produce tens
of thousands of run files and a merge heap with that many handles — plausibly slower and possibly
past the open-file limit. The 100k-document experiment that shows the memory saving cannot show that
cost, because it has only 101 runs. **Choosing the cap needs a run at a corpus size where the run
count is realistic**, and that measurement has not been made.

**A variance caveat.** The same configuration here reads 641.4 MB where the earlier sweep read
548.9 MB — a ~15% spread between runs of the identical build. Allocator behaviour and machine state
move these numbers, so differences smaller than that are not findings.

## The cap, costed at a realistic run count

The open question was whether lowering `block_tokens` is safe, which the earlier 100k-document run
could not answer because it had only 101 runs. Measured at **400,000 documents** (200,000 vocabulary,
100 tokens per document), where a small cap produces a realistic number of runs:

| `block_tokens` | runs | peak RSS | index MB | build s |
| --- | --- | --- | --- | --- |
| 100,000 | ~401 | **723.0 MB** | 333.8 | 241.5 |
| 1,000,000 | ~41 | 1,106.4 MB | 333.8 | 240.2 |
| 4,000,000 | ~11 | **1,256.9 MB** | 333.8 | 237.0 |

**The run count costs almost nothing** — 241.5 s against 237.0 s, about 1.8% — while the small cap buys
**42% less peak memory** at a byte-identical index. So a lower cap is a good trade, not the tradeoff
the 101-run experiment made it look like.

**It cannot ship on its own, though.** The merge opens *every* run at once
(`handles = [np.load(path, mmap_mode="r") for path in runs]`), so a 100k-token cap needs ~1,000
handles at a million documents and ~10,000 at ten million — past the usual file-descriptor limit, and
the failure would be a hard error rather than a slow run. The ordered fix is therefore: **bound the
merge's open runs first, then lower the default cap.** That is the remaining work on this item, and it
is now specified rather than guessed.

## What this does not show

- **Not a fix.** No index format changed. The cheaper per-block representation above is the next
  step and is parity-testable (identical runs), but it is not implemented here.
- **Not a full memory profile.** `tracemalloc` tracks Python-level allocations only, so numpy and
  mmap buffers are invisible to it. That is enough to show the accumulators are negligible and the
  peak is transient, but it does not attribute the transient peak between the block's dicts, the
  run/merge arrays, and allocator retention.
- **Not natural text.** Uniform document length and a uniform term distribution; the tokenizer's
  stemming and stopword behaviour does apply, since the real `OnDiskBM25` build is what runs.
