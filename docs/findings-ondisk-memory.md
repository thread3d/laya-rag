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

## What this does not show

- **Not a fix.** No index format changed. The bounded external merge / on-disk chunk-id list is a
  separate, parity-critical change (the 20/20 and 300/300 ranking-parity checks must stay green).
- **Not a profile.** Which structure dominates the unbounded component is unseparated, as above.
- **Not natural text.** Uniform document length and a uniform term distribution; the tokenizer's
  stemming and stopword behaviour does apply, since the real `OnDiskBM25` build is what runs.
