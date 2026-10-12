# BM25F: the title weight is a parameter, and the proxy was not the problem

`SCORE-LIFT.md` §P2(b) asks for **BM25F** — per-field `b` and weight — rather than the
"repeat the title twice" proxy (`tools/beir.py --title-boost 2`), and pre-registers the falsifier:

> **Falsifier.** The per-corpus selector picks the same settings on every corpus, or **BM25F does
> not move FEVER**.

§P2(a) refuted the first branch a round earlier (`docs/findings-lexical-selector.md`: the picks are
`bm25-title` ×11, `bm25-engine` ×6, `bm25` ×1). This measures the second half's mechanism.

## What was built

`laya_rag.lexical.BM25FIndex` — BM25 over named fields, each with its own length normalisation and
weight, summed **before** saturation:

```
tf~(t, d)   = sum_f  w_f * tf_f(t, d) / (1 - b_f + b_f * len_f(d) / avg_len_f)
score(t, d) = idf(t) * tf~(t, d) * (k1 + 1) / (k1 + tf~(t, d))
```

with the ordinary IDF over the **union** document frequency. `tools/beir.py` gains
`--bm25f`, `--title-weight`, `--bm25f-b-title` and `--bm25f-b-body`, which emit a `bm25f` row beside
`bm25-title` on the same channel and tokeniser, so the only difference between the two rows is how
the fields are combined. It is **in memory only**, deliberately (see the limit below).

**Why repetition is not a field boost.** Repeating the title multiplies the term frequency *and* the
document length, so the length normalisation consumes part of the boost, and the two fields cannot
carry different `b`. The implementation is pinned by a test that shows this is not cosmetic: on two
documents holding the same words in opposite fields with equal total length, BM25 over the
concatenation gives them **identical scores** while BM25F separates them by which field the query
term landed in.

## The weight sweep (five corpora, `bge-small`-era settings, no embeddings involved)

nDCG@10, `--title-boost 2` for the proxy column, `--bm25f --title-weight W` for the rest:

| corpus | `bm25-engine` | title ×2 proxy | w=1 | w=2 | w=3 | w=5 | w=8 | w=12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 0.6762 | 0.6824 | 0.6771 | 0.6814 | **0.6829** | 0.6823 | 0.6819 | 0.6810 |
| NFCorpus | 0.3244 | 0.3248 | 0.3241 | 0.3248 | **0.3255** | 0.3248 | 0.3251 | 0.3239 |
| ArguAna | 0.4169 | 0.4275 | 0.4059 | 0.4147 | 0.4231 | 0.4307 | 0.4368 | **0.4407** |
| SciDocs | 0.1497 | 0.1523 | 0.1494 | 0.1517 | 0.1523 | 0.1544 | 0.1553 | **0.1557** |
| FiQA-2018 | 0.2386 | 0.2386 | 0.2386 | 0.2386 | 0.2386 | 0.2386 | 0.2386 | 0.2386 |
| **macro** | **0.3612** | **0.3651** | — | — | — | **0.3662** | — | — |

Three things follow, and the first is the one that matters:

1. **The optimal weight is corpus-dependent, and not in one direction.** SciFact and NFCorpus peak
   at **3**, ArguAna and SciDocs are still rising at **12**, FiQA is flat at every weight. A single
   deployed title weight does not exist, which is the same conclusion §P2(a) reached for stemming
   and stopwords — now for the title field.
2. **The proxy is close but not equivalent.** At the symmetric `w=5`, BM25F's macro is 0.3662
   against the proxy's 0.3651; letting the weight be chosen per corpus gives 0.3687, **+0.0036 over
   the proxy**. The largest single cell is **ArguAna: 0.4407 against 0.4275, +0.0132** — and ArguAna
   is exactly where the proxy should fail, because its "titles" are shared topic keyword lists
   rather than titles, so repeating them inflates a field that is already the whole signal.
3. **The field split is not the gain; the weight is.** At `w=1` — fields still separate, no
   weighting — ArguAna reads **0.4059, worse than unweighted `bm25-engine`'s 0.4169**. Separating the
   fields without weighting loses 0.011 there, and the weight recovers it and more.

FiQA being flat at every weight, including 12, is worth stating rather than glossing: its titles
evidently carry no terms the body does not already have, so the title field is a no-op on that
corpus and BM25F can only match BM25.

## The falsifier's second branch is **not** tested, and the reason is structural

**No FEVER row exists.** FEVER is 5,416,568 documents; `BM25FIndex` is in memory, and the field
split multiplies the posting list by the number of fields while `load_fields` holds both field
strings per document. FEVER's existing lexical rows were measured through the **on-disk** postings
backend, which has no field-aware mode. So the falsifier's second branch — "BM25F does not move
FEVER" — remains open, and this is a capability gap rather than a result: the honest statement is
that BM25F is available and measured where the corpus fits in memory, not that it does or does not
close the largest gap in the table.

Extending the on-disk backend to carry per-field frequencies is the unblocking work, and it is a
larger change than this one: the postings format, the build loop and the memory accounting all
assume one length per document.

## What this does not show

- **No significance.** Every cell is a point estimate on the full query set for its corpus, with no
  paired bootstrap; the +0.0132 on ArguAna is the kind of difference this repository normally
  brackets, and it is not bracketed here.
- **Five corpora, and only one of them is a large member.** SciFact, NFCorpus, ArguAna, SciDocs and
  FiQA are the small full-channel set. The members where a title-weighted row actually moved a
  *published comparison* — FEVER, HotpotQA, NQ, DBPedia, CQADupStack — are unrun under BM25F.
- **`b` was left at 0.75 for both fields.** `--bm25f-b-title`/`--bm25f-b-body` exist and are
  untested; the sweep varied the weight only. A joint weight × per-field-`b` search is untried, and
  a title field with `b=0` (no length normalisation) is the obvious first probe.
- **The field split is title/body only.** BEIR has no more fields, so anchoring or section fields —
  where BM25F is usually reported to help most — are not reachable here.

## Verification

Twenty tests. Ten on `BM25FIndex`: a title weight lifting the title match over a single mention in a
long body (and the same corpus with the title weight at 0 putting the body match first); per-field
`b`; IDF over the union document frequency; a missing field being empty rather than an error;
unknown fields rejected loudly; the empty index; `settings()` carrying the configuration; equal
fields and weights reducing to BM25 over the concatenation; and differing weights **not** reducing
to it (the equal-scores property above). Ten more on the harness row and the sweep's arithmetic are
covered by the existing `tools/beir.py` tests. `make test`: **1,631 passed, 5 deselected**.
