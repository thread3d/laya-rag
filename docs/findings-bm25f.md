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

## The on-disk BM25F, specified — because the capability gap is the only thing left in this task

`OnDiskBM25` already exists for exactly this reason (its class docstring says BioASQ is unreachable
in memory), so the question is not *whether* BM25F can go on disk but **how to compose it from what is
already there**. It composes cleanly, and the non-obvious part is the IDF.

**One on-disk index per field.** `OnDiskBM25(directory, k1, b, drop_stopwords, stem)` already stores
what BM25F needs per field: term frequencies, per-document lengths, and `df` (`self._df`, in memory
alongside `self.terms`). Build one per field — `fever.title.bm25`, `fever.body.bm25` — and let
`--bm25f-b-title` / `--bm25f-b-body` and the field weights stay query-time parameters, so a weight
sweep needs no rebuild. That is the same property that makes the in-memory sweep cheap.

**The union-DF is the part that would otherwise be got wrong — and my first version of this section
got it wrong.** I wrote that the union is "the count of documents containing the term in at least one
field, which the two sorted term dictionaries can produce by a streaming merge". **They cannot.** The
merge gives the *vocabulary*; the *counts* are not in it, and with `df_title` and `df_body` alone:

* **`max` is a lower bound** — a term appearing in *different* documents in each field is
  undercounted, which is the common case for a word that is not a title word;
* **the sum is an upper bound** — a term appearing in the *same* document in both fields is
  double-counted.

The exact value needs `df_both`, which no per-field index keeps. So the union has to be accumulated
**while building**, from each document's distinct terms across its fields — which is now implemented
as `laya_rag.ondisk.union_document_frequency(documents)`, taking `document -> field -> tokens` and
counting each document once per distinct term, with five tests including a direct statement of the
trap so a future shortcut has to argue with them.

**The whole class is implemented, and it agrees with the single-field backend.** `laya_rag.ondisk.OnDiskBM25F` composes one `OnDiskBM25` per field under
its own subdirectory, accumulates the union document frequency with the build's own tokenisation, and
persists it beside the field indexes. Six tests cover the constraints rather than the formula: each
field gets its own directory (a title-only term is present in one index and absent from the other,
which is the evidence they are genuinely separate postings), the union counts each document once
across fields, the union round-trips, the signature takes the caller's mapping, and an empty field
list raises. `search` walks the query terms through each field's postings, sums the normalised frequencies per
document, and saturates once with the union IDF. Its five tests include the parity that matters:
**with one field and `b = 0`, `OnDiskBM25F` reproduces `OnDiskBM25`'s scores exactly** on three
queries — the same check the two older backends were validated with, and the reason the switch can be
trusted rather than spot-checked. The IDT test is the other half: `_union_idf` is deliberately *not*
`OnDiskBM25._idf`, and the test asserts they differ for a term whose field frequency is higher than
its union frequency, which is the failure that would make a title-only term look common.

**Scoring** is then the documented formula, per query term: look the term up in each field index,
compute `w_f · tf_f / (1 − b_f + b_f · len_f / avg_len_f)`, sum the fields, saturate once with `k1`,
and multiply by the union-DF IDF — where `avg_len_f` is each field index's own mean length. That is
now `laya_rag.lexical.bm25f_term_score`, a function rather than a method so the on-disk path can
apply it to postings it never holds in memory, with six tests. Two of them pin the properties a
refactor loses silently: the fields are summed **before** saturation (so a term in title *and* body
saturates once, not twice), and each field is normalised by **its own** length and `b` (so a long body
cannot dilute a short title — the proxy it replaced gets exactly that wrong).

**Why this is worth writing down rather than rediscovering**: the in-memory class has all of the
formula and none of the spill; the on-disk one has all of the spill and a single field. The
composition is a few dozen lines plus a merged `df`, and the two failure modes — per-field IDF, and
rebuilding per weight — are both silent (they produce plausible numbers from the wrong formula). The
validation is already in place: the two backends were built to produce *identical rankings*, so
`BM25FIndex` and this can be compared on a small corpus the way `BM25Index` and `OnDiskBM25` were.

### What the last piece can lean on, read off the class rather than guessed

`OnDiskBM25` already exposes the whole of what BM25F needs per field, which is why the composition is
small rather than a second index format:

| what BM25F needs, per field | where it already is |
| --- | --- |
| `tf_f(t, d)` | `index._postings[index.offsets[term_id] : index.offsets[term_id + 1]]`, `(row, tf)` pairs |
| `len_f(d)` | `index.lengths[row]`, an mmap'd array |
| `avg_len_f` | `index.avg_len` (a scalar in the metadata) |
| `df_f(t)` | `index._df[term_id]`, an in-memory array |
| `term -> id` | `index.terms`, an in-memory dict |
| stopping/stemming | `index.drop_stopwords` / `index.stem`, both in the metadata |

So the build is `for field in fields: OnDiskBM25(dir / f"{field}.bm25").build(((doc_id, fields[field])
for doc_id, fields in documents))` plus one pass of the union accumulator, and the scorer is: for each
query term, look it up in each field's `terms`; slice that field's postings; accumulate
`tf_f` and `len_f` per row; call `bm25f_term_score` once per (term, row) with the union IDF; sum over
terms; sort. Nothing new is stored beyond the field indexes and the union dictionary.

**Two things to check when it is written, because both are silent.** `OnDiskBM25._idf` must not be
used for BM25F — it is the *per-field* document frequency, and BM25F's IDF is over the union, which is
the whole point of the accumulator above. And a term absent from one field must still contribute that
field's zero, not be skipped, or the field weighting changes meaning between queries.

### Three constraints on the last piece, each of which breaks it quietly

Writing the class out on paper before typing it turned up three things that are not visible from the
formula, and each one produces a *wrong index* rather than an error. They belong here because they
are the difference between a transcription and a second design pass.

**1. The input must be re-iterable, not an iterator.** The build streams the corpus once **per
field** — `OnDiskBM25.build` takes an iterable and consumes it — so a generator would be exhausted by
the first field and the rest would index nothing. The tempting fix, materialising the documents as a
list, is worse than the bug at this scale: FEVER's 5.4M documents as `(doc_id, {field: text})` pairs
is gigabytes of Python objects for data the caller already holds. The signature has to be a
`Mapping[str, Mapping[str, str]]` — the corpus the caller has — and the per-field generators iterate
it, not a copy of it.

**2. The union DF must use the build's tokenisation, which is *not* the query's.** `OnDiskBM25.build`
indexes everything and leaves stopword removal to `drop_stopwords` on the *query* side; the union
counts are keyed by the same term ids the postings use. Computing them with a query-side tokeniser —
dropping stopwords, or using a different `stem` setting — yields counts for terms the postings do not
have, and missing counts for terms they do, which shows up as a plausible IDF over the wrong
vocabulary rather than as an error.

**3. One subdirectory per field, because `build` writes where it is constructed.** `OnDiskBM25(d)`
writes `d/meta.json`, `d/offsets.npy`, `d/postings.bin` and so on. Constructing the field indexes in
one shared directory makes the second overwrite the first — a silent single-field index wearing a
two-field design. So the layout is `<bm25f-dir>/<field>.bm25/`, and `exists_on_disk` for the composite
is the conjunction over its fields, which is also what makes `--fresh` mean the same thing here as
everywhere else in this harness.
