# The engine's extractive row, end to end

`SCORE-LIFT.md` §P12: `compose_extractive` emits only sentences that are verbatim substrings of a
retrieved passage, so an end-to-end score over its output has **no generator confound at all** — the
strongest form of the claim this project can make. It did not exist, because every end-to-end target
was priced at the *LLM* path's ~36 s/query; retrieval in the same pilot is ~3 s/query, so the
extractive row is cheap where the generated one is not. The surfaces landed first
(`POST /v1/extract`, MCP `extract_answer`, `RagEngine.query_extractive(include_evidence=True)`);
this is the measurement.

`tools/engine_extractive.py` (`make engine-extractive`) runs the shipped engine over MultiHop-RAG's
answerable records, composes an answer from the passages it returned, and scores four things that
are all the engine's. No model, no judge, no API is in the path.

## The row (2,255 answerable MultiHop-RAG queries — the full set)

```sh
make engine-extractive ENGINE_EXTRACTIVE_ARGS="\
  --index-dir eval/benchmarks/multihop-rag/index/engine-hybrid \
  --embed-model BAAI/bge-small-en-v1.5 --zero-thresholds"
```

| metric | full set | 400-query subset (seed 17) | what it is |
| --- | ---: | ---: | --- |
| Hits@10 | 0.9082 | 0.9075 | the benchmark's own retrieval metric |
| Hits@4 | 0.7632 | 0.7525 | |
| MAP@10 | 0.3009 | 0.3057 | |
| MRR@10 | 0.5983 | 0.5901 | |
| **citation clean** | **1.0000** | **1.0000** | every claim of every emitted answer cites in-range evidence |
| uncited sentences | **0 of 9,806** | 0 of 1,728 | |
| abstention | 0.0000 | 0.0000 | nothing in the retrieved evidence qualified for quoting |
| answer nugget recall | 0.1906 | 0.1921 | share of each gold fact's nuggets the **answer** contains |

**The retrieval half is exactly the retrieval row.** On the full set,
`eval/benchmarks/multihop-rag/engine-hybrid.json` reports Hits@10 **0.9082**, Hits@4 **0.7632**,
MAP@10 **0.3009**, MRR@10 **0.5983** — identical to four decimals on all four metrics; on the
400-query subset the same holds against `engine-hybrid-400.json` (0.9075 / 0.7525 / 0.3057 /
0.5901). The runs were configured to be comparable (`--zero-thresholds`, the same `engine-hybrid`
index, the same encoder), so this is a cross-check rather than a coincidence: **composing an answer
from the passages does not disturb the retrieval it composed from.** The subset reads 0.005 higher
on MAP@10, which is sampling, not a difference in configuration.

**Attribution is perfect, and that is a measurement rather than an assertion.** All 1,728 emitted
sentences across 400 answers are cited in range. The claim "a citation score below 1.0 is a bug" was
made in the finding for `extractive.py` before any end-to-end run existed; it is now the measured
value, and it took two real bugs to get there (below).

**Completeness is the finding.** The composed answer carries **0.1906** of each gold fact's
programmatically-derived nuggets (0.1921 on the subset), against the **0.7438** claim-recall@10 the
*retrieved pool* carries (`tools/nugget_recall.py`). A five-sentence extract of the top passages is not a written
answer: three quarters of the gold content is in the evidence and about a fifth of it is in the
answer. That gap is what an extractive-by-construction system costs, and it is now a number instead
of the caveat "what is guaranteed is attribution, not usefulness".

**Abstention is 0.0 here, and that is the configuration, not a claim about the engine.** The row
runs with the relevance and answerability thresholds zeroed, which is what every retrieval row in
this repository does and is what makes the retrieval half comparable. With the shipped gates active
the same queries abstain at the gate's rate — measured separately at **28.7%** on answerable
MultiHop queries, with the shipped scalar at AUC 0.509. A deployment abstains for both reasons, and
they are reported separately rather than added together.

## Two bugs the run found, and one wrong fix along the way

The row's own contract is that its output is checkable. It was not, twice.

1. **The citation marker trailed its sentence** (`"Refunds take five days. [1]"`).
   `verify_citations` splits an answer into claims *after* the terminal punctuation, so the marker
   was orphaned onto the next claim: the sentence it supported read as uncited and the last marker
   dangled. Fixed by leading with the marker; recorded in
   [`findings-extractive-composer.md`](findings-extractive-composer.md).
2. **A quotation cut in half by a sentence boundary.** The checker reads a quotation as everything
   between an opening and a closing mark *anywhere in the answer*, so a run carrying one mark of a
   pair left the other to pair with a mark in a later run — and the span between them contains the
   `[n]` markers the composer inserts, which no passage contains. A chunk is a slice of a document,
   so it can begin or end inside a quotation. **16 of 400 answers** failed the quoted-span check
   this way, and **none** of them had an uncited claim or a citation out of range — every failure
   was half a quotation.
3. **The wrong fix, recorded because it is instructive.** Suppressing the boundary *inside* a
   quotation so the pair stays in one run made the checker split there instead, and the sentences
   after it were emitted with one `[n]` and checked as several claims: **24 uncited sentences of
   1,637**, traded for the quote failures it was meant to cure. Both checks are computed from the
   same split, so they need the same boundary rule. The fix that works is to split exactly where the
   checker splits — terminal punctuation, wherever it falls — and then **refuse to emit a run that
   carries half of a quoted pair**. Both numbers went to their floor: clean 1.0, uncited 0.

A fourth defect is visible in the output and is **not** fixed: a chunk boundary can also cut a word,
so a sentence can be emitted mid-word (`"[1] nconsistently, particularly when it comes to content
related to Palestine."`). It is verbatim and it is cited, so it is correct by the contract and ugly
by any other reading. Fixing it means refusing runs that do not start at a word boundary in the
passage, which trades content for prose.

## Run 2 — the same row on FRAMES, where the gold answers are a few words

```sh
make frames-extractive FRAMES_EXTRACTIVE_ARGS="--embed-model BAAI/bge-small-en-v1.5 --zero-thresholds"
```

814 questions over 2,482 Wikipedia articles (50,743 chunks), the same engine and composer, gates
zeroed so the abstention is the composer's:

| metric | value | what it is |
| --- | ---: | --- |
| recall@1 / recall@5 / recall@10 / recall@20 | 0.8477 / 0.9742 / **0.9853** / 0.9853 | over the gold article set |
| MRR@10 | 0.9032 | |
| nDCG@10 | 0.7420 | |
| **citation clean** | 0.9779 | 18 of 814 answers fail — **all 18 on the quoted-span check, none on a citation** |
| uncited sentences | **0 of 3,751** | |
| abstention | 0.0000 | |
| answer nugget recall | 0.1922 | against the gold **answer** string itself |

**Retrieval saturates here.** recall@10 is 0.9853 and recall@5 is 0.9742, so the gold articles are
almost always in the top five — FRAMES is not a retrieval test for this engine, which is why the run's
interest is entirely in the answer half. That matches `frames_eval`'s own retrieval row and its
0.9977 gold coverage after redirect resolution.

**The completeness number is the surprise.** FRAMES' gold answers are usually one or two words
("2005", a name), so the naive expectation is that a five-sentence extract either contains the answer
or does not — and it contains it **19.2%** of the time, almost exactly the MultiHop-RAG figure
(19.1%). Two corpora in opposite regimes, the same number: what limits the answer is the
**sentence budget**, not the corpus, and `--max-sentences` is the lever. That is a more useful
reading than either row alone.

**The 18 failures are corpus provenance, not the composer.** All 18 are on answers that quote
**raw Wikipedia table markup** — `{| class="wikitable"`, `|-`, `! scope="row"`, `==Key==`. The
decomposition is exact: **227 of 814 answers contain markup and 18 of those fail; the other 587 are
all clean.** No answer has an uncited claim and none cites out of range. The deterministic quote
checker pairs `"` marks and looks for the span in the evidence, and when the text between two marks
is a table attribute separated by a `[n]` marker or a `!` cell boundary, the "quote" it extracts is
markup rather than a statement. So the finding is about the corpus adapter: `eval/frames/corpus.jsonl`
is raw wikitext, and an extractive composer over it emits table syntax as prose. **The fix belongs in
corpus preparation — strip wiki markup before indexing — not in the composer, which is doing exactly
what it promises with the text it was given.**

**A configuration mistake worth recording, and it changed the number twice.** The first attempt at
this row ran with the **shipped gates** active, so `abstention` read 0.163 — the gate refusing
evidence, added to the composer finding nothing, which are two different measurements reported
separately everywhere else in this repository. That run also read `citation clean 0.8372`, which was
arithmetically just `1 − abstention`: the abstention sentence is one uncited claim to the checker.
`--zero-thresholds` is now a flag on both extractive runners with a test asserting its default, so
which configuration a row used cannot be invisible again.

## What this does not show

- **Two corpora, and neither is a natural extractive target.** MultiHop-RAG is multi-hop over news
  sentences; FRAMES is single-hop over raw Wikipedia articles, 227 of whose answers quote wikitext.
  A corpus whose documents are already prose *and* whose answers are sentences — the own-domain set
  of §8/§9 — is unrun, and on the evidence here the sentence budget matters more than the corpus.
- **Five sentences is a fixed budget.** `--max-sentences` was left at its default; the answer
  nugget recall at 10, 20 and unlimited sentences is unmeasured, and it is the obvious next lever on
  the completeness number.
- **Extractive prose is not fluent and is not complete.** 0.1921 nugget recall is the measurement of
  that, and any reader comparing this row to a generated one should compare attribution (where this
  wins by construction) against completeness (where it loses by construction).
- **`answer nugget recall` is a lower bound on usefulness.** It counts programmatic nuggets —
  numbers, dates, entities, key terms — present in the answer; it does not know whether the answer
  *addresses* the question, and it cannot see that a correctly quoted sentence is responsive.
- **The engine-level `FUSION_MODE=convex` change from §P5(a) is not applied here.** This is the
  shipped RRF configuration.
- **No interval.** The row is a mean over 2,255 queries with per-query values persisted
  (`eval/benchmarks/multihop-rag/engine-extractive.json`, `per_query`), so a paired bootstrap against
  any other arm on the same queries is one command away, and none is quoted here.

## Verification

Ten tests on the row's components: a verbatim, cited answer scores clean; an uncited answer does
not; a span quoting something absent from the evidence does not; the retrieval half is the
benchmark's own metric; answer nugget recall is the answer's and not the passage's; a query with no
nuggets reports `None` rather than zero; `summarize` counts abstention separately and skips null
readings; and `render` states that a citation score below 1.0 is a bug. Sixteen more on the
composer, including the two properties this row depends on — the composed answer never carries half
of a quoted pair, and the splitter is deliberately not quote-aware because the checker splits there
too. `make test`: **1,611 passed, 5 deselected**.
