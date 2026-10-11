# The per-corpus lexical profile: the shipped BM25 settings lose to another one on twelve of eighteen

`SCORE-LIFT.md` §1.3 records a reversal the repository had never acted on: the lexical channel is
the **only** channel eight BEIR members have, its settings are chosen **globally**, and on
Signal-1M the shipped configuration loses to its own flat control by **five points** (0.3131
against 0.3640). §P2 asks for the settings to become a per-corpus choice, and pre-registers its own
falsifier so that a negative is as publishable as a positive:

> **Falsifier.** The per-corpus selector picks the same settings on every corpus, or BM25F does not
> move FEVER.

`tools/lexical_selector.py` (`make lexical-selector`) answers the first half from the rows already
measured — every number below is a retriever row in `eval/beir/*.json`, so the measurement costs no
new runs. It compares the four configurations the harness can express today: `bm25` (the published
Anserini control: `k1=0.9, b=0.4`, query stopwords kept, no stemming), `bm25-engine` (what ships:
stemming on, stopwords dropped, `1.2/0.75`), `bm25-title` (the title repeated twice) and
`bm25-passages` (long documents split at 600 characters).

## Result

| member | bm25 | bm25-engine | bm25-title | bm25-passages | picks | delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| arguana | 0.3935 | 0.4169 | **0.4275** | — | bm25-title | +0.0106 |
| bioasq | 0.5168 | **0.5193** | — | — | bm25-engine | +0.0000 |
| climate-fever | — | 0.1610 | **0.1657** | — | bm25-title | +0.0047 |
| cqadupstack | 0.2988 | 0.2943 | **0.3046** | — | bm25-title | +0.0103 |
| dbpedia-entity | — | 0.3053 | **0.3166** | — | bm25-title | +0.0113 |
| fever | — | 0.6323 | **0.6749** | — | bm25-title | +0.0426 |
| fiqa | 0.2345 | **0.2386** | 0.2386 | — | bm25-engine | +0.0000 |
| hotpotqa | — | 0.6352 | **0.6563** | — | bm25-title | +0.0211 |
| nfcorpus | 0.3042 | 0.3244 | **0.3248** | 0.3183 | bm25-title | +0.0004 |
| nq | — | 0.3118 | **0.3264** | — | bm25-title | +0.0146 |
| quora | — | **0.7689** | 0.7689 | — | bm25-engine | +0.0000 |
| robust04 | 0.3711 | **0.4077** | 0.4077 | — | bm25-engine | +0.0000 |
| scidocs | 0.1462 | 0.1497 | **0.1523** | — | bm25-title | +0.0026 |
| scifact | 0.6539 | 0.6762 | **0.6824** | 0.6592 | bm25-title | +0.0062 |
| signal1m | **0.3640** | 0.3131 | 0.3131 | — | bm25 | +0.0509 |
| trec-covid | 0.5573 | 0.6188 | **0.6389** | — | bm25-title | +0.0201 |
| trec-news | 0.3151 | **0.3351** | 0.3285 | — | bm25-engine | +0.0000 |
| webis-touche2020 | — | **0.4419** | 0.3896 | — | bm25-engine | +0.0000 |

- shipped `bm25-engine` macro over the 18 members: **0.4195**
- per-corpus best macro: **0.4303** — **+0.0109**, the *oracle* for a lexical selector
- **picks: `bm25` ×1, `bm25-engine` ×6, `bm25-title` ×11**

**The falsifier's first branch is refuted.** The selector does not pick the same settings
everywhere: eleven members want the title-weighted row, six want the shipped settings, and
**Signal-1M wants the flat published control** — the configuration that is explicitly *not* the
engine's. §P2's expected band was "+0.002 to +0.010 macro"; the oracle is **+0.0109**, at the top of
it, and it is an oracle: it selects on the same qrels it then scores, so a real selector — which
cannot see the labels — will capture some, not all, of it. Reporting the per-cell picks is the
result, as §P1 says of its own table.

Note what Signal-1M means: this is not a missing channel, it is a **misconfigured** one. The member
has no dense row at all, and the one channel it has was tuned on corpora where its settings won.

## What the tool excludes, and why

- A **superseded** report (`consistency_check.SUPERSEDED`) — the same registry the gate uses.
- A **named subset** (`hotpotqa-subset`) and a report that **scored fewer queries** than the best
  report for the same dataset: the 600-query HotpotQA estimate reads 0.6365 against the full
  7,405-query run's 0.6352, so a naive maximum would chart an estimate as a result.
- A **deliberately different corpus**: `arguana-notitles*` strips ArguAna's titles (they are shared
  topic keyword lists, so this is an experiment about the corpus, not about BM25).

A configuration's value is the **modal** value across the remaining reports, with the other values
reported as variants. On the committed reports **no configuration disagrees with itself**, which is
worth stating: the same retriever on the same corpus reads the same number in every report that
carries it.

## What this does not show

- **It is not the selector.** §P2(a) asks for the settings to become a per-corpus choice *in the
  harness*, selected on the corpus itself. This measures what such a choice is worth from the rows
  already on disk; it does not implement the choice, and the engine has no `bm25` profile switch.
- **It is oracle selection, not label-free selection.** The picks are made on the same qrels that
  score them. Whether a *transductive* rule (channel rank-agreement, query-length distribution,
  top-k self-consistency) can reach these picks without labels is §P5(b), and it is untried.
- **The second falsifier is untouched.** §P2(b) asks for **BM25F** — per-field `b` and weight, rather
  than the "repeat the title twice" proxy — and the FEVER gap (now −0.137 against the published
  column, the largest in the table) sits on exactly that axis. `bm25-title` recovers +0.0426 of it
  and is a proxy for field weighting, not field weighting.
- **The four configurations are not a search space.** `k1`/`b` were not swept, only the four
  combinations the harness already exposes; `bm25-passages` is measured on two members.
- **The macro here is the lexical-only macro**, over the shipped lexical row per member. It is not
  the BEIR-18 best-per-dataset macro of `SCORE-LIFT.md` §1.2 (0.4799), where a lexical gain only
  moves a cell whose best row is lexical.

## Verification

Seven tests: the modal value across reports and the variant list; a deliberately different corpus
contributing nothing; a named subset and a superseded report skipped; the picks and both macros
from a synthetic pair of reports; a tie going to the shipped configuration; and two integration
guards on the committed reports — no configuration disagreeing with itself, and the selector not
being degenerate (§P2's falsifier). `make test`: **1,582 passed, 5 deselected**.
