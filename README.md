# laya_rag

A retrieval-augmented generation system that uses [Laya](https://github.com/thread3d/laya) as its
retrieval and decision engine, with a command line, an HTTP API and a web interface.

Clients meet the engine over a local HTTP service:

```
┌──────────────────────────────┐            ┌───────────────────────────────┐
│  clients                     │            │  Laya RAG engine (Python)     │
│                              │            │                               │
│  browser ── the GUI at /     │─────► 8077 │  chunk → embed → index        │
│  curl ───── the HTTP API     │   ◄─────   │  hybrid recall (dense + BM25) │
│  cli ────── laya-rag         │            │  Laya re-rank + gate          │
│                              │            │  evidence bundle              │
│  the caller composes         │            │                               │
│  the answer, or not          │            │  Laya checkpoint (torch)      │
└──────────────────────────────┘            └───────────────────────────────┘
```

## Why the split is shaped this way

Laya is not a generative model. It is a **non-autoregressive decision engine**: it evaluates typed
questions (`choice`, `score`, `noul`) over a body of text in a single forward pass and returns
calibrated probabilities. Its README is explicit — *"No text generation, so nothing to parse and
nothing to hallucinate."* It also ships no document loader, no chunker and no vector store.

That determines the architecture:

| Concern | Owner | Why |
| --- | --- | --- |
| Chunking, persistence, retrieval | this project | Laya ships none of it |
| Embeddings | **Laya** | `embed_fn_from_agent` mean-pools the checkpoint encoder |
| Relevance re-ranking | **Laya** | calibrated `noul` decision, thresholdable |
| Answerability gating | **Laya** | calibrated `noul` decision over the evidence set |
| Writing the answer | **the caller**, or the optional compose step | Laya cannot generate text |

The engine returns **evidence, not prose**. It never invents an answer; it reports which passages
support one, with a calibrated probability that they do, and abstains when they do not. Whoever
consumes it — a person reading the interface, or the optional compose step — writes the final wording
from exactly those passages.

## What is actually measured here

Laya's decision head is dramatically sharper than the encoder's cosine similarity, which is why the
engine spends compute on re-ranking rather than trusting dense retrieval:

| Passage (query: *"How long does a refund take?"*) | Dense cosine | Laya relevance |
| --- | --- | --- |
| Refunds are processed within five business days… | 0.357 | **0.790** |
| Shipping is free on orders over fifty dollars… | 0.284 | 0.001 |
| To reset your password, open Settings… | 0.283 | 0.001 |
| Our API rate limit is 600 requests per minute… | 0.138 | 0.011 |

Cosine barely separates the correct passage from unrelated ones; a 0.5 threshold on it would be
meaningless. Laya's calibrated probability separates them by ~0.78, so thresholds are usable.
Answerability gates the same way: **0.735** for evidence that answers the query against **0.0007**
for a question the evidence cannot answer.

Cosine compression is also why the engine runs **hybrid recall**. Laya's encoder is mean-pooled and
not retrieval-trained — its own README says a dedicated bi-encoder "will usually shortlist better" —
and measured on this repository's own documentation it put the expected file outside the top 20 for
2 of 5 unambiguous queries. Adding BM25 and fusing with Reciprocal Rank Fusion fixed that:

| Query | Dense only | Hybrid (BM25 + dense) |
| --- | --- | --- |
| "How do I use Laya with LangChain?" | miss | **rank 1** (0.94) |
| "How do I run Laya in Docker?" | miss | **rank 1** (0.91) |
| "How does structured output with Pydantic work?" | rank 2 | **rank 1** (0.83) |

## Requirements

- macOS or Linux, Python 3.10+
- ~3 GB disk for the Laya checkpoints and ~1 GB for the virtualenv
- No GPU required. Laya runs fp32 on CPU, and picks up Metal (MPS) automatically when available

This machine is an Intel Mac with no CUDA; the engine auto-selects **MPS** and answers a query in
2–5 s end to end. On CPU-only hosts it works but is slower.

## Setup

The Laya checkout lives in the `vendor/laya` submodule and ships a setup script pinned for macOS
Intel (PyTorch stopped publishing macOS x86_64 wheels after 2.2.2):

```sh
git clone --recurse-submodules <repository-url> laya-rag
cd laya-rag

# 1. Laya itself: venv, pinned deps, checkpoints, integrity check, smoke test
PYTHON=python3 vendor/laya/setup_laya.sh

# 2. This project: its own venv plus the service dependencies
make setup
```

`make setup` creates `.venv`, installs the same pins, installs Laya and this package editable, and
reuses the pip cache from step 1 so it is quick.

Verify everything is wired up:

```sh
.venv/bin/laya-rag doctor
```

## Using it

### Command line

```sh
# Index a corpus (Markdown, source code, config, plain text)
.venv/bin/laya-rag index vendor/laya/docs --rebuild

# Ask for evidence
.venv/bin/laya-rag query "How do I run Laya in Docker?"

# Show configuration, model and corpus state
.venv/bin/laya-rag stats --json
```

`query` prints Laya's calibrated verdict, the passages it selected with their sources, and which
recall channel found each one. Use `--json` for machine-readable output.

### HTTP service

```sh
.venv/bin/laya-rag-serve          # 127.0.0.1:8077
```

| Route | Purpose |
| --- | --- |
| `GET /` | the web GUI (see below) |
| `GET /health` | liveness, checkpoint, whether the model and index are usable |
| `GET /v1/stats` | config, model and corpus statistics |
| `GET /v1/documents` | every indexed document with its chunk and character counts |
| `GET /v1/chunks` | indexed chunks in document order, optionally `?path=` one file |
| `POST /v1/index` | `{paths: [...], rebuild?: bool}` |
| `POST /v1/query` | `{query, top_k?, retrieve_k?, relevance_threshold?, answerability_threshold?, max_passage_chars?}` |
| `POST /v1/compose` | `{query, passages: [...]}` — streams composed prose as newline-delimited JSON |
| `POST /v1/verify` | `{answer, passages: [...]}` — deterministic citation and quote checks |
| `POST /v1/adjudicate` | `{answer, …}` — Laya decides whether the corpus supports each claim |
| `POST /v1/reset` | delete the index |

### Web GUI

The engine serves its own interface at the root, so one command is the whole setup:

```sh
.venv/bin/laya-rag-serve          # then open http://127.0.0.1:8077/
```

It is plain HTML, CSS and JavaScript shipped inside the Python package — no bundler, no npm, no CDN
and no network access, which keeps the "works offline" property the rest of the project has. Serving
the GUI from the same process as the JSON API is also what makes it need no configuration: the page
calls the API on its own origin.

What it does:

- **Evidence requests.** Ask a question and read Laya's verdict as the primary result — a calibrated
  `P(answerable)` with the decision threshold drawn on the meter, the candidate funnel
  (considered → kept → below relevance), and per-stage timings. Retrieval controls are seeded from
  the engine's own configuration rather than hard-coded, and each passage shows its Laya relevance
  beside its recall score and which recall channel found it (`dense`, `lexical`, or both).
- **Honest abstention.** When nothing clears the relevance bar the engine short-circuits and reports
  `answerability = 0.0` without ever asking Laya the answerability question. The GUI detects this and
  renders "gate not reached" instead of drawing that default as a measured probability.
- **Corpus browser.** Every indexed file with chunk counts, and a viewer for the exact chunks Laya
  sees, so a surprising ranking can be traced to the text that produced it.
- **Indexing and reset.** Index paths or directories (optionally rebuilding), with the engine's
  report — including which paths it refused and why.
- **Evidence export.** Copy a result as markdown or as raw JSON.

**Composing an answer (opt-in, and not Laya's).** There is a switch above the run button. Off — the
default — the engine behaves exactly as described above and returns evidence only. On, a generative
model writes a prose answer from the passages on screen, streamed into the page as it is written, and
**the footer banner turns into a warning that the answer may contain hallucinations**. It appears
below Laya's verdict rather than above it, badged as unverified, and it is never included in the
"Copy markdown" evidence export — that stays a pure record of what Laya returned.

This is the one part of the pipeline Laya is not responsible for, so it is worth being precise about
what is and is not calibrated:

| | Laya's verdict and passages | The composed answer |
| --- | --- | --- |
| Source | calibrated `noul` decisions | a generative model |
| Thresholdable | yes | no |
| Can invent facts | no — it cannot generate text | yes |

The model is configured here: `LAYA_RAG_COMPOSE_MODEL` names it and `LAYA_RAG_COMPOSE_BASE_URL` the
endpoint, both read from this project's own environment.
Composition is unavailable — with the reason shown in the interface — when no model can be resolved,
and the switch is disabled rather than offering something that cannot work. If the model is
unreachable or times out mid-answer, the prose is dropped or marked incomplete, the evidence below is
left untouched, and the warning stays up.

Three implementation details behind that. The request disables the model's *thinking* mode, because
otherwise the reasoning tokens can consume the entire budget and return empty content — measured at
41 s for no answer versus sub-second with it off. A truncated answer (the model hitting its token
limit) is reported as truncated, since a cut-off answer otherwise looks identical to a complete one.
And **the question is stated twice**, before *and* after the evidence, which is not redundancy: on a
short topical query (`Laya in docker`) the model asked for the question in **3 of 5** runs when it
appeared only after the evidence, in **3 of 3** when it appeared only before, and in **0 of 8** when
it appeared at both ends — with answers several times longer besides. A large evidence block buries a
short trailing question, so do not tidy that repetition away without re-measuring.

The first two depend on `think: false` actually being honoured, and **it is not honoured on every
route**. Ollama's OpenAI-compatible `/v1/chat/completions` silently ignores the flag: measured over
three calls it still produced 561–614 characters of reasoning each time, took 19–21 s, and once spent
an entire 1200-token budget thinking and wrote no answer at all. The native `/api/chat` route honours
it — `thinking: 0`, ~7 s for the same prompt. That is why `COMPOSE_API` defaults to `ollama` and
reaches for the native route, deriving `<root>/api/chat` by dropping the base URL's `/v1` suffix. Set
`COMPOSE_API=openai` only for an endpoint that is not Ollama.

#### The citation check

Every composed answer is followed by a **citation check**, which is deterministic: no model, no
threshold, no probability to misread. It reports whether each claim cites in-range evidence, whether
any citation points outside the passages, and whether each quoted span actually appears in the
evidence. Quotes are matched after folding case, whitespace and quote style — a fold, not a fuzzy
match, so "found" means the text really is there.

It is deliberately **not** a green "verified" badge. The wording bounds the claim, because the check
answers a narrower question than it looks like it does:

| The citation check does | The citation check does not |
| --- | --- |
| confirm every claim carries a citation | confirm the citation supports the claim |
| catch citations to passages that do not exist | catch a misreading of a real passage |
| catch quotes that are not in the evidence | catch a claim that is true but uncited |
| count the claims it could not check | establish that the answer is **true** |

The two directions matter differently. "Citation out of range" and "quote not found" are decided
mechanically, so they cannot themselves be wrong. "Uncited" is a prompt-compliance signal, not a
truth signal — which is why the composition prompt states the citation rule as a hard requirement: a
first real run produced twelve uncited claims out of twelve while every quoted span was verbatim, and
a check that always reports "uncited" tells you nothing. With the rule stated explicitly, the same
query produced 8 of 9 claims cited with all 8 quoted spans verbatim.

What is still missing is the harder half — whether a correctly-cited claim is a fair reading of its
source. That needs Laya's adjudication, not string matching.

#### Corpus support

Below the citation check, the second half of verification asks **Laya** — not the model, and not a
string comparison — whether the corpus actually supports each claim. For every claim it re-retrieves
candidates across the whole index (dense + BM25 fused by RRF) and adjudicates them with Laya's
calibrated decisions, so each claim ends up with a probability and the passages that produced it.

Three choices shape it:

- **The corpus, not just the shown passages.** A claim can be true of the corpus and simply not have
  been among the five passages the model was handed. Re-retrieval is what tells those apart.
- **The already-measured question.** It reuses `ANSWERABILITY_QUESTION` — *"Can `query` be answered
  using only `evidence`?"* — with the claim standing in as the query. A newly-worded claim-support
  question would carry a calibration nobody has measured, and Laya's own docs warn that `noul` is
  sensitive to wording and that the question type in use must be tested on the data you intend to
  serve. The report names the question it used so it cannot be mistaken for a benchmarked
  claim-support decision.
- **"Not measured" is not zero.** When no candidate clears the relevance bar, the engine
  short-circuits and Laya is never asked. Those claims are reported as *not measured* rather than
  drawn as a confident `0.000` — the same distinction the query verdict makes.

Cost is linear in claims × candidates, which is why the candidate pool here is narrower than the
query path's: measured warm over five claims, `CLAIM_RETRIEVE_K=20` took 21.6 s against 8.7 s at
`10`, for the same verdicts on the sample. The interface shows elapsed progress rather than a bare
spinner, because this stage takes seconds.

Together the two layers catch different failures, which is why both are worth having: a fabricated
claim that *does* carry a valid citation — "Laya was first released in 2019 by a team in Iceland
[1]" — passes the citation check cleanly and is caught only by adjudication, while a misquoted
passage is caught by the citation check alone.

Two small things worth knowing. The page itself loads without a token so it can ask for one: if
`LAYA_RAG_API_KEY` is set, paste it under **Connection** and it is kept in this browser's local
storage only, then sent as a bearer token. And the GUI needs no restart to reflect a change on disk —
it re-reads engine and corpus state, and has a **Refresh** button besides.

## Configuration

Every value is an environment variable with a working default, so the engine can be retargeted
without editing code. Prefix: `LAYA_RAG_`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHECKPOINT` | `multilingual` | `multilingual`, `laya` (English) or `typed-decisions` |
| `DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda` |
| `MODELS` | `./models` | checkpoint location |
| `MAX_LEN` | `2048` | token budget for the state (document/evidence) Laya reads |
| `HEAD_MAX_LEN` | `192` | token budget for the question and its options |
| `EMBED_MAX_LENGTH` | `512` | encoder token limit for one embedding |
| `EMBED_BATCH_SIZE` | `16` | texts per embedding forward pass |
| `TORCH_THREADS` | `16` | torch intra-op threads; oversubscription hurts on high-core hosts |
| `INDEX_DIR` | `./indexes/default` | where the index is persisted |
| `CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS` | `1200` / `200` | chunking budget |
| `MIN_CHUNK_CHARS` | `80` | a shorter trailing chunk is folded into the previous one |
| `LEXICAL_STEM` | `true` | English stemming for the lexical channel; `false` keeps exact terms only |
| `RETRIEVE_K` / `RERANK_K` | `20` / `5` | candidates considered / passages kept |
| `EMBED_MODEL` | unset | retrieval-trained bi-encoder for the dense channel; unset uses Laya's own encoder |
| `EMBED_POOLING` | `auto` | `cls` for BGE-family models, `mean` otherwise — BGE is trained with `[CLS]` |
| `EMBED_QUERY_INSTRUCTION` | `false` | prepend the model's retrieval instruction to queries |
| `DENSE_WEIGHT` / `LEXICAL_WEIGHT` | `auto` / `1.0` | RRF channel weights; `auto` is `0` for Laya's encoder and `1` for a configured `EMBED_MODEL` |
| `RERANK_MODEL` | unset | retrieval-trained cross-encoder that orders and filters evidence; Laya still gates |
| `RERANK_ORDER` | `auto` | `auto` orders by the scorer when `RERANK_MODEL` is set and keeps recall order otherwise; or force `recall` / `relevance` |
| `CLAIM_RETRIEVE_K` | `10` | candidates per claim when adjudicating; narrower than `RETRIEVE_K` because cost is linear in claims × candidates |
| `CLAIM_MAX` | `25` | claims adjudicated per answer, so one long answer cannot run unbounded |
| `RELEVANCE_THRESHOLD` | `0.5` | minimum Laya P(relevant) to keep a passage |
| `ANSWERABILITY_THRESHOLD` | `0.5` | minimum Laya P(answerable) to report an answer |
| `HOST` / `PORT` | `127.0.0.1` / `8077` | service bind address |
| `API_KEY` | unset | when set, requires `Authorization: Bearer <key>` |
| `WARMUP` | `true` | load the checkpoint during service startup instead of on the first request |
| `COMPOSE_BASE_URL` | Ollama's `http://localhost:11434/v1` | endpoint used to compose an answer |
| `COMPOSE_MODEL` | unset | model that writes the composed answer; composition is unavailable until this is set |
| `COMPOSE_API` | `ollama` | `ollama` calls the native `/api/chat`; `openai` calls any OpenAI-compatible `/v1/chat/completions` |
| `COMPOSE_TIMEOUT_S` | `300` | how long composition may take before it is abandoned |
| `COMPOSE_MAX_TOKENS` | `1200` | token budget for one composed answer; hitting it is reported as truncated |

The default checkpoint is `multilingual` (322M parameters, 768-dim embeddings): it matches the
English checkpoint's accuracy on the measured queries while being roughly twice as fast, and it
handles non-English text and long documents. Set `CHECKPOINT=laya` for an English-only corpus.

For example, against a local Ollama with that model pulled:

```sh
export LAYA_RAG_COMPOSE_MODEL=ornith-1.5:35b
export LAYA_RAG_COMPOSE_BASE_URL=http://localhost:11434/v1   # the default
```

Composition is the only place where a model name matters, and the engine will not invent one: with
no `COMPOSE_MODEL`, composition is simply unavailable and the interface says so rather than guessing
at a model that would fail later.

## Testing

```sh
make test          # Python suite, offline, deterministic (296 tests)
make test-gui      # drive the GUI in headless Chrome (needs the service)
make test-real     # the same path against real Laya weights (4 tests, ~40 s)
```

The offline suite runs against a test double that speaks the same `embed`/`predict` interface, so it
is fast and needs no checkpoints. It covers the GUI's assets and the endpoints that back them: that
the static files ship, that mounting the GUI at `/` does not shadow the API routes, and that the
corpus listing stays index-optional.

`make test-gui` covers what the Python suite structurally cannot — what the page does once a browser
parses it. It drives a real headless Chrome over the DevTools Protocol and asserts on the rendered
DOM: that the query controls are seeded from the engine's config, that the panes contain their own
scrolling instead of pushing the footer off-screen, that both colour themes elevate their surfaces
and keep the primary action above a 4.5:1 contrast ratio, that the compose switch and the footer's
hallucination warning move together in both directions, that an answerable query renders a measured
verdict with threshold marker and highlights, that an unanswerable one abstains *without* drawing the
engine's short-circuited `0.0` as a measurement, and that no console error escaped. It skips rather
than fails when Chrome or the engine is absent, and reads `LAYA_RAG_API_KEY` so it can test an
engine that requires a token. Add `GUI_SMOKE_COMPOSE=1` to also compose a real answer and check it
streams in badged as unverified; that one is off by default because it calls a model and takes tens
of seconds, while everything else stays deterministic.

`make test-real` is the suite that can fail on retrieval quality and asserts the measured behaviour
in the table above.

### Measuring quality and limits

`make test` says whether the plumbing works. It says nothing about whether retrieval
finds the right passage or whether Laya's probabilities are usable at the shipped
thresholds. Two opt-in harnesses answer that, and both need the real checkpoints:

```sh
make eval      # retrieval, ranking, margin, calibration on labelled queries
make bench     # evidence size, batch determinism, chunk boundaries, scale, concurrency
```

`tools/eval.py` scores a labelled query set — the synthetic `planted` corpus by default, whose
gold files and unanswerable questions are unambiguous, or `--corpus docs` for the vendored
documentation. It reports candidate recall for dense vs hybrid and after Laya's re-rank, the
relevance margin over the best negative, reliability diagrams and ECE for both decisions, and a
threshold sweep. Two choices there are deliberate. The corpus includes **hard negatives** that
clear the relevance bar but do not answer the query, because without them the answerability gate
is never reached on a negative and its abstention behaviour goes unmeasured. And answerability
calibration is reported both for gate-reached queries and for all queries, because a
short-circuit to `0.0` is not a probability Laya measured.

`tools/bench.py` bisects the operating envelope: an evidence-size sweep against `max_len`,
batch-shape determinism, chunk-boundary survival as a function of split distance, dense-vs-BM25
timing and memory as the corpus grows, parallel-query correctness, and the claims × candidates
cost of adjudication.

Both write JSON and markdown under `eval/`. They measure rather than assert, so the directory is
gitignored; pass `--baseline eval/planted.json` to a later run to see exactly which metric moved.

### Comparing against other systems

`tools/eval.py` says whether the engine works. It cannot say how it compares: it scores a
13-chunk corpus against its own labels. `tools/beir.py` runs the same retrieval channels over a
standard [BEIR](https://arxiv.org/abs/2104.08663) dataset with the official qrels, and reports the
metrics an IR leaderboard reports — `nDCG@10`, `nDCG@100`, `MRR@10`, `Recall@100`, `MAP@100`:

```sh
make beir                                            # scifact + nfcorpus
make beir BEIR_ARGS="--datasets scifact --retrievers bm25,dense"
make beir-ablate BEIR_ABLATE_ARGS="--datasets scifact"   # sweep weight / rerank depth
```

Four retrievers run over the same corpus, qrels, evaluator and top-k, so the difference between
rows is the retriever and nothing else:

| row | what it is |
| --- | --- |
| `bm25` | lexical, at the published BEIR configuration (`k1=0.9, b=0.4`, query stopwords kept) |
| `dense` | Laya's mean-pooled checkpoint encoder — the channel this engine ships |
| `hybrid` | dense + BM25 fused by RRF, exactly as `RagEngine._fuse` fuses them |
| `biencoder` | `bge-small-en-v1.5`, a retrieval-trained model, for the comparison Laya's README invites |
| `hybrid-biencoder` | bi-encoder + BM25 fused by RRF — separates "fusion is wrong" from "the encoder is wrong" |
| `crossencoder` | a retrieval-trained cross-encoder over the best first stage — the standard second stage BAAI recommends |

The rows are not a head-to-head with a leaderboard entry, and the report says so: published BM25
baselines use Anserini/Pyserini with a Lucene analyzer and stemming while this harness uses the
project's regex tokenizer without stemming, and `dense` is a 322M decision encoder rather than a
retrieval model. What they answer is the question that matters here: how far the shipped dense
channel falls short of a dedicated bi-encoder, and whether hybrid fusion recovers it.

Datasets download to `eval/beir/data/` and embeddings cache beside them, so a re-run skips the
expensive part.

#### What it found, and what changed

Measured on this machine, multilingual checkpoint, `bge-small-en-v1.5` as the bi-encoder:

| dataset | retriever | nDCG@10 | nDCG@100 | MRR@10 | Recall@100 | MAP@100 |
| --- | --- | --- | --- | --- | --- | --- |
| SciFact | `bm25` (published settings) | 0.654 | 0.678 | 0.622 | 0.875 | 0.616 |
| SciFact | `bm25-engine` (stemmed) | 0.676 | 0.701 | 0.641 | 0.912 | 0.636 |
| SciFact | `dense` (Laya) | 0.014 | 0.022 | 0.011 | 0.067 | 0.012 |
| SciFact | `hybrid` — **shipped default** | 0.676 | 0.701 | 0.641 | 0.912 | 0.636 |
| SciFact | `biencoder` (CLS) | 0.720 | 0.743 | 0.685 | 0.953 | 0.682 |
| SciFact | `hybrid-biencoder` | **0.732** | 0.758 | 0.701 | 0.965 | 0.696 |
| SciFact | `crossencoder` | 0.691 | 0.720 | 0.661 | 0.938 | 0.653 |
| NFCorpus | `bm25` (published settings) | 0.304 | 0.264 | 0.513 | 0.234 | 0.138 |
| NFCorpus | `bm25-engine` (stemmed) | 0.324 | 0.281 | 0.532 | 0.247 | 0.145 |
| NFCorpus | `dense` (Laya) | 0.015 | 0.019 | 0.046 | 0.028 | 0.002 |
| NFCorpus | `hybrid` — **shipped default** | 0.324 | 0.281 | 0.532 | 0.247 | 0.145 |
| NFCorpus | `biencoder` (CLS) | 0.339 | 0.314 | 0.528 | 0.306 | 0.157 |
| NFCorpus | `hybrid-biencoder` | **0.364** | 0.334 | 0.573 | 0.314 | 0.173 |
| NFCorpus | `crossencoder` | 0.360 | 0.311 | 0.587 | 0.271 | 0.167 |

**Two datasets flatter this.** Adding ArguAna, SciDocs and FiQA-2018 — see
[`RANKINGS.md`](RANKINGS.md) for the full five-dataset standing, the tuning sweeps and the plan —
moves the best per-dataset configuration to a 0.434 four-dataset mean: level with SPLADE++
(0.433) and 0.057 behind a 335M BGE-large (0.491), mostly because ArguAna is where a stronger
encoder dominates (published BGE-large 0.636 against our 0.429). The rows below remain the
detailed two-dataset comparison.

The `bm25` row is the control: it lands where the published Anserini BM25 baseline lands on both
datasets, so the evaluator is measuring what a leaderboard measures. What the rows show:

- **Laya's mean-pooled encoder is close to chance at retrieval.** nDCG@10 ≈ 0.014 against BM25's
  0.654 (SciFact) and 0.304 (NFCorpus) — not "a bi-encoder would shortlist better", but ~45x worse.
- **Stemming closes the gap to the published lexical baseline** on its own: 0.654 → 0.676 and
  0.304 → 0.324, against a published Lucene BM25 of 0.679 and 0.325.
- **Fusion helps once both channels are competent.** With an unstemmed lexical channel, RRF below
  BM25 on both; with stemming it beats either channel alone (0.732 / 0.364), which is the
  textbook behaviour and the reverse of the first round's finding.

Two rounds of `make beir-ablate` chose the configuration. The first turned the dense channel off,
moved the rerank order to recall, and added the optional bi-encoder:

| change | default now | effect |
| --- | --- | --- |
| `DENSE_WEIGHT` is `auto` | `0` for Laya's encoder, `1` with `EMBED_MODEL` | SciFact 0.370 → **0.654**, NFCorpus 0.173 → **0.304** |
| `RERANK_ORDER` is `auto` | recall order, unless a cross-encoder supplies a score | SciFact 0.654 (was 0.470 reordered); NFCorpus 0.304 (was 0.283) |
| `EMBED_MODEL` is opt-in | unset | fused with BM25: SciFact **0.732**, NFCorpus **0.364** (after the second round) |

The second round checked the remaining candidates — embedding pooling, query instruction, BM25
stemming and a retrieval-trained cross-encoder:

| change | default now | effect |
| --- | --- | --- |
| `LEXICAL_STEM` | `true` | SciFact 0.654 → **0.676**, NFCorpus 0.304 → **0.324**, matching the published stemmed Lucene baseline (0.679 / 0.325) — and it is what makes fusion pay off |
| `EMBED_POOLING` | `auto`: `[CLS]` for BGE-family models | SciFact 0.708 → **0.720**; NFCorpus 0.348 → 0.339. BGE is trained with `[CLS]`, so mean pooling was a mismatch; SciFact prefers the fix, NFCorpus slightly prefers the old behaviour, and the two-dataset mean prefers `[CLS]` |
| `EMBED_QUERY_INSTRUCTION` | `false` | a wash: SciFact 0.7200 off vs 0.7127 on, NFCorpus 0.3386 off vs 0.3447 on — so the simpler setting stays |
| `RERANK_MODEL` | unset (opt-in) | lifts the shipped lexical path from 0.676 to **0.691** (SciFact) and 0.324 to **0.360** (NFCorpus), but still slightly below fusion and the bi-encoder |

Two findings worth internalising. **Strengthening the lexical channel flipped fusion from harmful to
helpful**: with an unstemmed BM25, RRF sat below BM25 alone; with stemming it beats both channels
(0.732 / 0.364). RRF is only as good as its weaker input. And **a cross-encoder helps a weak first
stage more than a strong one** — it is now the *worst* of the three good SciFact configurations,
because retrieval is already strong enough that re-scoring adds less than fusing.

The best single configuration on the two-dataset mean is `EMBED_MODEL=bge-small-en-v1.5` with
stemming (**0.548**), which sits just below a published 335M BGE-large (0.551) and above SPLADE++
(0.526) and BM25 (0.502). The shipped default, with no extra model at all, is 0.500 — level with
that published BM25 baseline.

On this project's own planted corpus the first round moved candidate MRR **0.941 → 1.0**, the worst
gold-over-negative margin from **−0.044 → +0.165**, and the count of queries where a negative
outranked gold from **1 → 0**; Laya's rerank MRR goes 0.971 → 1.0. Stemming holds all of that
(relevance ECE 0.123 → 0.097). One more measurement fell out of the sweeps: the engine's stopword
drop is marginally *better* than the published BM25 settings (+0.004 / +0.007 nDCG@10), so it stays.
And **Laya's decision head is a stronger filter than a ranker** — reordering BM25's candidates by it
lowered nDCG@10 on both datasets (SciFact 0.654 → 0.470). It still filters and still gates
answerability, which is what those decisions are calibrated for.

## Repository layout

```
laya-rag/
├── src/laya_rag/          Python engine
│   ├── config.py            environment-driven configuration
│   ├── chunking.py          document loading and paragraph-aware chunking
│   ├── model.py             Laya agent, plus optional bi-encoder and cross-encoder
│   ├── index_store.py       persistent normalised vector index
│   ├── lexical.py           BM25 channel and reciprocal rank fusion
│   ├── rerank.py            relevance filtering and ordering, answerability gate
│   ├── engine.py            orchestration: index, hybrid recall, evidence bundle
│   ├── compose.py           optional answer composition (the one non-Laya step)
│   ├── verify.py            deterministic citation and quote checks
│   ├── api.py               FastAPI service
│   ├── cli.py               `laya-rag` command line
│   └── web/                 the GUI: static HTML/CSS/JS, no build step
├── tools/                 gui-smoke.mjs; evalkit.py, eval.py, bench.py, beir.py, beir_ablate.py
├── tests/                 Python suite, real-weights suite, metrics and adversarial checks
├── models/                Laya checkpoints (gitignored)
├── indexes/               persisted indexes (gitignored)
├── eval/                  generated evaluation reports and caches (gitignored)
└── vendor/                submodule: laya
```

## Known limits

These are measured, by `make eval` and `make bench`, rather than inferred. The numbers come from
the planted corpus and the synthetic scale run and will move with a different corpus or checkpoint.

**A fact longer than the chunk overlap can be lost, not merely truncated.** A fact shorter than
`CHUNK_OVERLAP_CHARS` (200 by default) survived every boundary split tested, intact and retrieved.
A 290-character fact placed across a boundary was present in no single chunk at one split distance,
and was not retrieved at either distance tried. Long facts need a larger `CHUNK_OVERLAP_CHARS`.

**The relevance gate does most of the abstaining; the answerability gate is rarely reached on a
negative.** Every plainly unrelated question was refused before Laya was ever asked the
answerability question. Only a hard negative — a question the corpus discusses but does not answer —
got that far, and it cleared both gates and was reported answerable. "Measured" is not "supported".

**Deleting a file does not remove its chunks.** Indexing is incremental: a path that is re-indexed
has its old chunks replaced, but a file that has been deleted or made unreadable is never loaded, so
its stale chunks stay in the index and remain retrievable until a rebuild. After removing files,
re-index the parent directory with `--rebuild` rather than just the files that are left.

**BM25 is pure Python and grows linearly, though an inverted index halved the constant.** The
lexical channel keeps `term -> postings` now rather than looping over every document per query
term, which is what unblocked ArguAna (~190-word queries): 40,000 synthetic chunks went from
~120 ms to ~37 ms per query, and ArguAna's query set from stalling past 50 minutes to 4.3. The
dense scan is still far cheaper per chunk, so millions of documents still need an ANN index and
a real postings store on disk.

**The dense channel is off by default because it never earned its weight.** On BEIR SciFact and
NFCorpus Laya's mean-pooled encoder scored nDCG@10 ≈ 0.014 against BM25's 0.654 and 0.304, and every
non-zero fusion weight lowered the score — including on this project's own planted corpus, where
lexical recall ranked gold first for every query and fusion dropped it. So `DENSE_WEIGHT` resolves
to `0` for Laya's encoder, and the shipped recall is lexical. That costs semantic recall: a query
with no lexical overlap retrieves nothing. Set `EMBED_MODEL` to a retrieval-trained bi-encoder
(e.g. `BAAI/bge-small-en-v1.5`) to bring the dense channel back with weight 1, which fuses with the
stemmed lexical channel to **0.732 / 0.364** — above both channels alone, and level with a published
335M model on SciFact. Run `make beir` for the numbers.

**A cross-encoder helps a weak first stage, not a strong one.** `RERANK_MODEL` lifts the shipped
lexical path (SciFact 0.676 → 0.691, NFCorpus 0.324 → 0.360) but lands *below* the bi-encoder and the
fusion on SciFact, so it is opt-in rather than a default. `RERANK_ORDER=auto` orders by it when
configured.

**Laya's relevance decision is a stronger filter than a ranker.** Reordering BM25's candidates by
Laya's probability lowered nDCG@10 on both BEIR datasets (SciFact 0.654 → 0.470, NFCorpus
0.304 → 0.283), which is why `RERANK_ORDER` resolves to `recall` unless a cross-encoder supplies the
score: the probability still decides what is kept and whether the evidence answers the query, but
the recall order decides the ordering. Set `RERANK_ORDER=relevance` for the calibrated ordering
instead.

**A fabricated claim about a topic the corpus never mentions is not refuted.** Adjudication reports
it `not_measured`, not `unsupported`: no candidate clears the relevance bar, so Laya is never asked.
That is the engine being honest, but it means "not measured" must be read as unverified, never as
fine.

**The state budget is a soft limit.** Answerability did not collapse when the evidence block
overflowed `max_len` — it stayed above 0.73 even at 2.9x the budget, because Laya truncates the
state and the relevant passage is usually read first. Chunking still keeps blocks well inside the
budget at the default thresholds.

## The vendored submodule

The Laya checkout lives under `vendor/`. It was originally at `laya/`, but a top-level `laya/`
directory shadows the installed `laya` Python package for any process started from the repository
root, so `import laya` resolved to the checkout folder. Moving it under `vendor/` removed the
collision. The Python package for this project is `laya_rag`, which does not collide.

## License

Apache-2.0. Laya carries its own license in its checkout under `vendor/`.
