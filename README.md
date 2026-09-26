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
| `POST /v1/query` | `{query, top_k?, retrieve_k?, relevance_threshold?, answerability_threshold?}` |
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
| `MODELS_DIR` | `./models` | checkpoint location |
| `INDEX_DIR` | `./indexes/default` | where the index is persisted |
| `CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS` | `1200` / `200` | chunking budget |
| `RETRIEVE_K` / `RERANK_K` | `20` / `5` | candidates considered / passages kept |
| `CLAIM_RETRIEVE_K` | `10` | candidates per claim when adjudicating; narrower than `RETRIEVE_K` because cost is linear in claims × candidates |
| `CLAIM_MAX` | `25` | claims adjudicated per answer, so one long answer cannot run unbounded |
| `RELEVANCE_THRESHOLD` | `0.5` | minimum Laya P(relevant) to keep a passage |
| `ANSWERABILITY_THRESHOLD` | `0.5` | minimum Laya P(answerable) to report an answer |
| `HOST` / `PORT` | `127.0.0.1` / `8077` | service bind address |
| `API_KEY` | unset | when set, requires `Authorization: Bearer <key>` |
| `COMPOSE_BASE_URL` | Ollama's `http://localhost:11434/v1` | endpoint used to compose an answer |
| `COMPOSE_MODEL` | unset | model that writes the composed answer; composition is unavailable until this is set |
| `COMPOSE_API` | `ollama` | `ollama` calls the native `/api/chat`; `openai` calls any OpenAI-compatible `/v1/chat/completions` |
| `COMPOSE_TIMEOUT_S` | `300` | how long composition may take before it is abandoned |
| `COMPOSE_MAX_TOKENS` | `600` | token budget for one composed answer; hitting it is reported as truncated |

The default checkpoint is `multilingual` (322M parameters, 768-dim embeddings): it matches the
English checkpoint's accuracy on the measured queries while being roughly twice as fast, and it
handles non-English text and long documents. Set `CHECKPOINT=laya` for an English-only corpus.

For example, against a local Ollama with that model pulled:

```sh
export LAYA_RAG_COMPOSE_MODEL=ornith-1.5:35b
export LAYA_RAG_COMPOSE_BASE_URL=http://localhost:11434/v1   # the default
```

Composition is the only place where a model name matters, and the engine will not invent one: with
no `COMPOSE_MODEL` and no readable `agent-default-model.model`, composition is simply unavailable and
the interface says so rather than guessing at a model that would fail later.

## Testing

```sh
make test          # Python suite, offline, deterministic (190 tests)
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

## Repository layout

```
laya-rag/
├── src/laya_rag/          Python engine
│   ├── config.py            environment-driven configuration
│   ├── chunking.py          document loading and paragraph-aware chunking
│   ├── model.py             the single loaded Laya Agent (embeddings + decisions)
│   ├── index_store.py       persistent normalised vector index
│   ├── lexical.py           BM25 channel and reciprocal rank fusion
│   ├── rerank.py            Laya relevance re-ranking and answerability gate
│   ├── engine.py            orchestration: index, hybrid recall, evidence bundle
│   ├── api.py               FastAPI service
│   ├── cli.py               `laya-rag` command line
│   └── web/                 the GUI: static HTML/CSS/JS, no build step
├── tools/                 gui-smoke.mjs, the headless-browser GUI check
├── tests/                 Python suite plus a real-weights suite
├── models/                Laya checkpoints (gitignored)
├── indexes/               persisted indexes (gitignored)
└── vendor/                submodule: laya
```

## Two things worth knowing

**The vendored submodule lives under `vendor/`.** It was originally at `laya/`, but a top-level
`laya/` directory shadows the installed `laya` Python package for any process started from the
repository root, so `import laya` resolved to the checkout folder. Moving it under `vendor/` removed
the collision. The Python package for this project is `laya_rag`, which does not collide.

## License

Apache-2.0. Laya carries its own license in its checkout under `vendor/`.
