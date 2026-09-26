# laya_dsh_rag

A retrieval-augmented generation system that uses [Laya](https://github.com/thread3d/laya) as its
retrieval and decision engine and [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
(`dsh`) as its user interface.

The two halves meet over a local HTTP service:

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  dsh  (Node, user interface)│  HTTP   │  Laya RAG engine (Python)    │
│                             │ ──────► │                              │
│  laya_dsh_rag_search            │  8077   │  chunk → embed → index       │
│  laya_dsh_rag_index             │         │  hybrid recall               │
│  laya_dsh_rag_status            │         │  Laya re-rank + gate         │
│                             │ ◄────── │  evidence bundle             │
│  harness agent writes the   │         │                              │
│  final answer from evidence │         │  Laya checkpoint (torch)     │
└─────────────────────────────┘         └──────────────────────────────┘
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
| Writing the answer | **dsh's agent** | Laya cannot generate text |

The engine returns **evidence, not prose**. It never invents an answer; it reports which passages
support one, with a calibrated probability that they do, and abstains when they do not. The harness
agent composes the final wording from exactly those passages.

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
git clone --recurse-submodules <repository-url> laya_dsh_rag
cd laya_dsh_rag

# 1. Laya itself: venv, pinned deps, checkpoints, integrity check, smoke test
PYTHON=python3 vendor/laya/setup_laya.sh

# 2. This project: its own venv plus the service dependencies
make setup
```

`make setup` creates `.venv`, installs the same pins, installs Laya and this package editable, and
reuses the pip cache from step 1 so it is quick.

Verify everything is wired up:

```sh
.venv/bin/laya-dsh-rag doctor
```

## Using it

### Command line

```sh
# Index a corpus (Markdown, source code, config, plain text)
.venv/bin/laya-dsh-rag index vendor/laya/docs --rebuild

# Ask for evidence
.venv/bin/laya-dsh-rag query "How do I run Laya in Docker?"

# Show configuration, model and corpus state
.venv/bin/laya-dsh-rag stats --json
```

`query` prints Laya's calibrated verdict, the passages it selected with their sources, and which
recall channel found each one. Use `--json` for machine-readable output.

### HTTP service

```sh
.venv/bin/laya-dsh-rag-serve          # 127.0.0.1:8077
```

| Route | Purpose |
| --- | --- |
| `GET /health` | liveness, checkpoint, whether the model and index are usable |
| `GET /v1/stats` | config, model and corpus statistics |
| `POST /v1/index` | `{paths: [...], rebuild?: bool}` |
| `POST /v1/query` | `{query, top_k?, retrieve_k?, relevance_threshold?, answerability_threshold?}` |
| `POST /v1/reset` | delete the index |

### DeepSeek Harness

The tools are registered by a cordis plugin that imports nothing from the harness, so it needs no
build step and no dependency install. Load it with a `--patch` overlay:

```sh
dsh --profile web --patch "$PWD/dsh-plugin/cordis.local.yml"
```

This gives the agent three tools:

| Tool | Purpose |
| --- | --- |
| `laya_dsh_rag_search` | retrieve Laya-ranked, Laya-gated evidence for a question |
| `laya_dsh_rag_index` | index files or directories into the corpus |
| `laya_dsh_rag_status` | report checkpoint, corpus size and reachability |

See [`dsh-plugin/README.md`](dsh-plugin/README.md) for installation into a profile, configuration,
and how the answerability verdict reaches the model.

## Configuration

Every value is an environment variable with a working default, so the engine can be retargeted
without editing code. Prefix: `LAYA_DSH_RAG_`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHECKPOINT` | `multilingual` | `multilingual`, `laya` (English) or `typed-decisions` |
| `DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda` |
| `MODELS_DIR` | `./models` | checkpoint location |
| `INDEX_DIR` | `./indexes/default` | where the index is persisted |
| `CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS` | `1200` / `200` | chunking budget |
| `RETRIEVE_K` / `RERANK_K` | `20` / `5` | candidates considered / passages kept |
| `RELEVANCE_THRESHOLD` | `0.5` | minimum Laya P(relevant) to keep a passage |
| `ANSWERABILITY_THRESHOLD` | `0.5` | minimum Laya P(answerable) to report an answer |
| `HOST` / `PORT` | `127.0.0.1` / `8077` | service bind address |
| `API_KEY` | unset | when set, requires `Authorization: Bearer <key>` |

The default checkpoint is `multilingual` (322M parameters, 768-dim embeddings): it matches the
English checkpoint's accuracy on the measured queries while being roughly twice as fast, and it
handles non-English text and long documents. Set `CHECKPOINT=laya` for an English-only corpus.

## Testing

```sh
make test          # Python suite, offline, deterministic (73 tests)
make test-plugin   # Node suite for the harness tools (10 tests)
make test-real     # the same path against real Laya weights (4 tests, ~40 s)
```

The offline suite runs against a test double that speaks the same `embed`/`predict` interface, so it
is fast and needs no checkpoints. `make test-real` is the suite that can fail on retrieval quality
and asserts the measured behaviour in the table above.

## Repository layout

```
laya_dsh_rag/
├── src/laya_dsh_rag/          Python engine
│   ├── config.py            environment-driven configuration
│   ├── chunking.py          document loading and paragraph-aware chunking
│   ├── model.py             the single loaded Laya Agent (embeddings + decisions)
│   ├── index_store.py       persistent normalised vector index
│   ├── lexical.py           BM25 channel and reciprocal rank fusion
│   ├── rerank.py            Laya relevance re-ranking and answerability gate
│   ├── engine.py            orchestration: index, hybrid recall, evidence bundle
│   ├── api.py               FastAPI service
│   └── cli.py               `laya-dsh-rag` command line
├── dsh-plugin/            DeepSeek Harness tools (zero harness imports)
├── tests/                 Python suite plus a real-weights suite
├── models/                Laya checkpoints (gitignored)
├── indexes/               persisted indexes (gitignored)
└── vendor/                submodules: laya, deepseek-harness
```

## Two things worth knowing

**The submodules live under `vendor/`.** They were originally at `laya/` and `deepseek-harness/`, but
a top-level `laya/` directory shadows the installed `laya` Python package for any process started
from the repository root, so `import laya` resolved to the checkout folder. Moving them under
`vendor/` removed the collision. The Python package for this project is `laya_dsh_rag`, which does not
collide.

**`dsh plugin add` is not used.** It forwards to `pnpm`, which is not installed on this machine, so
the plugin is loaded through a `--patch` overlay or a profile patch file instead. Both are documented
in [`dsh-plugin/README.md`](dsh-plugin/README.md).

## License

Apache-2.0. Laya and DeepSeek Harness carry their own licenses in their checkouts under `vendor/`.
