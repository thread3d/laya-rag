# laya-dsh-rag

DeepSeek Harness tools that expose the local Laya RAG engine to the harness agent.

The plugin imports **nothing** from the harness: a cordis plugin only has to export
`name`/`inject`/`apply`, and `ctx.tools.register()` accepts a raw JSON-Schema tool definition — the
same path MCP-sourced tools take. That keeps it installable with no build step and no dependency
resolution against the harness installation.

## Tools

| Tool | Purpose |
| --- | --- |
| `laya_dsh_rag_search` | Retrieve evidence for a question, ranked and gated by Laya |
| `laya_dsh_rag_index` | Index files or directories into the corpus |
| `laya_dsh_rag_status` | Report checkpoint, corpus size and engine reachability |

### How the answerability verdict reaches the model

`laya_dsh_rag_search` returns evidence, not an answer, and the rendered tool result leads with Laya's
verdict. When Laya judges the evidence insufficient, the rendered text tells the model to say the
corpus does not cover the question rather than answering from memory:

```
Laya RAG evidence for: How do I run Laya in Docker?
Verdict: answerable (Laya answerability 0.91)

[1] vendor/laya/docs/docker.md (relevance 0.91)
Build the Laya image with docker build and run it with docker compose up.
```

This is the whole point of using Laya here: the abstention decision is a calibrated probability from
a model that cannot generate text, so it cannot be talked into a confident wrong answer.

## Prerequisites

The Python engine must be running:

```sh
.venv/bin/laya-dsh-rag-serve          # 127.0.0.1:8077
```

Every tool reports a clear error when the engine is unreachable, and `laya_dsh_rag_status` returns
`reachable: false` instead of failing.

## Installing

### Option A: `--patch` overlay (no install, no pnpm)

The simplest path, and the one used during development:

```sh
dsh --profile web --patch "$PWD/dsh-plugin/cordis.local.yml"
```

`--patch` is a launcher flag, so it must come **before** the web app's own flags:

```sh
# correct
dsh --profile web --patch ./dsh-plugin/cordis.local.yml --port 8080
# rejected: --patch is not a web-app flag
dsh --profile web --port 8080 --patch ./dsh-plugin/cordis.local.yml
```

A `--patch` overlay is an argv input, so it applies at launch — restart the harness to pick it up.

### Option B: a profile patch file (hot-reloadable)

Write the same row into the profile's own patch layer, which the shipped web profile sets to
`patchReload: live`, so it applies without a restart:

```sh
# ~/.dsh/profiles/web/cordis.patch.yml
- insert:
    - id: laya-dsh-rag
      name: '/absolute/path/to/dsh-plugin/index.js'
      config:
        baseUrl: http://127.0.0.1:8077
```

### Option C: `dsh plugin add` (needs pnpm)

`dsh plugin` forwards to pnpm in the profile directory, and this package ships a bundle manifest
(`dsh.bundle.patch`), so the standard install works wherever pnpm is available:

```sh
dsh plugin --profile web add ./dsh-plugin
dsh --profile web --dump-config | grep laya     # confirm the layer composed
```

pnpm is **not installed** on this machine, so options A and B are the working paths here.

## Configuration

Configured through the plugin row's `config` block:

| Key | Default | Meaning |
| --- | --- | --- |
| `baseUrl` | `http://127.0.0.1:8077` | Where the Python engine serves. A trailing slash is trimmed. |
| `timeoutMs` | `120000` | Per-request deadline. Indexing a large corpus needs the headroom. |

## Verifying a load

Set `LAYA_DSH_RAG_TRACE_FILE` to a path and the plugin appends a JSON line when it loads, which is the
quickest way to confirm the wiring took effect:

```sh
DSH_HOME=/tmp/dsh-check \
LAYA_DSH_RAG_TRACE_FILE=/tmp/dsh-check/trace.jsonl \
  dsh --profile web --patch "$PWD/dsh-plugin/cordis.local.yml" --port 8199 --no-open
```

```json
{"event":"registered","tools":["laya_dsh_rag_search","laya_dsh_rag_index","laya_dsh_rag_status"],
 "baseUrl":"http://127.0.0.1:8077","at":"2026-09-26T11:26:02.339Z"}
```

Use a separate `DSH_HOME` for checks like this so the harness does not write into your live profile.

## Tests

```sh
node --test dsh-plugin/test/tools.test.mjs
```

The suite uses a fake cordis context, so it covers registration, the tool schemas, the renderers and
the HTTP calls without booting the harness. Tests that need the engine skip themselves when it is
not running; set `LAYA_DSH_RAG_BASE_URL` to point them at a different port.
