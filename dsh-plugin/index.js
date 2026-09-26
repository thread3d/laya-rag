/**
 * Laya RAG tools for DeepSeek Harness.
 *
 * The engine runs in Python (Laya is a torch library) and this plugin runs in the
 * harness's Node process, so the two halves meet over the engine's HTTP service.
 *
 * The plugin deliberately imports nothing from the harness. A cordis plugin only
 * needs to export `name`/`inject`/`apply`, and `ctx.tools.register()` accepts a raw
 * JSON-Schema tool definition — the same path MCP-sourced tools take. That keeps
 * this installable with no build step and no dependency resolution against the
 * harness installation.
 *
 * @module laya-dsh-rag
 */

import { appendFileSync } from 'node:fs'

const DEFAULT_BASE_URL = 'http://127.0.0.1:8077'
const DEFAULT_TIMEOUT_MS = 120000

export const name = 'laya-dsh-rag'

/** Wait for the tool registry before registering into it. */
export const inject = ['tools']

/**
 * Call the engine's HTTP service.
 *
 * @param {string} baseUrl - Engine origin, no trailing slash.
 * @param {string} path - Route path beginning with a slash.
 * @param {{method?: string, body?: unknown, signal?: AbortSignal, timeoutMs?: number}} options
 * @returns {Promise<any>} The decoded JSON response.
 * @throws {Error} With the service's own detail text when it answers with an error.
 */
async function callEngine(baseUrl, path, options = {}) {
  const { method = 'GET', body, signal, timeoutMs = DEFAULT_TIMEOUT_MS } = options
  const timeout = AbortSignal.timeout(timeoutMs)
  // Honour caller cancellation and the local deadline together.
  const composed = signal ? AbortSignal.any([signal, timeout]) : timeout

  let response
  try {
    response = await fetch(`${baseUrl}${path}`, {
      method,
      headers: body === undefined ? {} : { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: composed,
    })
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error)
    throw new Error(
      `Laya RAG engine is unreachable at ${baseUrl} (${reason}). ` +
        'Start it with: .venv/bin/laya-dsh-rag-serve',
    )
  }

  const text = await response.text()
  let payload
  try {
    payload = text ? JSON.parse(text) : {}
  } catch {
    throw new Error(`Laya RAG engine returned non-JSON (HTTP ${response.status}): ${text.slice(0, 300)}`)
  }
  if (!response.ok) {
    const detail = payload && typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload)
    throw new Error(`Laya RAG engine error (HTTP ${response.status}): ${detail}`)
  }
  return payload
}

/**
 * Format an engine evidence response as model-facing text.
 *
 * The engine returns evidence, never prose: Laya is a decision engine and does not
 * generate text. The model composes the answer from these passages, so the
 * answerability verdict and the source of every passage travel with it.
 *
 * @param {any} value - The canonical value returned by the search tool.
 * @returns {string} Markdown evidence block.
 */
function renderEvidence(value) {
  const lines = []
  const verdict = value.answerable
    ? `answerable (Laya answerability ${value.answerability})`
    : `NOT answerable (Laya answerability ${value.answerability})`
  lines.push(`Laya RAG evidence for: ${value.query}`)
  lines.push(`Verdict: ${verdict}`)
  if (value.note) lines.push(`Note: ${value.note}`)
  lines.push('')

  if (!value.passages || value.passages.length === 0) {
    lines.push('No passage cleared the relevance threshold. Do not answer from prior knowledge.')
    return lines.join('\n')
  }

  value.passages.forEach((passage, index) => {
    lines.push(`[${index + 1}] ${passage.path} (relevance ${passage.relevance})`)
    lines.push(passage.text)
    lines.push('')
  })

  if (!value.answerable) {
    lines.push(
      'Laya judged this evidence insufficient. Tell the user the indexed corpus does not answer ' +
        'the question rather than answering from memory.',
    )
  }
  return lines.join('\n')
}

/** JSON Schema for one returned passage. */
const PASSAGE_SCHEMA = {
  type: 'object',
  properties: {
    text: { type: 'string', description: 'Verbatim passage text from the corpus' },
    path: { type: 'string', description: 'Source file the passage was read from' },
    chunk_id: { type: 'string', description: 'Stable identifier of the indexed chunk' },
    relevance: { type: 'number', description: 'Calibrated P(relevant) from Laya, 0..1' },
    channels: {
      type: 'array',
      description: 'Recall channels that surfaced this passage',
      items: { type: 'string' },
    },
  },
  required: ['text', 'path', 'chunk_id', 'relevance'],
  additionalProperties: false,
}

/**
 * Register the Laya RAG tools.
 *
 * @param {any} ctx - Cordis context.
 * @param {{baseUrl?: string, timeoutMs?: number}} [config] - Row configuration.
 */
export function apply(ctx, config) {
  const baseUrl = (config?.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
  const timeoutMs = config?.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const registered = []

  /** Register one tool and remember it, so a load can be verified from outside. */
  const register = (definition) => {
    const dispose = ctx.tools.register(definition)
    registered.push(definition.name)
    return dispose
  }

  register({
    name: 'laya_dsh_rag_search',
    description:
      'Search the indexed local corpus for evidence answering a question. Returns source ' +
      'passages ranked and gated by the Laya decision engine, plus its calibrated verdict on ' +
      'whether the evidence actually answers the question. Use this for any question about the ' +
      'indexed documents instead of answering from memory, and answer only from the returned ' +
      'passages. When the verdict is not answerable, say the corpus does not cover it.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The question to find evidence for' },
        top_k: {
          type: 'integer',
          description: 'Maximum passages to return after re-ranking (default 5)',
        },
      },
      required: ['query'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          answerable: { type: 'boolean', description: "Laya's calibrated answerability verdict" },
          answerability: { type: 'number', description: 'Calibrated P(answerable), 0..1' },
          note: { type: 'string', description: 'Set when the engine could not search, e.g. an empty index' },
          considered: { type: 'integer', description: 'Candidate passages Laya judged' },
          kept: { type: 'integer', description: 'Passages that cleared the relevance threshold' },
          passages: { type: 'array', items: PASSAGE_SCHEMA },
        },
        required: ['query', 'answerable', 'passages'],
        additionalProperties: false,
      },
      render: (_args, value) => [{ type: 'text', text: renderEvidence(value) }],
    },
    async execute(args, exec) {
      const payload = await callEngine(baseUrl, '/v1/query', {
        method: 'POST',
        body: { query: args.query, top_k: args.top_k ?? undefined },
        signal: exec?.signal,
        timeoutMs,
      })
      return {
        query: payload.query,
        answerable: Boolean(payload.answerable),
        answerability: typeof payload.answerability === 'number' ? payload.answerability : 0,
        ...(payload.note ? { note: payload.note } : {}),
        considered: payload.considered ?? 0,
        kept: payload.kept ?? 0,
        passages: (payload.passages ?? []).map((passage) => ({
          text: passage.text,
          path: passage.path,
          chunk_id: passage.chunk_id,
          relevance: passage.relevance,
          channels: passage.channels ?? [],
        })),
      }
    },
  })

  register({
    name: 'laya_dsh_rag_index',
    description:
      'Index files or directories into the Laya RAG corpus so they can be searched with ' +
      'laya_dsh_rag_search. Reads text files (Markdown, source code, config, plain text), splits ' +
      'them into passages and embeds them with the Laya checkpoint. Run this after adding or ' +
      'changing documents in the corpus.',
    parameters: {
      type: 'object',
      properties: {
        paths: {
          type: 'array',
          description: 'Absolute file or directory paths to index',
          items: { type: 'string' },
        },
        rebuild: {
          type: 'boolean',
          description: 'Discard the existing index and rebuild from these paths only',
        },
      },
      required: ['paths'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          documents: { type: 'integer' },
          chunks: { type: 'integer' },
          total_chunks: { type: 'integer' },
          paths: { type: 'integer' },
          embed_seconds: { type: 'number' },
          total_seconds: { type: 'number' },
          note: { type: 'string' },
          skipped: {
            type: 'array',
            description: 'Files that were not indexed, with the reason',
            items: {
              type: 'object',
              properties: { path: { type: 'string' }, reason: { type: 'string' } },
              required: ['path', 'reason'],
              additionalProperties: false,
            },
          },
        },
        required: ['documents', 'chunks', 'total_chunks'],
        additionalProperties: false,
      },
      render: (_args, value) => {
        const lines = [
          `Indexed ${value.documents} documents into ${value.chunks} new passages ` +
            `(${value.total_chunks} passages now searchable across ${value.paths} files).`,
          `Embedding took ${value.embed_seconds}s.`,
        ]
        if (value.skipped?.length) {
          lines.push(`${value.skipped.length} file(s) skipped:`)
          for (const entry of value.skipped.slice(0, 10)) {
            lines.push(`  - ${entry.path}: ${entry.reason}`)
          }
        }
        if (value.note) lines.push(value.note)
        return [{ type: 'text', text: lines.join('\n') }]
      },
    },
    async execute(args, exec) {
      const payload = await callEngine(baseUrl, '/v1/index', {
        method: 'POST',
        body: { paths: args.paths, rebuild: Boolean(args.rebuild) },
        signal: exec?.signal,
        timeoutMs,
      })
      return {
        documents: payload.documents ?? 0,
        chunks: payload.chunks ?? 0,
        total_chunks: payload.total_chunks ?? 0,
        paths: payload.paths ?? 0,
        embed_seconds: payload.embed_seconds ?? 0,
        total_seconds: payload.total_seconds ?? 0,
        ...(payload.note ? { note: payload.note } : {}),
        skipped: payload.skipped ?? [],
      }
    },
  })

  register({
    name: 'laya_dsh_rag_status',
    description:
      'Report the state of the Laya RAG engine: which Laya checkpoint is loaded, how many ' +
      'passages are indexed, and which files are searchable. Use this to check whether the ' +
      'corpus is ready before searching, or to diagnose a failed search.',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    output: {
      schema: {
        type: 'object',
        properties: {
          reachable: { type: 'boolean' },
          checkpoint: { type: 'string' },
          model_loaded: { type: 'boolean' },
          chunks: { type: 'integer' },
          documents: { type: 'integer' },
          files: { type: 'integer' },
          embedding_dim: { type: 'integer' },
          index_dir: { type: 'string' },
          error: { type: 'string' },
        },
        required: ['reachable'],
        additionalProperties: false,
      },
      render: (_args, value) => {
        if (!value.reachable) {
          return [{ type: 'text', text: `Laya RAG engine unreachable: ${value.error}` }]
        }
        return [
          {
            type: 'text',
            text:
              `Laya RAG engine ready. Checkpoint ${value.checkpoint} ` +
              `(model loaded: ${value.model_loaded}, embedding dim ${value.embedding_dim}). ` +
              `Corpus: ${value.chunks} passages from ${value.documents} documents across ` +
              `${value.files} files. Index at ${value.index_dir}.`,
          },
        ]
      },
    },
    async execute(_args, exec) {
      try {
        const health = await callEngine(baseUrl, '/health', { signal: exec?.signal, timeoutMs: 10000 })
        const stats = await callEngine(baseUrl, '/v1/stats', { signal: exec?.signal, timeoutMs: 10000 })
        const index = stats.index ?? {}
        const model = stats.model ?? {}
        return {
          reachable: true,
          checkpoint: String(stats.config?.checkpoint ?? health.checkpoint ?? 'unknown'),
          model_loaded: Boolean(health.model_loaded),
          chunks: index.chunks ?? 0,
          documents: index.documents ?? 0,
          files: index.paths ?? 0,
          embedding_dim: index.dim ?? 0,
          index_dir: String(stats.index_dir ?? ''),
        }
      } catch (error) {
        return {
          reachable: false,
          error: error instanceof Error ? error.message : String(error),
        }
      }
    },
  })

  ctx.logger?.info?.(`laya-dsh-rag: registered ${registered.length} tools against ${baseUrl}`)

  // Diagnostics: prove the plugin loaded, and record what it registered, without
  // requiring a log-level change. Set LAYA_DSH_RAG_TRACE_FILE to a path to enable.
  const traceFile = process.env.LAYA_DSH_RAG_TRACE_FILE
  if (traceFile) {
    try {
      appendFileSync(
        traceFile,
        `${JSON.stringify({ event: 'registered', tools: registered, baseUrl, at: new Date().toISOString() })}\n`,
      )
    } catch (error) {
      // Tracing is best-effort: an unwritable path must not fail plugin loading.
      ctx.logger?.error?.(`laya-dsh-rag: could not write trace file: ${String(error)}`)
    }
  }
}
