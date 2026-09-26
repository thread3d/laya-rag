/**
 * Tests for the Laya RAG harness tools.
 *
 * The plugin imports nothing from the harness, so a fake cordis context is enough
 * to exercise registration, the tool schemas, and the `execute` bodies. Tests that
 * need the Python engine skip themselves when it is not running, so the suite stays
 * usable without a 2.3 GB checkpoint on disk.
 *
 * Run: node --test dsh-plugin/test/
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { apply, inject, name } from '../index.js'

const BASE_URL = process.env.LAYA_RAG_BASE_URL ?? 'http://127.0.0.1:8077'

/** Collect registrations from a fake context. */
function fakeContext() {
  const tools = new Map()
  const logged = []
  return {
    tools: {
      register(definition) {
        tools.set(definition.name, definition)
        return () => tools.delete(definition.name)
      },
    },
    logger: { info: (message) => logged.push(message), error: (message) => logged.push(message) },
    tools_registered: tools,
    logged,
  }
}

/** Whether the engine is answering right now. */
async function engineReachable() {
  try {
    const response = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(2000) })
    return response.ok
  } catch {
    return false
  }
}

const reachable = await engineReachable()
const skipIfOffline = reachable ? false : `Laya RAG engine not running at ${BASE_URL}`

test('plugin declares its cordis identity', () => {
  assert.equal(name, 'laya-rag')
  assert.deepEqual(inject, ['tools'])
})

test('apply registers the three RAG tools', () => {
  const ctx = fakeContext()
  apply(ctx, { baseUrl: BASE_URL })

  assert.deepEqual(
    [...ctx.tools_registered.keys()].sort(),
    ['laya_rag_index', 'laya_rag_search', 'laya_rag_status'],
  )
})

test('every tool declares the model-facing contract', () => {
  const ctx = fakeContext()
  apply(ctx, { baseUrl: BASE_URL })

  for (const [toolName, definition] of ctx.tools_registered) {
    assert.equal(definition.name, toolName)
    assert.ok(definition.description.length > 40, `${toolName} needs a useful description`)
    assert.equal(definition.parameters.type, 'object')
    assert.ok(definition.output, `${toolName} needs an output declaration`)
    assert.ok(definition.output.schema, `${toolName} needs an output schema`)
    assert.equal(typeof definition.output.render, 'function')
    assert.equal(typeof definition.execute, 'function')
  }
})

test('apply trims a trailing slash from baseUrl', () => {
  const ctx = fakeContext()
  apply(ctx, { baseUrl: `${BASE_URL}/` })
  assert.ok(ctx.tools_registered.has('laya_rag_search'))
})

test('registration is disposable', () => {
  const ctx = fakeContext()
  apply(ctx, { baseUrl: BASE_URL })
  const dispose = ctx.tools.register({ name: 'probe', description: 'x', parameters: {}, output: { schema: {}, render: () => [] }, execute: async () => null })
  assert.ok(ctx.tools_registered.has('probe'))
  dispose()
  assert.equal(ctx.tools_registered.has('probe'), false)
})

test('laya_rag_status reports the engine', { skip: skipIfOffline }, async () => {
  const ctx = fakeContext()
  apply(ctx, { baseUrl: BASE_URL })
  const status = ctx.tools_registered.get('laya_rag_status')
  const value = await status.execute({}, {})

  assert.equal(value.reachable, true)
  assert.equal(typeof value.checkpoint, 'string')
  assert.equal(typeof value.chunks, 'number')
  assert.ok(value.embedding_dim > 0, 'expected a real embedding dimension')
  assert.match(status.output.render({}, value)[0].text, /Corpus: \d+ passages/)
})

test('laya_rag_search returns Laya-gated evidence', { skip: skipIfOffline }, async () => {
  const ctx = fakeContext()
  apply(ctx, { baseUrl: BASE_URL })
  const search = ctx.tools_registered.get('laya_rag_search')

  const value = await search.execute({ query: 'How do I use Laya with LangChain?', top_k: 3 }, {})

  assert.equal(value.query, 'How do I use Laya with LangChain?')
  assert.equal(typeof value.answerable, 'boolean')
  assert.ok(value.passages.length > 0, 'expected at least one passage from the indexed docs')
  assert.ok(value.passages.length <= 3, 'top_k must cap the returned passages')

  for (const passage of value.passages) {
    assert.equal(typeof passage.text, 'string')
    assert.ok(passage.text.length > 0)
    assert.equal(typeof passage.path, 'string')
    assert.ok(passage.relevance >= 0 && passage.relevance <= 1, 'relevance is a probability')
    assert.ok(Array.isArray(passage.channels))
  }
  // Passages are ordered by Laya's relevance, strongest first.
  const relevance = value.passages.map((p) => p.relevance)
  assert.deepEqual(relevance, [...relevance].sort((a, b) => b - a))
})

test('search render is pure and cites sources', { skip: skipIfOffline }, async () => {
  const ctx = fakeContext()
  apply(ctx, { baseUrl: BASE_URL })
  const search = ctx.tools_registered.get('laya_rag_search')
  const value = await search.execute({ query: 'How do I run Laya in Docker?' }, {})

  const first = search.output.render({}, value)
  const second = search.output.render({}, value)
  assert.deepEqual(first, second, 'render must be a pure function of its arguments')
  assert.equal(first[0].type, 'text')
  assert.match(first[0].text, /Laya RAG evidence for:/)
  assert.match(first[0].text, /Verdict: /)
  assert.match(first[0].text, /\[1\] /)
})

test('search on an empty query is rejected by the engine', { skip: skipIfOffline }, async () => {
  const ctx = fakeContext()
  apply(ctx, { baseUrl: BASE_URL })
  const search = ctx.tools_registered.get('laya_rag_search')
  await assert.rejects(() => search.execute({ query: '' }, {}), /HTTP 422|error/i)
})

test('tools report an unreachable engine clearly', async () => {
  const ctx = fakeContext()
  apply(ctx, { baseUrl: 'http://127.0.0.1:9', timeoutMs: 2000 })
  const search = ctx.tools_registered.get('laya_rag_search')
  await assert.rejects(
    () => search.execute({ query: 'anything' }, {}),
    /unreachable/i,
  )

  const status = ctx.tools_registered.get('laya_rag_status')
  const value = await status.execute({}, {})
  assert.equal(value.reachable, false)
  assert.match(status.output.render({}, value)[0].text, /unreachable/i)
})
