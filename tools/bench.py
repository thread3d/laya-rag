#!/usr/bin/env python
"""Limit probes for the Laya RAG engine.

Where ``tools/eval.py`` measures quality on a labelled set, this measures where the
pipeline stops behaving and what each stage costs:

* ``evidence``      — how the answerability gate responds as the evidence block grows,
                      and how chunk size changes it;
* ``batch``         — whether Laya's predictions depend on the batch shape they arrive
                      in (they are floats, and Laya's own docs warn they can move);
* ``boundary``      — whether the chunk overlap really rescues a fact that straddles a
                      chunk boundary, and at what split distance;
* ``scale``         — dense vs BM25 query time, build time, disk and memory as the
                      corpus grows (hash embedder: the channels are what is measured);
* ``concurrency``   — whether parallel queries stay correct and actually speed up;
* ``adjudication``  — the claims x candidates cost of corpus support.

    .venv/bin/python tools/bench.py --probes evidence,batch
    .venv/bin/python tools/bench.py                 # everything

Writes ``eval/bench.json`` and ``eval/bench.md``.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import evalkit  # noqa: E402
from laya_rag.chunking import Chunk, chunk_text  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.engine import RagEngine  # noqa: E402
from laya_rag.index_store import VectorIndex  # noqa: E402
from laya_rag.lexical import BM25Index  # noqa: E402
from laya_rag.model import HashingEmbedder, LayaModel  # noqa: E402
from laya_rag.verify import split_claims  # noqa: E402

BENCH_DIR = ROOT / "eval" / ".bench"
GOLD_QUERY = "How long does a refund take?"
GOLD_DOC = "billing/refunds.md"
BOUNDARY_QUERY = "What is the secret rotation interval?"
BOUNDARY_FACT = "The secret rotation interval is forty two days."
ALL_PROBES = ("evidence", "batch", "boundary", "scale", "concurrency", "adjudication")


# ── helpers ──────────────────────────────────────────────────────────────

def rss_mb() -> float:
    """Resident high-water mark in MB (macOS reports bytes, Linux kilobytes)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(raw / 1024 / 1024, 1) if platform.system() == "Darwin" else round(raw / 1024, 1)


def write_files(directory: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return directory


def build_engine(
    model: Any,
    name: str,
    files: dict[str, str],
    chunk_chars: int | None = None,
) -> tuple[RagEngine, EngineConfig]:
    """Index a fresh corpus under ``eval/.bench/<name>`` and return the engine."""
    directory = write_files(BENCH_DIR / name / "corpus", files)
    config = EngineConfig(index_dir=BENCH_DIR / name / "index")
    if chunk_chars is not None:
        config.chunk_chars = chunk_chars
    engine = RagEngine(config, model=model)
    engine.index_paths([directory], rebuild=True)
    return engine, config


def top_paths(engine: RagEngine, query: str, k: int) -> list[Any]:
    """The fused candidate hits for a query, using the engine's own retrieval path."""
    index = engine.index
    dense = index.search(engine.model.embed_one(query), k=k)
    lexical = engine.lexical
    lexical_hits = lexical.search(query, k=k) if lexical else []
    return engine._fuse(index, dense, lexical_hits, limit=k)


# ── probes ───────────────────────────────────────────────────────────────

def probe_evidence(model: Any) -> dict[str, Any]:
    """Does the answerability gate survive a large evidence block?

    ``LayaReranker.check_answerability`` concatenates every kept passage into one
    state, which Laya truncates to ``max_len`` tokens. With the shipped
    ``chunk_chars=1200`` and ``rerank_k=5`` that concatenation is several thousand
    characters, so this sweeps both the number of passages and the chunk size and
    watches the gate.
    """
    files = evalkit.planted_corpus()
    engine, config = build_engine(model, "evidence", files)

    by_top_k: list[dict[str, Any]] = []
    for top_k in (1, 2, 3, 5, 8, 12):
        result = engine.query(GOLD_QUERY, top_k=top_k, retrieve_k=20, max_passage_chars=None)
        evidence_chars = sum(len(p["text"]) for p in result.passages)
        by_top_k.append({
            "top_k": top_k,
            "kept": result.kept,
            "evidence_chars": evidence_chars,
            "est_tokens": round(evidence_chars / 4),
            "pct_of_max_len": round(100 * (evidence_chars / 4) / config.max_len, 1),
            "answerability": result.answerability,
            "answerable": result.answerable,
        })

    by_chunk: list[dict[str, Any]] = []
    for chunk_chars in (300, 600, 1200, 2000):
        chunked, chunk_config = build_engine(model, "evidence", files, chunk_chars=chunk_chars)
        result = chunked.query(GOLD_QUERY, top_k=5, retrieve_k=20, max_passage_chars=None)
        evidence_chars = sum(len(p["text"]) for p in result.passages)
        by_chunk.append({
            "chunk_chars": chunk_chars,
            "chunks": len(chunked.index),
            "kept": result.kept,
            "evidence_chars": evidence_chars,
            "est_tokens": round(evidence_chars / 4),
            "pct_of_max_len": round(100 * (evidence_chars / 4) / chunk_config.max_len, 1),
            "answerability": result.answerability,
            "answerable": result.answerable,
        })

    # The default relevance threshold caps how many passages survive, so the block never
    # grows large enough to approach max_len no matter what top_k asks for. Forcing the
    # threshold to zero is what actually tests the gate against a block that overflows
    # the state budget.
    forced: list[dict[str, Any]] = []
    for top_k in (3, 5, 8, 12, 20):
        result = engine.query(
            GOLD_QUERY, top_k=top_k, retrieve_k=20, relevance_threshold=0.0, max_passage_chars=None
        )
        evidence_chars = sum(len(p["text"]) for p in result.passages)
        forced.append({
            "top_k": top_k,
            "kept": result.kept,
            "evidence_chars": evidence_chars,
            "est_tokens": round(evidence_chars / 4),
            "pct_of_max_len": round(100 * (evidence_chars / 4) / config.max_len, 1),
            "answerability": result.answerability,
            "answerable": result.answerable,
        })

    # The planted corpus is only ~4.5 KB, so even with the threshold forced to zero the
    # evidence block tops out well below max_len and truncation is unreachable. This
    # builds a corpus of long, all-relevant chunks so the block genuinely overflows the
    # state budget, which is the only way to see what the gate does when it does.
    filler = "Billing context and refund handling notes for the support team. " * 22
    overflow_files = {
        f"overflow/doc{index:02d}.md": (
            f"# Refund note {index}\n\n"
            "Refunds are processed within five business days to the original payment method. "
            f"{filler}\n"
        )
        for index in range(20)
    }
    overflow_engine, overflow_config = build_engine(model, "evidence-overflow", overflow_files)
    overflow: list[dict[str, Any]] = []
    for top_k in (1, 2, 4, 8, 16, 20):
        result = overflow_engine.query(
            GOLD_QUERY, top_k=top_k, retrieve_k=20, relevance_threshold=0.0, max_passage_chars=None
        )
        evidence_chars = sum(len(p["text"]) for p in result.passages)
        overflow.append({
            "top_k": top_k,
            "chunks_in_index": len(overflow_engine.index),
            "kept": result.kept,
            "evidence_chars": evidence_chars,
            "est_tokens": round(evidence_chars / 4),
            "pct_of_max_len": round(100 * (evidence_chars / 4) / overflow_config.max_len, 1),
            "answerability": result.answerability,
            "answerable": result.answerable,
        })

    return {
        "max_len": config.max_len,
        "chunk_chars": config.chunk_chars,
        "by_top_k": by_top_k,
        "by_chunk_chars": by_chunk,
        "forced_large_evidence": forced,
        "overflow": overflow,
    }


def probe_batch(model: Any) -> dict[str, Any]:
    """Is a Laya probability independent of the batch shape it is computed in?"""
    engine, _ = build_engine(model, "batch", evalkit.planted_corpus())
    queries = [q for q in evalkit.planted_queries() if q.answerable][:8]
    pairs: list[tuple[str, str]] = []
    for query in queries:
        pairs.extend((query.query, hit.chunk.text) for hit in top_paths(engine, query.query, 5))

    baseline, _ = engine.reranker.score_relevance_pairs(pairs, batch_size=8)
    rows: list[dict[str, Any]] = []
    for batch_size in (1, 2, 4, 16, 32):
        scores, _ = engine.reranker.score_relevance_pairs(pairs, batch_size=batch_size)
        deltas = [abs(left - right) for left, right in zip(baseline, scores)]
        # Re-rank each query's five candidates and see whether order moved.
        reordered = 0
        for offset in range(0, len(pairs), 5):
            base_block = baseline[offset:offset + 5]
            this_block = scores[offset:offset + 5]
            if sorted(range(5), key=lambda i: -base_block[i]) != sorted(range(5), key=lambda i: -this_block[i]):
                reordered += 1
        rows.append({
            "batch_size": batch_size,
            "queries_reordered": reordered,
            "max_abs_delta": round(max(deltas), 6),
            "mean_abs_delta": round(sum(deltas) / len(deltas), 8),
            "identical": max(deltas) == 0.0,
        })

    # predict() is system_one; predict_batch() is the throughput path. Do they agree?
    sample_query, sample_text = pairs[0]
    single = model.predict(
        {"query": sample_query, "passage": sample_text},
        {"relevant": {"type": "noul", "instructions": "Does `passage` help answer `query`?"}},
    )
    batched = model.predict_many(
        [{"query": sample_query, "passage": sample_text}],
        {"relevant": {"type": "noul", "instructions": "Does `passage` help answer `query`?"}},
    )[0]
    single_value = single["answers"]["relevant"]["noul"]
    batched_value = batched["answers"]["relevant"]["noul"]
    return {
        "pairs": len(pairs),
        "by_batch_size": rows,
        "single_vs_batch": {
            "predict": single_value,
            "predict_batch": batched_value,
            "delta": round(abs(single_value - batched_value), 6),
        },
    }


def probe_boundary(model: Any) -> dict[str, Any]:
    """Where does the chunk overlap's containment guarantee actually end?

    A fact no longer than ``overlap_chars`` is provably intact in at least one chunk at
    any split distance; the probe places it at a controlled distance before the first cut
    point and checks. The chunker also backs a cut inside a sentence up to that sentence's
    start, so a whole sentence up to ``chunk_chars`` survives even when it is longer than
    the overlap. A fact longer than the overlap *and* spanning more than one sentence is
    the residual limit: no single chunk contains it.
    """
    files = dict(evalkit.planted_corpus())
    filler = "word "
    # The engine that will answer uses the configured defaults, so measure the same
    # chunk size and overlap it actually indexes with rather than hard-coded values.
    _base, config = build_engine(model, "boundary", files)
    chunk_chars = config.chunk_chars
    overlap = config.chunk_overlap_chars

    def chunked(body: str) -> list[tuple[str, int, int]]:
        return chunk_text(body, chunk_chars=chunk_chars, overlap_chars=overlap, min_chunk_chars=80)

    def answer(name: str, body: str) -> dict[str, Any]:
        probe_files = dict(files)
        probe_files["misc/rotation.md"] = body
        engine, _ = build_engine(model, name, probe_files)
        result = engine.query(BOUNDARY_QUERY, top_k=5, retrieve_k=20, max_passage_chars=None)
        best = next((p for p in result.passages if "rotation.md" in p["path"]), None)
        return {
            "found_in_top_k": best is not None,
            "relevance": None if best is None else best["relevance"],
            "answerability": result.answerability,
            "answerable": result.answerable,
        }

    rows: list[dict[str, Any]] = []
    for shift in (0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400):
        prefix = filler * ((chunk_chars - shift) // len(filler))
        body = f"{prefix}{BOUNDARY_FACT} {' '.join(['tail'] * 400)}\n"
        intact = any(BOUNDARY_FACT in piece for piece, _, _ in chunked(body))
        rows.append({
            "shift_chars": shift,
            "fact_intact_in_a_chunk": intact,
            **answer("boundary", body),
        })

    # A fact longer than the overlap, placed so a cut lands inside it at several
    # distances. This is what the larger overlap is for: below it the fact must survive.
    long_fact = (
        "The secret rotation interval is forty two days and the break glass procedure "
        "requires two approvers from the platform team before any credential is rotated, "
        "with every request written to the audit channel and reviewed by the security team "
        "within one business day of the rotation completing."
    )
    long_rows: list[dict[str, Any]] = []
    for shift in (250, 300, overlap - 50, overlap + 50):
        prefix = filler * ((chunk_chars - shift) // len(filler))
        body = f"{prefix}{long_fact} {' '.join(['tail'] * 400)}\n"
        intact = any(long_fact in piece for piece, _, _ in chunked(body))
        long_rows.append({
            "fact_chars": len(long_fact),
            "shift_chars": shift,
            "fact_intact_in_a_chunk": intact,
            **answer("boundary-long", body),
        })

    # A single sentence longer than the overlap, in normal prose, must survive because
    # the chunker will not split it: the sentence-aware pass, not the overlap, is what
    # keeps it whole. The shifts reach past the overlap into the sentence's interior.
    sentence = (
        "The secret rotation interval is forty two days and the break glass procedure "
        "requires two approvers from the platform team before any credential is rotated, "
        "and every request is written to the audit channel with the requesting service, "
        "the two approvers, and the on-call security engineer recorded before the "
        "rotation is allowed to proceed, with the rotation blocked automatically if "
        "either approver is unavailable or the audit channel is not writable at the "
        "moment the request is made and the failure written to the same channel for "
        "the next business day's review."
    )
    unit = "Earlier background context. "
    sentence_rows: list[dict[str, Any]] = []
    for shift in (0, 250, 500, len(sentence)):
        prefix = (unit * (chunk_chars // len(unit) + 2))[: max(0, chunk_chars - shift)]
        body = f"{prefix} {sentence} {' '.join(['tail'] * 400)}\n"
        intact = any(sentence in piece for piece, _, _ in chunked(body))
        sentence_rows.append({
            "sentence_chars": len(sentence),
            "shift_chars": shift,
            "sentence_intact_in_a_chunk": intact,
            **answer("boundary-sentence", body),
        })

    return {
        "fact_chars": len(BOUNDARY_FACT),
        "sentence_chars": len(sentence),
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap,
        "rows": rows,
        "long_fact": long_rows,
        "sentence": sentence_rows,
    }


def synthetic_chunks(count: int, words_per_chunk: int = 200, seed: int = 13) -> list[Chunk]:
    rng = random.Random(seed)
    vocab = [f"token{i}" for i in range(400)]
    vocab += ["refund", "interval", "secret", "rotation", "api", "docker", "limit", "invoice"]
    chunks: list[Chunk] = []
    for position in range(count):
        words = [rng.choice(vocab) for _ in range(words_per_chunk)]
        text = " ".join(words)
        chunks.append(Chunk(
            chunk_id=f"c{position:07d}",
            doc_id=f"d{position // 10:06d}",
            path=f"doc{position // 10:06d}.md",
            text=text,
            index=position % 10,
            start_char=0,
            end_char=len(text),
        ))
    return chunks


def probe_scale(sizes: Sequence[int], embed_dim: int = 256) -> dict[str, Any]:
    """Dense vs BM25 as the corpus grows, with the hash embedder.

    The embedder is a stand-in on purpose: this measures the retrieval channels and
    the storage, neither of which depends on the real encoder. Timing dense and BM25
    separately is the point — the README claims an exhaustive dense scan scales to
    millions of chunks, and BM25 here is a pure-Python loop that may not.
    """
    embedder = HashingEmbedder(dim=embed_dim)
    query = "what is the refund interval for secret rotation"
    rows: list[dict[str, Any]] = []
    for size in sizes:
        chunks = synthetic_chunks(size)
        directory = BENCH_DIR / f"scale-{size}"
        directory.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        vectors = embedder.embed([chunk.text for chunk in chunks])
        embed_seconds = time.perf_counter() - started

        index = VectorIndex()
        index.add(chunks, vectors)
        lexical = BM25Index()
        started = time.perf_counter()
        lexical.build(chunks)
        bm25_build_seconds = time.perf_counter() - started

        started = time.perf_counter()
        index.save(directory)
        lexical.save(directory / "lexical.json")
        save_seconds = time.perf_counter() - started

        query_vector = embedder.embed_one(query)
        dense_samples = []
        for _ in range(5):
            mark = time.perf_counter()
            index.search(query_vector, k=20)
            dense_samples.append(time.perf_counter() - mark)
        lexical_samples = []
        for _ in range(5):
            mark = time.perf_counter()
            lexical.search(query, k=20)
            lexical_samples.append(time.perf_counter() - mark)

        rows.append({
            "chunks": size,
            "embed_seconds": round(embed_seconds, 2),
            "bm25_build_seconds": round(bm25_build_seconds, 2),
            "save_seconds": round(save_seconds, 2),
            "dense_ms_median": round(1000 * median(dense_samples), 2),
            "bm25_ms_median": round(1000 * median(lexical_samples), 2),
            "dense_per_1k_chunks_ms": round(1000 * median(dense_samples) / (size / 1000), 4),
            "bm25_per_1k_chunks_ms": round(1000 * median(lexical_samples) / (size / 1000), 4),
            "index_mb": round((directory / "index.json").stat().st_size / 1e6, 2),
            "vectors_mb": round((directory / "vectors.npy").stat().st_size / 1e6, 2),
            "lexical_mb": round((directory / "lexical.json").stat().st_size / 1e6, 2),
            "rss_mb": rss_mb(),
        })
        del index, lexical, vectors, chunks
    return {"embed_dim": embed_dim, "query": query, "rows": rows}


def probe_concurrency(model: Any, workers: int = 4) -> dict[str, Any]:
    """Do parallel queries stay correct, and does the wall clock actually improve?"""
    engine, _ = build_engine(model, "concurrency", evalkit.planted_corpus())
    queries = [q.query for q in evalkit.planted_queries()][:8]

    def run(query: str) -> dict[str, Any]:
        result = engine.query(query, top_k=5, retrieve_k=20)
        return {
            "query": query,
            "answerable": result.answerable,
            "answerability": result.answerability,
            "paths": [p["path"] for p in result.passages],
        }

    started = time.perf_counter()
    sequential = [run(query) for query in queries]
    sequential_seconds = time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parallel = list(pool.map(run, queries))
    parallel_seconds = time.perf_counter() - started

    mismatches = [
        {"query": left["query"], "sequential": left["paths"], "parallel": right["paths"]}
        for left, right in zip(sequential, parallel)
        if left["paths"] != right["paths"] or left["answerability"] != right["answerability"]
    ]
    return {
        "workers": workers,
        "queries": len(queries),
        "sequential_seconds": round(sequential_seconds, 3),
        "parallel_seconds": round(parallel_seconds, 3),
        "speedup": round(sequential_seconds / parallel_seconds, 2) if parallel_seconds else None,
        "mismatches": mismatches,
    }


def probe_adjudication(model: Any) -> dict[str, Any]:
    """The claims x candidates cost of asking Laya whether the corpus supports a claim."""
    # A corpus large enough that retrieve_k actually changes the candidate count: the
    # planted corpus has 13 chunks, so a sweep up to retrieve_k=40 would be capped at 13
    # for every setting and measure nothing. Six copies give 78 chunks.
    files = {
        f"copy{index}/{path}": text
        for index in range(6)
        for path, text in evalkit.planted_corpus().items()
    }
    engine, _ = build_engine(model, "adjudication", files)
    answer = (
        "Refunds are processed within five business days [1]. "
        "The API allows 600 requests per minute per API key [2]. "
        "Two-factor authentication supports hardware security keys [3]. "
        "Laya was founded in Iceland in 2015 [4]. "
        "The tracking number format is SH-0000-XX [5]."
    )
    claims = split_claims(answer)
    # Warm the forward path once so the first timed sweep is not measuring lazy init.
    engine.adjudicate_claims(["Warm up the adjudication path."], retrieve_k=5)
    rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for retrieve_k in (5, 10, 20, 40):
        started = time.perf_counter()
        report = engine.adjudicate_claims(claims, retrieve_k=retrieve_k)
        elapsed = time.perf_counter() - started
        actual_pairs = sum(claim["considered"] for claim in report["claims"])
        answerability_pairs = sum(1 for claim in report["claims"] if claim["kept"])
        if retrieve_k == 10:
            detail = [
                {
                    "claim": claim["text"],
                    "measured": claim["measured"],
                    "supported": claim["supported"],
                    "support_probability": claim["support_probability"],
                    "best_relevance": claim["best_relevance"],
                    "kept": claim["kept"],
                }
                for claim in report["claims"]
            ]
        rows.append({
            "retrieve_k": retrieve_k,
            "claims": report["total"],
            "candidates_per_claim": retrieve_k,
            "pairs": actual_pairs,
            "answerability_pairs": answerability_pairs,
            "supported": report["supported"],
            "unsupported": report["unsupported"],
            "not_measured": report["not_measured"],
            "seconds": round(elapsed, 3),
            "seconds_per_100_pairs": round(elapsed / max(1, actual_pairs) * 100, 3),
        })
    return {
        "claims": len(claims),
        "corpus_chunks": len(engine.index),
        "claim_texts": claims,
        "by_retrieve_k": rows,
        "detail_at_retrieve_k_10": detail,
    }


# ── reporting ────────────────────────────────────────────────────────────

def render_markdown(results: dict[str, Any]) -> str:
    failed = {
        name: block["error"]
        for name, block in results.items()
        if isinstance(block, dict) and "error" in block
    }
    results = {name: block for name, block in results.items() if name not in failed}
    parts: list[str] = ["# Limit probes", ""]
    if failed:
        parts += ["## Probes that failed", ""]
        parts += [f"- **{name}**: {message}" for name, message in failed.items()]
        parts += [""]
    if "evidence" in results:
        block = results["evidence"]
        parts += [
            "## Evidence size vs the answerability gate",
            "",
            f"`max_len={block['max_len']}` tokens, `chunk_chars={block['chunk_chars']}`. "
            "Token estimate is characters/4.",
            "",
            evalkit.markdown_table(block["by_top_k"]),
            "",
            "### Varying chunk size (top_k=5)",
            "",
            evalkit.markdown_table(block["by_chunk_chars"]),
            "",
            "### Evidence forced past the relevance filter (relevance_threshold=0.0)",
            "",
            evalkit.markdown_table(block["forced_large_evidence"]),
            "",
            "### Overflowing the state budget (long all-relevant chunks, threshold forced to 0)",
            "",
            evalkit.markdown_table(block["overflow"]),
            "",
        ]
    if "batch" in results:
        block = results["batch"]
        parts += [
            "## Batch-shape determinism",
            "",
            f"{block['pairs']} scored pairs, compared against batch_size=8.",
            "",
            evalkit.markdown_table(block["by_batch_size"]),
            "",
            evalkit.markdown_table([{"comparison": "predict() vs predict_batch()", **block["single_vs_batch"]}]),
            "",
        ]
    if "boundary" in results:
        block = results["boundary"]
        parts += [
            "## Chunk-boundary survival",
            "",
            f"Fact is {block['fact_chars']} chars, chunk is {block['chunk_chars']} chars, "
            f"overlap is {block['overlap_chars']} chars. `shift_chars` is how far the first "
            "cut point is pushed into the fact.",
            "",
            evalkit.markdown_table(block["rows"]),
            "",
            "### A fact longer than the overlap",
            "",
            evalkit.markdown_table(block["long_fact"]),
            "",
            "### A sentence longer than the overlap",
            "",
            f"Sentence is {block['sentence_chars']} chars; the chunker backs a cut inside it "
            "up to the sentence start, so it survives without a wider overlap.",
            "",
            evalkit.markdown_table(block["sentence"]),
            "",
        ]
    if "scale" in results:
        block = results["scale"]
        parts += [
            "## Scale (hash embedder; channels and storage only)",
            "",
            evalkit.markdown_table(block["rows"]),
            "",
        ]
    if "concurrency" in results:
        block = results["concurrency"]
        parts += [
            "## Concurrency",
            "",
            evalkit.markdown_table([{
                "workers": block["workers"],
                "queries": block["queries"],
                "sequential_s": block["sequential_seconds"],
                "parallel_s": block["parallel_seconds"],
                "speedup": block["speedup"],
                "mismatches": len(block["mismatches"]),
            }]),
            "",
        ]
    if "adjudication" in results:
        block = results["adjudication"]
        parts += [
            "## Adjudication cost",
            "",
            f"Corpus: {block['corpus_chunks']} chunks. `pairs` counts candidates actually "
            "scored, so it stops growing once `retrieve_k` exceeds the corpus.",
            "",
            evalkit.markdown_table(block["by_retrieve_k"]),
            "",
            "### Per claim (retrieve_k=10)",
            "",
            evalkit.markdown_table(block["detail_at_retrieve_k_10"]),
            "",
        ]
    return "\n".join(parts)


def render_console(results: dict[str, Any]) -> str:
    failed = {
        name: block["error"]
        for name, block in results.items()
        if isinstance(block, dict) and "error" in block
    }
    results = {name: block for name, block in results.items() if name not in failed}
    lines: list[str] = []
    if failed:
        lines.append("probes that failed: " + ", ".join(f"{name} ({msg})" for name, msg in failed.items()))
    if "evidence" in results:
        rows = results["evidence"]["by_top_k"]
        lines.append("evidence gate:")
        for row in rows:
            lines.append(
                f"  top_k={row['top_k']:<2} chars={row['evidence_chars']:<5} "
                f"~{row['pct_of_max_len']:>5}% of max_len  P(ans)={row['answerability']}"
            )
        for row in results["evidence"]["by_chunk_chars"]:
            lines.append(
                f"  chunk={row['chunk_chars']:<5} kept={row['kept']} "
                f"~{row['pct_of_max_len']:>5}% of max_len  P(ans)={row['answerability']}"
            )
        for row in results["evidence"]["forced_large_evidence"]:
            lines.append(
                f"  forced top_k={row['top_k']:<2} chars={row['evidence_chars']:<6} "
                f"~{row['pct_of_max_len']:>5}% of max_len  P(ans)={row['answerability']}"
            )
        for row in results["evidence"]["overflow"]:
            lines.append(
                f"  overflow top_k={row['top_k']:<2} kept={row['kept']:<3} "
                f"chars={row['evidence_chars']:<6} ~{row['pct_of_max_len']:>6}% of max_len  "
                f"P(ans)={row['answerability']}"
            )
    if "batch" in results:
        lines.append("batch determinism:")
        for row in results["batch"]["by_batch_size"]:
            lines.append(
                f"  batch={row['batch_size']:<3} max_delta={row['max_abs_delta']:<10} "
                f"reordered={row['queries_reordered']}/{results['batch']['pairs'] // 5}"
            )
    if "boundary" in results:
        lines.append("boundary survival:")
        for row in results["boundary"]["rows"]:
            lines.append(
                f"  shift={row['shift_chars']:<4} intact={row['fact_intact_in_a_chunk']!s:<5} "
                f"found={row['found_in_top_k']!s:<5} relevance={row['relevance']}"
            )
        for row in results["boundary"]["long_fact"]:
            lines.append(
                f"  long fact {row['fact_chars']} chars at shift={row['shift_chars']:<4} "
                f"intact={row['fact_intact_in_a_chunk']!s:<5} found={row['found_in_top_k']!s:<5} "
                f"relevance={row['relevance']}"
            )
        for row in results["boundary"]["sentence"]:
            lines.append(
                f"  sentence {row['sentence_chars']} chars at shift={row['shift_chars']:<4} "
                f"intact={row['sentence_intact_in_a_chunk']!s:<5} found={row['found_in_top_k']!s:<5} "
                f"relevance={row['relevance']}"
            )
    if "scale" in results:
        lines.append("scale:")
        for row in results["scale"]["rows"]:
            lines.append(
                f"  {row['chunks']:>7} chunks: dense {row['dense_ms_median']}ms "
                f"bm25 {row['bm25_ms_median']}ms  index {row['index_mb']}MB "
                f"vectors {row['vectors_mb']}MB rss {row['rss_mb']}MB"
            )
    if "concurrency" in results:
        block = results["concurrency"]
        lines.append(
            f"concurrency: {block['queries']} queries, {block['workers']} workers, "
            f"{block['sequential_seconds']}s -> {block['parallel_seconds']}s "
            f"({block['speedup']}x), mismatches={len(block['mismatches'])}"
        )
    if "adjudication" in results:
        lines.append("adjudication:")
        for row in results["adjudication"]["by_retrieve_k"]:
            lines.append(
                f"  retrieve_k={row['retrieve_k']:<3} pairs={row['pairs']:<4} "
                f"{row['seconds']}s  supported={row['supported']} unsupported={row['unsupported']} "
                f"not_measured={row['not_measured']}"
            )
        for claim in results["adjudication"]["detail_at_retrieve_k_10"]:
            verdict = (
                "supported" if claim["supported"]
                else ("unsupported" if claim["measured"] else "NOT MEASURED")
            )
            lines.append(
                f"    [{verdict:<12}] P={claim['support_probability']:<7} "
                f"best_rel={claim['best_relevance']:<7} {claim['claim'][:60]}"
            )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench.py", description="Find the Laya RAG engine's limits")
    parser.add_argument("--probes", default="all", help=f"comma list from {', '.join(ALL_PROBES)}")
    parser.add_argument("--out", default=str(ROOT / "eval" / "bench"), help="report prefix")
    parser.add_argument("--scale-sizes", default="500,2000,10000,40000")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    probes = ALL_PROBES if args.probes == "all" else tuple(args.probes.split(","))
    unknown = [probe for probe in probes if probe not in ALL_PROBES]
    if unknown:
        raise SystemExit(f"unknown probes: {', '.join(unknown)}")

    config = EngineConfig()
    if args.device:
        config.device = args.device

    results: dict[str, Any] = {}
    model: LayaModel | None = None

    def real_model() -> LayaModel:
        nonlocal model
        if model is None:
            model = LayaModel(config)
            model.ensure_loaded()
        return model

    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "evidence": lambda: probe_evidence(real_model()),
        "batch": lambda: probe_batch(real_model()),
        "boundary": lambda: probe_boundary(real_model()),
        "scale": lambda: probe_scale([int(size) for size in args.scale_sizes.split(",")]),
        "concurrency": lambda: probe_concurrency(real_model(), workers=args.workers),
        "adjudication": lambda: probe_adjudication(real_model()),
    }
    for probe in probes:
        started = time.perf_counter()
        print(f"running {probe} ...", flush=True)
        try:
            results[probe] = runners[probe]()
        except Exception as exc:  # noqa: BLE001 - one broken probe must not lose the rest
            results[probe] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  {probe} done in {time.perf_counter() - started:.1f}s", flush=True)

    payload = {"checkpoint": config.checkpoint, "probes": results}
    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), payload)
    evalkit.write_text(out.with_suffix(".md"), render_markdown(results))
    print()
    print(render_console(results))
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
