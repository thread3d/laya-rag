"""Evaluate the engine itself on MultiHop-RAG, under the engine's own chunking.

Every MultiHop-RAG number in this project comes from `tools/multihop_official.py`, a harness
that reimplements the channels at the benchmark's own 256-token chunking. `docs/FINDINGS.md`
names the consequence as *the largest known gap in the evidence*: the engine's multi-hop
retrieval under its own corpus-aware chunking is **not measured**, and the BEIR replica was
already shown to overstate the real engine by 0.0144. This runs the real `RagEngine` over the
609 articles, with the decision thresholds at zero so nothing is filtered and the returned
order is the retrieval order, and scores the benchmark's own metrics on the chunks the engine
actually returns.

Two configurations are worth running, and they answer different questions:

* **default** (no ``--embed-model``): the shipped path. Laya's encoder resolves to a dense
  weight of zero, so recall is lexical and the decision head still filters.
* **``--embed-model BAAI/bge-small-en-v1.5``**: the hybrid path the README's comparisons use,
  dense fused with the stemmed lexical channel.

Because the engine is the expensive part, this is a jobkit-backed long run: each query's
metrics are appended to `<out>.jsonl` and fsynced as it finishes, so a kill costs at most the
query in flight and re-running the identical command resumes. `--fresh` discards the
checkpoint, `--shard i/n` runs one contiguous slice, and `--merge N` folds finished shards.

    # the cheap smoke first, then the real run
    .venv/bin/python tools/engine_multihop.py --limit 25 --embed-model BAAI/bge-small-en-v1.5
    .venv/bin/python tools/engine_multihop.py --out eval/benchmarks/multihop-rag/engine-default
    .venv/bin/python tools/engine_multihop.py --embed-model BAAI/bge-small-en-v1.5 \
        --out eval/benchmarks/multihop-rag/engine-hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import jobkit  # noqa: E402
from engine_beir import ranking_from_passages, write_corpus  # noqa: E402
from multihop_official import official_metrics  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
METRIC_KEYS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")


def answerable_items(items: Sequence[Mapping[str, Any]]) -> list[tuple[int, Mapping[str, Any]]]:
    """The benchmark's answerable records, keeping their index in the original file.

    ``null_query`` records are excluded: their gold answer is literally "Insufficient
    information", so scoring retrieval against them would measure nothing.
    """
    return [(i, x) for i, x in enumerate(items) if x["question_type"] != "null_query"]


def gold_facts(item: Mapping[str, Any]) -> list[str]:
    return [e["fact"] for e in item["evidence_list"]]


def select_items(
    items: Sequence[Any], limit: int | None = None, sample_seed: int | None = None
) -> list[Any]:
    """Limit the scored records, reproducing the harness's sampling when a seed is given.

    `tools/multihop_official.py` samples by position over the answerable records and sorts the
    result; with the same `--limit` and `--sample-seed` this returns exactly that subset, which
    is what makes an engine number comparable to the harness's 400-query bar. Without a seed the
    limit is a plain prefix, which is fine for a smoke and not for a comparison.
    """
    items = list(items)
    if not limit:
        return items
    if sample_seed is not None and limit < len(items):
        sampled = sorted(random.Random(sample_seed).sample(range(len(items)), limit))
        return [items[i] for i in sampled]
    return items[:limit]


def summarize_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average the per-query metric rows — the aggregation `official_metrics` uses."""
    if not rows:
        return {key: 0.0 for key in METRIC_KEYS}
    return {key: sum(row[key] for row in rows) / len(rows) for key in METRIC_KEYS}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="engine_multihop.py")
    ap.add_argument("--embed-model", default=None,
                    help="bi-encoder for the engine's dense channel; omit for the shipped "
                         "default, in which Laya's encoder resolves to a dense weight of zero")
    ap.add_argument("--out", default=str(ROOT_BENCH / "engine-default"),
                    help="report stem; jobkit sidecars are written beside it")
    ap.add_argument("--top-k", type=int, default=10, help="the benchmark's metric is @10")
    ap.add_argument("--retrieve-k", type=int, default=None,
                    help="candidates considered before rerank (engine default when omitted)")
    ap.add_argument("--chunk-chars", type=int, default=None,
                    help="override the chunk size; the default is corpus-aware (p90)")
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--no-chunk-auto", action="store_true",
                    help="disable corpus-aware chunk sizing, for a fixed-size ablation")
    ap.add_argument("--context-prefix", action="store_true")
    ap.add_argument("--sparse", action="store_true",
                    help="index the learned-sparse channel as well")
    ap.add_argument("--sparse-model", default="opensearch-project/opensearch-neural-sparse-encoding-v1")
    ap.add_argument("--rerank-model", default=None)
    ap.add_argument("--tag", default="", help="suffix for the engine's index directory")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of queries (smoke)")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="with --limit, sample this seed's subset instead of the first N — "
                         "matching tools/multihop_official.py's --sample-seed, so the engine "
                         "number is comparable to the harness's 400-query bar")
    ap.add_argument("--shard", default="0/1", help="run one contiguous slice, 'i/n'")
    ap.add_argument("--fresh", action="store_true", help="discard any existing checkpoint")
    ap.add_argument("--merge", type=int, default=0, metavar="N",
                    help="only fold N finished shards into <out>.json, then exit")
    return ap.parse_args(argv)


def index_dir_for(args: argparse.Namespace) -> Path:
    """Where a run's index lives, from the report stem and tag alone — no environment change.

    Kept separate from :func:`configure_engine` so a caller can consult the index (its identity,
    whether it has to be rebuilt) *before* the engine is configured.
    """
    report_stem = Path(args.out)
    return report_stem.parent / "index" / f"{report_stem.name}{'-' + args.tag if args.tag else ''}"


def configure_engine(args: argparse.Namespace) -> Path:
    """Apply the run's knobs through the environment, as `tools/engine_beir.py` does.

    The index directory is derived from the report stem and tag, so the two configurations do
    not share one index — reusing a dense index built with a different encoder is the failure
    mode `engine_beir`'s `--tag` exists to prevent.
    """
    index_dir = index_dir_for(args)
    os.environ["LAYA_RAG_INDEX_DIR"] = str(index_dir)
    if args.embed_model:
        os.environ["LAYA_RAG_EMBED_MODEL"] = args.embed_model
    else:
        os.environ.pop("LAYA_RAG_EMBED_MODEL", None)
    # Nothing may be filtered: the returned order is the engine's retrieval order.
    os.environ["LAYA_RAG_RELEVANCE_THRESHOLD"] = "0.0"
    os.environ["LAYA_RAG_ANSWERABILITY_THRESHOLD"] = "0.0"
    for name, value in (
        ("CHUNK_CHARS", args.chunk_chars),
        ("CHUNK_OVERLAP_CHARS", args.chunk_overlap),
        ("RETRIEVE_K", args.retrieve_k),
    ):
        if value is not None:
            os.environ[f"LAYA_RAG_{name}"] = str(value)
        else:
            os.environ.pop(f"LAYA_RAG_{name}", None)
    os.environ["LAYA_RAG_CHUNK_CHARS_AUTO"] = "0" if args.no_chunk_auto else "1"
    os.environ["LAYA_RAG_CONTEXT_PREFIX"] = "1" if args.context_prefix else "0"
    if args.sparse:
        os.environ["LAYA_RAG_SPARSE_MODEL"] = args.sparse_model
    else:
        os.environ.pop("LAYA_RAG_SPARSE_MODEL", None)
    if args.rerank_model:
        os.environ["LAYA_RAG_RERANK_MODEL"] = args.rerank_model
    else:
        os.environ.pop("LAYA_RAG_RERANK_MODEL", None)
    return index_dir


def _ingest_settings(args: argparse.Namespace) -> dict[str, Any]:
    """The knobs that shape the index rather than the query, for the index's identity record."""
    return {
        "embed_model": args.embed_model or "engine-default",
        "sparse": args.sparse,
        "sparse_model": args.sparse_model if args.sparse else None,
        "chunk_chars": args.chunk_chars,
        "chunk_overlap": args.chunk_overlap,
        "chunk_auto": not args.no_chunk_auto,
        "context_prefix": args.context_prefix,
    }


def _fingerprint(args: argparse.Namespace, identity: str | None) -> str:
    """The run's fingerprint. `identity=None` reproduces it as it was before the index was in it."""
    return jobkit.config_fingerprint(
        tool="engine_multihop",
        embed_model=args.embed_model or "engine-default",
        top_k=args.top_k,
        retrieve_k=args.retrieve_k,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        chunk_auto=not args.no_chunk_auto,
        context_prefix=args.context_prefix,
        sparse=args.sparse,
        sparse_model=args.sparse_model if args.sparse else None,
        rerank_model=args.rerank_model,
        limit=args.limit,
        sample_seed=args.sample_seed,
        # What the numbers were measured *against*, not just how the run was configured. Absent
        # for the legacy form, so a checkpoint written before this can be recognised and rebound.
        **({"index": identity} if identity is not None else {}),
    )


def _build_report(rows: dict) -> dict:
    """Rebuild the report from merged rows, using the reserved shared-state row."""
    if jobkit.META_KEY not in rows:
        raise RuntimeError("no shard checkpoints found — run the shards before --merge")
    meta = rows.pop(jobkit.META_KEY)
    per_query = {k: v for k, v in rows.items()}
    return {
        "_settings": meta["_settings"],
        "queries": meta["queries"],
        "summary": summarize_rows([v["metrics"] for v in per_query.values()]),
        "per_query": per_query,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("engine_multihop.py")
    shard_index, shard_count = jobkit.parse_shard(args.shard)
    # The identity of the index this run is measured against: the corpus file and the knobs that
    # shape ingest. It goes into the run fingerprint, so a checkpoint cannot be resumed across a
    # different index, and it decides whether the index on disk has to be rebuilt. Computed before
    # the merge branch so shards and their fold agree on it.
    identity = jobkit.index_identity(
        _ingest_settings(args), [ROOT_BENCH / "corpus.json", ROOT_BENCH / "MultiHopRAG.json"]
    )
    fingerprint = _fingerprint(args, identity)
    index_dir = index_dir_for(args)

    if args.merge:
        try:
            merged = jobkit.merge_shards(args.out, args.merge, fingerprint=fingerprint,
                                         build=_build_report)
        except (RuntimeError, jobkit.FingerprintMismatch) as exc:
            print(f"merge failed: {exc}", file=sys.stderr)
            return 2
        print(f"merged {args.merge} shard(s) -> {args.out}.json "
              f"({merged['_job']['completed']} queries)")
        return 0

    stem = jobkit.shard_stem(args.out, shard_index, shard_count)
    if not args.fresh and jobkit.index_state(index_dir, identity) == "absent" and \
            Path(index_dir).exists() and \
            jobkit.rebind_legacy_fingerprint(stem, legacy=_fingerprint(args, None),
                                             current=fingerprint):
        print(f"  {Path(str(stem)).name}: rebound from the pre-identity fingerprint — the index "
              f"has no identity record, so it is adopted as measured", flush=True)
    if args.fresh:
        for suffix in (".jsonl", ".manifest.json"):
            Path(str(stem) + suffix).unlink(missing_ok=True)
    run = jobkit.ResumableRun(str(stem) + ".json", fingerprint=fingerprint,
                              shard=(shard_index, shard_count))

    raw = json.load(open(ROOT_BENCH / "corpus.json"))
    items_all = json.load(open(ROOT_BENCH / "MultiHopRAG.json"))
    items = select_items(answerable_items(items_all), args.limit, args.sample_seed)

    index_dir = configure_engine(args)
    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    docs_dir = ROOT_BENCH / "engine-corpus"
    doc_ids = write_corpus({a["url"]: a["title"] + " " + a["body"] for a in raw}, docs_dir)
    items_by_index = {i: x for i, x in items}
    queries = {str(i): x["query"] for i, x in items}

    config = EngineConfig()
    engine = RagEngine(config)
    state = jobkit.index_state(index_dir, identity)
    if state == "match":
        print(f"reusing the index at {index_dir} — its identity matches", flush=True)
    elif state == "absent" and engine.has_index():
        # Built before identities were recorded. Adopted once rather than rebuilt: every existing
        # engine index predates the sidecar, the same "adopt a legacy cache once" rule
        # `beir.cached_embed` applies, and the identity is written from here on.
        jobkit.write_index_identity(index_dir, identity, **{"multihop": True})
        print(f"reusing the index at {index_dir} — no identity was recorded; adopted", flush=True)
    else:
        detail = "the corpus or the ingest settings changed" if state == "mismatch" else "no index"
        started = time.perf_counter()
        report = engine.index_paths([docs_dir], rebuild=True)
        jobkit.write_index_identity(index_dir, identity, **{"multihop": True})
        print(f"indexed {report.documents} docs / {report.chunks} chunks in "
              f"{time.perf_counter() - started:.0f}s "
              f"(derived chunk size {report.derived_chunk_chars or config.chunk_chars}, "
              f"dense={config.resolved_dense_weight != 0.0}, sparse={config.sparse_enabled}; "
              f"{detail})",
              flush=True)

    if jobkit.META_KEY not in run:
        run.record(jobkit.META_KEY, {
            "_settings": {
                **vars(args),
                # The tool that wrote this report, and the arguments it was given. Added because four successive
                # static screens failed to identify a report's producer from its settings alone (see SCORE-LIFT.md):
                # the flags describe an interface that several tools share, and one of them is a cache builder
                # rather than a reporter. A field in the artefact settles in one look what inference cannot.
                "producer": Path(__file__).name,
                "argv": sys.argv[1:],
                "index_dir": str(index_dir),
                "index_identity": identity,
                "embed_model_effective": config.embed_model or "engine-default",
                "dense_weight": config.resolved_dense_weight,
                "chunk_chars_effective": config.chunk_chars,
                "sparse_enabled": config.sparse_enabled,
                "chunks": len(engine.list_chunks()),
            },
            "queries": {k: v for k, v in queries.items()},
        })

    mine = jobkit.shard_items(list(queries), shard_index, shard_count)
    todo = run.pending(mine)
    print(f"  {len(todo)} of {len(mine)} queries to score ({len(run) - 1} checkpointed)",
          flush=True)

    started = time.perf_counter()
    for position, key in enumerate(todo, 1):
        item = items_by_index[int(key)]
        result = engine.query(queries[key], top_k=args.top_k, max_passage_chars=None)
        texts = [str(p.get("text") or "") for p in result.passages[: args.top_k]]
        metrics, _ = official_metrics([texts], [gold_facts(item)])
        run.record(key, {
            "metrics": metrics,
            "ranking": ranking_from_passages(result.passages, doc_ids),
            "chunks": [str(p.get("chunk_id") or "") for p in result.passages[: args.top_k]],
            "considered": result.considered,
            "kept": result.kept,
        })
        if position % 25 == 0:
            run.touch(total=len(mine))
            print(f"    {position}/{len(todo)} in {time.perf_counter() - started:.0f}s", flush=True)

    shared = run.get(jobkit.META_KEY)
    per_query = {k: v for k, v in run.values().items() if k != jobkit.META_KEY}
    payload = {
        "_settings": {**shared["_settings"], "seconds": round(time.perf_counter() - started, 1)},
        "queries": shared["queries"],
        "summary": summarize_rows([v["metrics"] for v in per_query.values()]),
        "per_query": per_query,
    }
    run.finish(payload, total=len(mine))
    summary = payload["summary"]
    print(f"{Path(args.out).name}: Hits@10 {summary['Hits@10']:.4f} Hits@4 {summary['Hits@4']:.4f} "
          f"MAP@10 {summary['MAP@10']:.4f} MRR@10 {summary['MRR@10']:.4f} "
          f"over {len(per_query)} queries")
    print(f"wrote {stem}.json")
    if shard_count > 1:
        print(f"  run all {shard_count} shards, then: --merge {shard_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
