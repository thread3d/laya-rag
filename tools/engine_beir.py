"""Evaluate the engine itself on a BEIR dataset (not the ``tools/beir.py`` replica).

Every BEIR number in this project comes from `tools/beir.py`, which reimplements the
engine's channels rather than calling the engine. That is exactly how the engine fused at
`k=60` while the harness used `k=5` for several rounds. This runs the real `RagEngine` over a
dataset's test queries, with the decision thresholds set to zero so nothing is filtered and
the returned order is the retrieval order, and scores nDCG@10.

    # two runs, to isolate the sparse channel's contribution at the engine level
    .venv/bin/python tools/engine_beir.py --dataset scifact --tag base
    .venv/bin/python tools/engine_beir.py --dataset scifact --tag sparse --sparse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir, evalkit, jobkit  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SPARSE = "opensearch-project/opensearch-neural-sparse-encoding-v1"


def write_corpus(corpus: Mapping[str, str], docs_dir: Path) -> list[str]:
    """One ``.txt`` per document, named by position; returns the id order."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = list(corpus)
    for index, doc_id in enumerate(doc_ids):
        (docs_dir / f"{index:06d}.txt").write_text(corpus[doc_id], encoding="utf-8")
    return doc_ids


def ranking_from_passages(passages: Sequence[Mapping[str, Any]], doc_ids: Sequence[str]) -> list[str]:
    """Map returned passage paths back to document ids, de-duplicated, order preserved."""
    ranking: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        path = str(passage.get("path") or "")
        stem = Path(path).stem
        if not stem.isdigit():
            continue
        index = int(stem)
        if not (0 <= index < len(doc_ids)):
            continue
        doc_id = doc_ids[index]
        if doc_id not in seen:
            seen.add(doc_id)
            ranking.append(doc_id)
    return ranking


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="engine_beir.py")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    ap.add_argument("--embed-model", default=DEFAULT_MODEL)
    ap.add_argument("--sparse", action="store_true", help="also index the sparse channel")
    ap.add_argument("--sparse-model", default=DEFAULT_SPARSE)
    ap.add_argument("--tag", default="base", help="names the index directory and outputs")
    ap.add_argument("--min-docs", type=int, default=0,
                    help="promote the best passage of this many distinct documents first")
    ap.add_argument("--rerank-model", default=None,
                    help="cross-encoder used as the second stage")
    ap.add_argument("--rerank-coverage-lambda", type=float, default=None,
                    help="explicit coverage weight; omit to use the engine's auto default")
    ap.add_argument("--k1", type=float, default=None, help="BM25 k1 (harness baseline 0.9)")
    ap.add_argument("--b", type=float, default=None, dest="b_bm25",
                    help="BM25 b (harness baseline 0.4)")
    ap.add_argument("--chunk-chars", type=int, default=None,
                    help="chunk size; the default 1200 splits short abstracts the replica "
                         "embeds whole")
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--min-chunk-chars", type=int, default=None)
    ap.add_argument("--chunk-merge-short", action="store_true",
                    help="keep documents of at most chunk_chars+overlap as one chunk")
    ap.add_argument("--retrieve-k", type=int, default=None,
                    help="candidates considered before rerank; chunking packs several "
                         "chunks per document into this budget")
    ap.add_argument("--context-prefix", action="store_true",
                    help="index a structural breadcrumb (path + title line) on every chunk, "
                         "so later chunks keep the document's context")
    ap.add_argument("--chunk-auto", action="store_true",
                    help="derive the chunk size from the corpus document lengths (p90)")
    ap.add_argument("--chunk-auto-percentile", type=float, default=None,
                    help="quantile used by --chunk-auto (default 0.9)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="cap the number of test queries")
    ap.add_argument("--index-dir", default=None,
                    help="read an existing engine index instead of deriving one from --out/--tag")
    ap.add_argument("--fusion-mode", default="rrf", choices=("rrf", "convex"),
                    help="the first stage's fusion rule; `convex` is the engine-level arm of P5a")
    ap.add_argument("--dense-weight", type=float, default=None,
                    help="override DENSE_WEIGHT, which is also the convex dense weight")
    ap.add_argument("--no-second-stage", action="store_true",
                    help="return the first stage's own order with no decision model in the path")
    ap.add_argument("--out", default=str(ROOT / "eval" / "engine-beir"))
    ap.add_argument("--shard", default="0/1", help="run one contiguous slice, 'i/n'")
    ap.add_argument("--fresh", action="store_true", help="discard any existing checkpoint")
    ap.add_argument("--merge", type=int, default=0, metavar="N",
                    help="only fold N finished shards into the report, then exit")
    return ap.parse_args(argv)


def report_stem(args: argparse.Namespace) -> Path:
    return Path(args.out) / f"{args.dataset}-{args.tag}"


def index_dir(args: argparse.Namespace) -> Path:
    return Path(args.out) / "index" / f"{args.dataset}-{args.tag}"


def resolve_index_dir(args: argparse.Namespace) -> Path:
    """The index this run reads: `--index-dir` when given, otherwise the derived one.

    One accessor for both the engine's `LAYA_RAG_INDEX_DIR` and the identity sidecar, so a borrowed
    index is checked and stamped where it actually lives rather than where a derived name points.
    """
    return Path(args.index_dir) if args.index_dir else index_dir(args)


def _cpu_seconds() -> float:
    """This process's own user+sys CPU time, in seconds."""
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


_cpu_at_start = _cpu_seconds()


def engine_version() -> str:
    """The vendored Laya version this run measured, so a bump's effect is readable from the report.

    Without it a report cannot say which engine produced it, and comparing versions means re-running the
    old one — which is what had to be done by hand for the 0.4.0 to 0.4.14 check.
    """
    try:
        import laya

        return str(getattr(laya, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 - a missing engine must not break a settings block
        return "unavailable"


def _ingest_settings(args: argparse.Namespace) -> dict[str, Any]:
    """The knobs that shape the index rather than the query, for the index's identity record."""
    return {
        "embed_model": args.embed_model,
        "sparse": args.sparse,
        "sparse_model": args.sparse_model if args.sparse else None,
        "chunk_chars": args.chunk_chars,
        "chunk_overlap": args.chunk_overlap,
        "chunk_auto": args.chunk_auto,
        "chunk_auto_percentile": args.chunk_auto_percentile,
        "min_chunk_chars": args.min_chunk_chars,
        "chunk_merge_short": args.chunk_merge_short,
        "context_prefix": args.context_prefix,
    }


def _fingerprint(args: argparse.Namespace, identity: str | None) -> str:
    """The run's fingerprint. `identity=None` reproduces it as it was before the index was in it."""
    return jobkit.config_fingerprint(
        tool="engine_beir",
        dataset=args.dataset,
        tag=args.tag,
        embed_model=args.embed_model,
        sparse=args.sparse,
        sparse_model=args.sparse_model if args.sparse else None,
        rerank_model=args.rerank_model,
        rerank_coverage_lambda=args.rerank_coverage_lambda,
        min_docs=args.min_docs,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        chunk_auto=args.chunk_auto,
        context_prefix=args.context_prefix,
        retrieve_k=args.retrieve_k,
        top_k=args.top_k,
        limit=args.limit,
        fusion_mode=args.fusion_mode,
        dense_weight=args.dense_weight,
        second_stage=not args.no_second_stage,
        # What the numbers were measured *against*, not just how the run was configured. Absent
        # for the legacy form, so a checkpoint written before this can be recognised and rebound.
        **({"index": identity} if identity is not None else {}),
    )


def _build_report(rows: dict, args: argparse.Namespace, qrels: Mapping[str, Any],
                  corpus_size: int, identity: str = "") -> dict:
    """Rebuild the report from merged rankings; the metrics need every query at once."""
    if jobkit.META_KEY not in rows:
        raise RuntimeError("no shard checkpoints found — run the shards before --merge")
    rows.pop(jobkit.META_KEY)
    rankings = {qid: value["ranking"] for qid, value in rows.items()}
    evaluation = evalkit.evaluate_rankings(rankings, qrels, k_values=(10,),
                                           ignore_identical_ids=True)
    return {
        "_settings": {**vars(args), "index_identity": identity, "laya_version": engine_version()},
        "documents": corpus_size,
        "queries": len(rankings),
        "summary": evaluation["summary"],
        "per_query": evalkit.per_query_metric(evaluation["per_query"], "nDCG@10"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("engine_beir.py")
    shard_index, shard_count = jobkit.parse_shard(args.shard)
    out = Path(args.out)
    index_path = resolve_index_dir(args)
    docs_dir = out / "docs" / args.dataset

    dataset_dir = beir.ensure_dataset(args.dataset, Path(args.data_root), download=False)
    corpus = beir.load_corpus(dataset_dir, include_titles=True)
    queries = beir.load_queries(dataset_dir)
    qrels = beir.load_qrels(dataset_dir, split="test")
    # The identity of the index this run is measured against: the corpus file and the knobs that
    # shape ingest. It goes into the run fingerprint, so a checkpoint cannot be resumed across a
    # different index, and it decides whether the index on disk has to be rebuilt.
    identity = jobkit.index_identity(_ingest_settings(args), [dataset_dir / "corpus.jsonl"])
    fingerprint = _fingerprint(args, identity)

    if args.merge:
        try:
            merged = jobkit.merge_shards(
                report_stem(args), args.merge, fingerprint=fingerprint,
                build=lambda rows: _build_report(rows, args, qrels, len(corpus), identity),
            )
        except (RuntimeError, jobkit.FingerprintMismatch) as exc:
            print(f"merge failed: {exc}", file=sys.stderr)
            return 2
        print(f"merged {args.merge} shard(s) -> {report_stem(args)}.json "
              f"({merged['queries']} queries)")
        return 0

    os.environ["LAYA_RAG_INDEX_DIR"] = str(index_path)
    # Query-time knobs. They shape the fingerprint (below) but not the index, so they are not part
    # of its identity: the same index can be scored under RRF and convex without a rebuild.
    os.environ["LAYA_RAG_FUSION_MODE"] = args.fusion_mode
    os.environ["LAYA_RAG_SECOND_STAGE"] = "0" if args.no_second_stage else "1"
    if args.dense_weight is not None:
        os.environ["LAYA_RAG_DENSE_WEIGHT"] = str(args.dense_weight)
    else:
        os.environ.pop("LAYA_RAG_DENSE_WEIGHT", None)
    os.environ["LAYA_RAG_EMBED_MODEL"] = args.embed_model
    # Nothing may be filtered, so the returned order is the engine's retrieval order.
    os.environ["LAYA_RAG_RELEVANCE_THRESHOLD"] = "0.0"
    os.environ["LAYA_RAG_ANSWERABILITY_THRESHOLD"] = "0.0"
    if args.min_docs:
        os.environ["LAYA_RAG_RERANK_MIN_DOCS"] = str(args.min_docs)
    else:
        os.environ.pop("LAYA_RAG_RERANK_MIN_DOCS", None)
    if args.rerank_model:
        os.environ["LAYA_RAG_RERANK_MODEL"] = args.rerank_model
    else:
        os.environ.pop("LAYA_RAG_RERANK_MODEL", None)
    if args.rerank_coverage_lambda is not None:
        os.environ["LAYA_RAG_RERANK_COVERAGE_LAMBDA"] = str(args.rerank_coverage_lambda)
    else:
        os.environ.pop("LAYA_RAG_RERANK_COVERAGE_LAMBDA", None)
    if args.k1 is not None:
        os.environ["LAYA_RAG_BM25_K1"] = str(args.k1)
    else:
        os.environ.pop("LAYA_RAG_BM25_K1", None)
    if args.b_bm25 is not None:
        os.environ["LAYA_RAG_BM25_B"] = str(args.b_bm25)
    else:
        os.environ.pop("LAYA_RAG_BM25_B", None)
    for name, value in (
        ("CHUNK_CHARS", args.chunk_chars),
        ("CHUNK_OVERLAP_CHARS", args.chunk_overlap),
        ("MIN_CHUNK_CHARS", args.min_chunk_chars),
    ):
        if value is not None:
            os.environ[f"LAYA_RAG_{name}"] = str(value)
        else:
            os.environ.pop(f"LAYA_RAG_{name}", None)
    os.environ["LAYA_RAG_CHUNK_MERGE_SHORT"] = "1" if args.chunk_merge_short else "0"
    if args.retrieve_k is not None:
        os.environ["LAYA_RAG_RETRIEVE_K"] = str(args.retrieve_k)
    else:
        os.environ.pop("LAYA_RAG_RETRIEVE_K", None)
    os.environ["LAYA_RAG_CONTEXT_PREFIX"] = "1" if args.context_prefix else "0"
    os.environ["LAYA_RAG_CHUNK_CHARS_AUTO"] = "1" if args.chunk_auto else "0"
    if args.chunk_auto_percentile is not None:
        os.environ["LAYA_RAG_CHUNK_CHARS_AUTO_PERCENTILE"] = str(args.chunk_auto_percentile)
    else:
        os.environ.pop("LAYA_RAG_CHUNK_CHARS_AUTO_PERCENTILE", None)
    if args.sparse:
        os.environ["LAYA_RAG_SPARSE_MODEL"] = args.sparse_model
    else:
        os.environ.pop("LAYA_RAG_SPARSE_MODEL", None)

    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    test_queries = {qid: text for qid, text in queries.items() if qid in qrels}
    if args.limit:
        test_queries = dict(list(test_queries.items())[: args.limit])
        # Same dilution trap as `beir.py`: the evaluator macro-averages over the qrels it is handed,
        # so a capped query set must be paired with qrels capped to it.
        qrels = beir.restrict_qrels(qrels, test_queries)
    doc_ids = write_corpus(corpus, docs_dir)

    stem = jobkit.shard_stem(report_stem(args), shard_index, shard_count)
    config = EngineConfig()
    engine = RagEngine(config)
    state = jobkit.index_state(index_path, identity)
    # An index with no identity record predates the sidecar and is adopted as measured. Its
    # checkpoint then has to be migrated too, or folding the index into the fingerprint throws
    # away hours of scoring — and it must be migrated *before* the identity is written, so a
    # retry after an unrelated failure still recognises the adoption.
    adopting = state == "absent" and engine.has_index()
    if not args.fresh and adopting and jobkit.rebind_legacy_fingerprint(
            stem, legacy=_fingerprint(args, None), current=fingerprint):
        print(f"  {Path(str(stem)).name}: rebound from the pre-identity fingerprint",
              flush=True)
    if state == "match":
        print(f"reusing the index at {index_path} — its identity matches", flush=True)
    elif adopting:
        jobkit.write_index_identity(index_path, identity, dataset=args.dataset, tag=args.tag)
        print(f"reusing the index at {index_path} — no identity was recorded; adopted", flush=True)
    else:
        detail = "the corpus or the ingest settings changed" if state == "mismatch" else "no index"
        started = time.perf_counter()
        report = engine.index_paths([docs_dir], rebuild=True)
        jobkit.write_index_identity(index_path, identity, dataset=args.dataset, tag=args.tag)
        print(f"indexed {report.documents} docs / {report.chunks} chunks in "
              f"{time.perf_counter() - started:.0f}s ({detail})", flush=True)
    print(f"sparse configured: {config.sparse_enabled}", flush=True)

    if args.fresh:
        for suffix in (".jsonl", ".manifest.json"):
            Path(str(stem) + suffix).unlink(missing_ok=True)
    run = jobkit.ResumableRun(str(stem) + ".json", fingerprint=fingerprint,
                              shard=(shard_index, shard_count))
    if jobkit.META_KEY not in run:
        run.record(jobkit.META_KEY, {"_settings": {**vars(args), "index_identity": identity, "laya_version": engine_version()},
                                     "sparse": config.sparse_enabled})

    mine = jobkit.shard_items(list(test_queries), shard_index, shard_count)
    todo = run.pending(mine)
    print(f"  {len(todo)} of {len(mine)} queries to score ({len(run) - 1} checkpointed)",
          flush=True)

    started = time.perf_counter()
    for position, query_id in enumerate(todo, start=1):
        result = engine.query(test_queries[query_id], top_k=args.top_k)
        run.record(query_id, {"ranking": ranking_from_passages(result.passages, doc_ids)})
        if position % 50 == 0:
            run.touch(total=len(mine))
            print(f"  {position}/{len(todo)} in {time.perf_counter() - started:.0f}s", flush=True)
    seconds = time.perf_counter() - started

    rankings = {qid: value["ranking"] for qid, value in run.values().items()
                if qid != jobkit.META_KEY}
    evaluation = evalkit.evaluate_rankings(rankings, qrels, k_values=(10,), ignore_identical_ids=True)
    payload = {
        "_settings": {**vars(args), "index_identity": identity, "laya_version": engine_version()},
        "documents": len(corpus),
        "queries": len(rankings),
        "sparse": config.sparse_enabled,
        # Wall clock is NOT a performance signal on this host: it is shared with other users, so a
        # version comparison by `seconds` measures everyone's load. `cpu_seconds` is the process's own
        # CPU time, which is what a before/after can use. Both are recorded; only the second is usable
        # for comparing versions, and neither is comparable across machines.
        "seconds": round(seconds, 1),
        "cpu_seconds": round(_cpu_seconds() - _cpu_at_start, 1),
        "summary": evaluation["summary"],
        "per_query": evalkit.per_query_metric(evaluation["per_query"], "nDCG@10"),
    }
    run.finish(payload, total=len(mine))
    print(f"{args.dataset} {args.tag}: nDCG@10 {evaluation['summary']['nDCG@10']:.4f} "
          f"over {len(rankings)} queries in {seconds:.0f}s")
    print("wrote", str(stem) + ".json")
    if shard_count > 1:
        print(f"  run all {shard_count} shards, then: --merge {shard_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
