"""Evaluate the engine itself on MuSiQue, under the engine's own chunking.

`tools/musique.py` measures a harness that indexes paragraphs directly at depth 100; this runs
the real `RagEngine` over the same paragraph corpus, with the decision thresholds at zero so the
returned order is the retrieval order, and scores the same metric definitions against the
paragraph-level gold.

**The corpus is always the full split.** `tools/musique.py --limit-queries N` builds its corpus
from only the first N records, so its limited runs retrieve over a smaller paragraph set and are
not comparable to its full-split numbers. A subset run here keeps all 21,098 paragraphs and
narrows only the *queries*, which is what makes `--limit` a real subsample of the same task and
lets `--compare` pair against `retrieval.json`'s per-query MAP@10.

Resumable like `tools/engine_multihop.py`: per-query results are appended and fsynced, so a kill
costs the query in flight and re-running the identical command resumes. `--fresh` discards the
checkpoint, `--shard i/n` runs one contiguous slice, `--merge N` folds finished shards.

    .venv/bin/python tools/engine_musique.py --limit 200 \
        --embed-model BAAI/bge-small-en-v1.5 --out eval/benchmarks/musique/engine-hybrid-200
    .venv/bin/python tools/engine_musique.py --compare \
        --out eval/benchmarks/musique/engine-hybrid-200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit, jobkit  # noqa: E402
from engine_beir import ranking_from_passages, write_corpus  # noqa: E402
from engine_multihop import configure_engine  # noqa: E402
from multihop_rerank_experiment import metrics_from_matches  # noqa: E402
from musique import load_split  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "musique"
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")


def score_query(ranked_positions: Sequence[int], gold_positions: Sequence[int], top_k: int = 10) -> dict:
    """The harness's own metric definitions, over paragraph positions in retrieval order."""
    gold = set(gold_positions)
    matches = [{p} if p in gold else set() for p in ranked_positions]
    return metrics_from_matches(list(range(len(ranked_positions))), matches, max(len(gold), 1), top_k)


def summarize_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in METRICS}
    return {key: sum(row[key] for row in rows) / len(rows) for key in METRICS}


def pair_map10(engine_per_query: Mapping[str, Any], harness_per_query: Mapping[str, float]) -> tuple[dict, dict]:
    """Align the engine's and the harness's per-query MAP@10 on the shared query ids."""
    shared = [qid for qid in engine_per_query if qid in harness_per_query]
    base = {qid: float(harness_per_query[qid]) for qid in shared}
    cand = {qid: float(engine_per_query[qid]["metrics"]["MAP@10"]) for qid in shared}
    return base, cand


def load_task(split_path: Path, limit: int | None = None) -> tuple[dict, dict, dict, dict]:
    """`(corpus, golds, order, queries)` with the **full** corpus and optionally narrowed queries.

    This is the whole reason the function exists: `tools/musique.py --limit-queries` passes the
    limit into `load_split`, which assembles the corpus from only the first N records. That makes
    a limited harness run retrieve over a smaller paragraph set, so its numbers are not
    comparable to a full-split run. Here the corpus is always the whole split and only the query
    list shrinks, which keeps a subset a real subsample of the same task.
    """
    data = load_split(split_path)
    queries = dict(data["queries"])
    if limit:
        queries = dict(list(queries.items())[:limit])
    return data["corpus"], data["golds"], data["order"], queries


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="engine_musique.py")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--embed-model", default=None,
                    help="bi-encoder for the dense channel; omit for the shipped default")
    ap.add_argument("--out", default=str(ROOT_BENCH / "engine-default"))
    ap.add_argument("--harness", default=str(ROOT_BENCH / "retrieval.json"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--retrieve-k", type=int, default=None)
    ap.add_argument("--chunk-chars", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--no-chunk-auto", action="store_true")
    ap.add_argument("--context-prefix", action="store_true")
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--sparse-model", default="opensearch-project/opensearch-neural-sparse-encoding-v1")
    ap.add_argument("--rerank-model", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the QUERIES only; the corpus stays the full split")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--merge", type=int, default=0, metavar="N")
    ap.add_argument("--compare", action="store_true",
                    help="pair the finished report's MAP@10 against the harness's, then exit")
    return ap.parse_args(argv)


def _fingerprint(args: argparse.Namespace) -> str:
    return jobkit.config_fingerprint(
        tool="engine_musique",
        split=args.split,
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
    )


def _build_report(rows: dict) -> dict:
    if jobkit.META_KEY not in rows:
        raise RuntimeError("no shard checkpoints found — run the shards before --merge")
    meta = rows.pop(jobkit.META_KEY)
    per_query = dict(rows)
    return {
        "_settings": meta["_settings"],
        "queries": meta["queries"],
        "summary": summarize_rows([v["metrics"] for v in per_query.values()]),
        "per_query": per_query,
    }


def _compare(args: argparse.Namespace) -> int:
    report_path = Path(str(args.out) + ".json")
    if not report_path.is_file():
        print(f"compare: {report_path} not found — run the measurement first", file=sys.stderr)
        return 2
    engine = json.loads(report_path.read_text(encoding="utf-8"))
    harness = json.loads(Path(args.harness).read_text(encoding="utf-8"))
    harness_rows = harness["per_query"].get("fused")
    if not harness_rows:
        print(f"compare: {args.harness} has no per-query 'fused' MAP@10 to pair against",
              file=sys.stderr)
        return 2
    base, cand = pair_map10(engine["per_query"], harness_rows)
    if not base:
        print("compare: no shared query ids", file=sys.stderr)
        return 2
    boot = evalkit.paired_bootstrap(base, cand, samples=10_000)
    print(f"# Engine vs the MuSiQue harness — paired MAP@10 over {len(base)} queries\n")
    print("| metric | harness `fused` | engine | delta | 95% interval | p |")
    print("| --- | --- | --- | --- | --- | --- |")
    sig = "" if boot["significant"] else " (ns)"
    print(f"| MAP@10 | {sum(base.values()) / len(base):.4f} | {sum(cand.values()) / len(cand):.4f} | "
          f"**{boot['delta']:+.4f}**{sig} | [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] | "
          f"{boot['p_value']:.4f} |")
    print("\nOnly MAP@10 is paired: the harness report persists per-query MAP@10 alone, so the")
    print("other metrics are reported by the engine's own run and are not tested here.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare:
        return _compare(args)
    shard_index, shard_count = jobkit.parse_shard(args.shard)
    fingerprint = _fingerprint(args)

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
    if args.fresh:
        for suffix in (".jsonl", ".manifest.json"):
            Path(str(stem) + suffix).unlink(missing_ok=True)
    run = jobkit.ResumableRun(str(stem) + ".json", fingerprint=fingerprint,
                              shard=(shard_index, shard_count))

    data = load_task(ROOT_BENCH / f"{args.split}.json", args.limit)
    corpus, golds, order, queries = data
    print(f"[{args.split}] {len(corpus)} paragraphs, {len(queries)} queries "
          f"(corpus is always the full split)", flush=True)

    index_dir = configure_engine(args)
    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    docs_dir = ROOT_BENCH / f"engine-corpus-{args.split}"
    doc_ids = write_corpus({doc_id: corpus[doc_id] for doc_id in corpus}, docs_dir)

    config = EngineConfig()
    engine = RagEngine(config)
    if not engine.has_index():
        started = time.perf_counter()
        report = engine.index_paths([docs_dir], rebuild=True)
        print(f"indexed {report.documents} docs / {report.chunks} chunks in "
              f"{time.perf_counter() - started:.0f}s "
              f"(derived chunk size {report.derived_chunk_chars or config.chunk_chars})", flush=True)
    else:
        print(f"reusing the index at {index_dir}", flush=True)

    if jobkit.META_KEY not in run:
        run.record(jobkit.META_KEY, {
            "_settings": {
                **vars(args),
                "index_dir": str(index_dir),
                "dense_weight": config.resolved_dense_weight,
                "chunk_chars_effective": config.chunk_chars,
                "chunks": len(engine.list_chunks()),
                "corpus": len(corpus),
            },
            "queries": queries,
        })

    mine = jobkit.shard_items(list(queries), shard_index, shard_count)
    todo = run.pending(mine)
    print(f"  {len(todo)} of {len(mine)} queries to score ({len(run) - 1} checkpointed)",
          flush=True)

    started = time.perf_counter()
    for position, qid in enumerate(todo, 1):
        result = engine.query(queries[qid], top_k=args.top_k, max_passage_chars=None)
        retrieved = ranking_from_passages(result.passages, doc_ids)
        ranked = [order[doc_id] for doc_id in retrieved if doc_id in order]
        run.record(qid, {
            "metrics": score_query(ranked, golds.get(qid, []), args.top_k),
            "ranking": ranked[: args.top_k],
            "considered": result.considered,
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
    s = payload["summary"]
    print(f"{Path(args.out).name}: Hits@10 {s['Hits@10']:.4f} Hits@4 {s['Hits@4']:.4f} "
          f"MAP@10 {s['MAP@10']:.4f} MRR@10 {s['MRR@10']:.4f} over {len(per_query)} queries")
    print(f"wrote {stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
