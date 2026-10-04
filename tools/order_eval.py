"""Rerank the MuSiQue validation pool with a cross-encoder, and compare arms — ROADMAP Phase 2.2.

`tools/order_train.py` adapts a cross-encoder on MuSiQue's own train split. This measures whether
that helped, by reranking the *same* cached pool with the same architecture zero-shot and adapted,
so the comparison is adapted-versus-zero-shot rather than adapted-versus-whatever-was-default.

Three arms, all over the same candidates and scored by the same metric function:

* `fused` — the pool's own order, i.e. what the first stage already returned;
* `ce-cached` — the cross-encoder scores `tools/musique.py` cached (`bge-reranker-base`);
* `ce-model` — the cross-encoder passed to `--cross-encoder` (a Hub id or a saved model path).

`--limit` narrows the *queries*; the pool and the gold are always the full split, and the metrics
are scored against the subset that was reranked (a limited run against the full qrels would dilute
every number toward zero).

    .venv/bin/python tools/order_eval.py --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --out eval/benchmarks/musique/order-zero-shot
    .venv/bin/python tools/order_eval.py --cross-encoder eval/order-ce-musique \
        --out eval/benchmarks/musique/order-adapted
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
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "order-hf-cache"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit, jobkit  # noqa: E402
from multihop_rerank_experiment import metrics_from_matches  # noqa: E402
from musique import gold_matches, load_split  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "musique"
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")
ARMS = ("fused", "ce-cached", "ce-model")


def rank_positions(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def score_arms(pool: Sequence[str], gold: Sequence[int], order: Mapping[str, int],
               cached: Sequence[float], model_scores: Sequence[float],
               top_k: int) -> dict[str, dict[str, float]]:
    matches = gold_matches(list(pool), gold, order)
    gold_count = max(len(gold), 1)
    return {
        "fused": metrics_from_matches(list(range(min(top_k, len(pool)))), matches, gold_count, top_k),
        "ce-cached": metrics_from_matches(rank_positions(cached)[:top_k], matches, gold_count, top_k),
        "ce-model": metrics_from_matches(rank_positions(model_scores)[:top_k], matches, gold_count, top_k),
    }


def summarize(per_query: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        rows = [per_query[q][arm] for q in per_query]
        out[arm] = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}
    return out


def compare(per_query: Mapping[str, Any], baseline: str, candidate: str,
            samples: int = 10_000) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        base = {q: per_query[q][baseline][metric] for q in per_query}
        cand = {q: per_query[q][candidate][metric] for q in per_query}
        out[metric] = evalkit.paired_bootstrap(base, cand, samples=samples)
    return out


def render_markdown(report: Mapping[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# MuSiQue — a cross-encoder adapted on the corpus's own conventions",
        "",
        f"- {report['_settings']['queries']} queries, pool {report['_settings']['pool']}, "
        f"model `{report['_settings']['cross_encoder']}`",
        "",
        "| arm | Hits@10 | Hits@4 | MAP@10 | MRR@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        lines.append(f"| {arm} | " + " | ".join(f"{s[arm][m]:.4f}" for m in METRICS) + " |")
    lines += ["", "## Paired comparisons", ""]
    for name, block in report["comparisons"].items():
        lines.append(f"**{name}**")
        for metric, row in block.items():
            sig = "" if row["significant"] else " (ns)"
            lines.append(f"- {metric}: {row['delta']:+.4f} "
                         f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_value']:.4f}{sig}")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="order_eval.py")
    ap.add_argument("--cross-encoder", default=None,
                    help="a Hub id or a saved model path; required unless --compare-against")
    ap.add_argument("--compare-against", default=None, metavar="REPORT",
                    help="pair this report's `ce-model` arm against another report's, by query — "
                         "the adapted-versus-zero-shot test. Needs no cross-encoder")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--cache", default=str(ROOT_BENCH / "ce-scores-validation.json"))
    ap.add_argument("--pool", type=int, default=20, help="candidates reranked per query")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT_BENCH / "order-eval"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--merge", type=int, default=0, metavar="N")
    return ap.parse_args(argv)


def _compare(args: argparse.Namespace) -> int:
    """`--out` is the candidate (adapted) report; `--compare-against` is the baseline (zero-shot)."""
    mine_path = Path(str(args.out) + ".json")
    if not mine_path.is_file():
        print(f"compare: {mine_path} not found — run the measurement first", file=sys.stderr)
        return 2
    mine = json.loads(mine_path.read_text(encoding="utf-8"))
    other_path = Path(args.compare_against)
    other = json.loads(other_path.read_text(encoding="utf-8"))
    shared = [q for q in mine["per_query"] if q in other["per_query"]]
    if not shared:
        print("compare: no shared query ids", file=sys.stderr)
        return 2
    lines = [
        "# Adapted versus zero-shot cross-encoder on MuSiQue",
        "",
        f"- {len(shared)} shared queries; baseline `{other_path.name}` "
        f"({other['_settings'].get('cross_encoder')}) versus "
        f"`{mine['_settings'].get('cross_encoder')}`",
        "",
        "| metric | zero-shot | adapted | delta | 95% interval | p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for metric in METRICS:
        base = {q: other["per_query"][q]["ce-model"][metric] for q in shared}
        cand = {q: mine["per_query"][q]["ce-model"][metric] for q in shared}
        boot = evalkit.paired_bootstrap(base, cand, samples=10_000)
        sig = "" if boot["significant"] else " (ns)"
        lines.append(
            f"| {metric} | {sum(base.values()) / len(base):.4f} | "
            f"{sum(cand.values()) / len(cand):.4f} | **{boot['delta']:+.4f}**{sig} | "
            f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] | {boot['p_value']:.4f} |"
        )
    lines.append("")
    table = "\n".join(lines)
    print(table)
    out = Path(str(args.out) + ".vs.md")
    evalkit.write_text(out, table)
    print("wrote", out)
    return 0


def _fingerprint(args: argparse.Namespace) -> str:
    return jobkit.config_fingerprint(
        tool="order_eval", cross_encoder=args.cross_encoder, split=args.split,
        pool=args.pool, top_k=args.top_k, max_length=args.max_length, limit=args.limit,
    )


def _build_report(rows: dict) -> dict:
    if jobkit.META_KEY not in rows:
        raise RuntimeError("no shard checkpoints found — run the shards before --merge")
    meta = rows.pop(jobkit.META_KEY)
    per_query = dict(rows)
    return {**meta, "per_query": per_query}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare_against:
        return _compare(args)
    if not args.cross_encoder:
        raise SystemExit("--cross-encoder is required unless --compare-against is given")
    shard_index, shard_count = jobkit.parse_shard(args.shard)
    fingerprint = _fingerprint(args)
    if args.merge:
        try:
            merged = jobkit.merge_shards(args.out, args.merge, fingerprint=fingerprint,
                                         build=_build_report)
        except (RuntimeError, jobkit.FingerprintMismatch) as exc:
            print(f"merge failed: {exc}", file=sys.stderr)
            return 2
        print(f"merged {args.merge} shard(s) -> {args.out}.json")
        return 0

    data = load_split(ROOT_BENCH / f"{args.split}.json")
    corpus, golds, order = data["corpus"], data["golds"], data["order"]
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    rows = cache["queries"]
    if args.limit:
        rows = dict(list(rows.items())[: args.limit])
    # Keep only queries the split also has gold for; a cache entry without gold cannot be scored.
    rows = {q: row for q, row in rows.items() if q in golds}
    print(f"[{args.split}] {len(corpus)} paragraphs, {len(rows)} queries, pool {args.pool}", flush=True)

    stem = jobkit.shard_stem(args.out, shard_index, shard_count)
    if args.fresh:
        for suffix in (".jsonl", ".manifest.json"):
            Path(str(stem) + suffix).unlink(missing_ok=True)
    run = jobkit.ResumableRun(str(stem) + ".json", fingerprint=fingerprint,
                              shard=(shard_index, shard_count))
    if jobkit.META_KEY not in run:
        run.record(jobkit.META_KEY, {
            "_settings": {**vars(args), "queries": len(rows), "paragraphs": len(corpus)},
            "query_order": list(rows),
        })

    mine = jobkit.shard_items(list(rows), shard_index, shard_count)
    todo = run.pending(mine)
    print(f"  {len(todo)} of {len(mine)} queries to rerank ({len(run) - 1} checkpointed)",
          flush=True)

    model = None
    if todo:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(args.cross_encoder, num_labels=1, max_length=args.max_length,
                             device=args.device or ("mps" if _mps() else "cpu"))
    started = time.perf_counter()
    for position, qid in enumerate(todo, 1):
        pool = list(rows[qid]["pool"])[: args.pool]
        scores = model.predict([(data["queries"][qid], corpus[d]) for d in pool],
                               batch_size=args.batch_size, show_progress_bar=False)
        run.record(qid, {"scores": [round(float(s), 6) for s in scores]})
        if position % 50 == 0:
            run.touch(total=len(mine))
            print(f"    {position}/{len(todo)} in {time.perf_counter() - started:.0f}s", flush=True)

    shared = run.get(jobkit.META_KEY)
    per_query: dict[str, dict] = {}
    for qid in shared["query_order"]:
        row = run.get(qid)
        if row is None:
            continue
        pool = list(rows[qid]["pool"])[: args.pool]
        cached = list(rows[qid]["ce"])[: args.pool]
        per_query[qid] = score_arms(pool, golds[qid], order, cached, row["scores"], args.top_k)
    report = {
        "_settings": {**shared["_settings"], "seconds": round(time.perf_counter() - started, 1)},
        "summary": summarize(per_query),
        "comparisons": {
            "ce-model_vs_fused": compare(per_query, "fused", "ce-model"),
            "ce-model_vs_ce-cached": compare(per_query, "ce-cached", "ce-model"),
        },
        "per_query": per_query,
    }
    run.finish(report, total=len(mine))
    evalkit.write_text(Path(str(stem) + ".md"), render_markdown(report))
    print("\n" + render_markdown(report))
    print("wrote", str(stem) + ".json", "and", str(stem) + ".md")
    return 0


def _mps() -> bool:
    import torch

    return torch.backends.mps.is_available()


if __name__ == "__main__":
    sys.exit(main())
