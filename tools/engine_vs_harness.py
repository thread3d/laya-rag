"""The harness-vs-engine delta on MultiHop-RAG, with a paired interval over queries.

`tools/engine_multihop.py` measures the real engine under its own corpus-aware chunking;
`tools/multihop_official.py` measures a 256-token replica. The two are only comparable query by
query, on the *same* subset, so this pairs them and bootstraps the difference.

The pairing rule: both reports must have been produced with the same `--limit` and
`--sample-seed`, which the engine report records in `_settings`. The engine report's `queries`
map is in the subset's ascending-index order, and the harness's `_per_query` rows are in the
same order, so row *n* of each describes the same question. That is asserted here — the engine
report is re-derived from `MultiHopRAG.json` and the recorded seed, and a mismatch is an error
rather than a quietly wrong comparison.

    .venv/bin/python tools/engine_vs_harness.py \
        --engine eval/benchmarks/multihop-rag/engine-hybrid-400.json \
        --harness eval/benchmarks/multihop-rag/official-rerank-subset.json \
        --arm "bge-small hybrid+rerank@ov200"
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
METRICS = ("MAP@10", "Hits@4", "Hits@10", "MRR@10")


def subset_keys(items: Sequence[Mapping[str, Any]], limit: int, sample_seed: int) -> list[str]:
    """The engine's work keys for a `--limit/--sample-seed` subset, in report order.

    Mirrors `engine_multihop.select_items`: sample *positions* in the answerable list, sort, and
    keep each record's original index — which is the key the engine report records.
    """
    answerable = [(i, x) for i, x in enumerate(items) if x["question_type"] != "null_query"]
    if limit >= len(answerable):
        return [str(i) for i, _ in answerable]
    sampled = sorted(random.Random(sample_seed).sample(range(len(answerable)), limit))
    return [str(answerable[p][0]) for p in sampled]


def average(rows: Mapping[str, float]) -> float:
    return sum(rows.values()) / len(rows) if rows else 0.0


def compare(
    engine: Mapping[str, Any],
    harness_arm: Mapping[str, Any],
    keys: Sequence[str],
    *,
    samples: int = 10_000,
) -> list[dict[str, Any]]:
    """One paired bootstrap per metric, harness as base and engine as candidate."""
    harness_rows = harness_arm["_per_query"]
    if len(harness_rows) != len(keys):
        raise ValueError(
            f"harness arm has {len(harness_rows)} rows against the engine's {len(keys)} queries; "
            "they were not run on the same subset"
        )
    out: list[dict[str, Any]] = []
    for metric in METRICS:
        base = {key: float(harness_rows[n][metric]) for n, key in enumerate(keys)}
        cand = {key: float(engine["per_query"][key]["metrics"][metric]) for key in keys}
        boot = evalkit.paired_bootstrap(base, cand, samples=samples)
        out.append({
            "metric": metric,
            "harness": average(base),
            "engine": average(cand),
            **{k: boot[k] for k in ("delta", "ci_low", "ci_high", "p_value", "significant")},
        })
    return out


def render_markdown(engine_name: str, arm: str, rows: Sequence[Mapping[str, Any]],
                    queries: int) -> str:
    lines = [
        f"# The engine against the 256-token harness — `{arm}`",
        "",
        f"- engine report: `{engine_name}`; harness arm: `{arm}`; paired over **{queries}** queries",
        "",
        "| metric | harness | engine | delta | 95% interval | p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        sig = "" if row["significant"] else " (ns)"
        lines.append(
            f"| {row['metric']} | {row['harness']:.4f} | {row['engine']:.4f} | "
            f"**{row['delta']:+.4f}**{sig} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['p_value']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="engine_vs_harness.py")
    ap.add_argument("--engine", required=True, help="report from tools/engine_multihop.py")
    ap.add_argument("--harness", default=str(ROOT_BENCH / "official-rerank-subset.json"))
    ap.add_argument("--arm", default="bge-small hybrid@ov200", help="harness arm to compare")
    ap.add_argument("--items", default=str(ROOT_BENCH / "MultiHopRAG.json"))
    ap.add_argument("--samples", type=int, default=10_000)
    ap.add_argument("--out", default=None, help="write the markdown table here as well")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    engine = json.loads(Path(args.engine).read_text(encoding="utf-8"))
    harness = json.loads(Path(args.harness).read_text(encoding="utf-8"))
    if args.arm not in harness:
        raise SystemExit(f"{args.harness} has no arm {args.arm!r}; "
                         f"available: {', '.join(k for k in harness if k != '_settings')}")

    settings = engine.get("_settings", {})
    limit = settings.get("limit")
    sample_seed = settings.get("sample_seed")
    if not limit or sample_seed is None:
        raise SystemExit("the engine report must be a --limit/--sample-seed subset run; "
                         f"got limit={limit!r}, sample_seed={sample_seed!r}")
    items = json.loads(Path(args.items).read_text(encoding="utf-8"))
    keys = subset_keys(items, limit, sample_seed)

    # Assert the pairing instead of trusting it: the report's own keys must be this subset.
    recorded = list(engine["queries"])
    if recorded != keys:
        raise SystemExit(
            f"the engine report's queries are not the limit={limit} seed={sample_seed} subset "
            f"({len(recorded)} keys, first mismatch at "
            f"{next((n for n, (a, b) in enumerate(zip(recorded, keys)) if a != b), 'length')})"
        )

    rows = compare(engine, harness[args.arm], keys, samples=args.samples)
    table = render_markdown(Path(args.engine).name, args.arm, rows, len(keys))
    print(table)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
