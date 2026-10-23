"""Union per-query rankings written by `tools/beir.py --dump-rankings` and score them once.

This exists so that a sharded run can be **safe**. `evalkit.evaluate_rankings` macro-averages over every
query in the qrels it is given, so scoring a ranking set that covers only one slice counts every unsampled
query as zero — the dilution that has produced numbers ~10x too low in this repository four separate times.
The only correct shape is therefore: shards produce *rankings*, a merge unions them per query, and the
union is scored once against the **full** qrels.

So this tool takes sidecars, never reports. It refuses a merge whose union does not cover every query the
qrels names, because that would silently score an incomplete set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import beir  # noqa: E402
import evalkit  # noqa: E402


def load_parts(paths: list[Path]) -> dict[str, dict[str, list[str]]]:
    """Union the sidecars: retriever -> query id -> ranked ids, first part wins on a duplicate query."""
    merged: dict[str, dict[str, list[str]]] = {}
    seen: dict[str, set[str]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rankings = payload.get("rankings")
        if not isinstance(rankings, dict):
            raise SystemExit(f"{path}: no rankings block — this wants a --dump-rankings sidecar, not a report")
        for retriever, per_query in rankings.items():
            bucket = merged.setdefault(retriever, {})
            covered = seen.setdefault(retriever, set())
            for qid, ids in per_query.items():
                if qid in covered:
                    continue                      # a duplicate may only come from a retried shard
                bucket[qid] = list(ids)
                covered.add(qid)
    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="BEIR dataset the shards were run over")
    ap.add_argument("--parts", nargs="+", required=True, help="rankings sidecars to union, in order")
    ap.add_argument("--data-root", default=str(beir.ROOT / "eval" / "beir"), help="where corpora live")
    ap.add_argument("--out", help="report stem to write (not needed with --check)")
    ap.add_argument("--check", action="store_true",
                    help="report each retriever's coverage of the qrels and exit without scoring; the mode to run "
                         "before committing hours to a sharded sweep")
    args = ap.parse_args(argv)
    if not args.check and not args.out:
        ap.error("--out is required unless --check is given")

    dataset_dir = beir.ensure_dataset(args.dataset, Path(args.data_root), download=False)
    qrels = beir.load_qrels(dataset_dir, split="test")
    merged = load_parts([Path(p) for p in args.parts])
    if not merged:
        print("no rankings in the parts", file=sys.stderr)
        return 2

    if args.check:
        for retriever, ranking in merged.items():
            covered = len(set(qrels) & set(ranking))
            extra = len(set(ranking) - set(qrels))
            print(f"  {retriever:14s} covers {covered}/{len(qrels)} qrels queries"
                  + (f", {extra} ranked ids are not in the qrels" if extra else ""))
        return 0

    results: dict[str, object] = {}
    for retriever, ranking in merged.items():
        missing = len(set(qrels) - set(ranking))
        if missing:
            print(f"{retriever}: {missing} qrels queries are unranked by the union — refusing to score a "
                  f"partial set (see the qrels-dilution note in this file)", file=sys.stderr)
            return 2
        evaluation = evalkit.evaluate_rankings(ranking, qrels, k_values=(10, 100))
        row = dict(evaluation["summary"])
        row["scored_queries"] = evaluation["queries"]
        results[retriever] = row

    report = {
        "_settings": {"tool": "beir_merge_rankings.py", "dataset": args.dataset,
                      "parts": [str(p) for p in args.parts]},
        "datasets": {args.dataset: {"retrievers": results, "documents": None}},
    }
    evalkit.write_json(Path(str(args.out) + ".json"), report)
    for retriever, row in results.items():
        print(f"  {args.dataset}  {retriever:14s} nDCG@10 {float(row['nDCG@10']):.4f} "
              f"over {row['scored_queries']} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
