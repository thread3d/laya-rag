#!/usr/bin/env python
"""Is a **tuned** convex combination better than the RRF the engine ships, per corpus?

`SCORE-LIFT.md` §P5(a) asks for exactly this and names why the existing answer does not settle it:

> Replace `FUSION_K=5` RRF with a **tuned convex combination** of normalised channel scores,
> per-channel weights, and an ablated `k` … The one local measurement (`--convex`, SciFact,
> untuned weights) gave +0.0092 with p = 0.139 — not significant, and **not a tuned comparison**.

`docs/findings-fusion-ablation.md` measured the untuned convex arm at SciFact against RRF at
`k=60` — *not* the `k=5` the engine ships — and said so. This tool closes both gaps at once:

* the channels are computed **once** per corpus (`beir.channel_score_maps`) and the weight sweep
  re-fuses those score maps (`beir.fuse_score_maps`), so the sweep costs one index build rather
  than one per weight;
* the baseline is the engine's **RRF at `k=5`**, with `k=60` kept as the historical arm;
* the weights are chosen on a **held-out half** of the queries and scored on the other, so the
  reported delta is what a tuner that cannot see the labels would get, not an oracle.

    make fusion-tuning
    .venv/bin/python tools/fusion_tuning.py --datasets scifact,nfcorpus,arguana
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir, evalkit  # noqa: E402

#: The five members with a full channel set, plus the two where the channel *reversal* was
#: measured — so the sweep is run where the fusion question has already been shown to matter.
DEFAULT_DATASETS = "scifact,nfcorpus,arguana,scidocs,fiqa"

#: Dense weights swept against a fixed lexical weight of 1.0. `1.0` is the symmetric point the
#: existing reports use, `0.0` drops the dense channel and `> 1` over-weights it, so the sweep
#: brackets the whole range the engine can express through `DENSE_WEIGHT`.
DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

#: The fusion constants to compare against. 5 is what ships (`FUSION_K`), 60 is the historical
#: default the earlier convex comparison used.
RRF_K = (5, 60)

TRAIN_FRACTION = 0.5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="fusion_tuning.py")
    ap.add_argument("--datasets", default=DEFAULT_DATASETS)
    ap.add_argument("--data-root", default=str(beir.DEFAULT_DATA_ROOT))
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                    help="the encoder the cached vectors belong to; must match the cache")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--weights", default=",".join(str(w) for w in DEFAULT_WEIGHTS))
    ap.add_argument("--title-boost", type=int, default=1,
                    help="repeat titles N times in the lexical channel, when a title map exists")
    ap.add_argument("--sample-seed", type=int, default=17,
                    help="seeds the train/test query split, so the split is reproducible")
    ap.add_argument("--samples", type=int, default=10_000)
    ap.add_argument("--out", default=str(ROOT / "eval" / "beir" / "fusion-tuning"))
    return ap.parse_args(argv)


def _split(query_ids: Sequence[str], seed: int) -> tuple[list[str], list[str]]:
    """A deterministic half/half query split, so a tuned weight cannot be chosen on its own test set.

    Stratified by nothing: the BEIR test sets here are one query per topic, so a plain seeded
    shuffle is the honest split and, unlike a head slice, does not put a whole corpus's easy
    queries first.
    """
    ordered = sorted(query_ids)
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * TRAIN_FRACTION)
    return sorted(shuffled[:cut]), sorted(shuffled[cut:])


def _score(rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Any],
           query_ids: Sequence[str]) -> tuple[float, dict[str, float]]:
    """nDCG@10 over ``query_ids``, plus the per-query series the paired bootstrap needs."""
    subset = {qid: list(rankings[qid]) for qid in query_ids if qid in rankings}
    restricted = beir.restrict_qrels(qrels, subset)
    evaluation = evalkit.evaluate_rankings(
        subset, restricted, k_values=(10,), ignore_identical_ids=True
    )
    per_query = evalkit.per_query_metric(evaluation["per_query"], "nDCG@10")
    mean = sum(per_query.values()) / len(per_query) if per_query else 0.0
    return mean, per_query


def tune_one(
    name: str, args: argparse.Namespace, weights: Sequence[float]
) -> dict[str, Any] | None:
    """One corpus: the RRF baselines, the weight sweep, and the held-out reading."""
    dataset_dir = beir.ensure_dataset(name, Path(args.data_root), download=False)
    corpus = beir.load_corpus(dataset_dir, include_titles=True)
    queries = beir.load_queries(dataset_dir)
    qrels = beir.load_qrels(dataset_dir, split="test")
    scored = {qid: text for qid, text in queries.items() if qid in qrels}
    if not scored:
        return None
    titles = beir.load_titles(dataset_dir) if args.title_boost > 1 else None

    encoder = beir.BiEncoder(args.model)
    vectors = beir.cached_embed(
        encoder, corpus, ROOT / "eval" / "beir" / "cache" / f"{name}.{encoder.name}", chunk=0
    )
    started = time.perf_counter()
    lexical_scores, dense_scores = beir.channel_score_maps(
        corpus, scored, encoder, vectors, args.top_k,
        titles=titles, title_boost=args.title_boost,
    )
    channels_seconds = round(time.perf_counter() - started, 1)

    # The rank-fusion baselines, from the same channels.
    lexical_rankings = {
        qid: [doc for doc, _ in sorted(scores.items(), key=lambda kv: -kv[1])[: args.top_k]]
        for qid, scores in lexical_scores.items()
    }
    dense_rankings = {
        qid: [doc for doc, _ in sorted(scores.items(), key=lambda kv: -kv[1])[: args.top_k]]
        for qid, scores in dense_scores.items()
    }
    rrf: dict[int, dict[str, list[str]]] = {
        k: beir.hybrid_rankings(dense_rankings, lexical_rankings, args.top_k,
                                dense_weight=1.0, lexical_weight=1.0, rrf_k=k)
        for k in RRF_K
    }
    convex: dict[float, dict[str, list[str]]] = {
        w: beir.fuse_score_maps(lexical_scores, dense_scores, args.top_k,
                                dense_weight=w, lexical_weight=1.0, normalise="minmax")
        for w in weights
    }

    train, test = _split(list(scored), args.sample_seed)
    full_ids = list(scored)
    rows: list[dict[str, Any]] = []
    for label, rankings in (
        [(f"rrf_k{k}", rrf[k]) for k in RRF_K]
        + [(f"convex@{w:g}", convex[w]) for w in weights]
    ):
        mean, per_query = _score(rankings, qrels, full_ids)
        _train_mean, train_per_query = _score(rankings, qrels, train)
        test_mean, test_per_query = _score(rankings, qrels, test)
        rows.append({
            "arm": label,
            "full": mean,
            "train": _train_mean,
            "test": test_mean,
            "per_query_full": per_query,
            "per_query_test": test_per_query,
        })

    by_arm = {row["arm"]: row for row in rows}
    tuned = max((f"convex@{w:g}" for w in weights), key=lambda arm: by_arm[arm]["train"])
    baseline = "rrf_k5"
    paired = evalkit.paired_bootstrap(
        by_arm[baseline]["per_query_test"],
        by_arm[tuned]["per_query_test"],
        samples=args.samples,
        seed=args.sample_seed,
    )
    oracle = max(rows, key=lambda row: row["full"])
    return {
        "dataset": name,
        "documents": len(corpus),
        "queries": len(scored),
        "train_queries": len(train),
        "test_queries": len(test),
        "channels_seconds": channels_seconds,
        "arms": [
            {key: row[key] for key in ("arm", "full", "train", "test")} for row in rows
        ],
        "tuned_arm": tuned,
        "tuned_full": by_arm[tuned]["full"],
        "baseline_arm": baseline,
        "baseline_full": by_arm[baseline]["full"],
        "oracle_arm": oracle["arm"],
        "oracle_full": oracle["full"],
        "tuned_vs_rrf_k5_on_test": paired,
        "note": (
            "weights chosen on the train half, scored on the test half; the oracle column picks "
            "the best arm on the full set and is reported only as a ceiling"
        ),
    }


def render(result: Mapping[str, Any]) -> str:
    lines = [
        "| dataset | RRF k=5 | RRF k=60 | tuned convex | weight | held-out Δ vs k=5 | 95% interval | p | oracle |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in result["datasets"]:
        arms = {arm["arm"]: arm for arm in row["arms"]}
        paired = row["tuned_vs_rrf_k5_on_test"]
        sig = "" if paired["significant"] else " (ns)"
        lines.append(
            f"| {row['dataset']} | {arms['rrf_k5']['full']:.4f} | {arms['rrf_k60']['full']:.4f} | "
            f"{row['tuned_full']:.4f} | {row['tuned_arm'].split('@')[1]} | "
            f"**{paired['delta']:+.4f}**{sig} | "
            f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}] | {paired['p_value']:.3f} | "
            f"{row['oracle_full']:.4f} ({row['oracle_arm'].split('@')[1]}) |"
        )
    held = [row["tuned_vs_rrf_k5_on_test"] for row in result["datasets"]]
    wins = sum(1 for row in held if row["significant"] and row["delta"] > 0)
    losses = sum(1 for row in held if row["significant"] and row["delta"] < 0)
    mean_delta = sum(row["delta"] for row in held) / len(held) if held else 0.0
    picks = {}
    for row in result["datasets"]:
        picks[row["tuned_arm"]] = picks.get(row["tuned_arm"], 0) + 1
    lines += [
        "",
        f"{len(result['datasets'])} corpora; the tuned arm is chosen on half the queries and "
        f"scored on the other half.",
        f"significant wins **{wins}**, significant losses **{losses}**, mean held-out delta "
        f"**{mean_delta:+.4f}**.",
        "picked weights: " + ", ".join(f"{arm} ×{count}" for arm, count in sorted(picks.items())),
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("fusion_tuning.py")
    weights = [float(part) for part in str(args.weights).split(",") if part.strip()]
    datasets = [name.strip() for name in str(args.datasets).split(",") if name.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"settings": vars(args), "datasets": []}
    for name in datasets:
        try:
            row = tune_one(name, args, weights)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"{name}: skipped — {exc}", file=sys.stderr)
            continue
        if row is None:
            print(f"{name}: no queries in the test split", file=sys.stderr)
            continue
        result["datasets"].append(row)
        arms = {arm["arm"]: arm for arm in row["arms"]}
        print(f"{name}: {row['queries']} queries, channels in {row['channels_seconds']}s; "
              f"RRF k=5 {arms['rrf_k5']['full']:.4f}, tuned {row['tuned_full']:.4f} "
              f"({row['tuned_arm']}), held-out "
              f"{row['tuned_vs_rrf_k5_on_test']['delta']:+.4f} "
              f"[{row['tuned_vs_rrf_k5_on_test']['ci_low']:+.4f}, "
              f"{row['tuned_vs_rrf_k5_on_test']['ci_high']:+.4f}]", flush=True)
        evalkit.write_json(out.with_suffix(".json"), result)

    if not result["datasets"]:
        print("no dataset could be scored", file=sys.stderr)
        return 2
    evalkit.write_json(out.with_suffix(".json"), result)
    evalkit.write_text(out.with_suffix(".md"), render(result))
    print()
    print(render(result))
    print(f"wrote {out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
