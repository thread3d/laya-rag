"""Does a learned combination of the channels rank better than any single one?

Every ranking rule in this repository orders by **one** signal at a time — the fused recall
order, a cross-encoder logit, Laya's calibrated relevance, the bridge-conditioned score — or
greedily combines two by hand with a fixed weight. The channels are individually weak and
partially independent, and the answerability gate already showed what happens when several
weak signals are combined by a *fitted* model rather than by hand: ROC-AUC 0.51 from the single
scalar became 0.815 from twelve features. The same experiment has never been run for ranking.

This is a deliberately small prototype of it. Per candidate it uses only signals that already
exist in the caches — the fused pool position, the cross-encoder logit, and (optionally) the
bridge-conditioned decision — fits a logistic scorer on half the queries, and scores the other
half. The comparison that matters is against the best single channel on the same held-out
queries, with a paired interval.

    .venv/bin/python tools/multihop_ltr.py \
        --tri-cache eval/benchmarks/multihop-rag/tri-scores-real-bridge.json

Honest limits, stated up front: this is *supervised on the benchmark's own labels*, so it is
"recalibrate on the target domain", exactly like the answerability gate — it is not evidence
that a scorer trained here transfers anywhere else. It is evidence about whether the
combination has headroom at all, which is the question the ranking programme depends on.
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
import numpy as np  # noqa: E402
from multihop_official import chunk_text  # noqa: E402
from multihop_rerank_experiment import (  # noqa: E402
    blend_order,
    coverage_doc_quota_order,
    document_of,
    gold_matches_for_pool,
    metrics_from_matches,
)

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")


def load_corpus(overlap: int) -> dict[str, str]:
    raw = json.loads((ROOT_BENCH / "corpus.json").read_text(encoding="utf-8"))
    corpus: dict[str, str] = {}
    for article in raw:
        for j, text in enumerate(chunk_text(article["title"] + " " + article["body"], 256, overlap)):
            corpus[f"{article['url']}#{j}"] = text
    return corpus


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score using **training** statistics only.

    Computing them over both halves would leak the held-out queries' distribution into the
    fitted model, which is the quiet way a supervised comparison stops being held out.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0.0] = 1.0
    return (train - mean) / std, (other - mean) / std


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = 600,
    learning_rate: float = 0.5,
    l2: float = 1e-3,
) -> np.ndarray:
    """Batch-gradient logistic regression with an L2 penalty; deterministic.

    Full-batch rather than stochastic so the result does not depend on a shuffle seed: with
    ~20k training candidates this converges in under a second and there is no reason to accept
    seed variance in a comparison this small.
    """
    weights = np.zeros(features.shape[1], dtype=np.float64)
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float(len(labels) - labels.sum()), 1.0)
    # Class weighting, because a gold-matching candidate is ~7% of the pool and an
    # unweighted fit is dominated by the negatives.
    sample_weight = np.where(labels > 0, len(labels) / (2.0 * positives), len(labels) / (2.0 * negatives))
    for _ in range(iterations):
        logits = features @ weights
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = features.T @ (sample_weight * (probabilities - labels)) / len(labels)
        gradient += l2 * weights
        weights -= learning_rate * gradient
    return weights


def rank_by(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="multihop_ltr.py")
    ap.add_argument("--ce-cache", default=str(ROOT_BENCH / "ce-scores.json"))
    ap.add_argument("--tri-cache", default=None,
                    help="optional bridge-conditioned cache; adds its score as a feature")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "ltr"))
    args = ap.parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("multihop_ltr.py")

    cache = json.loads(Path(args.ce_cache).read_text(encoding="utf-8"))
    corpus = load_corpus(int(cache["_settings"]["overlap"]))
    tri = None
    if args.tri_cache:
        tri = json.loads(Path(args.tri_cache).read_text(encoding="utf-8"))["scores"]

    qids = sorted(cache["queries"], key=lambda v: (len(v), v))
    shuffled = list(qids)
    random.Random(args.seed).shuffle(shuffled)
    half = len(shuffled) // 2
    train_ids, test_ids = set(shuffled[:half]), set(shuffled[half:])

    # ── build the per-candidate design matrices ──────────────────────────
    feature_names = ["ce", "fused_rank", "tri"]
    if tri is None:
        feature_names = ["ce", "fused_rank"]
    train_rows: list[list[float]] = []
    train_labels: list[float] = []
    test_rows: list[list[float]] = []
    test_labels: list[float] = []
    test_offset: dict[str, int] = {}
    per_query_index: dict[str, dict[str, Any]] = {}
    channels: dict[str, dict[str, list[int]]] = {
        "fused": {}, "ce": {}, "tri": {}, "ltr": {}, "blend-0.75": {}, "covquota-3-0.7": {},
    }

    for q in qids:
        payload = cache["queries"][q]
        pool = list(payload["pool"])
        ce = [float(v) for v in payload["ce"]]
        matches = gold_matches_for_pool(corpus, pool, cache["golds"][q])
        gold_count = len(cache["golds"][q])
        positions = list(range(len(pool)))
        n = max(len(pool) - 1, 1)
        rows = []
        for i in positions:
            fused_rank = 1.0 - i / n          # 1.0 = best first-stage position
            row = [ce[i], fused_rank]
            if tri is not None:
                row.append(float(tri[q][i]))
            rows.append(row)
        labels = [1.0 if m else 0.0 for m in matches]
        if q in train_ids:
            train_rows.extend(rows)
            train_labels.extend(labels)
        else:
            # Record where this query's rows start, so the held-out scores can be sliced back
            # out without recomputing positions from the query order.
            test_offset[q] = len(test_rows)
            test_rows.extend(rows)
            test_labels.extend(labels)
        per_query_index[q] = {
            "matches": matches, "gold_count": gold_count, "ce": ce, "rows": rows,
            "pool_len": len(pool), "pool": pool, "atoms": payload["atoms"],
            "atom_weight": cache["_settings"]["atom_weight"],
        }

    train_x = np.asarray(train_rows, dtype=np.float64)
    train_y = np.asarray(train_labels, dtype=np.float64)
    test_x_raw = np.asarray(test_rows, dtype=np.float64)
    train_x, test_x = standardize(train_x, test_x_raw)
    weights = fit_logistic(train_x, train_y)
    print(f"fitted on {len(train_y)} candidates from {len(train_ids)} queries; "
          f"weights: " + ", ".join(f"{n}={w:+.3f}" for n, w in zip(feature_names, weights)))

    # ── score the held-out half with every channel ───────────────────────
    per_query: dict[str, dict[str, dict[str, float]]] = {
        name: {metric: {} for metric in METRICS} for name in channels
    }
    for q in test_ids:
        info = per_query_index[q]
        matches = info["matches"]
        gold_count = int(info["gold_count"])
        rows = test_x[test_offset[q]: test_offset[q] + info["pool_len"]]
        ltr_scores = (rows @ weights).tolist()
        tri_scores = [float(tri[q][i]) for i in range(info["pool_len"])] if tri is not None else None
        rankings = {
            "fused": list(range(info["pool_len"])),
            "ce": rank_by(info["ce"]),
            "ltr": rank_by(ltr_scores),
            # The hand-built rules the fitted scorer has to beat to be worth anything: the
            # best blend already measured, and the best selection rule already measured.
            "blend-0.75": [
                int(i) for i in blend_order(list(range(info["pool_len"])), info["ce"], 0.75, 60)
            ][: args.top_k],
            "covquota-3-0.7": coverage_doc_quota_order(
                info["ce"], info["atoms"], info["atom_weight"],
                [document_of(c) for c in info["pool"]], args.top_k, 0.7, 3,
            ),
        }
        if tri_scores is not None:
            rankings["tri"] = rank_by(tri_scores)
        for name, ranked in rankings.items():
            metrics = metrics_from_matches(ranked, matches, gold_count, args.top_k)
            for metric in METRICS:
                per_query[name][metric][q] = metrics[metric]

    summary = {
        name: {metric: round(float(np.mean(list(rows.values()))), 4) for metric, rows in block.items()}
        for name, block in per_query.items() if any(block[metric] for metric in METRICS)
    }
    comparisons: dict[str, Any] = {}
    for baseline in ("ce", "fused", "blend-0.75", "covquota-3-0.7"):
        for metric in ("MAP@10", "Hits@4", "Hits@10"):
            if not per_query.get(baseline, {}).get(metric) or not per_query["ltr"][metric]:
                continue
            comparisons[f"ltr vs {baseline} {metric}"] = {
                "bootstrap": evalkit.paired_bootstrap(
                    per_query[baseline][metric], per_query["ltr"][metric], samples=args.samples
                ),
                "permutation": evalkit.permutation_test(
                    per_query[baseline][metric], per_query["ltr"][metric], samples=args.samples
                ),
            }

    result = {
        "_settings": {
            "ce_cache": Path(args.ce_cache).name,
            "tri_cache": Path(args.tri_cache).name if args.tri_cache else None,
            "features": feature_names,
            "weights": [round(float(w), 4) for w in weights],
            "train_queries": len(train_ids),
            "test_queries": len(test_ids),
            "seed": args.seed,
            "caveat": "supervised on the benchmark's own labels; recalibrate-on-target-domain, "
                      "not evidence of transfer",
        },
        "summary": summary,
        "comparisons": comparisons,
        "per_query": {name: block["MAP@10"] for name, block in per_query.items() if block["MAP@10"]},
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), result)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(result: Mapping[str, Any]) -> str:
    settings = result["_settings"]
    lines = [
        "# Learned rank fusion over the existing channels (held-out half)",
        "",
        f"- features: {', '.join(settings['features'])}; weights "
        + ", ".join(f"{w:+.3f}" for w in settings["weights"]),
        f"- fitted on {settings['train_queries']} queries, scored on "
        f"{settings['test_queries']} unseen ones (seed {settings['seed']})",
        f"- **caveat:** {settings['caveat']}",
        "",
        "| ranker | Hits@10 | Hits@4 | MAP@10 | MRR@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in result["summary"].items():
        lines.append(
            f"| {name} | {row['Hits@10']:.4f} | {row['Hits@4']:.4f} | "
            f"{row['MAP@10']:.4f} | {row['MRR@10']:.4f} |"
        )
    lines += ["", "## Paired comparisons (bootstrap 95% CI, permutation p)", ""]
    for key, value in sorted(result["comparisons"].items()):
        boot = value["bootstrap"]
        perm = value["permutation"]
        lines.append(
            f"- `{key}`: {boot['delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
            f"p={perm['p_value']:.4f} ({'sig' if boot['significant'] else 'ns'})"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
