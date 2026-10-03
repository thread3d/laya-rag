"""Does the true bridge carry signal, separately from whether the score ranks well?

The selection experiment answers "does ordering by ``s(q, b, c)`` beat the fused order", and
the answer is no — Laya's decision head is a weak orderer, which this repository already
measured for the two-way question. That result conflates two different failures:

* the head's scores are poorly *ordered* across the whole pool (a known property), and
* the bridge content adds nothing over an arbitrary passage (the actual hypothesis).

This separates them. For each query every pool candidate is labelled by whether it matches a
gold fact, and the per-query **AUC** of the conditional score against that label is computed
for the true bridge and for the shuffled-bridge control. AUC asks only "does a
gold-matching candidate score above a non-matching one" — it is invariant to the monotone
distortions that make the head a bad ranker, so it is the right instrument for the
discrimination question.

The paired comparison is between the two arms on the *same* queries and candidates, so a
difference is attributable to the bridge and not to the instruction text, which both arms
share.

    .venv/bin/python tools/multihop_tri_discriminate.py \
        --real eval/benchmarks/multihop-rag/tri-scores-real-bridge.json \
        --shuffled eval/benchmarks/multihop-rag/tri-scores-shuffled-bridge.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402
import numpy as np  # noqa: E402
from multihop_official import chunk_text  # noqa: E402
from multihop_rerank_experiment import gold_matches_for_pool  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U); 0.5 is chance, ties handled by average rank.

    Returns ``nan`` when a query has only one class — an undefined discrimination, which must
    not be silently counted as chance.
    """
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for k in range(position, end + 1):
            ranks[order[k]] = average
        position = end + 1
    rank_sum = sum(ranks[i] for i, label in enumerate(labels) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def load_corpus(overlap: int) -> dict[str, str]:
    raw = json.loads((ROOT_BENCH / "corpus.json").read_text(encoding="utf-8"))
    corpus: dict[str, str] = {}
    for article in raw:
        for j, text in enumerate(chunk_text(article["title"] + " " + article["body"], 256, overlap)):
            corpus[f"{article['url']}#{j}"] = text
    return corpus


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="multihop_tri_discriminate.py")
    ap.add_argument("--ce-cache", default=str(ROOT_BENCH / "ce-scores.json"))
    ap.add_argument("--real", default=str(ROOT_BENCH / "tri-scores-real-bridge.json"))
    ap.add_argument("--shuffled", default=str(ROOT_BENCH / "tri-scores-shuffled-bridge.json"))
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "tri-discriminate"))
    args = ap.parse_args(argv)

    cache = json.loads(Path(args.ce_cache).read_text(encoding="utf-8"))
    corpus = load_corpus(int(cache["_settings"]["overlap"]))
    arms = {
        "real": json.loads(Path(args.real).read_text(encoding="utf-8"))["scores"],
        "shuffled": json.loads(Path(args.shuffled).read_text(encoding="utf-8"))["scores"],
    }

    per_query: dict[str, dict[str, float]] = {arm: {} for arm in arms}
    per_query["ce"] = {}
    counts = {"queries": 0, "defined": 0, "positives": 0, "candidates": 0}
    for q, payload in cache["queries"].items():
        pool = list(payload["pool"])
        matches = gold_matches_for_pool(corpus, pool, cache["golds"][q])
        labels = [1 if m else 0 for m in matches]
        counts["queries"] += 1
        counts["candidates"] += len(labels)
        counts["positives"] += sum(labels)
        if sum(labels) == 0 or sum(labels) == len(labels):
            continue
        counts["defined"] += 1
        for arm, scores in arms.items():
            value = auc([float(s) for s in scores[q]], labels)
            if value == value:  # not NaN
                per_query[arm][q] = value
        # The cross-encoder as a reference discriminator on the same candidates.
        per_query["ce"][q] = auc([float(s) for s in payload["ce"]], labels)

    summary = {
        arm: {
            "mean_auc": round(float(np.mean(list(values.values()))), 4),
            "queries": len(values),
        }
        for arm, values in per_query.items()
    }
    comparisons: dict[str, Any] = {}
    for left, right in (("real", "shuffled"), ("real", "ce"), ("shuffled", "ce")):
        shared = sorted(set(per_query[left]) & set(per_query[right]))
        if not shared:
            continue
        comparisons[f"{left} vs {right} AUC"] = {
            "bootstrap": evalkit.paired_bootstrap(
                {q: per_query[right][q] for q in shared},
                {q: per_query[left][q] for q in shared},
                samples=args.samples,
            ),
            "permutation": evalkit.permutation_test(
                {q: per_query[right][q] for q in shared},
                {q: per_query[left][q] for q in shared},
                samples=args.samples,
            ),
        }

    result = {
        "_settings": {
            "ce_cache": Path(args.ce_cache).name,
            "real": Path(args.real).name,
            "shuffled": Path(args.shuffled).name,
            "candidates": counts["candidates"],
            "positive_candidates": counts["positives"],
        },
        "counts": counts,
        "summary": summary,
        "comparisons": comparisons,
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), result)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(result: Mapping[str, Any]) -> str:
    counts = result["counts"]
    lines = [
        "# Does the true bridge discriminate gold facts better than a shuffled one?",
        "",
        f"- {counts['queries']} queries, {counts['candidates']} pool candidates, "
        f"{counts['positives']} gold-matching; AUC defined on {counts['defined']} queries",
        "",
        "| arm | mean per-query AUC | queries |",
        "| --- | --- | --- |",
    ]
    for arm, row in result["summary"].items():
        lines.append(f"| {arm} | {row['mean_auc']:.4f} | {row['queries']} |")
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
