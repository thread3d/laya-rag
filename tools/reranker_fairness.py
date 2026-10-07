"""The reranker-fairness pack: R@k on a reranker-excluded pool, and the ordering-only claim.

`docs/research/07` §4 (RESEARCH.md Tier 0.5) asks for **R@10/100/1000 and a reranker-excluded
union pool** alongside MAP@10/Hits@k, because the cross-encoder's gain is measured on a pool the
reranker did not build and a single pooled judgment set at a fixed scoring depth can inflate a
reranker's measured gain.

Everything here is computed from cached runs — no model, no network, seconds of CPU:

* **Pool provenance.** The pool is the fused first stage (BM25 + dense, RRF), and the cross-encoder
  contributes no candidate: `tools/multihop_ce_cache.py` builds the pool first and only then scores
  it. The cache's own settings are echoed and the pool is *refused* if they name a reranker in pool
  construction, so the fairness property is checked rather than asserted.
* **The recall curve.** Gold-fact recall@k of the pool itself, at 10/100 from the cached pool and at
  10/100/500 from the depth-stratified diagnostics. A second stage that only **reorders** a pool
  cannot change set membership, so this is also where the claim "a reranker improves R@1000" is
  refuted by construction — and the pack measures it, by reporting R@k for the fused order and for
  the cross-encoder order over the same pool and checking that the top-k *sets* are identical.
* **The ordering gain.** MAP@10 / Hits@4 / Hits@10 / MRR@10 for both arms on that same pool, so the
  reranker's contribution is isolated to ordering.

`--deep` folds in a deeper diagnostics report (e.g. a depth-1000 run) for the R@1000 point; without
it, R@1000 is reported as **bounded** by the deepest cached ceiling and the bound is stated, which
is what the Tier 0.5 falsifier asks for ("if gold support is not exhaustive at depth 1000 the
ceiling is still a floor").

    .venv/bin/python tools/reranker_fairness.py
    .venv/bin/python tools/reranker_fairness.py --check     # offline invariants, no cache needed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import nugget_recall as nr  # noqa: E402  (the evidence/coverage machinery, already tested)

BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
DEFAULT_SCORES = BENCH / "ce-scores-full.json"
DEFAULT_DIAGNOSTICS = BENCH / "diagnostics.json"
DEFAULT_RERANK = BENCH / "rerank-experiment-full.json"

#: Cache settings that would mean the pool was built *with* a reranker in the loop. The pack
#: refuses such a pool: reporting fairness on a reranker-built pool is the thing Tier 0.5 exists
#: to prevent, and a silent pass would be worse than no number.
RERANKER_MARKERS = ("reranker", "cross_encoder", "cross-encoder", "ce_", "cepool")


def pool_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the cached pool is reranker-excluded, and the settings that say so."""
    settings = payload.get("_settings") or {}
    named = sorted(key for key in settings if any(m in key.lower() for m in RERANKER_MARKERS))
    return {
        "excluded": not named,
        "reranker_keys_in_pool_settings": named,
        "settings": {k: settings[k] for k in sorted(settings)},
    }


def arm_rankings(
    payload: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """``(fused, ce)`` rankings per query: the pool as-is, and the pool sorted by cached CE score."""
    fused: dict[str, list[str]] = {}
    ce: dict[str, list[str]] = {}
    for query_id, row in payload["queries"].items():
        pool = list(row["pool"])
        scores = list(row["ce"])
        fused[query_id] = pool
        ce[query_id] = [
            chunk_id for chunk_id, _score in sorted(zip(pool, scores), key=lambda pair: -pair[1])
        ]
    return fused, ce


def membership_invariance(
    fused: Mapping[str, Sequence[str]],
    ce: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """At the pool's own depth the two arms' candidate *sets* must be identical.

    A reranker only reorders: it can change which candidates are in the top-k for k below the pool
    depth — that is the ordering gain — but it cannot change the set at the pool depth, and it
    cannot add a candidate at all. So R@k for k >= the pool depth is the first stage's number.
    """
    checked = identical = 0
    for query_id, pool in fused.items():
        reordered = ce.get(query_id)
        if reordered is None:
            continue
        checked += 1
        identical += set(pool) == set(reordered)
    return {
        "checked": checked,
        "identical": identical,
        "all_identical": bool(checked) and identical == checked,
        "note": "at the pool's own depth; below it the arms may differ, and that is ordering",
    }


def divergent_top_k(
    fused: Mapping[str, Sequence[str]],
    ce: Mapping[str, Sequence[str]],
    k: int,
) -> dict[str, Any]:
    """How many queries the reordering moves at depth ``k`` — the size of the ordering effect."""
    checked = divergent = 0
    for query_id, pool in fused.items():
        reordered = ce.get(query_id)
        if reordered is None:
            continue
        checked += 1
        divergent += set(pool[:k]) != set(reordered[:k])
    return {"k": k, "checked": checked, "divergent": divergent}


def recall_table(
    rankings: Mapping[str, Iterable[str]],
    golds: Mapping[str, Sequence[str]],
    evidence: nr.Evidence,
    depths: Sequence[int],
) -> dict[str, Any]:
    """Gold-fact recall@k per arm per depth, using the benchmark's own substring test."""
    return {str(depth): nr.fact_recall(rankings, golds, evidence, k=int(depth)) for depth in depths}


def pool_depth_curve(path: Path) -> dict[str, Any]:
    """The fused pool's fact-recall curve from a depth-stratified diagnostics report."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    depths = (payload.get("_settings") or {}).get("depths") or []
    rows = payload.get("per_query") or []
    curve: dict[str, Any] = {}
    for depth in depths:
        key = f"pool_fact_recall@{depth}"
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            curve[str(depth)] = {"mean": round(sum(values) / len(values), 4), "queries": len(values)}
    return {
        "source": str(Path(path).name),
        "max_depth": (payload.get("_settings") or {}).get("max_depth"),
        "curve": curve,
    }


def ordering_gain(path: Path) -> dict[str, Any]:
    """MAP@10 and the Hits/MRR rows for the fused and cross-encoder arms over one pool."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    arms = {arm: summary.get(arm) for arm in ("fused", "ce") if summary.get(arm)}
    comparison = ((payload.get("comparisons") or {}).get("ce vs fused MAP@10") or {})
    return {"source": str(Path(path).name), "arms": arms, "ce_vs_fused": comparison}


def build_report(
    *,
    scores: Mapping[str, Any],
    golds: Mapping[str, Sequence[str]],
    evidence: nr.Evidence,
    diagnostics: Mapping[str, Any],
    rerank: Mapping[str, Any],
    pool_depths: Sequence[int] = (10, 100),
    deep: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the pack. Pure: everything comes in as already-loaded structures."""
    fused, ce = arm_rankings(scores)
    provenance = pool_provenance(scores)
    deepest = max(int(d) for d in diagnostics.get("curve", {}) or {0})
    report: dict[str, Any] = {
        "provenance": provenance,
        "queries": len(fused),
        "recall": recall_table({"fused": fused, "ce": ce}, golds, evidence, pool_depths),
        "membership": membership_invariance(fused, ce),
        "reordering": [divergent_top_k(fused, ce, int(k)) for k in pool_depths],
        "pool_curve": dict(diagnostics),
        "ordering": dict(rerank),
        "r1000": {
            "measured": False,
            "bound": (
                f"The deepest cached pool is depth {deepest}, and the cross-encoder pool is depth "
                "100. R@k for k at or beyond the pool depth is the first stage's number, so R@1000 "
                "needs a deeper pool, not a better reranker."
            ),
        },
    }
    if deep:
        report["r1000"] = {
            "measured": True,
            "curve": dict(deep),
            "note": "the pool's own recall at depth; a reranker cannot change it",
        }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    """A reader-facing rendering of the pack."""
    lines = [
        "# Reranker-fairness pack",
        "",
        "R@k on the reranker-excluded pool, and the part of the reranker's gain that is ordering.",
        "",
        "## Pool provenance",
        "",
    ]
    provenance = report["provenance"]
    if provenance["excluded"]:
        lines.append(
            "The pool is the fused first stage and the cross-encoder contributed no candidate "
            "(`tools/multihop_ce_cache.py` builds the pool, then scores it), so the pool is "
            "**reranker-excluded**. Cache settings:"
        )
        lines.append("")
        for key, value in provenance["settings"].items():
            if isinstance(value, (list, tuple, dict)):
                rendered = f"{type(value).__name__}[{len(value)}]"
            else:
                rendered = str(value)
                if len(rendered) > 120:
                    rendered = rendered[:117] + "..."
            lines.append(f"- `{key}`: {rendered}")
    else:
        lines.append(
            "**The cache names a reranker in pool construction** "
            f"({', '.join(provenance['reranker_keys_in_pool_settings'])}); the pool is not "
            "reranker-excluded and no fairness claim can be made from it."
        )
    lines += ["", f"Queries: **{report['queries']}**.", "", "## Gold-fact recall@k", ""]
    depths = sorted((int(d) for d in report["recall"]), key=int)
    lines.append("| depth | fused | cross-encoder | note |")
    lines.append("| --- | --- | --- | --- |")
    for depth in depths:
        row = report["recall"][str(depth)]
        fused_mean = row.get("fused", {}).get("mean")
        ce_mean = row.get("ce", {}).get("mean")
        note = "ordering only" if fused_mean == ce_mean else "the CE lifts whole facts into the top-k"
        lines.append(f"| {depth} | {fused_mean} | {ce_mean} | {note} |")
    lines += ["", "## Membership: what a reranker can and cannot move", ""]
    membership = report["membership"]
    lines.append(
        f"- At the pool's own depth the two arms carry the **same candidate set** for "
        f"**{membership['identical']}/{membership['checked']}** queries — a second stage only "
        "reorders, so it cannot add a candidate."
    )
    for row in report.get("reordering", []):
        if row["divergent"]:
            lines.append(
                f"- At depth {row['k']} the reordering changes the top-{row['k']} set for "
                f"**{row['divergent']}/{row['checked']}** queries — that difference *is* the "
                "ordering gain."
            )
        else:
            lines.append(
                f"- At depth {row['k']} the reordering changes the top-{row['k']} set for "
                f"**{row['divergent']}/{row['checked']}** queries: at the pool's own depth the two "
                "orders carry the same candidates."
            )
    lines += [
        "",
        "So **R@k for k at or beyond the pool depth is the first stage's number**, and a claim that a "
        "reranker improves R@1000 describes a retriever, not a reranker.",
        "",
        "## The pool's own depth curve",
        "",
        f"From `{report['pool_curve'].get('source')}` (max depth {report['pool_curve'].get('max_depth')}):",
        "",
        "| depth | pool gold-fact recall | queries |",
        "| --- | --- | --- |",
    ]
    for depth, row in sorted(report["pool_curve"].get("curve", {}).items(),
                             key=lambda item: int(item[0])):
        lines.append(f"| {depth} | {row['mean']} | {row['queries']} |")
    bound = report["r1000"]
    lines += ["", "## R@1000", ""]
    if bound.get("measured"):
        lines.append(
            "Measured on the deeper pool: "
            + ", ".join(f"R@{d} = {r['mean']}" for d, r in sorted(bound["curve"]["curve"].items()))
            + ". The pool's recall, not the reranker's — see the membership note above."
        )
    else:
        lines.append(f"**Not measurable from cache.** {bound['bound']}.")
    ordering = report["ordering"]
    lines += ["", "## The ordering gain on that pool", ""]
    lines.append(f"From `{ordering.get('source')}`:")
    lines.append("")
    lines.append("| arm | MAP@10 | Hits@4 | Hits@10 | MRR@10 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for arm in ("fused", "ce"):
        row = ordering.get("arms", {}).get(arm)
        if row:
            lines.append(
                f"| {arm} | {row.get('MAP@10')} | {row.get('Hits@4')} | "
                f"{row.get('Hits@10')} | {row.get('MRR@10')} |"
            )
    lines.append("")
    return "\n".join(lines)


def self_check() -> list[str]:
    """Offline invariants on a synthetic cache — no files, no model, no network.

    The synthetic pool has three candidates; the cross-encoder order is a permutation of it, so
    the top-k *sets* must match while the MAP@10-like ordering changes. A cache that names a
    reranker in pool construction must be refused.
    """
    problems: list[str] = []
    pool = ["a#0", "b#0", "c#0"]
    payload = {
        "_settings": {"rrf_k": 5, "model": "tiny"},
        "queries": {"0": {"pool": list(pool), "ce": [0.1, 0.9, 0.5]}},
    }
    fused, ce = arm_rankings(payload)
    if fused["0"] != pool:
        problems.append("the fused arm is not the pool order")
    if ce["0"] != ["b#0", "c#0", "a#0"]:
        problems.append(f"the CE arm is not sorted by score: {ce['0']}")
    membership = membership_invariance(fused, ce)
    if not membership["all_identical"]:
        problems.append("the candidate set changed under a pure reordering")
    if divergent_top_k(fused, ce, 1)["divergent"] != 1:
        problems.append("a reordering did not move the top-1 set, so the ordering effect is lost")
    if not pool_provenance(payload)["excluded"]:
        problems.append("a pool with no reranker settings was refused")
    dirty = {"_settings": {"pool_reranker": "bge-reranker-base"}, "queries": {}}
    if pool_provenance(dirty)["excluded"]:
        problems.append("a pool built with a reranker was accepted as reranker-excluded")
    if pool_provenance(dirty)["reranker_keys_in_pool_settings"] != ["pool_reranker"]:
        problems.append("the reranker key was not named in the refusal")
    # An empty pool must not report a vacuous "all identical".
    if membership_invariance({}, {})["all_identical"]:
        problems.append("an empty comparison reported all-identical")
    return problems


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="reranker_fairness.py")
    ap.add_argument("--scores", default=str(DEFAULT_SCORES))
    ap.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS))
    ap.add_argument("--rerank", default=str(DEFAULT_RERANK))
    ap.add_argument("--corpus", default=str(nr.CORPUS))
    ap.add_argument("--depths", default="10,100")
    ap.add_argument("--deep", default=None,
                    help="a deeper diagnostics report to fold in for the R@1000 point")
    ap.add_argument("--out", default=str(ROOT / "eval" / "reranker-fairness"))
    ap.add_argument("--check", action="store_true",
                    help="run the offline invariants and exit; reads no cache")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        problems = self_check()
        if problems:
            print("reranker-fairness --check: FAILED")
            for line in problems:
                print(f"  - {line}")
            return 1
        print("ok: reranker-fairness invariants hold (reranker-excluded pool, membership stable)")
        return 0

    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    # The refusal comes first: a fairness claim from a reranker-built pool is invalid whatever the
    # golds say, and reporting "no golds" instead would read as a data problem rather than a
    # methodological one.
    provenance = pool_provenance(scores)
    if not provenance["excluded"]:
        print(
            "reranker-fairness: the cached pool is NOT reranker-excluded "
            f"({', '.join(provenance['reranker_keys_in_pool_settings'])}); refusing to report"
        )
        return 1
    golds = {qid: facts for qid, facts in (scores.get("golds") or {}).items()}
    if not golds:
        print(f"reranker-fairness: {args.scores} carries no golds; cannot score recall")
        return 1
    evidence = nr.Evidence(nr.build_chunks(Path(args.corpus), overlap=200))
    depths = [int(part) for part in str(args.depths).split(",") if part.strip()]
    report = build_report(
        scores=scores,
        golds=golds,
        evidence=evidence,
        diagnostics=pool_depth_curve(Path(args.diagnostics)),
        rerank=ordering_gain(Path(args.rerank)),
        pool_depths=depths,
        deep=pool_depth_curve(Path(args.deep)) if args.deep else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
