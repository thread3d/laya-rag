"""Three more cross-system charts: compute, cost, and the channel reversal.

`tools/compare.py` draws the charts that were already there. These three came later and answer
questions it does not:

1. **Quality per parameter** — our arms against the ten published systems whose 4-dataset means and
   parameter counts are already pinned in `compare.PUBLISHED_4`. This is the axis this project can
   actually win, and it is invisible in a table sorted by score alone.
2. **The channel reversal** — where dense beats lexical and where it loses, across every corpus
   measured: the seven BEIR sets, MultiHop-RAG, MuSiQue and FRAMES. It is the programme's central
   finding and it has never been in one figure.
3. **FRAMES by arm** — recall@1/10/20 for the four arms. It carries **no published marker**, and the
   chart says why: the FRAMES paper reports end-to-end *accuracy* for generators, not retrieval, and
   this project has not run a generator on it. An external marker would have to be invented.

    .venv/bin/python tools/compare_more.py --charts docs/charts
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

import compare  # noqa: E402

FRAMES_REPORT = ROOT / "eval" / "frames-eval-ensemble.json"
BENCH = ROOT / "eval" / "benchmarks"
OURS = compare.OURS
OURS_ALT = compare.OURS_ALT
OURS_LIGHT = compare.OURS_LIGHT
ENSEMBLE = compare.ENSEMBLE
ACCENT = compare.ACCENT
PUBLISHED_DARK = compare.PUBLISHED_DARK


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def dataset_mean(dataset: str, retriever: str) -> float:
    """The dataset's macro nDCG@10 for one arm, from whichever report measured it."""
    return _mean(compare.per_query_for(dataset, retriever))


def parse_params(label: str) -> float:
    """`335M`/`1.2B`/`—` to millions, so the cost axis is numeric."""
    text = str(label).strip().upper()
    if text in ("—", "-", "", "N/A"):
        return 0.0
    scale = 1000.0 if text.endswith("B") else 1.0
    return float(text.rstrip("MB")) * scale


def chart_quality_per_parameter(outdir: Path, plt) -> dict[str, float]:
    """Quality against parameter count: the axis this project can win."""
    four = compare.FOUR
    ours = [
        ("this system\nlexical (BM25)", 0.0,
         _mean([dataset_mean(d, "bm25-engine") for d in four]), OURS_LIGHT, "o"),
        ("this system\nbge-small + fused", 33.0,
         _mean([dataset_mean(d, "hybrid-biencoder") for d in four]), OURS, "o"),
        ("this system\n3-model ensemble", 478.0,
         _mean([_mean(compare.per_query_ensemble(d)) for d in four]), ENSEMBLE, "o"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for name, params, score in compare.PUBLISHED_4:
        ax.scatter(parse_params(params), score, marker="x", s=52, color=PUBLISHED_DARK, zorder=4)
        ax.annotate(name, (parse_params(params), score), textcoords="offset points",
                    xytext=(6, -3), fontsize=7.5, color="#333")
    offsets = [(9, -16), (9, 5), (9, 5)]     # the BM25 point sits in a crowded corner
    for (name, params, score, colour, marker), offset in zip(ours, offsets):
        ax.scatter(params, score, marker=marker, s=110, color=colour, zorder=6,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(name, (params, score), textcoords="offset points", xytext=offset,
                    fontsize=8.5, color=colour, fontweight="bold")
    ax.set_xlabel("encoder parameters (millions; BM25 at 0)")
    ax.set_ylabel("4-dataset mean nDCG@10 (SciFact, NFCorpus, ArguAna, SciDocs)")
    ax.set_title("Quality per parameter: this system against published systems")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(-40, 1350)
    ax.set_ylim(0.30, 0.56)
    compare._save(fig, outdir, "quality_per_parameter.png", plt)
    return {"lexical": ours[0][2], "fused_33m": ours[1][2], "ensemble_478m": ours[2][2],
            "bge_large_335m": dict((n, s) for n, _p, s in compare.PUBLISHED_4)["BGE-large-v1.5"]}


#: (label, metric, lexical, dense) for every corpus measured, in the corpus's own headline metric.
#: BEIR and FRAMES are nDCG@10; the engine rows are MAP@10 and are labelled as such rather than
#: silently mixed, which is why the chart plots a *percentage* change.
REVERSALS = [
    ("SciFact", "nDCG@10", "scifact", "bm25-engine", "biencoder"),
    ("NFCorpus", "nDCG@10", "nfcorpus", "bm25-engine", "biencoder"),
    ("ArguAna", "nDCG@10", "arguana", "bm25-engine", "biencoder"),
    ("SciDocs", "nDCG@10", "scidocs", "bm25-engine", "biencoder"),
    ("FiQA", "nDCG@10", "fiqa", "bm25-engine", "biencoder"),
    ("Quora", "nDCG@10", "quora", "bm25-engine", "biencoder"),
    ("Touché-2020", "nDCG@10", "webis-touche2020", "bm25-engine", "biencoder"),
]


def engine_pair(default_report: str, hybrid_report: str) -> tuple[float, float]:
    """(lexical, dense) MAP@10 from a pair of engine runs over the same queries."""
    def total(name: str) -> float:
        rows = json.loads((BENCH / name).read_text())["per_query"]
        shared = sorted(set(rows) & set(json.loads((BENCH / hybrid_report).read_text())["per_query"]))
        return _mean([rows[q]["metrics"]["MAP@10"] for q in shared])
    return total(default_report), total(hybrid_report)


def reversal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, metric, dataset, lexical_arm, dense_arm in REVERSALS:
        lexical = dataset_mean(dataset, lexical_arm)
        dense = dataset_mean(dataset, dense_arm)
        rows.append({"label": label, "metric": metric, "lexical": lexical, "dense": dense})
    multihop_l, multihop_d = engine_pair("multihop-rag/engine-default.json",
                                         "multihop-rag/engine-hybrid.json")
    rows.append({"label": "MultiHop-RAG", "metric": "MAP@10", "lexical": multihop_l,
                 "dense": multihop_d})
    musique_l, musique_d = engine_pair("musique/engine-default-200.json",
                                       "musique/engine-hybrid-200.json")
    rows.append({"label": "MuSiQue (200)", "metric": "MAP@10", "lexical": musique_l,
                 "dense": musique_d})
    if FRAMES_REPORT.is_file():
        frames = json.loads(FRAMES_REPORT.read_text())["summary"]
        rows.append({"label": "FRAMES", "metric": "nDCG@10",
                     "lexical": frames["lexical"]["ndcg@10"], "dense": frames["dense"]["ndcg@10"]})
    for row in rows:
        row["delta_pct"] = 100.0 * (row["dense"] - row["lexical"]) / row["lexical"]
    return rows


def chart_channel_reversal(outdir: Path, plt) -> list[dict[str, Any]]:
    rows = reversal_rows()
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    y = list(range(len(rows)))
    for position, row in enumerate(rows):
        colour = ENSEMBLE if row["delta_pct"] > 0 else ACCENT
        ax.barh(position, row["delta_pct"], color=colour, height=0.62)
        offset = 3 if row["delta_pct"] > 0 else -3
        ax.text(row["delta_pct"] + offset, position, f"{row['delta_pct']:+.0f}%",
                va="center", ha="left" if offset > 0 else "right", fontsize=8.5,
                color=colour, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['label']}  ({row['metric']})" for row in rows], fontsize=9)
    ax.axvline(0, color="#333", linewidth=0.9)
    ax.set_xlabel("dense channel relative to lexical, percent")
    ax.set_title("Which channel wins, by corpus: the same two arms, opposite answers")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.invert_yaxis()
    compare._save(fig, outdir, "channel_reversal.png", plt)
    return rows


def chart_frames_arms(outdir: Path, plt) -> dict[str, Any]:
    report = json.loads(FRAMES_REPORT.read_text())
    summary: Mapping[str, Mapping[str, float]] = report["summary"]
    arms = [("lexical", "lexical", OURS_LIGHT),
            ("dense", "dense (bge-small)", OURS_ALT),
            ("ensemble", "3-model ensemble", ENSEMBLE),
            ("ensemble+bm25", "ensemble + BM25", ACCENT)]
    ks = (1, 10, 20)
    import numpy as np

    x = np.arange(len(ks))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for index, (arm, label, colour) in enumerate(arms):
        values = [summary[arm][f"recall@{k}"] for k in ks]
        bars = ax.bar(x + (index - 1.5) * width, values, width, color=colour, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"recall@{k}" for k in ks])
    ax.set_ylabel("recall (document level)")
    ax.set_ylim(0, 1.05)
    ax.set_title("FRAMES: retrieval by arm — no published marker, because none exists")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    ax.text(0.5, -0.20,
            "The FRAMES paper reports end-to-end accuracy for generators, not retrieval, and this "
            "project has run no generator on it.\nAn external marker here would have to be invented, "
            "so the chart compares our arms only.",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#444")
    compare._save(fig, outdir, "frames_arms.png", plt)
    return {"arms": {label: {f"recall@{k}": summary[arm][f"recall@{k}"] for k in ks}
                     for arm, label, _c in arms}}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="compare_more.py")
    ap.add_argument("--charts", default=str(ROOT / "docs" / "charts"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.charts)
    outdir.mkdir(parents=True, exist_ok=True)
    plt = compare._plt_mod()
    quality = chart_quality_per_parameter(outdir, plt)
    reversals = chart_channel_reversal(outdir, plt)
    frames = chart_frames_arms(outdir, plt)

    print("quality per parameter (4-dataset mean nDCG@10):")
    for name, value in quality.items():
        print(f"  {name:<18} {value:.4f}")
    print("\nchannel reversal (dense relative to lexical):")
    for row in reversals:
        print(f"  {row['label']:<16} {row['metric']:<8} lexical {row['lexical']:.4f} "
              f"dense {row['dense']:.4f}  {row['delta_pct']:+.1f}%")
    print("\nFRAMES by arm:")
    for arm, values in frames["arms"].items():
        print(f"  {arm:<18} " + "  ".join(f"{k} {v:.4f}" for k, v in values.items()))
    print(f"\nwrote 3 charts to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
