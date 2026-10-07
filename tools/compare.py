"""Cross-system comparison tables, error bars and text charts for the benchmark write-up.

**On error bars.** Retrieval in this project is deterministic: re-running a configuration
reproduces its nDCG@10 exactly, so repeating a run yields no interval at all — it just costs
hours. The uncertainty that actually matters is sampling uncertainty over *queries*: 300 SciFact
queries, 1,406 ArguAna queries, and a different draw of queries would move the mean. So every
interval here is a 10,000-resample bootstrap over the per-query scores each report already
persists (`per_query_nDCG@10`), and for MultiHop-RAG over the per-query rows
`tools/multihop_official.py` emits.

    .venv/bin/python tools/compare.py

Printed output is what the README's comparison section quotes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import evalkit  # noqa: E402

BEIR_DIR = ROOT / "eval" / "beir"
MH_DIR = ROOT / "eval" / "benchmarks" / "multihop-rag"
SAMPLES = 10_000

#: Published 4-dataset BEIR means, from the independent BEIR matrix compiled at
#: github.com/ruvnet/ruflo (commit 56819391), which sources the BEIR and BGE papers.
PUBLISHED_4 = [
    ("BGE-large-v1.5", "335M", 0.491),
    ("SPLADE++", "110M", 0.433),
    ("GTR-XL", "1.2B", 0.405),
    ("GenQ", "110M", 0.400),
    ("BM25 (Lucene)", "—", 0.390),
    ("Contriever", "110M", 0.387),
    ("TAS-B", "66M", 0.385),
    ("DocT5query", "60M", 0.378),
    ("ColBERT", "110M", 0.339),
    ("SBERT msmarco", "110M", 0.330),
]

#: Published MultiHop-RAG retrieval (paper Table 5) and generation (Table 6).
PUBLISHED_MH = [
    ("voyage-02 + rerank", 0.7467, 0.6625, 0.4795, 0.5860),
    ("bge-large + rerank", 0.7183, 0.6364, 0.4759, 0.5630),
    ("bge-large, no rerank", 0.6718, 0.5221, 0.3423, 0.4298),
]

DATASET_ORDER = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa",
                 "quora", "webis-touche2020"]
#: Datasets measured in their own reports rather than the original five-dataset run.
#: Datasets measured in their own reports. The large corpora are lexical-only runs, so they
#: appear here for the BM25 control but deliberately not in DATASET_ORDER, whose chart needs a
#: dense arm that was never run for them.
DATASET_REPORT = {"quora": "quora", "webis-touche2020": "touche",
                  "nq": "nq", "hotpotqa": "hotpotqa", "dbpedia-entity": "dbpedia-entity",
                  "climate-fever": "climate-fever", "fever": "fever",
                  "robust04": "robust04", "trec-news": "trec-news", "signal1m": "signal1m",
                  "cqadupstack": "cqadupstack", "bioasq": "bioasq"}
#: Display names, because `capitalize()` mangles the hyphenated one.
DATASET_LABEL = {"webis-touche2020": "Touché-2020", "quora": "Quora"}
FOUR = ["scifact", "nfcorpus", "arguana", "scidocs"]

#: The per-dataset best configuration, as recorded in RANKINGS.md.
BEST_PER_DATASET = {
    "scifact": ("large-splade-fixed", "hybrid-biencoder", "bge-large + fused BM25"),
    "nfcorpus": ("large-splade-fixed", "hybrid-biencoder", "bge-large + fused BM25"),
    "arguana": ("dense-text-only-fixed", "biencoder", "bge-small dense, text-only"),
    "scidocs": ("five-small-fixed", "hybrid-biencoder", "bge-small + fused BM25"),
}


def bootstrap_ci(values, samples: int = SAMPLES, seed: int = 20210617, confidence: float = 0.95):
    """Mean with a percentile bootstrap interval over the values (queries)."""
    x = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    blocks, remaining = [], samples
    while remaining > 0:
        chunk = min(256, remaining)
        blocks.append(x[rng.integers(0, len(x), size=(chunk, len(x)))].mean(axis=1))
        remaining -= chunk
    means = np.sort(np.concatenate(blocks))
    alpha = (1 - confidence) / 2
    return float(x.mean()), float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def load(name: str) -> dict:
    return json.loads((BEIR_DIR / f"{name}.json").read_text())


def per_query(name: str, dataset: str, retriever: str) -> list[float]:
    report = load(name)["datasets"][dataset]
    return list(report["per_query_nDCG@10"][retriever].values())


def per_query_for(dataset: str, retriever: str) -> list[float]:
    """Per-query nDCG@10 for a dataset from whichever report measured it."""
    return per_query(DATASET_REPORT.get(dataset, "five-small-fixed"), dataset, retriever)


#: The three-model dense ensemble was measured in two runs; route each dataset to its report.
ENSEMBLE_REPORT = {
    "scifact": "ensemble3", "nfcorpus": "ensemble3", "arguana": "ensemble3",
    "scidocs": "ensemble2", "fiqa": "ensemble2",
}


def per_query_ensemble(dataset: str, retriever: str = "hybrid-ensemble") -> list[float]:
    """Per-query nDCG@10 for the single three-model ensemble configuration."""
    return per_query(ENSEMBLE_REPORT[dataset], dataset, retriever)


def macro_with_ci(pools: dict[str, list[float]], samples: int = 4000, seed: int = 7):
    """Mean of per-dataset means, bootstrapping queries within each dataset."""
    rng = np.random.default_rng(seed)
    arrays = {d: np.asarray(v, dtype=np.float64) for d, v in pools.items()}
    point = float(np.mean([a.mean() for a in arrays.values()]))
    draws = [
        np.mean([a[rng.integers(0, len(a), size=len(a))].mean() for a in arrays.values()])
        for _ in range(samples)
    ]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def bar(value: float, scale: float, width: int = 40) -> str:
    full = int(round(width * value / scale))
    return "█" * full + "·" * (width - full)


def ranking_chart(rows, label: str, label_width: int = 34, scale: float = 0.52) -> None:
    print()
    print(label)
    print("-" * len(label))
    for name, params, mean in sorted(rows, key=lambda row: -row[2]):
        print(f"  {name:<{label_width}} {bar(mean, scale)} {mean:.3f}")


# ── charts ───────────────────────────────────────────────────────────────

OURS = "#1b6ca8"
OURS_ALT = "#79b4d9"
OURS_LIGHT = "#bdd7e7"
ENSEMBLE = "#1f7a4d"
PUBLISHED = "#c9c9c9"
PUBLISHED_DARK = "#8a8a8a"
ACCENT = "#c0392b"

#: Published BM25 nDCG@10. The first five are the anchors this project has always quoted; the last
#: two come from the same table and the same column, checked by reproducing three existing anchors —
#: ArguAna 0.397, SciFact 0.679 and FiQA 0.236 all match the source's BM25 column exactly. Source:
#: "Resources for Brewing BEIR" (arXiv:2306.07471), reference-model table.
PUBLISHED_BM25 = {"scifact": 0.679, "nfcorpus": 0.325, "arguana": 0.397,
                  "scidocs": 0.158, "fiqa": 0.236,
                  "quora": 0.789, "webis-touche2020": 0.442}
PUBLISHED_BGE = {"scifact": 0.722, "nfcorpus": 0.380, "arguana": 0.636, "scidocs": 0.225}
PUBLISHED_MH_DICT = {"bge-large + rerank": (0.7183, 0.6364, 0.4759, 0.5630),
                     "voyage-02 + rerank": (0.7467, 0.6625, 0.4795, 0.5860)}

_PLT = None


def _plt_mod():
    """Import matplotlib lazily, with a cache dir the sandbox will actually write to."""
    global _PLT
    if _PLT is None:
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "eval" / ".mplconfig"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": True,
            "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
        })
        _PLT = plt
    return _PLT


def _save(fig, outdir: Path, name: str, plt) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Print a repo-relative path when the output is inside the checkout, and the absolute
    # one otherwise — a temp directory in a test is not a subpath of ROOT.
    try:
        shown = path.resolve().relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"  wrote {shown}  ({path.stat().st_size // 1024} KB)")


def chart_beir_ranking(outdir: Path, best: float, ci, deploy: float, ensemble, ensemble_ci, plt) -> None:
    systems = [("this system - 3-model ensemble + BM25", ensemble, OURS, ensemble_ci),
               ("this system - best per dataset", best, OURS, ci)]
    systems += [(f"{name} ({params})", value, PUBLISHED, None) for name, params, value in PUBLISHED_4]
    systems.append(("this system - one encoder + BM25", deploy, OURS_ALT, None))
    systems.sort(key=lambda row: -row[1])

    labels = [s[0] for s in systems]
    values = [s[1] for s in systems]
    colors = [s[2] for s in systems]
    lo = np.array([0.0 if s[3] is None else max(0.0, s[1] - s[3][0]) for s in systems])
    hi = np.array([0.0 if s[3] is None else max(0.0, s[3][1] - s[1]) for s in systems])

    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.66, xerr=np.vstack([lo, hi]),
            error_kw=dict(ecolor="#333", capsize=3, lw=1.1))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    span = max(values)
    for yy, value, hi_delta in zip(y, values, hi):
        ax.text(value + hi_delta + span * 0.018, yy, f"{value:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, span * 1.22)
    ax.set_xlabel("4-dataset mean nDCG@10  (SciFact, NFCorpus, ArguAna, SciDocs)")
    ax.set_title("BEIR retrieval: this system against published systems")
    fig.text(0.005, -0.02, "Bars for this system carry a 95% bootstrap interval over queries; "
                           "published figures carry none.", fontsize=7.5, color="#555")
    _save(fig, outdir, "beir_ranking.png", plt)


def chart_beir_datasets(outdir: Path, plt) -> None:
    series = [("shipped lexical", "bm25-engine", OURS_LIGHT, False),
              ("bi-encoder (bge-small)", "biencoder", OURS_ALT, False),
              ("fused BM25 + bi-encoder", "hybrid-biencoder", OURS, False),
              ("3-model ensemble + BM25", "hybrid-ensemble", ENSEMBLE, True)]
    x = np.arange(len(DATASET_ORDER))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    for i, (label, key, color, is_ensemble) in enumerate(series):
        vals, lo, hi = [], [], []
        for ds in DATASET_ORDER:
            if is_ensemble and ds not in ENSEMBLE_REPORT:
                # The ensemble was not run on the two datasets added later; leave a gap rather than
                # implying a measurement that does not exist.
                vals.append(float("nan")); lo.append(float("nan")); hi.append(float("nan"))
                continue
            values = per_query_ensemble(ds) if is_ensemble else per_query_for(ds, key)
            m, low, high = bootstrap_ci(values)
            vals.append(m)
            lo.append(m - low)
            hi.append(high - m)
        ax.bar(x + (i - 1.5) * width, vals, width, color=color, label=label,
               yerr=np.vstack([lo, hi]), capsize=2.5, error_kw=dict(lw=0.9, ecolor="#444"))

    anchored = [d for d in DATASET_ORDER if d in PUBLISHED_BM25]
    ax.scatter([DATASET_ORDER.index(d) for d in anchored], [PUBLISHED_BM25[d] for d in anchored],
               marker="x", s=46,
               color=PUBLISHED_DARK, zorder=5, label="published BM25")
    four = [d for d in DATASET_ORDER if d in PUBLISHED_BGE]
    ax.scatter([DATASET_ORDER.index(d) for d in four], [PUBLISHED_BGE[d] for d in four],
               marker="D", s=30, facecolor="none", edgecolor=ACCENT, zorder=5,
               label="published BGE-large")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL.get(d, d.capitalize()) for d in DATASET_ORDER],
                       rotation=20, ha="right")
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 1.02)
    ax.set_title("BEIR per dataset: this system against published BM25 and BGE-large")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    _save(fig, outdir, "beir_datasets.png", plt)


def chart_multihop(outdir: Path, official: dict, plt) -> None:
    keys = ["Hits@10", "Hits@4", "MAP@10", "MRR@10"]
    ours = [("this system - hybrid", "hybrid (ours)@ov200", OURS),
            ("this system - BM25", "bm25@ov200", OURS_ALT),
            ("this system - bge-small dense", "bge-small dense@ov200", OURS_LIGHT)]
    blocks = []
    for label, key, color in ours:
        vals, lo, hi = [], [], []
        for metric in keys:
            m, low, high = bootstrap_ci([r[metric] for r in official[key]["_per_query"]])
            vals.append(m)
            lo.append(m - low)
            hi.append(high - m)
        blocks.append((label, vals, color, np.vstack([lo, hi])))
    for label, values in PUBLISHED_MH_DICT.items():
        blocks.append((f"published - {label}",
                       list(values), PUBLISHED_DARK if "bge" in label else PUBLISHED, None))

    x = np.arange(len(keys))
    width = 0.16
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, (label, vals, color, err) in enumerate(blocks):
        ax.bar(x + (i - 2) * width, vals, width, color=color, label=label, yerr=err,
               capsize=2, error_kw=dict(lw=0.9, ecolor="#444"))
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("score")
    ax.set_ylim(0, 0.95)
    ax.set_title("MultiHop-RAG under the benchmark's own protocol")
    ax.legend(frameon=False, fontsize=7.5, ncol=3)
    ax.annotate("the gap:\nall evidence, not one piece",
                xy=(2 - 0.02, 0.30), xytext=(1.15, 0.62), fontsize=8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))
    _save(fig, outdir, "multihop_rag.png", plt)


def chart_abstention(outdir: Path, plt) -> None:
    gate = json.loads((MH_DIR / "gate.json").read_text())
    groups = [("null_query\n(abstention correct)", gate["null"]),
              ("answerable\n(abstention is an error)", gate["positive"])]
    rates, errors, counts = [], [], []
    for _, rows in groups:
        n = len(rows)
        rate = sum(1 for row in rows if not row["answerable"]) / n
        rates.append(rate)
        errors.append(1.96 * (rate * (1 - rate) / n) ** 0.5)
        counts.append(n)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.bar([0, 1], rates, 0.5, color=[OURS, PUBLISHED_DARK], yerr=errors, capsize=5,
           error_kw=dict(lw=1.2, ecolor="#333"))
    for i, (rate, n) in enumerate(zip(rates, counts)):
        ax.text(i, rate + 0.035, f"{rate*100:.1f}%", ha="center", fontweight="bold")
        ax.text(i, 0.03, f"n={n}", ha="center", fontsize=8, color="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("abstention rate")
    ax.set_ylim(0, 1)
    ax.set_title("Answerability gate: no discrimination")
    fig.text(0.005, -0.04, "If the gate worked, the left bar would be near 1 and the right near 0. "
                           "The 0.022 gap is within noise.", fontsize=7.5, color="#555")
    _save(fig, outdir, "abstention.png", plt)


def chart_overlap(outdir: Path, official: dict, plt) -> None:
    names = ["bm25", "bm25-engine", "bge-small dense", "hybrid (ours)"]
    a = [official[f"{n}@ov200"]["Hits@10"] for n in names]
    b = [official[f"{n}@ov0"]["Hits@10"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.19, a, 0.38, color=OURS, label="overlap 200 (llama-index default)")
    ax.bar(x + 0.19, b, 0.38, color=OURS_ALT, label="overlap 0")
    for xi, (va, vb) in enumerate(zip(a, b)):
        ax.text(xi - 0.19, va + 0.012, f"{va:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.19, vb + 0.012, f"{vb:.3f}", ha="center", fontsize=8)
        ax.text(xi, 0.03, f"{vb-va:+.3f}", ha="center", fontsize=8, color="white",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Hits@10")
    ax.set_ylim(0, 1.0)
    ax.set_title("MultiHop-RAG: one undeclared chunking constant")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, outdir, "chunk_overlap.png", plt)


def second_stage_panels(
    outdir: Path,
    plt,
    report_path: Path,
    names: list[str],
    filename: str,
    title: str,
) -> bool:
    """Render the two-panel comparison for one report and a named strategy subset.

    Left panel: the level, as a dot with its 95% bootstrap interval — not a bar from zero,
    because the differences are a few points and a truncated bar would misrepresent them.
    Right panel: the question a reader actually has, *is this better than the baseline by
    more than the noise?*, answered with a **paired** interval on the per-query difference,
    so a whisker crossing zero reads as the null result it is.
    """
    report = json.loads(report_path.read_text())
    per_query = report.get("per_query") or {}
    names = [name for name in dict.fromkeys(names) if name in per_query]
    if not names:
        print(f"  ({report_path.name} has none of the requested strategies; skipping {filename})")
        return False

    means = {name: float(np.mean(list(per_query[name].values()))) for name in names}
    baseline = "fused" if "fused" in means else min(means, key=means.get)
    rows = []
    for name in names:
        values = [float(v) for v in per_query[name].values()]
        mean, low, high = bootstrap_ci(values)
        rows.append((name, mean, mean - low, high - mean))
    rows.sort(key=lambda row: row[1])
    order = [row[0] for row in rows]
    position = {name: index for index, name in enumerate(order)}

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.4, 0.4 * len(rows) + 2.0), gridspec_kw={"width_ratios": [1, 1]}
    )
    ys = np.arange(len(rows))
    # The shuffled-bridge control must not share a colour with the treatment it controls for:
    # reading a control as a treatment is the mistake this whole comparison exists to prevent.
    def series_color(name: str) -> str:
        if "shuffled" in name:
            return PUBLISHED_DARK
        if name.startswith(("tri", "pit-")):
            return ACCENT
        return OURS

    colors = [series_color(name) for name in order]
    left.errorbar(
        [row[1] for row in rows], ys,
        xerr=np.vstack([[row[2] for row in rows], [row[3] for row in rows]]),
        fmt="none", elinewidth=1.2, capsize=3, ecolor="#444",
    )
    for row, y, color in zip(rows, ys, colors):
        left.plot([row[1]], [y], "o", ms=7, color=color)
    left.set_yticks(ys)
    left.set_yticklabels(order, fontsize=8)
    left.set_xlabel("MAP@10 (95% bootstrap interval)")
    left.set_title("level", fontsize=10)
    lo = min(row[1] - row[2] for row in rows)
    hi = max(row[1] + row[3] for row in rows)
    pad = max(0.005, (hi - lo) * 0.35)
    left.set_xlim(lo - pad, hi + pad)
    left.grid(axis="x", alpha=0.25, lw=0.6)

    baseline_values = per_query[baseline]
    deltas, lows, highs = [], [], []
    for name in order:
        stats = evalkit.paired_bootstrap(baseline_values, per_query[name])
        deltas.append(float(stats["delta"]))
        lows.append(float(stats["delta"]) - float(stats["ci_low"]))
        highs.append(float(stats["ci_high"]) - float(stats["delta"]))
    right.axvline(0.0, color=PUBLISHED_DARK, lw=0.9, ls="--")
    right.errorbar(
        deltas, ys, xerr=np.vstack([lows, highs]), fmt="none",
        elinewidth=1.2, capsize=3, ecolor="#444",
    )
    for name, y, delta, color in zip(order, ys, deltas, colors):
        right.plot([delta], [y], "o", ms=7, color=color)
    right.set_yticks(ys)
    right.set_yticklabels([])
    right.set_xlabel(f"MAP@10 delta vs `{baseline}` (95% paired interval)")
    right.set_title("does it beat the baseline beyond the noise?", fontsize=10)
    right.set_xlim(min(-0.01, min(deltas) - max(lows) - 0.005), max(deltas) + max(highs) + 0.005)
    right.grid(axis="x", alpha=0.25, lw=0.6)

    left.set_ylim(-0.6, len(rows) - 0.4)
    right.set_ylim(-0.6, len(rows) - 0.4)
    fig.suptitle(
        f"{title} — {report.get('_settings', {}).get('limit_queries', '?')} "
        "MultiHop-RAG queries (orange: bridge-conditioned, grey: shuffled-bridge control)",
        fontsize=11,
    )
    _save(fig, outdir, filename, plt)
    return True


def chart_second_stage(outdir: Path, plt) -> None:
    """The broad second-stage comparison, over the largest report available."""
    candidates = [
        MH_DIR / "rerank-experiment-full.json",
        MH_DIR / "rerank-experiment.json",
    ]
    report_path = next((path for path in candidates if path.is_file()), None)
    if report_path is None:
        print(f"  (no second-stage report in {MH_DIR}; skipping that chart)")
        return
    report = json.loads(report_path.read_text())
    per_query = report.get("per_query") or {}
    if not per_query:
        print(f"  ({report_path.name} has no per-query scores; skipping that chart)")
        return
    wanted = ["fused", "ce", "coverage-0.7", "covquota-3-0.7"]
    names = [n for n in wanted if n in per_query]
    names += sorted(n for n in per_query if n.startswith("tri") or n.startswith("pit-"))
    second_stage_panels(
        outdir, plt, report_path, names, "second_stage.png", "Second-stage selection"
    )


def chart_bridge_stage(outdir: Path, plt) -> None:
    """The bridge-conditioned comparison: the treatment, its control, and the two baselines."""
    report_path = MH_DIR / "rerank-bridge.json"
    if not report_path.is_file():
        print(f"  ({report_path.name} absent; skipping the bridge chart)")
        return
    wanted = [
        "fused",
        "ce",
        "covquota-3-0.7",
        "tri",
        "tri-covquota-3-0.7",
        "ce-tri-filter-0.3",
        "ce-tri-top40",
        "tri-shuffled",
        "tri-shuffled-covquota-3-0.7",
        "ce-tri-shuffled-filter-0.3",
        "pit-tri-ce",
        "pit-tri-shuffled-ce",
    ]
    second_stage_panels(
        outdir, plt, report_path, wanted, "bridge_stage.png",
        "Bridge-conditioned relevance asked of Laya's decision head",
    )


def make_charts(outdir: Path, best: float, ci, deploy: float, ensemble, ensemble_ci, official: dict) -> None:
    plt = _plt_mod()
    chart_beir_ranking(outdir, best, ci, deploy, ensemble, ensemble_ci, plt)
    chart_beir_datasets(outdir, plt)
    chart_multihop(outdir, official, plt)
    chart_abstention(outdir, plt)
    chart_overlap(outdir, official, plt)
    chart_second_stage(outdir, plt)
    chart_bridge_stage(outdir, plt)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare.py",
        description="Cross-system comparison tables, error bars and charts.",
    )
    parser.add_argument(
        "--charts", nargs="?", const="docs/charts", default=None, metavar="DIR",
        help="write PNG charts here (default when passed without a value: docs/charts)",
    )
    args = parser.parse_args(argv)

    print("=" * 78)
    print("BEIR — per-dataset nDCG@10 with 95% bootstrap intervals over queries")
    print("=" * 78)
    retrievers = [("bm25-engine", "shipped lexical"), ("biencoder", "bi-encoder bge-small"),
                  ("hybrid-biencoder", "fused BM25 + bi-encoder")]
    header = f"{'dataset':<10} {'queries':>7} " + " ".join(f"{label:>26}" for _, label in retrievers)
    print(header)
    for ds in DATASET_ORDER:
        cell = []
        for r, _ in retrievers:
            try:
                m, lo, hi = bootstrap_ci(per_query_for(ds, r))
                cell.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]".rjust(26))
            except KeyError:
                cell.append("—".rjust(26))
        n = len(per_query_for(ds, "bm25-engine"))
        print(f"{ds:<10} {n:>7} " + " ".join(cell))

    print()
    print("4-dataset means (SciFact, NFCorpus, ArguAna, SciDocs)")
    print("-" * 78)
    deploy = [np.mean([np.mean(per_query("five-small-fixed", d, "hybrid-biencoder")) for d in FOUR])]
    best_pools = {d: per_query(BEST_PER_DATASET[d][0], d, BEST_PER_DATASET[d][1]) for d in FOUR}
    ensemble_pools = {d: per_query_ensemble(d) for d in FOUR}
    best = [np.mean([np.mean(v) for v in best_pools.values()])]
    _, best_lo, best_hi = macro_with_ci(best_pools)
    ens_mean, ens_lo, ens_hi = macro_with_ci(ensemble_pools)
    print(f"  this system — 3-model ensemble   {ens_mean:.3f}  [{ens_lo:.3f},{ens_hi:.3f}]")
    print(f"  this system — best per dataset   {best[0]:.3f}  [{best_lo:.3f},{best_hi:.3f}]")
    print(f"  this system — one deployable cfg {deploy[0]:.3f}")
    print(f"  BGE-large-v1.5 (published)       0.491")

    rows = [("this system — 3-model ensemble", "478M", ens_mean),
            ("this system — best per dataset", "mixed", best[0]),
            ("BGE-large-v1.5", "335M", 0.491),
            ("this system — deployable config", "33M", deploy[0]),
            ("SPLADE++", "110M", 0.433)] + PUBLISHED_4[2:]
    ranking_chart(rows, "BEIR 4-dataset mean, nDCG@10 — this system vs published")

    print()
    print("=" * 78)
    print("MultiHop-RAG — official protocol, 95% bootstrap intervals over queries")
    print("=" * 78)
    official = json.loads((MH_DIR / "official.json").read_text())
    metric_keys = ["Hits@10", "Hits@4", "MAP@10", "MRR@10"]
    print(f"{'system':<30} " + " ".join(f"{k:>18}" for k in metric_keys))
    for name, h10, h4, mp, mrr in PUBLISHED_MH:
        print(f"{(name + ' (published)'):<30} " + " ".join(
            f"{v:>18.3f}" for v in (h10, h4, mp, mrr)))
    for key, label in [("hybrid (ours)@ov200", "hybrid (ours)"),
                       ("bm25@ov200", "BM25"),
                       ("bge-small dense@ov200", "bge-small dense")]:
        rows_pq = official[key]["_per_query"]
        cells = []
        for k in metric_keys:
            m, low, high = bootstrap_ci([r[k] for r in rows_pq])
            cells.append(f"{m:.3f} [{low:.2f},{high:.2f}]".rjust(18))
        print(f"{label:<30} " + " ".join(cells))

    mh_rows = [("this system — hybrid (ours)", "", official["hybrid (ours)@ov200"]["Hits@10"]),
               ("this system — BM25", "", official["bm25@ov200"]["Hits@10"])]
    mh_rows += [(n + " (published)", "", h10) for n, h10, _, _, _ in PUBLISHED_MH]
    mh_rows.append(("this system — bge-small dense", "",
                    official["bge-small dense@ov200"]["Hits@10"]))
    ranking_chart(mh_rows, "MultiHop-RAG Hits@10 (overlap=200) — this system vs published",
                  label_width=32, scale=0.86)

    print()
    print("Chunk-overlap sensitivity (same rankings, different splitter constant)")
    print("-" * 78)
    for key in ["bm25", "bm25-engine", "bge-small dense", "hybrid (ours)"]:
        a = official[f"{key}@ov200"]["Hits@10"]
        b = official[f"{key}@ov0"]["Hits@10"]
        print(f"  {key:<18} overlap=200 {a:.3f}   overlap=0 {b:.3f}   delta {b-a:+.3f}")

    if args.charts:
        outdir = Path(args.charts)
        if not outdir.is_absolute():
            outdir = ROOT / outdir
        print()
        print("charts")
        print("-" * 78)
        make_charts(outdir, best[0], (best_lo, best_hi), deploy[0], ens_mean, (ens_lo, ens_hi), official)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
