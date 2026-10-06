"""Charts for the 2026-10-06 pass: generator-free iteration and quantization.

Two measurements added in this pass had no picture. Both are pure report-reading — the experiment
runs write JSON under `eval/`, and this renders it. Skips silently when a report is absent, because
`eval/` is gitignored and regenerable.

    .venv/bin/python tools/charts_pass.py --charts docs/charts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
GFI = BENCH / "generator-free-iteration.json"
QUANT = ROOT / "eval" / "quantization-probe-{name}.json"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        print(f"charts-pass: skip {path.relative_to(ROOT)} (absent)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def chart_generator_free_iteration(outdir: Path, plt: Any) -> bool:
    """Coverage, pool ceiling and the per-question-type gain, real probes vs shuffled control."""
    report = _load(GFI)
    if report is None:
        return False
    rows = report["iterations"]
    rounds = [row["iteration"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    coverage, ceiling, by_type = axes

    coverage.plot(rounds, [row["real_coverage"] for row in rows], "o-", color="#1f77b4",
                  label="entity probes")
    coverage.plot(rounds, [row["shuffled_coverage"] for row in rows], "s--", color="#7f7f7f",
                  label="shuffled control")
    coverage.set_title("Gold-fact coverage")
    coverage.set_xlabel("expansion round")
    coverage.set_ylim(0.78, 0.92)
    coverage.legend(fontsize=8)

    ceiling.plot(rounds, [row["real_ceiling_map@10"] for row in rows], "o-", color="#1f77b4",
                 label="entity probes")
    ceiling.plot(rounds, [row["shuffled_ceiling_map@10"] for row in rows], "s--", color="#7f7f7f",
                 label="shuffled control")
    ceiling.set_title("Pool ceiling MAP@10")
    ceiling.set_xlabel("expansion round")
    ceiling.set_ylim(0.58, 0.64)

    kinds = list((report.get("by_question_type") or {}).keys())
    if kinds:
        import numpy as np

        positions = np.arange(len(kinds))
        width = 0.38
        baseline = [report["by_question_type"][k]["coverage_baseline"] for k in kinds]
        final = [report["by_question_type"][k]["coverage_final"] for k in kinds]
        by_type.bar(positions - width / 2, baseline, width, label="round 0", color="#c7c7c7")
        by_type.bar(positions + width / 2, final, width, label="round 1", color="#1f77b4")
        by_type.set_xticks(positions)
        by_type.set_xticklabels([k.replace("_query", "") for k in kinds], fontsize=8)
        by_type.set_ylim(0.7, 1.0)
        by_type.set_title("Coverage by question type")
        by_type.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "generator_free_iteration.png", dpi=140)
    plt.close(fig)
    print("charts-pass: wrote generator_free_iteration.png")
    return True


def chart_quantization(outdir: Path, plt: Any) -> bool:
    """Recall@10 against index bytes per vector, on the shipped encoder's real vectors."""
    reports = [(name, _load(QUANT.with_name(f"quantization-probe-{name}.json")))
               for name in ("scidocs", "fiqa")]
    reports = [(name, report) for name, report in reports if report]
    if not reports:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colours = {"scidocs": "#1f77b4", "fiqa": "#d62728"}
    labelled: set[str] = set()
    for name, report in reports:
        rows = report["rows"]
        ax.plot([row["bytes_per_vector"] for row in rows], [row["recall@10"] for row in rows],
                "o-", color=colours.get(name, "black"), label=name)
        # The codec names are the same for both corpora, so label one series only.
        if name != "scidocs":
            continue
        for row in rows:
            kind = row["kind"]
            if kind in labelled or kind == "flat-exact":
                continue
            labelled.add(kind)
            label = "ivf-flat / flat-exact" if kind == "ivf-flat" else kind
            # Fixed offsets: the top-right codecs sit within a few pixels of each other.
            xytext, ha = {
                "ivf-pq": ((4, -10), "left"),
                "ivf-sq8": ((5, -11), "left"),
                "ivf-fp16": ((-3, 7), "left"),
                "ivf-flat": ((-6, -14), "right"),
            }.get(kind, ((4, -10), "left"))
            ax.annotate(label, (row["bytes_per_vector"], row["recall@10"]),
                        textcoords="offset points", xytext=xytext, ha=ha, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("index bytes per vector (log scale)")
    ax.set_ylabel("recall@10 (self-retrieval)")
    ax.set_title("Compression cost on bge-small's own vectors")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "quantization.png", dpi=140)
    plt.close(fig)
    print("charts-pass: wrote quantization.png")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="charts_pass.py")
    ap.add_argument("--charts", default=str(ROOT / "docs" / "charts"))
    args = ap.parse_args(argv)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "eval" / ".mplconfig"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        print("charts-pass: matplotlib is not installed; skipping")
        return 0
    outdir = Path(args.charts)
    outdir.mkdir(parents=True, exist_ok=True)
    wrote = chart_generator_free_iteration(outdir, plt)
    wrote = chart_quantization(outdir, plt) or wrote
    if not wrote:
        print("charts-pass: nothing to render (no reports on disk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
