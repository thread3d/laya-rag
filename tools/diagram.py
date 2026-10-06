"""Render the README's architecture diagram.

The ASCII version at the top of the README is compact and easy to edit, but it renders as a
code block — the first thing a reader sees is text art rather than a picture. This draws the
same two-panel diagram as an image, styled to match the comparison charts, and keeps it
reproducible in the repository instead of a file somebody has to re-make by hand.

    .venv/bin/python tools/diagram.py            # -> docs/architecture.png
    .venv/bin/python tools/diagram.py --out docs/architecture.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Matches the palette in tools/compare.py so the README reads as one document.
INK = "#12395c"
BLUE = "#1b6ca8"
BLUE_DEEP = "#0f4c75"
PANEL_CLIENT = "#f4f6f8"
EDGE_CLIENT = "#c3ccd6"
PANEL_ENGINE = "#eef5fb"
EDGE_ENGINE = "#a8c9e2"
STAGE_FACE = "#ffffff"
STAGE_EDGE = "#8fb8d8"
STAGE_LAST = "#dceaf7"
CHECK_FACE = "#fdf3e3"
CHECK_EDGE = "#d9a441"
GREY = "#6b7684"


def _plt():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "eval" / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "savefig.dpi": 150})
    return plt


def _box(ax, x, y, w, h, text, *, face, edge, fontsize=10, weight="normal",
         color=INK, radius=0.07, lw=1.2, zorder=2, style="round"):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"{style},pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=zorder,
        mutation_aspect=1,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=color, zorder=zorder + 1, linespacing=1.45)


def _arrow(ax, start, end, *, color=BLUE, lw=1.7, style="-|>", linestyle="-",
           zorder=5, scale=15):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=scale, linewidth=lw,
        color=color, linestyle=linestyle, zorder=zorder,
        shrinkA=0, shrinkB=0, joinstyle="miter",
    ))


def draw(out: Path) -> None:
    plt = _plt()
    fig = plt.figure(figsize=(12, 6.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0.42, 6.36)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── panels ───────────────────────────────────────────────────────────
    _box(ax, 0.15, 1.55, 3.00, 4.70, "", face=PANEL_CLIENT, edge=EDGE_CLIENT,
         radius=0.10, lw=1.3, zorder=1)
    _box(ax, 4.75, 1.55, 7.10, 4.70, "", face=PANEL_ENGINE, edge=EDGE_ENGINE,
         radius=0.10, lw=1.3, zorder=1)

    ax.text(0.45, 6.05, "Clients", fontsize=11.5, fontweight="bold", color=INK,
            ha="left", va="center")
    ax.text(5.05, 6.05, "Laya RAG engine  (Python)", fontsize=11.5, fontweight="bold",
            color=BLUE_DEEP, ha="left", va="center")

    # ── clients ──────────────────────────────────────────────────────────
    clients = [
        ("browser", "the GUI at  /"),
        ("HTTP client", "curl, SDKs"),
        ("MCP client", "Claude, Cursor, \u2026"),
        ("cli", "the laya-rag CLI"),
    ]
    for i, (name, detail) in enumerate(clients):
        y = 5.10 - i * 0.80
        _box(ax, 0.45, y, 2.40, 0.62, "", face="#ffffff", edge="#9aa7b4", radius=0.06)
        ax.text(0.60, y + 0.31, name, fontsize=9.6, fontweight="bold", color=INK,
                ha="left", va="center", zorder=4)
        ax.text(1.62, y + 0.31, f"\u2014  {detail}", fontsize=8.6, color=GREY,
                ha="left", va="center", zorder=4)

    ax.text(1.65, 2.15, "the caller composes the answer, or not", fontsize=8.8,
            style="italic", color=GREY, ha="center", va="center")

    # ── the three interfaces the engine exposes, then the pipeline ───────
    xs, width = 5.05, 6.50
    _box(ax, xs, 5.20, width, 0.60,
         "interfaces:   HTTP API  (:8077)   \u00b7   MCP server  (stdio / HTTP)   \u00b7   CLI",
         face=CHECK_FACE, edge=CHECK_EDGE, fontsize=9.4, weight="bold",
         color="#7a5410", radius=0.05, lw=1.2)

    stages = [
        "chunk  \u2192  embed  \u2192  index\n(delete-by-id tombstones; compact reclaims)",
        "hybrid recall  (dense + BM25)\ntrust/ACL-screened before fusion; pool shape reported",
        "Laya re-rank  +  answerability gate\n(optional feature gate; bridge column opt-in)",
        "evidence bundle, with calibrated probabilities\nextractive composer emits verbatim spans",
    ]
    height = 0.55
    ys = [4.35, 3.55, 2.75, 1.95]
    _arrow(ax, (xs + width / 2, 5.20), (xs + width / 2, ys[0] + height), lw=1.5, scale=13)
    for i, (text, y) in enumerate(zip(stages, ys)):
        last = i == len(stages) - 1
        _box(ax, xs, y, width, height, text,
             face=STAGE_LAST if last else STAGE_FACE,
             edge=BLUE if last else STAGE_EDGE,
             fontsize=8.6 if not last else 8.8,
             weight="bold" if last else "normal", radius=0.05, lw=1.2)
        if i:
            _arrow(ax, (xs + width / 2, ys[i - 1]), (xs + width / 2, y + height),
                   lw=1.5, scale=13)

    ax.text(xs + 0.02, 1.72,
            "index mutations are hash-chained (make audit-verify)",
            fontsize=8.4, style="italic", color=GREY, ha="left", va="center")

    # ── the wire across to the interfaces: HTTP and MCP ──────────────────
    _arrow(ax, (3.15, 5.62), (5.05, 5.62))
    _arrow(ax, (5.05, 5.38), (3.15, 5.38))
    ax.text(4.10, 5.84, "HTTP  \u00b7  MCP", fontsize=9.4, fontweight="bold", color=BLUE,
            ha="center", va="center")
    ax.text(4.10, 5.16, "request  /  evidence", fontsize=8.2, color=GREY,
            ha="center", va="center")

    # ── checkpoint the engine loads ──────────────────────────────────────
    cx = 4.75 + 7.10 / 2
    _box(ax, cx - 1.85, 0.72, 3.70, 0.62, "Laya checkpoint  (torch)",
         face=CHECK_FACE, edge=CHECK_EDGE, fontsize=9.8, weight="bold",
         color="#7a5410", radius=0.05, lw=1.2)
    _arrow(ax, (cx, 1.55), (cx, 1.34), color=CHECK_EDGE, lw=1.3,
           linestyle=(0, (4, 3)), style="-|>", scale=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="diagram.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/architecture.png")
    args = parser.parse_args(argv)
    out = Path(args.out)
    draw(out if out.is_absolute() else ROOT / out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
