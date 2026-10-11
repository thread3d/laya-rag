#!/usr/bin/env python
"""Regenerate the channel-coverage map and the deficit arithmetic in `SCORE-LIFT.md` §1.

`SCORE-LIFT.md` §1.1 makes a load-bearing claim — that "BEIR coverage 18 of 18" is a
statement about *dataset membership*, not about *channel coverage*, and that eleven members
have never had a dense or learned-sparse row — and §1.2 turns that into a score arithmetic
(a **0.8054** summed deficit, worth +0.022 to the macro at half-closed and +0.040 at nine
tenths). Neither is reproducible from the other documents in this repository, and
`tools/consistency_check.py` computes membership (`measured_beir`) but not the per-member
*retriever rows*. So the claim that matters most in that document was the one thing a reader
could not check.

This tool closes that. It reads the retriever rows actually present in `eval/beir/*.json` and
prints:

* the per-member channel map (§1.1), with each member classified by the richest channel it has;
* the lexical-only set, which is the claim;
* the deficit arithmetic (§1.2), from an explicit and cited "ours" column.

The "ours" column is **hand-entered** and carries its source in :data:`OUR_BEST`; the published
column is read from :mod:`published_beir18`, which was itself parsed from the paper's HTML. Only
the *sum* is computed here, so the arithmetic is auditable even though one input is transcribed.

**This module is an authority, not just a script.** `consistency_check.py` imports
:func:`channel_map` and :func:`deficit` and requires the §1.1 table rows and the §1.2 summary
lines to appear verbatim in `SCORE-LIFT.md`, so the document cannot drift from the reports it
is derived from without failing `make consistency-check`.

    .venv/bin/python tools/channel_coverage.py
    make channel-coverage
"""

from __future__ import annotations

import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import published_beir18 as published  # noqa: E402  (the parsed BEIR-18 table)
from consistency_check import BEIR_18  # noqa: E402  (the membership set, not re-typed)

BEIR_DIR = ROOT / "eval" / "beir"

#: Retrievers that carry no retrieval-trained model: the lexical channel and its variants.
LEXICAL = {"bm25", "bm25-engine", "bm25-title", "bm25-passages", "bm25-prf", "lexical"}

#: Retrievers that mean a stock bi-encoder was actually run over the corpus.
STOCK_DENSE = {"biencoder", "hybrid-biencoder", "dense", "hybrid"}

#: Retrievers that mean a multi-model dense ensemble was run.
ENSEMBLE = {"dense-ensemble", "hybrid-ensemble"}

#: Retrievers that mean a learned-sparse channel was run.
SPARSE = {"splade", "hybrid-splade"}

#: Report file stems whose `biencoder` rows are Laya's *adapted* encoder, not a stock one.
#: The distinction matters: the adapted encoder is this project's own and regresses against
#: `bge-small`, so a member with only an adapted dense row has still never seen a
#: retrieval-trained stock encoder.
ADAPTED_STEMS = {"robust04-adapted", "trec-news-adapted", "trec-covid-adapted"}

#: The published table keys members by the source's display name; this is the way back.
REVERSE: dict[str, str] = {repo: source for source, repo in published.CANONICAL.items()}

#: The row labels §1.2 uses on the left. They differ from `published.CANONICAL` only where the
#: paper's spelling carries a transcription artefact (`Tóuche-2020 (v2)`) or loses the suffix
#: (`DBPedia`); the document keeps the corpus's own name, and this is the one place that says so.
#: The paper's names come first so the two overrides below win.
DISPLAY: dict[str, str] = {
    **{repo: source for source, repo in published.CANONICAL.items()},
    "webis-touche2020": "webis-touche2020",
    "dbpedia-entity": "DBPedia-entity",
}

#: The engine's best measured row per BEIR member, nDCG@10.
#: Source: `README.md` §"What cannot be claimed", the BEIR-18 paragraph, which states these are
#: best-per-dataset and not one configuration. Six cells moved in the title-weighted and
#: adapted-dense pass; those are the values below.
OUR_BEST: dict[str, tuple[float, str]] = {
    "scifact": (0.767, "ensemble + BM25"),
    "nfcorpus": (0.387, "ensemble + BM25"),
    "arguana": (0.651, "ensemble + BM25"),
    "scidocs": (0.225, "ensemble + BM25"),
    "fiqa": (0.444, "ensemble + BM25"),
    "quora": (0.8876, "dense only"),
    "webis-touche2020": (0.4419, "lexical (dense loses)"),
    "bioasq": (0.5193, "lexical only"),
    "trec-covid": (0.6871, "adapted dense + lexical"),
    "robust04": (0.4405, "adapted dense + lexical"),
    "trec-news": (0.3745, "adapted dense + lexical"),
    "cqadupstack": (0.2943, "lexical only"),
    "signal1m": (0.3131, "lexical only"),
    "hotpotqa": (0.6352, "lexical only"),
    "climate-fever": (0.1657, "lexical only"),
    "dbpedia-entity": (0.3166, "lexical only"),
    "fever": (0.6323, "lexical only"),
    "nq": (0.3264, "lexical only"),
}

#: Every member §1.2 measures against a published *model* column, i.e. the whole membership.
#: The macro divides the summed deficit by this, not by the number of losing cells.
DENOMINATOR = len(BEIR_18)


def rows_by_dataset(directory: Path) -> dict[str, dict[str, set[str]]]:
    """Every retriever row measured per dataset, keyed dataset -> stem -> retriever names.

    A dataset name appears in every report that carries it (`nq.json` and `nq-title.json` both
    contribute to `nq`), and the *stem* is kept because it is what distinguishes a stock
    bi-encoder run from the adapted one — the row name is `biencoder` in both cases.
    """
    found: dict[str, dict[str, set[str]]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        datasets = report.get("datasets")
        if not isinstance(datasets, Mapping):
            continue
        for dataset, payload in datasets.items():
            if not isinstance(payload, Mapping):
                continue
            # The report writer emits `retrievers`; older reports used `results`. Accept both,
            # because a report that silently contributes no rows would understate coverage.
            rows = payload.get("retrievers") or payload.get("results")
            if not isinstance(rows, Mapping):
                continue
            found.setdefault(str(dataset), {})[path.stem] = {str(r) for r in rows}
    return found


def documents_by_dataset(directory: Path) -> dict[str, int]:
    """The corpus size each report records, for the table's documents column."""
    sizes: dict[str, int] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        datasets = report.get("datasets")
        if not isinstance(datasets, Mapping):
            continue
        for dataset, payload in datasets.items():
            if isinstance(payload, Mapping) and isinstance(payload.get("documents"), int):
                sizes[str(dataset)] = max(sizes.get(str(dataset), 0), int(payload["documents"]))
    return sizes


def classify(stems: Mapping[str, set[str]]) -> str:
    """The richest channel this member has actually been measured with."""
    every = set().union(*stems.values()) if stems else set()
    if every & ENSEMBLE:
        return "full"
    stock = bool(every & STOCK_DENSE) and not all(
        stem in ADAPTED_STEMS for stem, rows in stems.items() if rows & STOCK_DENSE
    )
    if every & SPARSE:
        return "sparse"
    if stock:
        return "dense"
    if every & STOCK_DENSE:
        return "adapted dense"
    if every - LEXICAL:
        return "other"
    return "lexical only"


def channel_map(directory: Path | None = None) -> dict[str, Any]:
    """§1.1, computed: one row per member with its documents, rows and richest channel."""
    beir = BEIR_DIR if directory is None else directory
    found = rows_by_dataset(beir)
    sizes = documents_by_dataset(beir)
    rows: list[dict[str, Any]] = []
    lexical_only: list[str] = []
    for dataset in sorted(found):
        stems = found[dataset]
        every = sorted(set().union(*stems.values()))
        kind = classify(stems)
        if kind == "lexical only" and dataset in BEIR_18:
            lexical_only.append(dataset)
        rows.append(
            {
                "dataset": dataset,
                "member": dataset in BEIR_18,
                "documents": sizes.get(dataset, 0),
                "retrievers": every,
                "kind": kind,
            }
        )
    return {
        "rows": rows,
        "lexical_only": lexical_only,
        "missing": sorted(BEIR_18 - set(found)),
        "extras": sorted(set(found) - BEIR_18),
        "total": len(BEIR_18),
    }


def deficit() -> dict[str, Any]:
    """§1.2, computed: the per-cell gap against the best published *model* column, and its sum.

    `BM25` is excluded by convention: the paper's BM25 is a multi-field index and this
    repository's control is flat, so the two are not the same system (`published_beir18`'s
    docstring makes the same point). Both inputs are static — a transcribed "ours" column and the
    parsed paper table — so this does not read `eval/` and cannot skip.
    """
    cells: list[dict[str, Any]] = []
    for dataset in OUR_BEST:
        ours, channel = OUR_BEST[dataset]
        source = REVERSE.get(dataset)
        models = {
            k: v for k, v in published.NDCG10.get(source or "", {}).items() if k != "BM25"
        }
        if not models:
            raise KeyError(
                f"{dataset!r} has no published model column under {source!r}; the deficit "
                f"cannot be computed against a lexical column"
            )
        best_key = max(models, key=lambda k: models[k])
        best = float(models[best_key])
        # The gap is taken between the two *stated* values, in decimal, so that 0.4405 − 0.473 is
        # −0.0325 and renders as −0.033 rather than the float's −0.03249999999999997 → −0.032.
        gap = float(
            Decimal(str(ours)).quantize(Decimal("0.0001")) - Decimal(str(best))
        )
        cells.append(
            {
                "dataset": dataset,
                "display": DISPLAY.get(dataset, dataset),
                "ours": float(ours),
                "best": best,
                "best_system": best_key,
                "gap": gap,
                "channel": channel,
            }
        )
    cells.sort(key=lambda c: -c["ours"])
    # Summed in decimal from the stated per-cell gaps, so the printed total is the arithmetic the
    # document shows rather than a float accumulation of it.
    total = sum(Decimal(str(-c["gap"])) for c in cells if c["gap"] < 0)
    macro = total / DENOMINATOR
    return {
        "cells": cells,
        "counted": sum(1 for c in cells if c["gap"] < 0),
        "total": float(total),
        "macro": float(macro),
        "half": float(macro / 2),
        "nine_tenths": float(macro * Decimal("0.9")),
        "denominator": DENOMINATOR,
        "measured": len(cells),
    }


def half_up(value: float, decimals: int) -> float:
    """Round half-up, the convention `consistency_check` states and every doc table uses.

    Python's `.3f` is half-even, so 0.0505 renders 0.050 there and 0.051 in the document.
    """
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def gap_text(gap: float) -> str:
    """§1.2's gap cell for a computed gap: three decimals, half-up, with the document's minus.

    The document adds its own bold to the cells it wants to emphasise; this is the number under
    the asterisks, and `consistency_check` searches for it inside the row rather than requiring
    the whole row, so emphasis is the document's business and the arithmetic is not.
    """
    sign = "\u2212" if gap < 0 else "+"          # U+2212, the minus the document uses
    return f"{sign}{half_up(abs(gap), 3):.3f}"


def coverage_table(coverage: Mapping[str, Any]) -> list[str]:
    """The §1.1 table and its summary lines, as they must appear in the document."""
    lines = [
        "| dataset | documents | rows measured | richest channel measured |",
        "| --- | ---: | --- | --- |",
    ]
    for row in coverage["rows"]:
        every = row["retrievers"]
        label = ", ".join(every)
        if len(every) > 6:
            label = f"{len(every)} rows"
        marker = "**lexical only**" if row["kind"] == "lexical only" else row["kind"]
        if not row["member"]:
            marker += " *(not a member)*"
        lines.append(f"| {row['dataset']} | {row['documents']:,} | {label} | {marker} |")
    lines.append("")
    lines.append(
        f"**lexical-only members ({len(coverage['lexical_only'])} of {coverage['total']}):** "
        f"{', '.join(coverage['lexical_only']) or 'none'}"
    )
    lines.append(f"**members with no report:** {', '.join(coverage['missing']) or 'none'}")
    lines.append(
        f"**non-members present (not coverage):** {', '.join(coverage['extras']) or 'none'}"
    )
    return lines


def deficit_table(deficit_map: Mapping[str, Any]) -> list[str]:
    """The §1.2 table as the tool prints it, for a human reading the arithmetic."""
    lines = [
        "| cell | ours | best published model column | gap | channel we have there |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for cell in deficit_map["cells"]:
        lines.append(
            f"| {cell['display']} | {cell['ours']:.4f} | {cell['best']:.3f} | "
            f"{gap_text(cell['gap'])} | {cell['channel']} |"
        )
    return lines


def arithmetic_lines(deficit_map: Mapping[str, Any]) -> list[str]:
    """The §1.2 closing arithmetic, as it must appear in the document."""
    return [
        f"summed deficit: **{deficit_map['total']:.4f} nDCG@10** over "
        f"{deficit_map['counted']} of {deficit_map['denominator']} members",
        f"macro cost: **{deficit_map['macro']:.4f}**",
        f"half closed: **+{deficit_map['half']:.3f}** · nine tenths closed: "
        f"**+{deficit_map['nine_tenths']:.3f}**",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    coverage = channel_map()
    if not coverage["rows"]:
        print(f"no BEIR reports under {BEIR_DIR}; nothing to check", file=sys.stderr)
        return 2
    print("## §1.1 — channel coverage per BEIR member\n")
    print("\n".join(coverage_table(coverage)))

    print("\n## §1.2 — the deficit arithmetic\n")
    deficit_map = deficit()
    print("\n".join(deficit_table(deficit_map)))
    print()
    print("\n".join(arithmetic_lines(deficit_map)))
    return 0 if not coverage["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
