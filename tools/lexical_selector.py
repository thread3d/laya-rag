#!/usr/bin/env python
"""The per-corpus lexical profile: which BM25 configuration does each BEIR member actually want?

`SCORE-LIFT.md` §1.3 and §P2 make a claim the repository had never acted on: the shipped lexical
settings are a **global** choice, while the evidence says the instruments reverse per corpus. On
Signal-1M the shipped `bm25-engine` costs **five points** against the flat `bm25` control
(0.3131 against 0.3640) — the shipped channel losing to its own control on a member the channel is
the *only* channel for.

§P2 asks for the settings to be a per-corpus choice, and pre-registers its own falsifier: *the
selector picks the same settings on every corpus*. This tool answers that from the rows already
measured, so the answer costs no new runs — every number here is a row in `eval/beir/*.json`:

* the measured nDCG@10 per member and per lexical configuration;
* which configuration wins per member, and therefore whether a selector has anything to choose;
* the macro of the shipped configuration against the macro of the per-corpus best, which is the
  **oracle** for P2(a) — the ceiling a selector can reach, not what one will reach.

A configuration's value is the **modal** value across the current reports, with the other values
reported as `variants`: a few reports measure a deliberately different corpus (`arguana-notitles`
strips titles), and a value that disagrees with no such explanation is drift worth seeing.

    .venv/bin/python tools/lexical_selector.py
    make lexical-selector
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from consistency_check import BEIR_18, BOUNDED, SUPERSEDED  # noqa: E402  (the membership set)

BEIR_DIR = ROOT / "eval" / "beir"

#: The lexical configurations the harness can express today, in the order ties are broken.
#: `bm25-engine` is what ships; `bm25` is the published Anserini control (`k1=0.9, b=0.4`,
#: query stopwords kept, no stemming); `bm25-title` repeats the title twice; `bm25-passages`
#: splits long documents at 600 characters.
CONFIGS: tuple[str, ...] = ("bm25", "bm25-engine", "bm25-title", "bm25-passages")

#: What ships when no per-corpus choice is made.
SHIPPED = "bm25-engine"

#: Report stems that measure a deliberately different corpus and so must not contribute a
#: configuration's value. `arguana-notitles*` strips titles (ArguAna's "titles" are shared topic
#: keyword lists); `*-subset` is a seeded query subset whose estimate is not the full row.
def _excluded(stem: str) -> bool:
    return (
        stem in SUPERSEDED
        or stem in BOUNDED
        or "notitles" in stem
        or stem.endswith("-subset")
        or stem in {"coverage", "lexical-control", "smoke"}
    )


def collect(directory: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """member -> config -> {"value", "reports", "variants"} over the reports still current.

    Superseded reports, bounded runs and deliberately different corpora are excluded by name, the
    same set `consistency_check` quarantines. The value is the modal one, because a variant that
    appears once is an experiment rather than the configuration's measurement.
    """
    values: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for path in sorted(directory.glob("*.json")):
        if _excluded(path.stem):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        datasets = report.get("datasets")
        if not isinstance(datasets, Mapping):
            continue
        for member, payload in datasets.items():
            if member not in BEIR_18 or not isinstance(payload, Mapping):
                continue
            rows = payload.get("retrievers") or payload.get("results")
            if not isinstance(rows, Mapping):
                continue
            for name, row in rows.items():
                if not isinstance(row, Mapping) or "nDCG@10" not in row:
                    continue
                # Passage rows are excluded: `bm25-passages@600` carries its budget (and, since
                # the aggregation work, its aggregation mode) in the row name, and different
                # budgets are *different retrievers* rather than one configuration measured
                # twice. Collapsing them made two legitimately different rows look like drift —
                # `passages-long.json` and `passages-sum.json` read 0.3522 and 0.2428 for
                # Robust04's 600-character row. Passage size is its own axis
                # (`docs/findings-passages-long.md`), so it is not this tool's question.
                if str(name).startswith("bm25-passages"):
                    continue
                config = str(name)
                if config not in CONFIGS:
                    continue
                values.setdefault(member, {}).setdefault(config, []).append(
                    (path.stem, float(row["nDCG@10"]))
                )

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for member, configs in values.items():
        for config, pairs in configs.items():
            counts = Counter(round(value, 4) for _stem, value in pairs)
            value, _n = counts.most_common(1)[0]
            variants = sorted(
                {round(v, 4) for _s, v in pairs if round(v, 4) != value}
            )
            out.setdefault(member, {})[config] = {
                "value": value,
                "reports": len(pairs),
                "variants": variants,
            }
    return out


def profile(collected: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """The per-member winner, and the two macros that bracket it."""
    members: list[dict[str, Any]] = []
    for member in sorted(collected):
        measured = collected[member]
        available = {c: float(row["value"]) for c, row in measured.items() if c in CONFIGS}
        if not available:
            continue
        # Ties break on `CONFIGS`' order, so the shipped configuration wins a tie it does not lose.
        best = max(CONFIGS, key=lambda c: (available.get(c, float("-inf")), -CONFIGS.index(c)))
        members.append(
            {
                "member": member,
                "values": available,
                "shipped": available.get(SHIPPED),
                "best": best,
                "best_value": available[best],
                "delta": available[best] - available.get(SHIPPED, available[best]),
                "variants": {c: row["variants"] for c, row in measured.items() if row["variants"]},
            }
        )
    scored = [m for m in members if m["shipped"] is not None]
    shipped_macro = sum(m["shipped"] for m in scored) / len(scored) if scored else 0.0
    best_macro = sum(m["best_value"] for m in scored) / len(scored) if scored else 0.0
    picks = Counter(m["best"] for m in members)
    return {
        "members": members,
        "shipped_macro": shipped_macro,
        "best_macro": best_macro,
        "gain": best_macro - shipped_macro,
        "picks": dict(picks),
        "scored": len(scored),
    }


def render(result: Mapping[str, Any]) -> str:
    members = result["members"]
    columns = ["member"] + list(CONFIGS) + ["picks", "delta"]
    lines = ["| " + " | ".join(columns) + " |", "| --- " + "| ---: " * (len(columns) - 2) + "| --- |"]
    for row in members:
        cells = [row["member"]]
        for config in CONFIGS:
            value = row["values"].get(config)
            text = "—" if value is None else f"{value:.4f}"
            if config == row["best"] and len(result["picks"]) > 1:
                text = f"**{text}**"
            cells.append(text)
        cells.append(row["best"])
        cells.append("—" if row["shipped"] is None else f"{row['delta']:+.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    header = [
        f"{result['scored']} members carry at least one lexical row.",
        "",
        f"shipped `{SHIPPED}` macro: **{result['shipped_macro']:.4f}**",
        f"per-corpus best macro: **{result['best_macro']:.4f}** "
        f"(**{result['gain']:+.4f}**, the oracle for a selector)",
        "picks: " + ", ".join(f"{name} ×{count}" for name, count in sorted(result["picks"].items())),
        "",
    ]
    variants = [
        (row["member"], config, values)
        for row in members
        for config, values in row["variants"].items()
    ]
    tail = [""]
    if variants:
        tail.append("Configurations whose reports disagree, with the other values seen:")
        tail.append("")
        for member, config, values in variants:
            tail.append(f"- `{member}` `{config}`: also {values}")
    else:
        tail.append("No configuration disagrees with itself across the current reports.")
    return "\n".join(header + lines + tail) + "\n"


def main(argv: list[str] | None = None) -> int:
    collected = collect(BEIR_DIR)
    if not collected:
        print(f"no lexical rows under {BEIR_DIR}; nothing to profile", file=sys.stderr)
        return 2
    result = profile(collected)
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
