"""The lexical-control summary, generated so the prose describing it can be pinned.

The claim "eight of nine agree within ±0.008" appeared in the README and was wrong — five of ten,
counted from the table. Generated tables cannot drift; the sentence summarising one can, and did.
So the summary becomes a report: this writes the gaps, the counts inside two tolerances and the
outliers, and `claim_check` pins the prose to those numbers instead of to my arithmetic.

    .venv/bin/python tools/lexical_control.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import compare_more, evalkit  # noqa: E402

TIGHT = 0.008
LOOSE = 0.02


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts and outliers for the lexical control, from the rows the chart already builds."""
    def inside(tolerance: float) -> int:
        return sum(1 for row in rows if abs(row["gap"]) <= tolerance)

    return {
        "corpora": len(rows),
        "within_tight": inside(TIGHT),
        "within_loose": inside(LOOSE),
        "tight_tolerance": TIGHT,
        "loose_tolerance": LOOSE,
        "in_our_favour": [row["dataset"] for row in rows if row["gap"] > LOOSE],
        "against_us": [row["dataset"] for row in rows if row["gap"] < -LOOSE],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="lexical_control.py")
    ap.add_argument("--out", default=str(ROOT / "eval" / "beir" / "lexical-control.json"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = compare_more.lexical_vs_published()
    summary = summarise(rows)
    report = {"summary": {k: v for k, v in summary.items() if k != "corpora"},
              "counts": {"corpora": summary["corpora"],
                         "within_0.008": summary["within_tight"],
                         "within_0.02": summary["within_loose"]},
              "rows": rows}
    out = Path(args.out)
    evalkit.write_json(out, report)
    print(f"{summary['corpora']} corpora: {summary['within_tight']} within ±{TIGHT}, "
          f"{summary['within_loose']} within ±{LOOSE}")
    print(f"  in our favour (+{LOOSE}): {', '.join(summary['in_our_favour']) or 'none'}")
    print(f"  against us  (-{LOOSE}): {', '.join(summary['against_us']) or 'none'}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
