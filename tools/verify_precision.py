"""Precision and recall for the deterministic verifier — ROADMAP Phase 5.3.

`verify.content_issues` raises four *advisory* findings — a number, a date or a unit that disagrees
with the cited evidence, and a possible negation flip — and the roadmap's shipping bar is precision
≥ ~0.9. Advisory checks are the ones that can quietly become noise: a detector that flags everything
is worse than none, because a caller learns to ignore it.

This measures each detector on a **constructed** labelled set: for every kind, claims that *should*
trip that specific finding, and matched claims that are faithful to the same evidence and should
trip nothing. The evidence and claims are generated from templates with varied slots, so the set is
deterministic and reproducible rather than sampled.

    .venv/bin/python tools/verify_precision.py
    .venv/bin/python tools/verify_precision.py --check      # exit 1 below the bar

**The set is constructed, not hand-labelled.** The roadmap asks for precision on hand-labelled
composed answers; this is a proxy that is reproducible and adversarially clear about each detector's
blind spots, and the difference is stated wherever the number is quoted.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402

from laya_rag.verify import content_issues  # noqa: E402

KINDS = ("number_not_in_evidence", "date_not_in_evidence", "unit_mismatch",
         "possible_negation_flip")
SIZE = 50


@dataclass(frozen=True)
class Case:
    """One labelled pair: the kind expected to fire, whether it *should* fire, and the texts."""

    kind: str
    positive: bool
    claim: str
    evidence: str


def build_cases(size: int = SIZE) -> list[Case]:
    """A deterministic labelled set, `size` positives and `size` negatives per kind."""
    cases: list[Case] = []
    for index in range(size):
        # Number: the claim's figure is absent from the evidence; the control repeats it.
        produced = 100 + index * 7
        evidence = f"The plant produced {produced} widgets during the quarter."
        cases.append(Case("number_not_in_evidence", True,
                          f"The plant produced {produced + 11} widgets during the quarter.", evidence))
        cases.append(Case("number_not_in_evidence", False,
                          f"The plant produced {produced} widgets during the quarter.", evidence))

        # Date: same sentence, a different signing date.
        month = 1 + (index % 12)
        day = 1 + (index % 28)
        evidence = f"The contract was signed on 2024-{month:02d}-{day:02d}."
        other = 1 + ((day + 6) % 28)
        cases.append(Case("date_not_in_evidence", True,
                          f"The contract was signed on 2024-{month:02d}-{other:02d}.", evidence))
        cases.append(Case("date_not_in_evidence", False, evidence, evidence))

        # Unit: same quantity and family, a different unit. A size pair and a time pair, so the
        # detector is exercised on two families rather than one.
        if index % 2 == 0:
            evidence = "The backup file is 40 MB compressed."
            cases.append(Case("unit_mismatch", True, "The backup file is 40 GB compressed.", evidence))
            cases.append(Case("unit_mismatch", False, "The backup file is 40 MB compressed.", evidence))
        else:
            evidence = "Refunds take 5 days to appear on the statement."
            cases.append(Case("unit_mismatch", True,
                              "Refunds take 5 hours to appear on the statement.", evidence))
            cases.append(Case("unit_mismatch", False,
                              "Refunds take 5 days to appear on the statement.", evidence))

        # Negation: the claim denies what the evidence asserts, with shared vocabulary.
        evidence = "The backup completed successfully and the service restarted cleanly."
        cases.append(Case("possible_negation_flip", True,
                          "The backup did not complete successfully and the service restarted cleanly.",
                          evidence))
        cases.append(Case("possible_negation_flip", False, evidence, evidence))
    return cases


def flagged_kinds(case: Case) -> set[str]:
    return {issue["kind"] for issue in content_issues(case.claim, [case.evidence])}


#: Near-misses a naive detector flags and a reader would not. Every one is a *legitimate
#: exception* of the kind `verify.py` documents — rounding, a range endpoint, a spelled number, the
#: same date in another format — so each should be **quiet**, and how many are not is the honest
#: precision limit rather than a failure to hide.
HARD_CASES: tuple[tuple[str, str, str], ...] = (
    ("same date, another format", "Signed on 2024-03-12.", "The contract was signed on March 12, 2024."),
    ("rounding", "Revenue was 1205 units.", "Revenue was about 1200 units."),
    ("range endpoint", "Delivery takes 10 days.", "Delivery takes between 5 and 10 days."),
    ("number spelled out", "The plant produced 5 widgets.", "The plant produced five widgets."),
    ("percent as decimal", "Growth was 5.0%.", "Growth was 5%."),
    ("negation in both texts", "The service did not restart.", "The service did not restart."),
)


def hard_case_results() -> list[dict[str, Any]]:
    results = []
    for label, claim, evidence in HARD_CASES:
        kinds = sorted({issue["kind"] for issue in content_issues(claim, [evidence])})
        results.append({"label": label, "kinds": kinds, "quiet": not kinds})
    return results


def summarize(cases: Sequence[Case]) -> dict[str, Any]:
    """Per-kind precision and recall, plus the false-alarm rate on the faithful controls."""
    per_kind: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        positives = [case for case in cases if case.kind == kind and case.positive]
        negatives = [case for case in cases if case.kind == kind and not case.positive]
        hits = [case for case in positives if kind in flagged_kinds(case)]
        false_positives = [case for case in negatives if kind in flagged_kinds(case)]
        flagged_total = len(hits) + len(false_positives)
        per_kind[kind] = {
            "positives": len(positives),
            "negatives": len(negatives),
            "recall": round(len(hits) / len(positives), 4) if positives else 0.0,
            "precision": round(len(hits) / flagged_total, 4) if flagged_total else 0.0,
            "false_positives": len(false_positives),
        }
    controls = [case for case in cases if not case.positive]
    noisy = [case for case in controls if flagged_kinds(case)]
    hard = hard_case_results()
    return {
        "cases": len(cases),
        "controls": len(controls),
        "control_false_alarm_rate": round(len(noisy) / len(controls), 4) if controls else 0.0,
        "control_false_alarms": len(noisy),
        "per_kind": per_kind,
        "hard_cases": hard,
        "hard_case_false_alarm_rate": round(
            sum(1 for row in hard if not row["quiet"]) / len(hard), 4) if hard else 0.0,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Deterministic verifier — precision and recall, constructed set",
        "",
        f"- {report['cases']} labelled cases, {report['controls']} of them faithful controls",
        "",
        "| kind | positives | recall | precision | false positives |",
        "| --- | --- | --- | --- | --- |",
    ]
    for kind, row in report["per_kind"].items():
        lines.append(f"| `{kind}` | {row['positives']} | {row['recall']:.4f} | "
                     f"{row['precision']:.4f} | {row['false_positives']} |")
    lines += [
        "",
        f"- **false-alarm rate on faithful controls: {report['control_false_alarm_rate']:.4f}** "
        f"({report['control_false_alarms']} of {report['controls']})",
        "",
        "## Known-hard near-misses",
        "",
        "Cases a reader would call faithful but a naive detector flags. Every one is a documented",
        "legitimate exception, so each should be **quiet**; the ones that are not are the detector's",
        "honest precision limit.",
        "",
        "| case | flagged kinds | quiet |",
        "| --- | --- | --- |",
    ]
    for row in report["hard_cases"]:
        lines.append(f"| {row['label']} | {', '.join(row['kinds']) or '—'} | "
                     f"{'yes' if row['quiet'] else '**no**'} |")
    lines += [
        "",
        f"- **false-alarm rate on hard near-misses: {report['hard_case_false_alarm_rate']:.4f}**",
        "",
        "The set is **constructed**, not hand-labelled: each detector is exercised on claims that",
        "should trip it, on matched faithful claims that should trip nothing, and on the near-misses",
        "above. It is a proxy for the hand-labelled evaluation the roadmap asks for, and is",
        "reproducible rather than sampled.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="verify_precision.py")
    ap.add_argument("--size", type=int, default=SIZE, help="positives per kind")
    ap.add_argument("--min-precision", type=float, default=0.9)
    ap.add_argument("--max-false-alarm", type=float, default=0.05)
    ap.add_argument("--check", action="store_true", help="exit 1 below the bar")
    ap.add_argument("--out", default=str(ROOT / "eval" / "verify-precision"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = summarize(build_cases(args.size))
    report["_settings"] = vars(args)
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), report)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(report))
    print(render_markdown(report))
    print("wrote", str(out) + ".json")
    failures: list[str] = []
    for kind, row in report["per_kind"].items():
        if row["precision"] < args.min_precision:
            failures.append(f"{kind} precision {row['precision']:.4f} < {args.min_precision}")
    if report["control_false_alarm_rate"] > args.max_false_alarm:
        failures.append(f"control false-alarm rate {report['control_false_alarm_rate']:.4f} "
                        f"> {args.max_false_alarm}")
    if failures and args.check:
        print("verify-precision: FAILED", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
