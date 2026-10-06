"""A leakage audit over the repository's own measurement reports.

The evaluation brief (§9) asks for a contamination/leakage audit, and this repository has already
been bitten by the exact trap it needs: `--limit-queries` passed the **full qrels** alongside a
capped ranking set, so `evaluate_rankings` macro-averaged over queries that were never scored and
divided by the wrong denominator. It was found in four tools. What made it fixable was that a report
persists per-query arrays: if a limited run scored the full qrels, its per-query array is **longer
than the limit it recorded**.

This tool formalizes the check that was done by hand — "no recorded report used a limited run, which
was checked rather than assumed" — and adds a second, weaker check that tuning reports still carry
their held-out reading:

1. **Dilution**: a report whose settings record a query limit must not have any `per_query*` array
   with more entries than that limit.
2. **Held-out reading present**: a report that records a `holdout`/`held_out`/`transfer` block must
   carry the fields that make the held-out claim checkable (`selected_on_a` + `selected_b_score`, a
   `gate_*` metric, or `splits`).

    make leakage-audit
    .venv/bin/python tools/leakage_audit.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]

#: Settings keys that record a capped query set.
_LIMIT_KEYS = ("limit_queries", "limit")
#: Held-out blocks and the fields that make each checkable; any one field satisfies it.
_HELD_OUT = {
    "holdout": ("selected_on_a", "selected_b_score", "selected_a_score"),
    "held_out": ("gate_auc", "gate_brier", "gate_risk@80", "auc"),
    "transfer": ("splits", "test_balanced_accuracy_mean"),
}


def _settings(report: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("_settings", "settings"):
        value = report.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def per_query_lengths(node: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, int]]:
    """Every ``per_query*`` mapping/list under ``node`` and its length, up to a shallow depth."""
    out: list[tuple[str, int]] = []
    if depth > 3 or not isinstance(node, Mapping):
        return out
    for key, value in node.items():
        if str(key).startswith("per_query") and isinstance(value, (Mapping, list)):
            out.append((f"{prefix}/{key}", len(value)))
        elif isinstance(value, Mapping):
            out.extend(per_query_lengths(value, f"{prefix}/{key}", depth + 1))
    return out


def audit_report(path: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    settings = _settings(report)
    limit = next(
        (int(settings[key]) for key in _LIMIT_KEYS if isinstance(settings.get(key), int)), None
    )
    if limit:
        for where, count in per_query_lengths(report):
            if count > limit:
                findings.append({
                    "kind": "dilution",
                    "path": str(path),
                    "detail": f"{where} has {count} per-query entries but the run recorded "
                              f"limit_queries={limit} — the dilution signature",
                })
    for block, fields in _HELD_OUT.items():
        value = report.get(block)
        if isinstance(value, Mapping) and not any(field in value for field in fields):
            findings.append({
                "kind": "held_out_missing",
                "path": str(path),
                "detail": f"{block!r} is recorded but carries none of {fields}",
            })
    return findings


def iter_reports(root: Path) -> Iterator[Path]:
    for path in sorted(root.glob("eval/**/*.json")):
        if path.is_file():
            yield path


def audit_tree(root: Path = ROOT) -> dict[str, Any]:
    reports = 0
    findings: list[dict[str, Any]] = []
    for path in iter_reports(root):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, Mapping):
            continue
        reports += 1
        findings.extend(audit_report(path, report))
    return {
        "reports": reports,
        "findings": findings,
        "ok": not findings,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Leakage audit",
        "",
        f"- {result['reports']} JSON reports scanned under `eval/`",
        "- checks: a limited run must not hold more per-query entries than its limit (the",
        "  dilution signature); a held-out block must carry a field that makes it checkable",
        "",
    ]
    if result["ok"]:
        lines.append("**No dilution signature and no empty held-out block found.**")
    else:
        lines += ["| kind | report | detail |", "| --- | --- | --- |"]
        for finding in result["findings"]:
            lines.append(f"| {finding['kind']} | `{finding['path']}` | {finding['detail']} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="leakage_audit.py")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true", help="exit 1 if any finding is present")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    if not (root / "eval").is_dir():
        print("leakage-audit: SKIPPED — no eval/ directory")
        return 0
    result = audit_tree(root)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_markdown(result))
    if args.check and not result["ok"]:
        print(f"FAILED: {len(result['findings'])} finding(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
