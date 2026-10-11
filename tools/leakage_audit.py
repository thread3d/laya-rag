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

#: A file above this size is not a report. `eval/` is 166 GB and holds 27 GB of JSON: a 24 GB
#: MeSH dump, 228 MB corpus splits, 166 MB posting caches, 33 MB tokenizer files. Every report a
#: tool writes here is megabytes at most, even with per-query arrays for thousands of queries, so
#: the cap costs no coverage — and the exclusions are *named* in the output rather than dropped,
#: because an audit's coverage is itself a claim.
DEFAULT_MAX_BYTES = 32 * 1024 * 1024

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


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def audit_tree(root: Path = ROOT, max_bytes: int = DEFAULT_MAX_BYTES) -> dict[str, Any]:
    """Every JSON file under `eval/`, with the ones that are not reports **named** rather than lost.

    `eval/` also holds downloaded source data and caches — `bioasq-src/allMeSH_2020.json` is a
    latin-1 MeSH dump, and `beir/cache/` holds posting stores — so an audit that skipped them
    silently would overstate its own coverage. They are counted and listed, so "N reports scanned"
    is a statement about N *readable reports* and a reader can see what was left out. Only a
    decodable JSON object is audited: every report this repository writes goes through
    `json.dumps` to UTF-8, so a file that is neither cannot carry the dilution signature.
    """
    reports = 0
    skipped: list[dict[str, str]] = []
    findings: list[dict[str, Any]] = []
    for path in iter_reports(root):
        try:
            size = path.stat().st_size
        except OSError as exc:
            skipped.append({"path": _relative(path, root), "reason": type(exc).__name__})
            continue
        if size > max_bytes:
            skipped.append({
                "path": _relative(path, root),
                "reason": f"{size / 1e6:.1f} MB > {max_bytes / 1e6:.0f} MB cap",
            })
            continue
        if size == 0:
            # The HuggingFace cache's `.no_exist/` sentinels are empty files: a name reserved for
            # a file that was *not* downloaded. They cannot be a report, so they are summarised
            # rather than named one by one.
            skipped.append({"path": _relative(path, root), "reason": "empty file"})
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            skipped.append({"path": _relative(path, root), "reason": type(exc).__name__})
            continue
        try:
            report = json.loads(text)
        except json.JSONDecodeError as exc:
            skipped.append({"path": _relative(path, root), "reason": f"JSONDecodeError: {exc.msg}"})
            continue
        if not isinstance(report, Mapping):
            skipped.append(
                {"path": _relative(path, root), "reason": f"top level is {type(report).__name__}"}
            )
            continue
        reports += 1
        findings.extend(audit_report(path, report))
    return {
        "reports": reports,
        "max_bytes": max_bytes,
        "skipped": skipped,
        "findings": findings,
        "ok": not findings,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    skipped = result.get("skipped") or []
    cap = result.get("max_bytes") or DEFAULT_MAX_BYTES
    lines = [
        "# Leakage audit",
        "",
        f"- {result['reports']} JSON reports scanned under `eval/`",
        f"- {len(skipped)} JSON file(s) under `eval/` not scanned: not a UTF-8 JSON object under "
        f"{cap / 1e6:.0f} MB (source data, caches, tokenizer files — not reports)",
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
    if skipped:
        # An exclusion that could *hide* a report — an undecodable or malformed file — is named
        # individually. Bulk categories that cannot (a list-shaped id cache, an oversized corpus)
        # are summarised by reason with examples, so the table stays readable without the coverage
        # claim going unstated. The complete list is always in `--json`.
        suspicious = [
            entry for entry in skipped
            if entry["reason"].startswith(("JSONDecodeError", "UnicodeDecodeError", "OSError"))
        ]
        grouped: dict[str, list[str]] = {}
        for entry in skipped:
            if entry in suspicious:
                continue
            grouped.setdefault(entry["reason"], []).append(entry["path"])
        lines += [
            "",
            "**Not scanned** — an audit's coverage is a claim, so the exclusions are stated:",
            "",
            "| reason | files | examples |",
            "| --- | ---: | --- |",
        ]
        for reason, paths in sorted(grouped.items(), key=lambda item: -len(item[1])):
            examples = ", ".join(f"`{p}`" for p in paths[:3])
            if len(paths) > 3:
                examples += f", … (+{len(paths) - 3})"
            lines.append(f"| {reason} | {len(paths)} | {examples} |")
        if suspicious:
            lines += ["", "Files that could not be read at all, named individually:", ""]
            lines += [f"- `{entry['path']}` — {entry['reason']}" for entry in suspicious]
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
