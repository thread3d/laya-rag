"""Resolve the headline numbers quoted in prose against the report that produced them.

The project's claim is that its numbers are checkable. Nothing enforced that until this:
the reorder ceiling was corrected in `tools/multihop_coverage.py` and in six documents, but
the *source* finding and the findings index kept the refuted figure and the refuted
conclusion, and three further hand-typed numbers (811 pools, the 0.451/0.695 ceilings, the
+0.086 headroom) had no committed derivation at all. A hand-copied number drifts; a pinned
one cannot.

The registry below is **explicit rather than inferred**. Each entry names a document, a
regex that locates the line(s) stating a number, the value the document states, and the
report path (or derived statistic) that is the source of truth. The check is:

* every line the anchor matches must contain the stated value (a prose edit that moves the
  number off its anchor fails loudly rather than silently passing);
* the stated value must equal the source value within tolerance (so the document and the
  report cannot disagree); and
* a `Forbidden` entry asserts a known-stale value is **absent**, which is how a corrected
  figure is kept corrected.

A missing report is a *skip*, not a failure: `eval/` is gitignored and regenerable, so a
fresh checkout has no reports to check against. Run `make multihop-diagnostics` first to
materialise the one this reads.

    make claim-check
    .venv/bin/python tools/claim_check.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
MULTIHOP = ROOT / "eval" / "benchmarks" / "multihop-rag"
DIAGNOSTICS = MULTIHOP / "diagnostics.json"


# ── derived statistics, computed from the diagnostics per-query rows ─────
#
# These are the figures the prose quotes but the report does not store as a scalar: a
# query is "incomplete" at depth D when its gold-fact recall at D is below 1.0. They are
# computed here so the documents do not have to hand-copy them.

def _incomplete_at(report: Mapping[str, Any], depth: int) -> list[Mapping[str, Any]]:
    rows = report["per_query"]
    key = f"pool_fact_recall@{depth}"
    return [r for r in rows if r.get(key) is not None and r[key] < 1.0]


def _complete_at(report: Mapping[str, Any], depth: int) -> list[Mapping[str, Any]]:
    rows = report["per_query"]
    key = f"pool_fact_recall@{depth}"
    return [r for r in rows if r.get(key) is not None and r[key] >= 1.0]


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def incomplete_100_count(report: Mapping[str, Any]) -> float:
    return float(len(_incomplete_at(report, 100)))


def incomplete_100_share(report: Mapping[str, Any]) -> float:
    return len(_incomplete_at(report, 100)) / len(report["per_query"])


def ceiling_100_incomplete(report: Mapping[str, Any]) -> float:
    return _mean(_incomplete_at(report, 100), "pool_ceiling_MAP@10@100")


def ceiling_100_complete(report: Mapping[str, Any]) -> float:
    return _mean(_complete_at(report, 100), "pool_ceiling_MAP@10@100")


def fused_map_incomplete(report: Mapping[str, Any]) -> float:
    return _mean(_incomplete_at(report, 100), "fused_MAP@10")


def fused_map_complete(report: Mapping[str, Any]) -> float:
    return _mean(_complete_at(report, 100), "fused_MAP@10")


def headroom_completing(report: Mapping[str, Any]) -> float:
    """How much higher the depth-100 ceiling would be if every pool were complete."""
    return ceiling_100_complete(report) - float(report["summary"]["depth"]["100"]["pool_ceiling_MAP@10"]["mean"])


DERIVED: dict[str, Callable[[Mapping[str, Any]], float]] = {
    "incomplete_100_count": incomplete_100_count,
    "incomplete_100_share": incomplete_100_share,
    "ceiling_100_incomplete": ceiling_100_incomplete,
    "ceiling_100_complete": ceiling_100_complete,
    "fused_map_incomplete": fused_map_incomplete,
    "fused_map_complete": fused_map_complete,
    "headroom_completing": headroom_completing,
}


# ── the registry ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Claim:
    """A number a document states, and the report value it must equal."""

    label: str
    doc: str
    anchor: str
    value: float
    source: tuple[str, ...] | str
    scale: float = 1.0          # doc_value = source * scale (100 for a percentage)
    decimals: int = 4
    tol: float = 5e-4


@dataclass(frozen=True)
class Forbidden:
    """A known-stale value that must not appear on the anchored line(s)."""

    label: str
    doc: str
    anchor: str
    value: float
    decimals: int = 4


_D = ("summary", "depth")
_PAIRED = ("summary", "paired")
DIAG = "docs/findings-multihop-diagnostics.md"
README = "README.md"

CHECKS: list[Claim] = [
    # The source report: the headline table and the paired comparisons.
    Claim("diag ceiling @10", DIAG, r"^\|\s*10\s*\|", 0.3970, _D + ("10", "pool_ceiling_MAP@10", "mean")),
    Claim("diag ceiling @100", DIAG, r"^\|\s*100\s*\|", 0.6088, _D + ("100", "pool_ceiling_MAP@10", "mean")),
    Claim("diag ceiling @500", DIAG, r"^\|\s*500\s*\|", 0.6592, _D + ("500", "pool_ceiling_MAP@10", "mean")),
    Claim("diag fact recall @100", DIAG, r"^\|\s*100\s*\|", 0.8666, _D + ("100", "pool_fact_recall", "mean")),
    Claim("diag 10->100 paired", DIAG, r"depth 10 → 100", 0.2118,
          _PAIRED + ("pool_ceiling_MAP@10:100-10", "delta")),
    Claim("diag 100->500 paired", DIAG, r"100 → 500", 0.0505,
          _PAIRED + ("pool_ceiling_MAP@10:500-100", "delta")),
    Claim("diag incomplete share", DIAG, r"depth-100 pools", 31.9,
          "incomplete_100_share", scale=100.0, decimals=1, tol=0.05),
    Claim("diag incomplete count", DIAG, r"depth-100 pools", 719,
          "incomplete_100_count", decimals=0, tol=0.5),
    Claim("diag ceiling, incomplete pools", DIAG, r"complete ones", 0.4469,
          "ceiling_100_incomplete"),
    Claim("diag ceiling, complete pools", DIAG, r"complete ones", 0.6845,
          "ceiling_100_complete"),
    Claim("diag fused, incomplete pools", DIAG, r"fused order reaches", 0.1498,
          "fused_map_incomplete"),
    Claim("diag fused, complete pools", DIAG, r"fused order reaches", 0.3052,
          "fused_map_complete"),
    Claim("diag headroom from completing", DIAG, r"Making every pool complete", 0.076,
          "headroom_completing", decimals=3, tol=1e-3),

    # README: the diagnostics table and the two derived claims.
    Claim("readme ceiling @10", README, r"^\|\s*10\s*\|", 0.397, _D + ("10", "pool_ceiling_MAP@10", "mean"),
          decimals=3, tol=1e-3),
    Claim("readme ceiling @100", README, r"^\|\s*100\s*\|", 0.609, _D + ("100", "pool_ceiling_MAP@10", "mean"),
          decimals=3, tol=1e-3),
    Claim("readme ceiling @500", README, r"^\|\s*500\s*\|", 0.659, _D + ("500", "pool_ceiling_MAP@10", "mean"),
          decimals=3, tol=1e-3),
    Claim("readme incomplete share", README, r"depth-100 pools", 31.9,
          "incomplete_100_share", scale=100.0, decimals=1, tol=0.05),
    Claim("readme incomplete count", README, r"depth-100 pools", 719,
          "incomplete_100_count", decimals=0, tol=0.5),
    Claim("readme headroom from completing", README, r"depth-100 ceiling from", 0.076,
          "headroom_completing", decimals=3, tol=1e-3),

    # RESEARCH.md: the standing table row.
    Claim("research ceiling", "RESEARCH.md", r"MultiHop pool-reorder oracle", 0.609,
          _D + ("100", "pool_ceiling_MAP@10", "mean"), decimals=3, tol=1e-3),
    Claim("research incomplete share", "RESEARCH.md", r"MultiHop pool-reorder oracle", 31.9,
          "incomplete_100_share", scale=100.0, decimals=1, tol=0.05),

    # docs/FINDINGS.md: the one-line index entry.
    Claim("findings ceiling", "docs/FINDINGS.md", r"Multi-hop selection", 0.6088,
          _D + ("100", "pool_ceiling_MAP@10", "mean")),
    Claim("findings incomplete share", "docs/FINDINGS.md", r"Multi-hop selection", 31.9,
          "incomplete_100_share", scale=100.0, decimals=1, tol=0.05),
]

FORBIDDEN: list[Forbidden] = [
    Forbidden("stale depth-10 ceiling", DIAG, r"^\|\s*10\s*\|", 0.7321),
    Forbidden("stale depth-100 ceiling", DIAG, r"^\|\s*100\s*\|", 0.7304),
    Forbidden("stale depth-500 ceiling", DIAG, r"^\|\s*500\s*\|", 0.6741),
    Forbidden("stale incomplete share", DIAG, r"depth-100 pools", 36.0, decimals=0),
    Forbidden("stale incomplete share", README, r"depth-100 pools", 36.0, decimals=0),
    Forbidden("stale incomplete count", README, r"depth-100 pools", 811.0, decimals=0),
    Forbidden("stale incomplete ceiling", README, r"depth-100 pools", 0.451),
    Forbidden("stale complete ceiling", README, r"depth-100 pools", 0.695),
    Forbidden("stale headroom", README, r"depth-100 ceiling from", 0.086, decimals=3),
    Forbidden("stale incomplete share", "RESEARCH.md", r"MultiHop pool-reorder oracle", 36.0, decimals=0),
]


# ── checking ─────────────────────────────────────────────────────────────

def resolve(source: tuple[str, ...] | str, report: Mapping[str, Any]) -> float:
    """The source-of-truth value: a JSON path into the report, or a derived statistic."""
    if isinstance(source, str):
        if source not in DERIVED:
            raise KeyError(f"unknown derived statistic {source!r}")
        return DERIVED[source](report)
    current: Any = report
    for key in source:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"report has no path {source!r} (failed at {key!r})")
        current = current[key]
    return float(current)


def _number_present(line: str, text: str) -> bool:
    """`text` as a standalone number: not embedded in a longer numeric token."""
    return re.search(r"(?<![\d.])" + re.escape(text) + r"(?![\d])", line) is not None


def _regions(lines: Sequence[str], anchor: str) -> list[str]:
    """The block each anchor match belongs to, as text safe to search for a number.

    A table row is its own region — the row must carry the number, and one row cannot
    vouch for another. Prose wraps, so a prose match expands to its blank-line-delimited
    paragraph, which keeps a re-wrap from failing a check that is semantically intact.
    """
    out: list[str] = []
    for index, line in enumerate(lines):
        if not re.search(anchor, line):
            continue
        if line.lstrip().startswith("|"):
            out.append(line)
            continue
        start = index
        while start > 0 and lines[start - 1].strip():
            start -= 1
        end = index
        while end + 1 < len(lines) and lines[end + 1].strip():
            end += 1
        out.append("\n".join(lines[start : end + 1]))
    return out


def run_checks(
    checks: Sequence[Claim],
    forbiddens: Sequence[Forbidden],
    *,
    root: Path,
    report: Mapping[str, Any],
) -> list[str]:
    """Return one failure message per violated claim or forbidden value."""
    failures: list[str] = []
    cache: dict[str, list[str]] = {}

    def lines_of(doc: str) -> list[str]:
        if doc not in cache:
            path = root / doc
            cache[doc] = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        return cache[doc]

    for claim in checks:
        regions = _regions(lines_of(claim.doc), claim.anchor)
        if not regions:
            failures.append(f"{claim.label}: anchor {claim.anchor!r} matched no line in {claim.doc}")
            continue
        try:
            expected = resolve(claim.source, report) * claim.scale
        except KeyError as exc:
            failures.append(f"{claim.label}: {exc}")
            continue
        text = f"{claim.value:.{claim.decimals}f}"
        missing = [r for r in regions if not _number_present(r, text)]
        if missing:
            first = missing[0].splitlines()[0].strip()
            failures.append(
                f"{claim.label}: {claim.doc} does not state {text} in an anchored block "
                f"(first miss: {first!r})"
            )
        if abs(claim.value - expected) > claim.tol:
            failures.append(
                f"{claim.label}: {claim.doc} states {text} but the report says "
                f"{expected:.{claim.decimals}f} (tolerance {claim.tol})"
            )

    for forbid in forbiddens:
        regions = _regions(lines_of(forbid.doc), forbid.anchor)
        text = f"{forbid.value:.{forbid.decimals}f}"
        present = [r for r in regions if _number_present(r, text)]
        if present:
            first = present[0].splitlines()[0].strip()
            failures.append(
                f"{forbid.label}: stale {text} still in {forbid.doc} (first: {first!r})"
            )

    return failures


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="claim_check.py")
    ap.add_argument("--root", default=str(ROOT), help="repository root the doc paths resolve under")
    ap.add_argument("--report", default=str(DIAGNOSTICS), help="the report the claims resolve against")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = root / report_path
    if not report_path.is_file():
        print(f"claim-check: SKIPPED — {report_path} not found (run `make multihop-diagnostics`)")
        return 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = run_checks(CHECKS, FORBIDDEN, root=root, report=report)
    checked = len(CHECKS) + len(FORBIDDEN)
    if failures:
        print(f"claim-check: {len(failures)} failure(s) across {checked} checks")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"claim-check: {checked} checks passed against {report_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
