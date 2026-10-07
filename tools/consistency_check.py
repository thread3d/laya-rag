"""Cross-document consistency: one quantity, one authoritative value, every document pinned.

`claim_check` pins prose against the *report* that produced a number. It does not pin documents
against *each other*, and that gap let the same quantity drift between two hand-written tables:

* the BEIR coverage count was stated as **5** of 18, **7** of 18 and **10** of 18;
* the published `voyage-02` + reranker row was round-truncated to **0.662 / 0.479** in one table and
  round-half-up to **0.663 / 0.480** in another — the paper's Table 5 says exactly 0.6625 / 0.4795;
* the recalibration half-split count is **300** in the report that computed it and **500** in the two
  documents that quote it;
* two `bench.md` / `bench-scale.md` grids report different peak RSS for the same 500–40,000-chunk
  sweep, because they measured against different process baselines and neither said so.

The registry below names each contested quantity once, resolves its authoritative value (from a
report on disk, or from a literal with its source), and records every document site that states it.
The gate fails when a site's anchor is missing, when a site states a value that is not the authority
rounded half-up to the site's precision, or when a superseded report is read as current. A missing
report is a *skip*, as in `claim_check`: `eval/` is gitignored and regenerable.

The audit set one level above the repository (`best-in-the-world/`) reads these same numbers and is
not committed to this repository, so it is held to two conventions instead of a registry:
`check_audit_documents` requires a **Currency** banner in every audit document and the computed
coverage count in every unfenced "of 18"; a dated finding is fenced with
`<!-- consistency: snapshot YYYY-MM-DD -->` … `<!-- consistency: /snapshot -->`. A checkout without
the audit directory skips both, exactly like a report that has not been generated.

    make consistency-check
    .venv/bin/python tools/consistency_check.py --fix     # write quarantine markers + coverage.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import claim_check  # noqa: E402  (reuse the region/number helpers rather than re-implementing them)

BEIR = ROOT / "eval" / "beir"
MH = ROOT / "eval" / "benchmarks" / "multihop-rag"
RECALIBRATION = MH / "recalibration.json"
COMPARE = ROOT / "tools" / "compare.py"

#: The audit set that reads this repository's numbers from outside it: the workspace sibling
#: `best-in-the-world/`. It is not committed to this repository, so its sites are optional and its
#: guards are the two conventions in `check_audit_documents` rather than a full registry entry.
AUDIT_DIR = ROOT.parent / "best-in-the-world"
AUDIT_PREFIX = "best-in-the-world/"

#: The reader-facing contract for a dated finding. A **Currency** banner says which sections are
#: pinned and which are snapshots; a fence pair marks a superseded number so the computed-count
#: guard below does not read history as a current claim.
CURRENCY_BANNER = re.compile(r"^> \*\*Currency:")
SNAPSHOT_OPEN = re.compile(r"^\s*<!--\s*consistency:\s*snapshot\s+(\d{4}-\d{2}-\d{2})\b.*-->\s*$")
SNAPSHOT_CLOSE = re.compile(r"^\s*<!--\s*consistency:\s*/snapshot\s*-->\s*$")


def _audit_dir(root: Path) -> Path:
    """The audit set beside a given repository root, so `--root` moves both together."""
    return root.parent / "best-in-the-world"


#: The denominator every "of 18" claim is measured against.
BEIR_TOTAL = 18

#: Reports kept on disk whose numbers are not comparable to the corrected ones: the ArguAna fix
#: (`8f7feb6`-era flat control) rewrote the BM25 row, and these predate it. Each names the report
#: that superseded it. They stay on disk for provenance but must carry a quarantine marker so that
#: neither a tool nor a reader can mistake them for a current result.
SUPERSEDED: dict[str, tuple[str, str]] = {
    "three-base": ("large-splade-fixed", "pre-fix ArguAna BM25 row (0.295 vs 0.3935)"),
    "three-large": ("large-splade-fixed", "pre-fix ArguAna BM25 row (0.295 vs 0.3935)"),
    "splade-more": ("large-splade-fixed", "pre-fix ArguAna BM25 row (0.295 vs 0.3935)"),
    "arguana-notitles": ("arguana-notitles-fixed", "pre-fix ArguAna BM25 row (0.2695 vs 0.36)"),
    "arguana-notitlesqueryinstruction": ("arguana-notitles-fixed", "pre-fix ArguAna BM25 row"),
    "dense-text-only": ("dense-text-only-fixed", "pre-fix ArguAna BM25 row (0.295 vs 0.3935)"),
    "five-small": ("five-small-fixed", "pre-fix ArguAna BM25 row (0.295 vs 0.3935)"),
    "large-splade": ("large-splade-fixed", "pre-fix ArguAna BM25 row (0.295 vs 0.3935)"),
    "report": ("report-final", "pre-fix fused order; superseded by report-after then report-final"),
    "report-after": ("report-final", "superseded by report-final, which adds the cross-encoder arm"),
}

#: Bounded runs: measured, but not on the full collection, so they do not count toward "of 18".
BOUNDED: dict[str, str] = {"trec-covid": "40,000-document subsample (full collection not run)"}
#: Datasets measured on a seeded query subset rather than the full qrels.
SUBSET: dict[str, str] = {"hotpotqa": "600-query seeded subset (seed 17)"}


# ── authoritative values ─────────────────────────────────────────────────

def half_up(value: float, decimals: int) -> float:
    """The stated form of a value: round-half-up, not Python's round-half-even.

    The drift this gate exists for is exactly this: 0.6625 at three decimals is 0.663 half-up and
    0.662 half-even/truncated, and two documents each picked a different convention.
    """
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def measured_beir(directory: Path | None = None) -> dict[str, Any]:
    """The datasets with a report in `eval/beir/`, computed rather than hand-typed.

    A report contributes every dataset key it carries. Superseded stems are excluded, and a
    bounded run is listed but does not count: the count is the claim, so it may not be a literal.
    """
    beir = BEIR if directory is None else directory
    contributors: dict[str, list[str]] = {}
    for path in sorted(beir.glob("*.json")):
        stem = path.stem
        if stem in SUPERSEDED or stem in {"coverage", "lexical-control"}:
            continue
        report = _read_json(path)
        if not report:
            continue
        datasets = report.get("datasets")
        if not isinstance(datasets, Mapping):
            continue
        for dataset in datasets:
            contributors.setdefault(dataset, []).append(stem)
    counted = sorted(d for d in contributors if d not in BOUNDED)
    return {
        "total": BEIR_TOTAL,
        "count": len(counted),
        "measured": counted,
        "subsets": {d: reason for d, reason in SUBSET.items() if d in counted},
        "bounded": {d: reason for d, reason in BOUNDED.items() if d in contributors},
        "contributors": {d: contributors[d] for d in sorted(contributors)},
    }


def voyage02() -> dict[str, float]:
    """The paper's Table 5 best reranked row, read from the one place it is pinned.

    `tools/compare.py`'s `PUBLISHED_MH` is the committed transcription; reading it keeps the docs
    and the chart from disagreeing about the same external number.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("laya_rag_compare_published", COMPARE)
    if spec is None or spec.loader is None:                     # pragma: no cover - import guard
        raise RuntimeError("cannot load tools/compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, hits10, hits4, map10, mrr in module.PUBLISHED_MH:
        if name == "voyage-02 + rerank":
            return {"Hits@10": hits10, "Hits@4": hits4, "MAP@10": map10, "MRR@10": mrr}
    raise RuntimeError("voyage-02 + rerank is not in compare.PUBLISHED_MH")


def recalibration_splits() -> float | None:
    report = _read_json(RECALIBRATION)
    try:
        return float(report["transfer"]["splits"])          # type: ignore[index]
    except (TypeError, KeyError):
        return None


def beir_report_metric(report: str, dataset: str, key: str) -> Callable[[], float | None]:
    """A BEIR report's engine row, so an audit's prose is pinned to the report and not to a copy."""

    def authority() -> float | None:
        data = _read_json(BEIR / f"{report}.json")
        if not data:
            return None
        try:
            return float(data["datasets"][dataset]["retrievers"]["bm25-engine"][key])
        except (KeyError, TypeError, ValueError):
            return None

    return authority


def _doc_path(doc: str, root: Path) -> Path:
    """Resolve a registered document. Audit documents live one level above the repository.

    A path under `best-in-the-world/` is tried against the repository first so a test root can hold
    its own copy, then against the workspace sibling. An absent document is the caller's decision:
    an `optional` site skips it, a required one fails on a missing anchor.
    """
    direct = root / doc
    if direct.is_file() or not doc.startswith(AUDIT_PREFIX):
        return direct
    return _audit_dir(root) / doc[len(AUDIT_PREFIX):]


# ── the registry ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Site:
    """One document's statement of a quantity, located by an anchor and stated to a precision."""

    doc: str
    anchor: str
    value: float
    decimals: int = 3
    #: A document outside the repository (the audit set): absent, the site is skipped, not failed.
    optional: bool = False


@dataclass(frozen=True)
class Quantity:
    """A contested quantity: one authority, every document that states it."""

    name: str
    sites: tuple[Site, ...]
    authority: Callable[[], Mapping[str, float] | float | None]
    provenance: str
    #: Which key of the authority mapping this site set states; None when the authority is a scalar.
    key: str | None = None


_VOYAGE: dict[str, float] | None = None


def _voyage_authority() -> Mapping[str, float] | None:
    global _VOYAGE
    if _VOYAGE is None:
        if not COMPARE.is_file():
            return None
        try:
            _VOYAGE = voyage02()
        except (RuntimeError, OSError):
            return None
    return _VOYAGE


def registry() -> list[Quantity]:
    return [
        Quantity(
            name="BEIR coverage count",
            sites=(
                Site("BENCHMARKS.md", r"\*\*BEIR\*\*.*of 18", 12, 0),
                Site("docs/PROGRAMME-STATUS.md", r"4\.2 at \d+ of 18", 12, 0),
                Site("ROADMAP.md", r"Coverage today: \d+ of 18", 12, 0),
                Site("README.md", r"BEIR's 18 datasets", 12, 0),
                Site("RESEARCH.md", r"^\| Scale \|", 12, 0),
                Site("docs/findings-consistency-gate.md", r"^\| BEIR coverage", 12, 0),
            ),
            authority=lambda: {"count": measured_beir()["count"]},
            key="count",
            provenance="eval/beir/*.json — the datasets with a report present",
        ),
        Quantity(
            name="HotpotQA full-set nDCG@10",
            sites=(
                Site(f"{AUDIT_PREFIX}measurement-audit.md", r"^\| HotpotQA ", 0.6352, 4, optional=True),
                Site(f"{AUDIT_PREFIX}proposal-ledger.md", r"^\| HotpotQA full", 0.6352, 4, optional=True),
            ),
            authority=beir_report_metric("hotpotqa", "hotpotqa", "nDCG@10"),
            provenance="eval/beir/hotpotqa.json datasets.hotpotqa.retrievers.bm25-engine.nDCG@10",
        ),
        Quantity(
            name="FEVER lexical nDCG@10",
            sites=(
                Site(f"{AUDIT_PREFIX}measurement-audit.md", r"^\| FEVER ", 0.6323, 4, optional=True),
                Site(f"{AUDIT_PREFIX}proposal-ledger.md", r"^\| Open-loop items", 0.6323, 4, optional=True),
            ),
            authority=beir_report_metric("fever", "fever", "nDCG@10"),
            provenance="eval/beir/fever.json datasets.fever.retrievers.bm25-engine.nDCG@10",
        ),
        Quantity(
            name="Climate-FEVER lexical nDCG@10",
            sites=(
                Site(f"{AUDIT_PREFIX}measurement-audit.md", r"^\| Climate-FEVER ", 0.161, 4, optional=True),
            ),
            authority=beir_report_metric("climate-fever", "climate-fever", "nDCG@10"),
            provenance=(
                "eval/beir/climate-fever.json datasets.climate-fever.retrievers.bm25-engine.nDCG@10"
            ),
        ),
        Quantity(
            name="voyage-02 + reranker Hits@4",
            sites=(
                Site("README.md", r"^\|\s*voyage-02 \+ rerank \(published\)", 0.663),
                Site("BENCHMARKS.md", r"^\|\s*published best \+ reranker \(`voyage-02`\)", 0.663),
            ),
            authority=_voyage_authority,
            key="Hits@4",
            provenance="MultiHop-RAG paper Table 5, pinned in tools/compare.py PUBLISHED_MH",
        ),
        Quantity(
            name="voyage-02 + reranker MAP@10",
            sites=(
                Site("README.md", r"^\|\s*voyage-02 \+ rerank \(published\)", 0.480),
                Site("BENCHMARKS.md", r"^\|\s*published best \+ reranker \(`voyage-02`\)", 0.480),
            ),
            authority=_voyage_authority,
            key="MAP@10",
            provenance="MultiHop-RAG paper Table 5, pinned in tools/compare.py PUBLISHED_MH",
        ),
        Quantity(
            name="voyage-02 + reranker Hits@10",
            sites=(
                Site("README.md", r"^\|\s*voyage-02 \+ rerank \(published\)", 0.747),
                Site("BENCHMARKS.md", r"^\|\s*published best \+ reranker \(`voyage-02`\)", 0.747),
            ),
            authority=_voyage_authority,
            key="Hits@10",
            provenance="MultiHop-RAG paper Table 5, pinned in tools/compare.py PUBLISHED_MH",
        ),
        Quantity(
            name="recalibration half-split count",
            sites=(
                Site("README.md", r"stratified half-splits", 300, 0),
                Site("BENCHMARKS.md", r"stratified half-splits", 300, 0),
                Site("eval/benchmarks/multihop-rag/recalibration.md", r"across \d+ splits", 300, 0),
            ),
            authority=lambda: ({"splits": s} if (s := recalibration_splits()) is not None else None),
            key="splits",
            provenance="eval/benchmarks/multihop-rag/recalibration.json transfer.splits",
        ),
    ]


# ── checking ─────────────────────────────────────────────────────────────

def _authority_value(quantity: Quantity) -> float | None:
    """Resolve the authority, or None when the report is absent (a skip)."""
    resolved = quantity.authority()
    if resolved is None:
        return None
    if isinstance(resolved, Mapping):
        if quantity.key is None:
            raise KeyError(f"{quantity.name}: mapping authority without a key")
        if quantity.key not in resolved:
            raise KeyError(f"{quantity.name}: authority has no {quantity.key!r}")
        return float(resolved[quantity.key])
    return float(resolved)


def check_superseded(*, root: Path, markers: bool) -> tuple[list[str], list[str]]:
    """Return (failures, notes). A superseded report present without its marker is a failure."""
    failures: list[str] = []
    notes: list[str] = []
    for stem, (replacement, reason) in SUPERSEDED.items():
        report = root / "eval" / "beir" / f"{stem}.json"
        if not report.is_file():
            continue
        marker = report.with_suffix(".superseded.json")
        if marker.is_file():
            data = _read_json(marker) or {}
            if data.get("superseded_by") != replacement:
                failures.append(
                    f"superseded marker for {stem} points at {data.get('superseded_by')!r}, "
                    f"registry says {replacement!r}"
                )
            continue
        if markers:
            marker.write_text(
                json.dumps(
                    {
                        "superseded_by": replacement,
                        "reason": reason,
                        "note": "Generated by tools/consistency_check.py; do not read as a result.",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            notes.append(f"wrote quarantine marker {marker.relative_to(root)}")
        else:
            failures.append(
                f"superseded report eval/beir/{stem}.json has no marker "
                f"(superseded by {replacement}); run with --fix"
            )
    return failures, notes


def check_references(*, root: Path, audit_dir: Path | None = None) -> list[str]:
    """A superseded stem must not be a claim source or a path any document cites."""
    failures: list[str] = []
    for claim in claim_check.CHECKS:
        stem = Path(claim.report).stem if claim.report else ""
        if stem in SUPERSEDED:
            failures.append(
                f"claim_check claim {claim.label!r} sources superseded eval/beir/{stem}.json"
            )
    cited = re.compile(r"eval/beir/([\w.-]+)\.(?:json|md)")
    for doc in _document_paths(root, audit_dir=audit_dir):
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for stem in cited.findall(line):
                if stem in SUPERSEDED:
                    failures.append(f"{_label(doc, root)}:{lineno} cites superseded {stem!r}")
    return failures


def _document_paths(root: Path, *, audit_dir: Path | None = None) -> list[Path]:
    """Every document the gate scans for stale references: the repo's, plus the audit set."""
    docs = sorted(root.glob("*.md")) + sorted((root / "docs").glob("**/*.md"))
    directory = _audit_dir(root) if audit_dir is None else audit_dir
    if directory.is_dir():
        docs += sorted(directory.glob("*.md"))
    return docs


def _label(doc: Path, root: Path) -> Path:
    """A doc's name relative to the repository; an audit doc keeps its workspace-relative name."""
    try:
        return doc.relative_to(root)
    except ValueError:
        return Path(os.path.relpath(doc, root))


def snapshot_mask(lines: Sequence[str]) -> tuple[list[str], list[str]]:
    """Blank the lines inside snapshot fences, and report a fence that is unclosed or undated.

    A fence is `<!-- consistency: snapshot YYYY-MM-DD -->` … `<!-- consistency: /snapshot -->`. The
    marker lines are blanked with the body, so a fenced block is invisible to the count guard and,
    being blank, still terminates a paragraph the way a blank line does.
    """
    masked = list(lines)
    failures: list[str] = []
    opened: int | None = None
    for index, line in enumerate(lines):
        if SNAPSHOT_OPEN.match(line):
            if opened is not None:
                failures.append(f"line {index + 1}: a snapshot fence opens inside another")
            else:
                opened = index
            masked[index] = ""
        elif SNAPSHOT_CLOSE.match(line):
            if opened is None:
                failures.append(f"line {index + 1}: a snapshot fence closes with nothing open")
            opened = None
            masked[index] = ""
        elif opened is not None:
            masked[index] = ""
    if opened is not None:
        failures.append(f"line {opened + 1}: a snapshot fence is never closed")
    return masked, failures


def check_audit_documents(
    *, root: Path, coverage_count: int | None, audit_dir: Path | None = None
) -> list[str]:
    """Pin the audit set above the repository, which is not committed here and has no registry.

    `best-in-the-world/` reads this repository's numbers and drifted from them: the BEIR coverage
    count stayed at 10 and 11 of 18 after FEVER and the full HotpotQA run landed, and its corpus
    table still called FEVER "queued". So two conventions are enforced on every audit document:

    * a `> **Currency:**` banner, which states which sections are pinned and which are dated;
    * the computed coverage count on every line that says "of 18", unless the line is inside a
      snapshot fence — a dated finding may keep a superseded number, but it may not read as current.

    The audit directory is a sibling of the repository, so a checkout that does not carry it skips
    this check entirely (returning no failures), like a report that has not been generated.
    """
    directory = _audit_dir(root) if audit_dir is None else audit_dir
    if not directory.is_dir():
        return []
    expected = None if coverage_count is None else f"{coverage_count:.0f}"
    failures: list[str] = []
    for path in sorted(directory.glob("*.md")):
        name = f"{AUDIT_PREFIX}{path.name}"
        lines = path.read_text(encoding="utf-8").splitlines()
        masked, fence_failures = snapshot_mask(lines)
        failures += [f"{name}: {message}" for message in fence_failures]
        if not any(CURRENCY_BANNER.match(line) for line in lines):
            failures.append(
                f"{name}: no `> **Currency:**` banner — a reader cannot tell a dated finding from "
                f"a pinned quantity"
            )
        if expected is None:
            continue
        for lineno, line in enumerate(masked, 1):
            if "of 18" in line and not claim_check._number_present(line, expected):
                failures.append(
                    f"{name}:{lineno} states a coverage count that is not the computed {expected} "
                    f"of 18 (fence a dated finding with `<!-- consistency: snapshot YYYY-MM-DD -->`)"
                )
    return failures


def run_checks(quantities: Sequence[Quantity], *, root: Path) -> list[str]:
    """One failure message per violated site or disagreeing document."""
    failures: list[str] = []
    cache: dict[str, list[str]] = {}

    def lines_of(doc: str) -> list[str]:
        if doc not in cache:
            path = _doc_path(doc, root)
            cache[doc] = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        return cache[doc]

    for quantity in quantities:
        authority = _authority_value(quantity)
        if authority is None:
            continue                       # its report has not been generated: a skip
        for site in quantity.sites:
            if site.optional and not _doc_path(site.doc, root).is_file():
                continue                   # the audit set is not part of this repository
            regions = claim_check._regions(lines_of(site.doc), site.anchor)
            if not regions:
                failures.append(
                    f"{quantity.name}: anchor {site.anchor!r} matched no line in {site.doc}"
                )
                continue
            expected = half_up(authority, site.decimals)
            stated = half_up(site.value, site.decimals)
            text = f"{stated:.{site.decimals}f}"
            missing = [r for r in regions if not claim_check._number_present(r, text)]
            if missing:
                first = missing[0].splitlines()[0].strip()
                failures.append(
                    f"{quantity.name}: {site.doc} does not state {text} at {site.anchor!r} "
                    f"(first miss: {first!r})"
                )
            if abs(stated - expected) > 1e-9:
                failures.append(
                    f"{quantity.name}: {site.doc} states {site.value} but the authority "
                    f"({quantity.provenance}) renders {expected:.{site.decimals}f}"
                )
    return failures


def checked_site_count(quantities: Sequence[Quantity], *, root: Path) -> int:
    """The sites actually checked: an authority absent or an optional document missing is a skip."""
    total = 0
    for quantity in quantities:
        if _authority_value(quantity) is None:
            continue
        for site in quantity.sites:
            if site.optional and not _doc_path(site.doc, root).is_file():
                continue
            total += 1
    return total


# ── entry point ──────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="consistency_check.py")
    ap.add_argument("--root", default=str(ROOT), help="repository root the doc paths resolve under")
    ap.add_argument("--fix", action="store_true",
                    help="write quarantine markers and eval/beir/coverage.json, then check")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()

    coverage = measured_beir()
    if args.fix:
        out = root / "eval" / "beir" / "coverage.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
        print(f"coverage: {coverage['count']} of {coverage['total']} — wrote {out.relative_to(root)}")

    failures, notes = check_superseded(root=root, markers=args.fix)
    for note in notes:
        print(note)
    failures += check_references(root=root)
    failures += check_audit_documents(
        root=root, coverage_count=coverage["count"] if coverage["count"] else None
    )

    if coverage["count"] == 0:
        print("consistency-check: SKIPPED — no eval/beir reports (nothing to pin)")
        return 1 if failures else 0

    quantities = registry()
    failures += run_checks(quantities, root=root)
    checked = checked_site_count(quantities, root=root)
    if failures:
        print(f"consistency-check: {len(failures)} failure(s) across {checked} sites")
        for line in failures:
            print(f"  - {line}")
        return 1
    audit = f", plus {AUDIT_PREFIX}*" if _audit_dir(root).is_dir() else ""
    print(
        f"consistency-check: {checked} sites agree; coverage {coverage['count']} of "
        f"{coverage['total']} ({', '.join(coverage['measured'])}){audit}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
