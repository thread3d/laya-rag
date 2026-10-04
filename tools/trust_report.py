"""One deterministic trust report — ROADMAP Phase 5.1.

The engine's differentiating claim is not retrieval rank (which a CPU-only, 33M-encoder system
cannot win) but **verifiable trust**: citation mechanics that cannot be wrong the way a model can,
a measured abstention curve, and a corpus that cannot be poisoned into the answer. Each of those
already exists in the repository as a separate surface; this is the one artefact that states them
together, deterministically and offline, so the claim is a report rather than a paragraph.

It aggregates, and does not re-implement:

* `tools/adversarial_audit.py` — poison exclusion, gold retention, negative rejection, injection
  flagging, tenant isolation, with its own committed thresholds;
* `verify.verify_citations` via the audit's `verification_probe` — citation mechanics, a fabricated
  number, and an injection marker on a known-bad answer;
* the answerability selective-prediction suite (`answerability-selective.json`, Phase 0.3) —
  held-out AUC, AURC, AUGRC, Brier and ECE for the gate.

    make trust-report          # writes eval/trust-report.{json,md}
    make trust-report TRUST_ARGS="--check"   # exit 1 if any check fails

A *missing* input is reported as `null` (skipped), never as a pass: the selective suite needs
`make multihop-answerability` to have run, and treating its absence as success would be the exact
failure mode `claim_check.py` exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402
from adversarial_audit import build_report, verification_probe  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
SELECTIVE = ROOT_BENCH / "answerability-selective.json"
DEFAULT_OUT = ROOT / "eval" / "trust-report"

#: The gate must beat chance out of sample for the abstention claim to mean anything.
MIN_GATE_AUC = 0.6


def _mean(value: Any) -> float:
    """Selective-prediction metrics are stored as `{mean, lo, hi}`; read the point estimate."""
    if isinstance(value, Mapping):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


def collect(selective_path: Path = SELECTIVE) -> dict[str, Any]:
    """Every deterministic trust surface, with a pass/fail/skip per claim."""
    audit = build_report()
    verification = verification_probe()

    abstention: dict[str, Any] | None = None
    path = Path(selective_path)
    if path.is_file():
        abstention = json.loads(path.read_text(encoding="utf-8")).get("held_out")

    # The gate earns its place only if it beats both chance and the shipped single scalar.
    gate_auc = _mean(abstention["gate_auc"]) if abstention else 0.0
    scalar_risk80 = _mean(abstention["head_risk@80"]) if abstention else 0.0
    gate_risk80 = _mean(abstention["gate_risk@80"]) if abstention else 0.0

    checks: dict[str, bool | None] = {
        **{f"audit_{name}": bool(value) for name, value in audit["checks"].items()},
        "verification_citation_mechanics_clean": bool(verification["citation_mechanics_clean"]),
        "verification_injection_flagged": bool(verification["injection_flagged"]),
        "verification_number_flagged": bool(verification["number_flagged"]),
        "abstention_beats_chance": (None if not abstention else gate_auc >= MIN_GATE_AUC),
        "abstention_beats_scalar_risk@80": (None if not abstention else gate_risk80 < scalar_risk80),
    }
    return {
        "audit": audit,
        "verification": verification,
        "abstention": abstention,
        "checks": checks,
    }


def summarize(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report["checks"]
    failed = sorted(name for name, value in checks.items() if value is False)
    skipped = sorted(name for name, value in checks.items() if value is None)
    return {
        "passed": sum(1 for value in checks.values() if value is True),
        "failed": failed,
        "skipped": skipped,
        "ok": not failed,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    audit = report["audit"]
    verification = report["verification"]
    abstention = report["abstention"]
    summary = summarize(report)
    lines = [
        "# Trust report — deterministic, offline",
        "",
        f"- checks: **{summary['passed']} passed**, {len(summary['failed'])} failed, "
        f"{len(summary['skipped'])} skipped",
        f"- adversarial corpus: `{audit['settings']['poison_path']}` planted, k="
        f"{audit['settings']['k']}",
        "",
        "## Corpus poisoning",
        "",
        "| reading | trust off | trust on |",
        "| --- | --- | --- |",
        f"| gold hit rate | {audit['trust_off']['gold_hit_rate']:.4f} | "
        f"{audit['trust_on']['gold_hit_rate']:.4f} |",
        f"| poison leak rate | {audit['trust_off']['poison_leak_rate']:.4f} | "
        f"{audit['trust_on']['poison_leak_rate']:.4f} |",
        f"| negative reject rate | {audit['trust_off']['negative_reject_rate']:.4f} | "
        f"{audit['trust_on']['negative_reject_rate']:.4f} |",
        f"| injection flag rate | {audit['trust_off']['injection_flag_rate']:.4f} | "
        f"{audit['trust_on']['injection_flag_rate']:.4f} |",
        "",
        f"- false-refusal cost of enabling trust: **{audit['false_refusal_rate']:.4f}** "
        "(gold answers lost, not gained)",
        f"- tenant isolation: alpha sees beta = {audit['tenants']['alpha_sees_beta']}, "
        f"beta sees alpha = {audit['tenants']['beta_sees_alpha']}, "
        f"anonymous passages = {audit['tenants']['anonymous_passages']}",
        "",
        "## Deterministic verification (fabricated answer)",
        "",
        f"- citation mechanics clean: **{verification['citation_mechanics_clean']}**",
        f"- fabricated number flagged: **{verification['number_flagged']}**",
        f"- injection marker flagged: **{verification['injection_flagged']}**",
        f"- content issues raised: {', '.join(verification['content_kinds']) or 'none'}",
        "",
        "## Abstention (held-out, selective prediction)",
        "",
    ]
    if abstention:
        canonical = ("gate_auc", "gate_balanced_accuracy", "gate_risk@80", "gate_risk@90",
                     "gate_aurc", "gate_augrc", "gate_brier", "gate_ece")
        # Only render columns both arms actually carry; a missing metric must not print as 0.0000.
        keys = [k for k in canonical
                if k in abstention and k.replace("gate_", "head_") in abstention]
        header = " | ".join(key.replace("gate_", "") for key in keys)
        lines += [
            f"| arm | {header} |",
            "| --- | " + " | ".join("---" for _ in keys) + " |",
            "| **gate (12 features)** | " + " | ".join(f"{_mean(abstention[k]):.4f}" for k in keys) + " |",
            "| scalar head (shipped) | " + " | ".join(
                f"{_mean(abstention[k.replace('gate_', 'head_')]):.4f}" for k in keys) + " |",
            "",
            "The scalar head is the shipped out-of-domain answerability decision; the gate is the",
            "recalibrated feature model. `risk@80` is the false-accept rate at 80% coverage, so",
            "**lower is better** for risk, AURC and AUGRC, and higher is better for AUC.",
        ]
    else:
        lines.append("_not measured — run `make multihop-answerability` first_")
    lines += ["", "## Checks", ""]
    for name, value in report["checks"].items():
        mark = "pass" if value is True else ("FAIL" if value is False else "skip")
        lines.append(f"- `{name}`: {mark}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="trust_report.py")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--selective", default=str(SELECTIVE))
    ap.add_argument("--check", action="store_true", help="exit 1 if any check fails")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect(Path(args.selective))
    summary = summarize(report)
    report["summary"] = summary
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), report)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(report))
    print(render_markdown(report))
    print(f"wrote {out}.json and {out}.md")
    if args.check and not summary["ok"]:
        print(f"trust-report: FAILED checks: {', '.join(summary['failed'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
