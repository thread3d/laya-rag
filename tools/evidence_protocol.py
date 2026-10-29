#!/usr/bin/env python
"""The evidence-layer protocol: one command, one report, four axes.

The programme's winnable claim is that this engine is scored where evidence is scored — TREC RAG's
first layer and nothing above it — and that the abstention decision is a first-class measured
quantity rather than a caveat. This tool is that claim made checkable: it reads the artefacts the
measurement programme already produces and prints one report over the four axes, every number
naming the file and field it came from.

1. **Evidence quality** — MAP@10 / Hits@10 / MRR@10 of the retrieval the answers were composed from.
2. **Evidence-side abstention** — AUC, risk@80 and AURC of the decision that says "this corpus cannot
   answer", scored on the benchmark's own negative class, against the shipped scalar as a baseline.
3. **Joint score** — evidence quality × (1 − false-accept rate). TREC 2025's own support table is led
   by a run with 0.84 support precision and a 0.06 strict-vital score, so a system that says little
   tops an attribution column; the joint score is what stops abstention being bought with refusals.
4. **Cost** — per-query seconds and CPU seconds, and the device they were measured on.

It deliberately **does not score generated text**: no axis here needs a generator, a judge model or an
API key, and any number that did would belong in the `engine + <model>` column instead.

    python tools/evidence_protocol.py \\
        --answer-row eval/benchmarks/multihop-rag/engine-extractive-rank150.json \\
        --gate-rows eval/gate/multihop.json \\
        --out eval/evidence-protocol

Every input is optional except one: an axis whose artefact is absent is reported as **not measured**
with the command that would produce it, never as a zero and never as a pass.
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

import numpy as np  # noqa: E402

import evalkit  # noqa: E402

#: What each axis needs, and how to get it. The report prints this for an absent input rather than
#: leaving a hole: a protocol that silently skips an axis is a protocol that can be gamed by omission.
RECIPES = {
    "answer_row": "make engine-extractive  (tools/engine_extractive.py)",
    "gate_rows": "python tools/gate_dataset.py --limit 300  (needs the MultiHop-RAG index)",
    "transfer": "python tools/gate_transfer.py --train A --test B",
    "cost": "make cost-report  (tools/cost_report.py)",
}


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evidence_quality(answer_row: Mapping[str, Any]) -> dict[str, Any]:
    """Axis 1: the retrieval the composed answers came from, and the answer's own two numbers.

    The answer row is the engine's extractive end-to-end run; its retrieval metrics are the same ones
    the retrieval rows report, and its ``citation_clean_all`` is a mechanical check rather than a
    judgement. Both are reported because "the evidence was good" and "the answer carried it" are
    different claims and this protocol must not let one stand for the other.
    """
    summary = answer_row.get("summary") or {}
    wanted = ("MAP@10", "Hits@10", "Hits@4", "MRR@10")
    metrics = {name: float(summary[name]) for name in wanted if name in summary}
    if not metrics:
        raise SystemExit("the answer row carries no retrieval metrics; is it an engine-extractive report?")
    return {
        "retrieval": metrics,
        "queries": int(summary.get("queries") or 0),
        "answer_nugget_recall": summary.get("answer_nugget_recall"),
        "citation_clean": summary.get("citation_clean_all"),
        "abstention_rate": summary.get("abstention_rate"),
        "provenance": {"file": "answer row", "fields": "summary.*"},
    }


def abstention_axis(rows: Sequence[Mapping[str, Any]], *, coverage: float = 0.80) -> dict[str, Any]:
    """Axis 2: the gate's decision on a corpus carrying both classes.

    Two scorers are reported, because the interesting claim is the *difference*: the shipped
    single-scalar answerability decision, and the twelve-feature gate fitted on a stratified half of
    these rows and scored on the other. A gate that only beats its own control is not evidence, and
    the scalar is the thing actually in production, so it is the baseline that matters.
    """
    from laya_rag.answerability import FEATURE_NAMES, LogisticGate, roc_auc

    labels = np.array([int(row["label"]) for row in rows], dtype=np.float64)
    scalars = np.array([float(row.get("answerability") or 0.0) for row in rows], dtype=np.float64)
    if len(np.unique(labels)) < 2:
        raise SystemExit("the gate rows carry a single class; nothing can be scored")

    X = np.array(
        [[float(row["features"].get(name, 0.0)) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    # Stratified halves, so both classes appear on each side regardless of row order.
    train: list[int] = []
    test: list[int] = []
    for label in (0.0, 1.0):
        members = [i for i, value in enumerate(labels) if value == label]
        train.extend(members[::2])
        test.extend(members[1::2])
    train_idx = np.asarray(sorted(train))
    test_idx = np.asarray(sorted(test))
    if len(np.unique(labels[test_idx])) < 2:
        raise SystemExit("the held-out half has a single class; more rows are needed")

    gate = LogisticGate().fit(X[train_idx], labels[train_idx])
    gate_scores = gate.predict_proba(X[test_idx])

    def arm(name: str, scores: np.ndarray) -> dict[str, Any]:
        probabilities = [float(v) for v in scores]
        outcomes = [int(v) for v in labels[test_idx]]
        return {
            "scorer": name,
            "auc": round(float(roc_auc(labels[test_idx], scores)), 4),
            "risk@80": evalkit.risk_at_coverage(probabilities, outcomes, coverage),
            "aurc": evalkit.aurc(probabilities, outcomes),
            "n": int(len(outcomes)),
            "positives": int(sum(outcomes)),
        }

    # The operating point a deployment actually uses, which `risk@80` is not: the shipped scalar's
    # answerability threshold is 0.5, so this is the share of the benchmark's unanswerable queries
    # the engine as shipped would answer. It is the number a reader can act on.
    negatives = labels[test_idx] == 0
    false_accept = float(np.mean(scalars[test_idx][negatives] >= 0.5)) if negatives.any() else None

    return {
        "held_out": arm("feature gate (12 features, held-out half)", gate_scores),
        "scalar": arm("shipped scalar, same half", scalars[test_idx]),
        "scalar_false_accept_at_0.5": None if false_accept is None else round(false_accept, 4),
        "rows": int(len(rows)),
        "provenance": {"file": "gate rows", "fields": "label, features, answerability"},
        "coverage": coverage,
    }


def joint_score(evidence_map: float, abstention: Mapping[str, Any]) -> dict[str, Any]:
    """Axis 3: quality × (1 − false-accept rate), the score that stops refusal-as-a-strategy.

    Both factors are on the *same* half of the queries when the gate rows and the answer row cover
    different corpora, so this is a **joint reading, not a product over one sample** — reported with
    that caveat rather than quietly multiplied and called a single number.
    """
    risk = float(abstention["held_out"]["risk@80"])
    return {
        "evidence_map@10": round(float(evidence_map), 4),
        "false_access_at_80": round(risk, 4),
        "joint": round(float(evidence_map) * (1.0 - risk), 4),
        "caveat": "quality and abstention come from different samples; read as a joint report, not a product",
    }


#: The settings that change what a number means. A report that does not carry these cannot be compared
#: with another, which is the failure the resume fingerprint exists to prevent in a different place.
CONFIG_KEYS = (
    "limit", "sample_seed", "rerank_model", "rerank_depth", "embed_model",
    "chunk_chars", "context_prefix", "max_sentences", "index_dir", "index_identity",
)


def configuration(answer_row: Mapping[str, Any] | None,
                  gate_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Which configuration produced each axis, and whether the two axes share a query sample.

    The joint score multiplies evidence quality from one sample by abstention from another. That is
    stated as a caveat today; here it is *computed*: if the answer row and the gate rows were drawn
    with a different `limit` or `sample_seed`, the report says so instead of leaving a reader to notice.
    """
    answer = (answer_row or {}).get("_settings") or {}
    gate = (gate_payload or {}).get("_settings") or {}
    answer_cfg = {key: answer.get(key) for key in CONFIG_KEYS if key in answer}
    gate_cfg = {key: gate.get(key) for key in CONFIG_KEYS if key in gate}
    if not gate_cfg:  # gate_dataset records its own settings at the top level, not under `_settings`
        gate_cfg = {
            key: (gate_payload or {}).get(key) for key in ("limit", "sample_seed", "records", "index_dir")
            if (gate_payload or {}).get(key) is not None
        }
    shared = (
        answer_cfg.get("limit") == gate_cfg.get("limit")
        and answer_cfg.get("sample_seed") == gate_cfg.get("sample_seed")
        and answer_cfg.get("limit") is not None
    )
    return {
        "answer_row": answer_cfg,
        "gate_rows": gate_cfg,
        "same_query_sample": bool(shared),
        "note": (
            "the two axes share a query sample"
            if shared
            else "the axes were measured on different samples; the joint score is a joint report, "
                 "not a product over one sample"
        ),
    }


def cost_axis(path: Path | None) -> dict[str, Any] | None:
    """Axis 4: seconds, CPU seconds and the device, when a cost report exists."""
    if path is None or not Path(path).is_file():
        return None
    report = load_json(Path(path))
    latency = report.get("latency_seconds") or {}
    cpu = report.get("cpu_seconds") or {}
    return {
        "device": report.get("device", "unknown"),
        "wall_mean_seconds": latency.get("mean"),
        "cpu_mean_seconds": cpu.get("mean"),
        "torch_threads": report.get("torch_threads"),
        "index_bytes": report.get("index_bytes"),
        "provenance": {"file": str(path), "fields": "latency_seconds, cpu_seconds, device"},
    }


def render(report: Mapping[str, Any]) -> str:
    lines: list[str] = ["# Evidence-layer protocol", ""]
    quality = report.get("evidence_quality")
    lines += ["## 1. Evidence quality", ""]
    if quality is None:
        lines += [f"**not measured** — {RECIPES['answer_row']}", ""]
    else:
        lines += [
            "| metric | value |",
            "| --- | ---: |",
        ]
        for name, value in quality["retrieval"].items():
            lines.append(f"| {name} | {value:.4f} |")
        for name in ("answer_nugget_recall", "citation_clean", "abstention_rate"):
            value = quality.get(name)
            if value is not None:
                lines.append(f"| {name.replace('_', ' ')} | {value:.4f} |")
        lines += ["", f"over {quality['queries']} queries.", ""]

    abstention = report.get("abstention")
    lines += ["## 2. Evidence-side abstention", ""]
    if abstention is None:
        lines += [f"**not measured** — {RECIPES['gate_rows']}", ""]
    else:
        lines += [
            "| scorer | n | positives | AUC | risk@80 | AURC |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for key in ("held_out", "scalar"):
            arm = abstention[key]
            lines.append(
                f"| {arm['scorer']} | {arm['n']} | {arm['positives']} | {arm['auc']:.4f} | "
                f"{arm['risk@80']:.4f} | {arm['aurc']:.4f} |"
            )
        false_accept = abstention.get("scalar_false_accept_at_0.5")
        if false_accept is not None:
            lines += [
                "",
                f"At the **deployed** threshold (answerability ≥ 0.5) the shipped scalar would answer "
                f"**{false_accept:.4f}** of the benchmark's unanswerable queries.",
            ]
        lines += ["", "negative class: the benchmark's own unanswerable records.", ""]

    joint = report.get("joint")
    lines += ["## 3. Joint score", ""]
    if joint is None:
        lines += ["**not measured**", ""]
    else:
        lines += [
            f"evidence MAP@10 **{joint['evidence_map@10']:.4f}** × "
            f"(1 − false-access **{joint['false_access_at_80']:.4f}**) = **{joint['joint']:.4f}**",
            "",
            f"_{joint['caveat']}_",
            "",
        ]

    cost = report.get("cost")
    lines += ["## 4. Cost", ""]
    if cost is None:
        lines += [f"**not measured** — {RECIPES['cost']}", ""]
    else:
        lines += [
            f"device **{cost['device']}** at {cost['torch_threads']} threads: "
            f"{cost['wall_mean_seconds']} s wall, {cost['cpu_mean_seconds']} s CPU per query, "
            f"index {cost['index_bytes']} bytes",
            "",
        ]
    config = report.get("configuration") or {}
    if config:
        lines += ["## 5. Which configuration produced these", ""]
        for name in ("answer_row", "gate_rows"):
            settings = config.get(name) or {}
            if settings:
                rendered = ", ".join(f"`{key}={value}`" for key, value in settings.items())
                lines.append(f"- **{name}**: {rendered}")
        lines += ["", f"_{config.get('note', '')}_", ""]
    lines += ["---", "", "Generated by `tools/evidence_protocol.py`. No generator, judge or API key "
              "is in any path above."]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="evidence_protocol.py", description=__doc__)
    ap.add_argument("--answer-row", default=None, help="engine-extractive report JSON")
    ap.add_argument("--gate-rows", default=None, help="gate dataset JSON (labelled feature rows)")
    ap.add_argument("--cost", default=None, help="cost-report JSON")
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--out", default=str(ROOT / "eval" / "evidence-protocol"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report: dict[str, Any] = {"coverage": args.coverage}

    if args.answer_row and Path(args.answer_row).is_file():
        row = load_json(Path(args.answer_row))
        report["evidence_quality"] = evidence_quality(row)
    else:
        report["evidence_quality"] = None

    if args.gate_rows and Path(args.gate_rows).is_file():
        payload = load_json(Path(args.gate_rows))
        rows = payload.get("rows") if isinstance(payload, Mapping) else payload
        report["abstention"] = abstention_axis(rows, coverage=args.coverage)
    else:
        report["abstention"] = None

    if report["evidence_quality"] and report["abstention"]:
        report["joint"] = joint_score(
            report["evidence_quality"]["retrieval"].get("MAP@10", 0.0), report["abstention"]
        )
    else:
        report["joint"] = None

    report["cost"] = cost_axis(Path(args.cost) if args.cost else None)
    report["configuration"] = configuration(
        load_json(Path(args.answer_row)) if args.answer_row and Path(args.answer_row).is_file() else None,
        load_json(Path(args.gate_rows)) if args.gate_rows and Path(args.gate_rows).is_file() else None,
    )

    rendered = render(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"\nwrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
