"""Citation mechanics over composed answers — ROADMAP Phase 3.3.

The end-to-end row is `engine + <generator>`, and most of what it measures belongs to the model.
This measures the part that belongs to the **engine**: whether the prose's citations and quoted
spans actually exist in the evidence the engine handed over. It is deterministic, model-free
(`verify.py` uses no weights and no thresholds), and separate from the generated answer's
correctness.

Evidence is not stored in the end-to-end report, so it is re-retrieved here. Retrieval is
deterministic at fixed settings, so the passages are the ones the generator saw.

    .venv/bin/python tools/citation_report.py \
        --report eval/benchmarks/multihop-rag/endtoend-pilot.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "beir" / "hf-cache"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402

from laya_rag.verify import verify_citations  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
DEFAULT_INDEX = ROOT_BENCH / "index" / "engine-default"


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-answer verification rows into the engine's own citation numbers."""
    groups: dict[str, list[Mapping[str, Any]]] = {"all": list(rows.values())}
    for label, predicate in (("answerable", lambda r: not r.get("is_null")),
                             ("null", lambda r: bool(r.get("is_null")))):
        groups[label] = [row for row in rows.values() if predicate(row)]

    out: dict[str, Any] = {}
    for label, group in groups.items():
        claims = sum(row["claims"] for row in group)
        cited = sum(row["cited"] for row in group)
        quotes_checked = sum(row["quotes_checked"] for row in group)
        quotes_found = sum(row["quotes_found"] for row in group)
        out[label] = {
            "answers": len(group),
            "clean": sum(1 for row in group if row["clean"]),
            "clean_rate": _ratio(sum(1 for row in group if row["clean"]), len(group)),
            "claims": claims,
            "cited_claims": cited,
            "citation_coverage": _ratio(cited, claims),
            "invalid_citations": sum(row["invalid_citations"] for row in group),
            "quotes_checked": quotes_checked,
            "quotes_found": quotes_found,
            "quote_validity": _ratio(quotes_found, quotes_checked),
            "injection_flags": sum(row["injection"] for row in group),
        }
    return out


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Citation mechanics over composed answers",
        "",
        f"- {report['_settings']['answers']} answered queries re-retrieved from "
        f"`{Path(report['_settings']['report']).name}`",
        "",
        "| group | answers | citation coverage | quote validity | clean | invalid citations | injection flags |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label in ("all", "answerable", "null"):
        row = report["summary"][label]
        lines.append(
            f"| {label} | {row['answers']} | {row['citation_coverage']:.4f} | "
            f"{row['quote_validity']:.4f} | {row['clean']} ({row['clean_rate']:.4f}) | "
            f"{row['invalid_citations']} | {row['injection_flags']} |"
        )
    lines += [
        "",
        "Citation coverage is the share of sentences carrying a citation; quote validity is the",
        "share of quoted spans found in the evidence; `clean` is `verify.py`'s deterministic",
        "citation verdict. **This is the engine's number, not the generator's** — it asks whether",
        "the prose's citations exist, never whether the prose is right.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="citation_report.py")
    ap.add_argument("--report", default=str(ROOT_BENCH / "endtoend-pilot.json"))
    ap.add_argument("--index-dir", default=str(DEFAULT_INDEX))
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "eval" / "citation-report"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = json.loads(Path(args.report).read_text(encoding="utf-8"))
    answered = {k: row for k, row in source["per_query"].items() if row.get("answer")}
    if not answered:
        raise SystemExit(f"{args.report} has no composed answers to verify")

    os.environ["LAYA_RAG_INDEX_DIR"] = str(args.index_dir)
    if args.embed_model:
        os.environ["LAYA_RAG_EMBED_MODEL"] = args.embed_model
    else:
        os.environ.pop("LAYA_RAG_EMBED_MODEL", None)

    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    engine = RagEngine(EngineConfig())
    if not engine.has_index():
        raise SystemExit(f"no complete index at {args.index_dir}")

    rows: dict[str, dict[str, Any]] = {}
    for key, row in answered.items():
        result = engine.query(row["query"], top_k=args.top_k, max_passage_chars=None)
        passages = [str(p.get("text") or "") for p in result.passages]
        report = verify_citations(row["answer"], passages, query=row["query"])
        counts = report["counts"]
        quotes = report.get("quotes") or []
        rows[key] = {
            "is_null": bool(row.get("is_null")),
            "clean": bool(report.get("clean")),
            "claims": counts.get("claims", 0),
            "cited": counts.get("cited", 0),
            "invalid_citations": counts.get("invalid_citations", 0),
            "quotes_checked": counts.get("quotes_checked", 0),
            "quotes_found": counts.get("quotes_found", 0),
            "injection": report.get("injection", {}).get("count", 0),
            "quote_statuses": [q.get("status") for q in quotes],
        }

    payload = {
        "_settings": {**vars(args), "answers": len(rows)},
        "summary": summarize(rows),
        "per_query": rows,
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), payload)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(payload))
    print(render_markdown(payload))
    print("wrote", str(out) + ".json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
