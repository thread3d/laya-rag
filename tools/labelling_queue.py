"""A labelling harness for the two owed hand-labelled evaluations.

The roadmap owes two human-labelled sets: **~300 claim/corpus pairs** for the three-state
adjudication's no-coverage recall (Phase 5.2) and **~200 composed answers** for the verifier's
precision (Phase 5.3). A harness cannot label them — a model-generated label would defeat the point
of a hand-labelled set — but it can make the human work bounded and its result scorable:

* `build` turns records into a review queue, carrying a **proposed** label (from the NLI head or the
  verifier, supplied by whoever generated the records) that a reviewer accepts or corrects;
* `score` reads the labelled queue and reports agreement with the proposal plus the metric the
  roadmap actually owes — recall of the `no_coverage` class, and the verifier's per-label counts.

Input records are a JSON list or JSONL, one item each:

    {"id": "q17-3", "query": "...", "claim": "...", "evidence": "...",
     "proposed_label": "supported" | "contradicted" | "no_coverage" | null}

    .venv/bin/python tools/labelling_queue.py build --input pairs.json --out queue.jsonl
    .venv/bin/python tools/labelling_queue.py score --input labelled.jsonl

The queue is deliberately plain JSONL: a reviewer edits `label` and `labelled_by` in place, and
nothing here ever writes a label on its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

#: The three-state adjudication labels the first set asks a human to assign.
ADJUDICATION_LABELS = ("supported", "contradicted", "no_coverage")


def load_records(path: Path) -> list[dict[str, Any]]:
    """A JSON list or a JSONL file of records."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{path} is not a JSON list")
        return [dict(item) for item in payload]
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, Mapping):
            raise ValueError(f"{path} line {number} is not an object")
        records.append(dict(item))
    return records


def build_queue(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add the review fields to each record, preserving everything else.

    ``label`` is always null in a fresh queue and ``proposed_label`` is never copied into it;
    the reviewer's value is the only thing that counts.
    """
    queue: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        item = dict(record)
        item.setdefault("id", f"item-{index}")
        item.setdefault("proposed_label", None)
        item["label"] = None
        item["labelled_by"] = None
        item["status"] = "unlabelled"
        queue.append(item)
    return queue


def score(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Agreement with the proposal, per-label counts, and no-coverage recall."""
    labelled = [record for record in records if record.get("label")]
    result: dict[str, Any] = {
        "items": len(records),
        "labelled": len(labelled),
        "unlabelled": len(records) - len(labelled),
    }
    if not labelled:
        result["note"] = "no labelled items yet; run `build`, label the queue, then score"
        return result
    counts = Counter(str(record.get("label")) for record in labelled)
    result["label_counts"] = dict(sorted(counts.items()))
    with_proposal = [
        record for record in labelled if record.get("proposed_label")
    ]
    result["with_proposal"] = len(with_proposal)
    if with_proposal:
        agree = sum(
            1 for record in with_proposal
            if str(record["proposed_label"]) == str(record["label"])
        )
        result["agreement"] = round(agree / len(with_proposal), 4)
    # The roadmap owes recall of the `no_coverage` class specifically: of the human-labelled
    # no-coverage items, how many did the proposal also call no_coverage?
    gold_no_coverage = [
        record for record in labelled if str(record.get("label")) == "no_coverage"
    ]
    if gold_no_coverage:
        caught = sum(
            1 for record in gold_no_coverage
            if str(record.get("proposed_label")) == "no_coverage"
        )
        result["no_coverage_items"] = len(gold_no_coverage)
        result["no_coverage_recall"] = round(caught / len(gold_no_coverage), 4)
    else:
        result["no_coverage_items"] = 0
        result["note"] = "no human-labelled no_coverage items yet"
    by_reviewer = Counter(str(record.get("labelled_by") or "unknown") for record in labelled)
    result["reviewers"] = dict(sorted(by_reviewer.items()))
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = ["# Labelling queue score", ""]
    for key in ("items", "labelled", "unlabelled", "with_proposal", "agreement"):
        if key in result:
            lines.append(f"- {key}: **{result[key]}**")
    if "label_counts" in result:
        lines += ["", "| label | count |", "| --- | --- |"]
        lines += [f"| {label} | {count} |" for label, count in result["label_counts"].items()]
    if result.get("no_coverage_items"):
        lines += ["", f"- human `no_coverage` items: **{result['no_coverage_items']}**, "
                  f"recall **{result['no_coverage_recall']}**"]
    if result.get("reviewers"):
        lines += ["", f"- reviewers: {result['reviewers']}"]
    if result.get("note"):
        lines += ["", result["note"]]
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="labelling_queue.py")
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="turn records into an unlabelled review queue")
    build.add_argument("--input", required=True)
    build.add_argument("--out", required=True, help="JSONL queue to write")
    score_cmd = sub.add_parser("score", help="score a labelled queue")
    score_cmd.add_argument("--input", required=True)
    score_cmd.add_argument("--json", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.input)
    if not source.is_file():
        print(f"labelling-queue: no input at {source}", file=sys.stderr)
        return 2
    records = load_records(source)
    if args.command == "build":
        queue = build_queue(records)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            for item in queue:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"labelling-queue: wrote {len(queue)} unlabelled items to {out}")
        return 0
    result = score(records)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
