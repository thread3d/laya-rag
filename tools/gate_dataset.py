#!/usr/bin/env python
"""Build a *properly sized* answerability-gate dataset from a labelled benchmark — R3's sample.

`tools/gate_transfer.py` carries the transfer protocol; this produces the rows it needs at a size
that can actually decide something. The two-corpus documentation pilot had 19 and 17 rows, a
permutation null whose 95th percentile was 0.82, and therefore no answer.

MultiHop-RAG is the right source because it ships a **real negative class**: 301 `null_query`
records whose gold answer is literally "Insufficient information", which the benchmark's own
retrieval metric *excludes*. Scoring refusal on the records everyone else drops is the cheapest
new measurement in this programme — and it is a negative class the gate has never seen.

    # all 301 null_query rows plus a seeded sample of answerable ones
    python tools/gate_dataset.py --limit 300 --sample-seed 17 \
        --index-dir eval/benchmarks/multihop-rag/index/engine-default \
        --out eval/gate/multihop

Rows are written one per line and fsynced as they are produced, so a run killed at row *k* loses at
most the row in flight and a re-run of the same command resumes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import jobkit  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
DEFAULT_RECORDS = ROOT_BENCH / "MultiHopRAG.json"


def load_records(path: Path) -> list[dict[str, Any]]:
    """The benchmark's records, refusing a file that is not the expected list."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"{path}: expected a non-empty list of records")
    for index, record in enumerate(payload[:5]):
        if not isinstance(record, dict) or "query" not in record:
            raise SystemExit(f"{path}: record {index} has no 'query'")
    return payload


def is_null_query(record: dict[str, Any]) -> bool:
    """The benchmark's own marker for "the corpus cannot answer this"."""
    return str(record.get("question_type") or "").strip() == "null_query"


def select(
    records: Sequence[dict[str, Any]],
    *,
    limit: int | None = None,
    sample_seed: int | None = None,
    include_nulls: bool = True,
) -> list[tuple[str, int, str]]:
    """``(query, label, question_type)`` triples: every null_query, plus a seeded sample of answerables.

    The nulls are **not** sampled, because there are only 301 of them and they are the class the
    gate is missing; sampling them would throw away the very thing being measured. The positives are
    sampled, so the two classes end up comparable in size rather than 7:1.
    """
    def pair(record: dict[str, Any], label: int) -> tuple[str, int, str]:
        return (str(record["query"]), label, str(record.get("question_type") or "").strip())

    nulls = [pair(record, 0) for record in records if include_nulls and is_null_query(record)]
    answerable = [pair(record, 1) for record in records if not is_null_query(record)]
    if limit is not None and limit >= 0 and len(answerable) > limit:
        rng = random.Random(sample_seed) if sample_seed is not None else random
        answerable = rng.sample(answerable, limit)
    return nulls + answerable


def row_for(engine: Any, query: str, label: int, question_type: str = "") -> dict[str, Any]:
    """One query through the engine, as a feature row.

    ``question_type`` is carried because it is the **grouping key for the only real transfer test**: splitting
    at random reuses queries of the same type on both sides, so a gate can learn that type's idiosyncrasies and
    look better than it is. Recording the key is what lets `--group-by` hold a whole type out — and a row that
    does not carry it can only be evaluated in the weak regime.
    """
    from laya_rag.answerability import FEATURE_NAMES, extract_features, passage_view

    result = engine.query(query, retrieve_k=20)
    vector = extract_features(query, [passage_view(p) for p in result.passages])
    return {
        "query": query,
        "label": int(label),
        "question_type": str(question_type or ""),
        "features": {name: float(value) for name, value in zip(FEATURE_NAMES, vector)},
        "answerability": result.answerability,
        "answerable_verdict": result.answerable,
        "kept": result.kept,
        "considered": result.considered,
        "device": result.device,
        "total_cpu_seconds": round(result.total_cpu_seconds, 3),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="gate_dataset.py", description=__doc__)
    ap.add_argument("--records", default=str(DEFAULT_RECORDS))
    ap.add_argument("--limit", type=int, default=300,
                    help="how many ANSWERABLE records to sample; the null_query records are all kept")
    ap.add_argument("--sample-seed", type=int, default=17)
    ap.add_argument("--no-nulls", action="store_true", help="answerable rows only")
    ap.add_argument("--index-dir", default=str(ROOT_BENCH / "index" / "engine-default"))
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "eval" / "gate" / "multihop"))
    ap.add_argument("--fresh", action="store_true", help="discard a previous partial run")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("gate_dataset.py")

    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    stderr_engine = EngineConfig(index_dir=Path(args.index_dir))
    engine = RagEngine(stderr_engine)
    if not engine.has_index():
        raise SystemExit(f"no complete index at {args.index_dir}")

    rows = select(
        load_records(Path(args.records)),
        limit=args.limit,
        sample_seed=args.sample_seed,
        include_nulls=not args.no_nulls,
    )
    stem = Path(args.out)
    if args.fresh:
        for suffix in (".jsonl", ".manifest.json"):
            Path(str(stem) + suffix).unlink(missing_ok=True)
    fingerprint = jobkit.config_fingerprint(
        tool="gate_dataset",
        index_dir=str(args.index_dir),
        retrieve_k=args.retrieve_k,
        limit=args.limit,
        sample_seed=args.sample_seed,
        nulls=not args.no_nulls,
    )
    run = jobkit.ResumableRun(str(stem) + ".json", fingerprint=fingerprint)
    todo = run.pending(rows, key=lambda triple: triple[0])
    print(f"{len(rows)} rows ({sum(1 for _, label, _ in rows if label == 0)} null_query), "
          f"{len(todo)} to do ({len(run)} checkpointed)", flush=True)

    for query, label, question_type in todo:
        run.record(query, row_for(engine, query, label, question_type))
        if len(run) % 25 == 0:
            print(f"  {len(run)}/{len(rows)}", flush=True)

    payload = {
        "records": args.records,
        "index_dir": args.index_dir,
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "rows": list(run.values().values()),
    }
    out = Path(str(stem) + ".json")
    jobkit.write_json_atomic(out, payload)
    run.manifest(status="complete", total=len(rows))
    nulls = sum(1 for row in payload["rows"] if row["label"] == 0)
    print(f"wrote {out}: {len(payload['rows'])} rows, {nulls} null_query / "
          f"{len(payload['rows']) - nulls} answerable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
