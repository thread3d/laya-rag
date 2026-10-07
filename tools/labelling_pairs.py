"""Build the two owed human-labelling sets from what is already on disk.

The roadmap owes two hand-labelled evaluations, and `tools/labelling_queue.py` can queue and score
them but cannot *produce* the records. This tool produces them; it never writes a label.

* **Adjudication (~300 claim/corpus pairs)** — for Phase 5.2's no-coverage recall. Each record is a
  query, a claim, and the evidence the engine actually retrieved for that query, drawn from the
  cached run (`ce-scores-full.json`) and the corpus itself, so no retrieval is re-run. Two kinds:
  a claim taken from *this* query's gold facts, which the evidence supports, and a claim taken from
  **another** query's gold facts, which this evidence is silent about — the `no_coverage` class the
  roadmap owes recall for. The reviewer assigns `supported` / `contradicted` / `no_coverage`.
* **Verifier (~200 composed answers)** — for Phase 5.3's verifier precision. Records carry a
  composed answer and the evidence it was written from; the reviewer decides whether the answer is
  faithful, and the verifier's flags are scored against that. Answers come from the shipped
  generator (`--generate N`, via `laya_rag.compose.Composer`) or from an existing report
  (`--answers`).

    .venv/bin/python tools/labelling_pairs.py --mode adjudication --limit 300
    .venv/bin/python tools/labelling_pairs.py --mode verifier --generate 200
    .venv/bin/python tools/labelling_queue.py build --input eval/labelling/adjudication-pairs.json \
        --out eval/labelling/adjudication-queue.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import nugget_recall as nr  # noqa: E402

BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
DATASET = BENCH / "MultiHopRAG.json"
CORPUS = BENCH / "corpus.json"
SCORES = BENCH / "ce-scores-full.json"
DEFAULT_OUT = ROOT / "eval" / "labelling"


def answerable_records(path: Path = DATASET) -> list[dict[str, Any]]:
    """The dataset's answerable records, in order."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [item for item in items if item.get("question_type") != "null_query"]


def answerable_indexed(path: Path = DATASET) -> list[tuple[int, dict[str, Any]]]:
    """``(dataset_index, record)`` for each answerable record.

    The cached run is keyed by **dataset** index — the 301 null records are simply absent — so the
    position in the answerable subsequence is *not* the cache key once a null has been passed. Using
    the position silently mis-aligns every later query's evidence, which shows up as a suspiciously
    low coverage count rather than an error.
    """
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        (index, item)
        for index, item in enumerate(items)
        if item.get("question_type") != "null_query"
    ]


def gold_facts(record: Mapping[str, Any]) -> list[str]:
    """The record's gold facts, in order, from its evidence list."""
    facts = [
        str(entry.get("fact") or "").strip()
        for entry in (record.get("evidence_list") or [])
        if str(entry.get("fact") or "").strip()
    ]
    return facts


def evidence_for(
    chunks: Mapping[str, str],
    pool: Sequence[str],
    *,
    top_k: int,
) -> list[dict[str, str]]:
    """The first ``top_k`` retrieved chunks as ``{path, text}``, skipping any with no text."""
    out: list[dict[str, str]] = []
    for chunk_id in list(pool)[:top_k]:
        text = chunks.get(chunk_id)
        if text:
            out.append({"path": chunk_id, "text": text})
    return out


def _covered(evidence: Mapping[str, str], coverage: nr.Evidence, fact: str) -> bool:
    return any(coverage.covers_fact(chunk_id, fact) for chunk_id in evidence)


def adjudication_records(
    records: Sequence[tuple[int, Mapping[str, Any]]],
    pools: Mapping[str, Sequence[str]],
    chunks: Mapping[str, str],
    *,
    limit: int,
    seed: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """A `supported`/`no_coverage` pair set: same evidence, one claim from it and one it is silent on.

    The claim for a `no_coverage` item comes from a *different* query, rotated deterministically, and
    is only kept when the evidence genuinely does not contain it — checked against the corpus text,
    not assumed, because a false negative here would be labelled `no_coverage` while being supported.
    """
    coverage = nr.Evidence(chunks)
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    chosen = order[: max(0, limit)]
    out: list[dict[str, Any]] = []
    for position, index in enumerate(chosen):
        dataset_index, record = records[index]
        query = str(record.get("query") or "").strip()
        pool = list(pools.get(str(dataset_index), []))
        if not query or not pool:
            continue
        evidence = evidence_for(chunks, pool, top_k=top_k)
        if not evidence:
            continue
        # A `supported` item needs a claim the evidence actually contains. Whole-fact coverage is
        # the benchmark's own normalised substring test, so it is searched over the *whole* pool and
        # the covering chunk is pulled into the shown window even when it ranks below `top_k`;
        # checking the top-k alone would report a fact as uncovered that a deeper rank supports.
        facts = gold_facts(record)
        supported: str | None = None
        covering: str | None = None
        for chunk_id in pool:
            hit = next((fact for fact in facts if coverage.covers_fact(chunk_id, fact)), None)
            if hit is not None:
                supported, covering = hit, chunk_id
                break
        if supported is not None and covering is not None:
            shown = [row for row in evidence if row["path"] != covering]
            if len(shown) >= top_k:
                shown = shown[: top_k - 1]
            shown.append({"path": covering, "text": chunks[covering]})
            out.append(
                {
                    "id": f"adj-{position:04d}-supported",
                    "kind": "supported",
                    "query": query,
                    "claim": supported,
                    "evidence": shown,
                    "claim_from_query": query,
                    "proposed_label": None,
                }
            )
        by_path = {row["path"]: row["text"] for row in evidence}
        # Another query's gold fact, rotated so a neighbouring item is not reused by accident,
        # and kept only when this evidence is silent about it. The rotation can land back on the
        # same record (small sets, unlucky offsets); step past it rather than ask a query to be
        # silent about its own gold.
        other_index = (index + 1 + position) % len(records)
        if other_index == index and len(records) > 1:
            other_index = (other_index + 1) % len(records)
        other = records[other_index][1]
        foreign = next(
            (
                fact
                for fact in gold_facts(other)
                if fact not in (supported or "",) and not _covered(by_path, coverage, fact)
            ),
            None,
        )
        if foreign is not None:
            out.append(
                {
                    "id": f"adj-{position:04d}-no-coverage",
                    "kind": "no_coverage",
                    "query": query,
                    "claim": foreign,
                    "evidence": evidence,
                    "claim_from_query": str(other.get("query") or ""),
                    "proposed_label": None,
                }
            )
    return out


def verifier_records(
    answers: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """~200 composed answers with the evidence they were written from, for verifier precision."""
    out: list[dict[str, Any]] = []
    for position, row in enumerate(answers[:limit]):
        answer = str(row.get("answer") or "").strip()
        query = str(row.get("query") or "").strip()
        if not answer or not query or row.get("abstained"):
            continue
        evidence = row.get("evidence") or []
        out.append(
            {
                "id": f"ver-{position:04d}",
                "kind": "composed",
                "query": query,
                "answer": answer,
                "evidence": evidence,
                "generator": row.get("generator"),
                "proposed_label": None,
            }
        )
    return out


def self_check() -> list[str]:
    """Offline invariants on a synthetic corpus: no files, no model, no network."""
    problems: list[str] = []
    chunks = {"s#0": "The company was founded in 1999 in Berlin.", "n#0": "An unrelated passage."}
    records = [
        (0, {"query": "When was it founded?",
             "evidence_list": [{"fact": "The company was founded in 1999 in Berlin."}]}),
        (1, {"query": "Where is it?",
             "evidence_list": [{"fact": "It is headquartered in Munich."}]}),
    ]
    pools = {"0": ["s#0"], "1": ["n#0"]}
    rows = adjudication_records(records, pools, chunks, limit=2, seed=17, top_k=5)
    if not rows:
        problems.append("no records built from a supported fixture")
        return problems
    supported = [r for r in rows if r["kind"] == "supported"]
    no_cov = [r for r in rows if r["kind"] == "no_coverage"]
    if not supported:
        problems.append("the fixture's covered claim was not emitted as supported")
    if not no_cov:
        problems.append("the fixture produced no no_coverage item")
    for row in rows:
        if row["proposed_label"] is not None:
            problems.append("a proposal was written; this tool must never label")
        if not row["evidence"]:
            problems.append(f"{row['id']} carries no evidence")
    if any(r["claim"] == "It is headquartered in Munich." and r["query"] == "Where is it?"
           for r in no_cov):
        problems.append("a query's own gold fact was used as a no_coverage claim")
    if adjudication_records(records, pools, chunks, limit=2, seed=17, top_k=5) != rows:
        problems.append("the build is not deterministic for a fixed seed")
    if verifier_records([{"query": "q", "answer": "a", "evidence": [{"path": "x"}]}], limit=5)[0]["id"] != "ver-0000":
        problems.append("verifier record ids are not positional")
    if verifier_records([{"query": "q", "answer": "", "evidence": []}], limit=5):
        problems.append("an empty answer produced a verifier record")
    return problems


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="labelling_pairs.py")
    ap.add_argument("--mode", choices=("adjudication", "verifier"), default="adjudication")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--scores", default=str(SCORES))
    ap.add_argument("--answers", default=None, help="verifier mode: a report with per-query answers")
    ap.add_argument("--generate", type=int, default=0,
                    help="verifier mode: compose this many answers with the shipped generator")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=5, help="passages shown per claim")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=str(DEFAULT_OUT / "pairs.json"))
    ap.add_argument("--check", action="store_true", help="run the offline invariants and exit")
    return ap.parse_args(argv)


def _generate_answers(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Compose answers with the shipped generator over the first ``--generate`` answerable queries.

    Resumable: each row is appended and flushed to ``<out>.jsonl`` as it is produced, and a re-run
    skips the queries already there. Two hours of generation that a crash can lose is the same
    mistake this repository has recorded three times.
    """
    import os

    # The MultiHop tools all pin the index and the model cache before the engine is built: the
    # default index directory is `indexes/default`, which holds no part of this benchmark, and
    # `~/.cache` is outside the workspace when the file sandbox is active.
    os.environ.setdefault("LAYA_RAG_INDEX_DIR", str(BENCH / "index"))
    os.environ.setdefault("LAYA_RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "beir" / "hf-cache"))

    from laya_rag.compose import Composer
    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    sidecar = Path(args.out).with_suffix(".jsonl")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if sidecar.is_file():
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row.get("query") or "")] = row

    config = EngineConfig()
    composer = Composer(config)
    described = composer.describe()
    if not described.get("available"):
        raise SystemExit(f"the composer is unavailable: {described.get('reason')}")
    engine = RagEngine(config)
    records = answerable_records(Path(args.dataset))[: args.generate]
    made = 0
    handle = sidecar.open("a", encoding="utf-8")
    try:
        for record in records:
            query = str(record["query"])
            if query in done:
                continue
            result = engine.query(query)
            row: dict[str, Any] = {"query": query, "generator": described.get("model")}
            if not result.answerable or not result.passages:
                row["abstained"] = True
            else:
                passages = [
                    (str(p.get("path") or ""), str(p.get("text") or "")) for p in result.passages
                ]
                pieces: list[str] = []
                for event in composer.stream(query, passages, max_tokens=args.max_tokens):
                    if event["type"] == "delta":
                        pieces.append(event["text"])
                row["answer"] = "".join(pieces).strip()
                row["abstained"] = False
                row["evidence"] = [{"path": path, "text": text} for path, text in passages]
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done[query] = row
            made += 1
            if made % 5 == 0:
                print(f"  composed {made} new ({len(done)}/{len(records)} total)", flush=True)
    finally:
        handle.close()
    return [done[str(record["query"])] for record in records if str(record["query"]) in done]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        problems = self_check()
        if problems:
            print("labelling-pairs --check: FAILED")
            for line in problems:
                print(f"  - {line}")
            return 1
        print("ok: labelling-pairs invariants hold (no labels written, evidence attached, deterministic)")
        return 0

    if args.mode == "adjudication":
        records = answerable_indexed(Path(args.dataset))
        payload = json.loads(Path(args.scores).read_text(encoding="utf-8"))
        pools = {qid: row.get("pool") or [] for qid, row in (payload.get("queries") or {}).items()}
        chunks = nr.build_chunks(Path(args.corpus), overlap=200)
        rows = adjudication_records(
            records, pools, chunks, limit=args.limit, seed=args.seed, top_k=args.top_k
        )
    else:
        if args.answers:
            report = json.loads(Path(args.answers).read_text(encoding="utf-8"))
            per_query = report.get("per_query") or {}
            answers = list(per_query.values()) if isinstance(per_query, Mapping) else list(per_query)
        elif args.generate:
            answers = _generate_answers(args)
        else:
            print("labelling-pairs: verifier mode needs --answers or --generate N")
            return 1
        rows = verifier_records(answers, limit=args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[str(row.get("kind"))] = kinds.get(str(row.get("kind")), 0) + 1
    print(f"wrote {out}: {len(rows)} records {kinds} — every label is still the reviewer's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
