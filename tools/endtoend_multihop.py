"""The end-to-end row on MultiHop-RAG, with a labelled generator — ROADMAP Phase 3.2/3.3.

`BENCHMARKS.md` says the project has **no end-to-end ranking**, because no end-to-end benchmark has
been run to completion. This produces one, under the project's own attribution rule: the answer
accuracy is `engine + <model>` and is labelled with the model that wrote the prose, while the
*engine's* half — what it retrieved, whether it chose to answer, and whether the cited spans exist —
is scored separately.

Protocol, deliberately split so the two halves cannot be confused:

* the engine runs at its **shipped defaults** (relevance and answerability gates active), so
  abstention is a real decision rather than a threshold switched off for measurement;
* if the engine abstains, no prose is written and the query is recorded as `abstained`;
* otherwise the configured generator writes an answer from the retrieved evidence;
* **EM and F1** are SQuAD-style string scores against the benchmark's gold `answer` — deterministic,
  no judge, and labelled as string matching rather than the benchmark's own assessment;
* abstention is scored on the benchmark's 301 `null_query` records, whose gold answer is literally
  "Insufficient information".

    .venv/bin/python tools/endtoend_multihop.py --answerable 30 --null 15
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
# Project-local model cache, following tools/beir.py: the encoder the index was built with is
# already cached under eval/beir/hf-cache, and the global cache is neither written to nor relied on.
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "beir" / "hf-cache"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit, jobkit  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
DEFAULT_INDEX = ROOT_BENCH / "index" / "engine-hybrid-400"
_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """SQuAD's normalization: lowercase, drop punctuation, articles and extra whitespace."""
    lowered = str(text or "").lower()
    stripped = "".join(" " if ch in string.punctuation else ch for ch in lowered)
    words = [word for word in stripped.split() if word not in _ARTICLES]
    return " ".join(words)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction: str, gold: str) -> float:
    """Token-overlap F1, the standard companion to EM for short answers."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    shared = sum(common.values())
    if shared == 0:
        return 0.0
    precision = shared / len(pred_tokens)
    recall = shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_gold(prediction: str, gold: str) -> float:
    """Whether the normalized gold answer appears in the prose.

    EM and F1 compare token multisets, which a *prose* answer almost never matches against a short
    entity gold even when it is correct ("Sam Bankman-Fried was the founder…" against
    "Sam Bankman-Fried"). Containment is the deterministic proxy for a reader's judgement, and it
    is labelled as a proxy, not as the benchmark's own assessment.
    """
    gold_normalized = normalize_answer(gold)
    if not gold_normalized:
        return 0.0
    return float(gold_normalized in normalize_answer(prediction))


def summarize(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Answer quality on answerable records, and rejection on the null records, kept apart."""
    answerable = [row for row in rows.values() if not row.get("is_null")]
    nulls = [row for row in rows.values() if row.get("is_null")]
    answered = [row for row in answerable if not row.get("abstained") and row.get("answer")]
    summary: dict[str, Any] = {
        "answerable": len(answerable),
        "null": len(nulls),
        "answered": len(answered),
        "abstained_on_answerable": sum(1 for row in answerable if row.get("abstained")),
        "rejected_on_null": sum(1 for row in nulls if row.get("abstained")),
        "errors": sum(1 for row in rows.values() if row.get("error")),
    }
    if nulls:
        summary["rejection_rate"] = round(summary["rejected_on_null"] / len(nulls), 4)
    if answerable:
        summary["abstention_rate"] = round(summary["abstained_on_answerable"] / len(answerable), 4)
    if answered:
        summary["exact_match"] = round(sum(exact_match(r["answer"], r["gold"]) for r in answered)
                                       / len(answered), 4)
        summary["f1"] = round(sum(f1(r["answer"], r["gold"]) for r in answered) / len(answered), 4)
        summary["gold_contained"] = round(
            sum(contains_gold(r["answer"], r["gold"]) for r in answered) / len(answered), 4)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="endtoend_multihop.py")
    ap.add_argument("--records", default=str(ROOT_BENCH / "MultiHopRAG.json"))
    ap.add_argument("--index-dir", default=str(DEFAULT_INDEX),
                    help="an existing engine index; the corpus is not re-indexed")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5",
                    help="must match the encoder the index was built with")
    ap.add_argument("--compose-model", default=None, help="default: the configured model")
    ap.add_argument("--answerable", type=int, default=30)
    ap.add_argument("--null", type=int, default=15, dest="null_count")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT_BENCH / "endtoend"))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--merge", type=int, default=0, metavar="N")
    return ap.parse_args(argv)


def _fingerprint(args: argparse.Namespace) -> str:
    return jobkit.config_fingerprint(
        tool="endtoend_multihop", index=Path(args.index_dir).name,
        embed_model=args.embed_model, compose_model=args.compose_model,
        answerable=args.answerable, null=args.null_count, top_k=args.top_k,
        max_tokens=args.max_tokens,
    )


def _build_report(rows: dict) -> dict:
    if jobkit.META_KEY not in rows:
        raise RuntimeError("no shard checkpoints found — run the shards before --merge")
    meta = rows.pop(jobkit.META_KEY)
    return {**meta, "per_query": dict(rows)}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    shard_index, shard_count = jobkit.parse_shard(args.shard)
    fingerprint = _fingerprint(args)
    if args.merge:
        try:
            merged = jobkit.merge_shards(args.out, args.merge, fingerprint=fingerprint,
                                         build=_build_report)
        except (RuntimeError, jobkit.FingerprintMismatch) as exc:
            print(f"merge failed: {exc}", file=sys.stderr)
            return 2
        print(f"merged {args.merge} shard(s) -> {args.out}.json")
        return 0

    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    answerable = [r for r in records if r["question_type"] != "null_query"][: args.answerable]
    nulls = [r for r in records if r["question_type"] == "null_query"][: args.null_count]
    selected = answerable + nulls

    os.environ["LAYA_RAG_INDEX_DIR"] = str(Path(args.index_dir))
    os.environ["LAYA_RAG_EMBED_MODEL"] = args.embed_model
    if args.compose_model:
        os.environ["LAYA_RAG_COMPOSE_MODEL"] = args.compose_model

    from laya_rag.compose import Composer
    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    config = EngineConfig()
    engine = RagEngine(config)
    if not engine.has_index():
        raise SystemExit(f"no index at {args.index_dir}; build it with `make engine-multihop` first")
    composer = Composer(config)
    described = composer.describe()
    if not described.get("available"):
        raise SystemExit(f"the composer is unavailable: {described.get('reason')}")
    print(f"retrieval index {Path(args.index_dir).name}, generator "
          f"{described['model']} at {described['base_url']}", flush=True)

    stem = jobkit.shard_stem(args.out, shard_index, shard_count)
    if args.fresh:
        for suffix in (".jsonl", ".manifest.json"):
            Path(str(stem) + suffix).unlink(missing_ok=True)
    run = jobkit.ResumableRun(str(stem) + ".json", fingerprint=fingerprint,
                              shard=(shard_index, shard_count))
    if jobkit.META_KEY not in run:
        run.record(jobkit.META_KEY, {
            "_settings": {**vars(args), "records": len(selected),
                          "generator_model": described["model"],
                          "generator_base_url": described["base_url"],
                          "thresholds": {"relevance": config.relevance_threshold,
                                         "answerability": config.answerability_threshold}},
            "query_order": [str(i) for i in range(len(selected))],
        })

    mine = jobkit.shard_items(list(range(len(selected))), shard_index, shard_count)
    todo = run.pending(mine, key=str)
    print(f"  {len(todo)} of {len(mine)} queries to answer ({len(run) - 1} checkpointed)",
          flush=True)

    started = time.perf_counter()
    for position, item in enumerate(todo, 1):
        key = str(item)
        record = selected[int(key)]
        row: dict[str, Any] = {
            "query": record["query"], "gold": record["answer"],
            "is_null": record["question_type"] == "null_query",
        }
        query_started = time.perf_counter()
        result = engine.query(record["query"], top_k=args.top_k, max_passage_chars=None)
        row["answerable"] = result.answerable
        row["answerability"] = result.answerability
        row["kept"] = result.kept
        row["abstained"] = not result.answerable
        if result.answerable and result.passages:
            passages = [(str(p.get("path") or ""), str(p.get("text") or ""))
                        for p in result.passages]
            chunks: list[str] = []
            error = ""
            for event in composer.stream(record["query"], passages, max_tokens=args.max_tokens):
                if event["type"] == "delta":
                    chunks.append(event["text"])
                elif event["type"] == "error":
                    error = str(event.get("message") or "composer error")
            row["answer"] = "".join(chunks).strip()
            if error:
                row["error"] = error
        row["seconds"] = round(time.perf_counter() - query_started, 2)
        run.record(key, row)
        if position % 5 == 0:
            run.touch(total=len(mine))
            print(f"    {position}/{len(todo)} in {time.perf_counter() - started:.0f}s", flush=True)

    shared = run.get(jobkit.META_KEY)
    per_query = {key: run.get(key) for key in shared["query_order"] if run.get(key) is not None}
    report = {
        "_settings": {**shared["_settings"], "seconds": round(time.perf_counter() - started, 1)},
        "summary": summarize(per_query),
        "per_query": per_query,
    }
    run.finish(report, total=len(mine))
    s = report["summary"]
    print(f"\nanswerable {s['answerable']}: abstained {s['abstained_on_answerable']}, "
          f"answered {s['answered']}, EM {s.get('exact_match')}, F1 {s.get('f1')}, "
          f"gold contained {s.get('gold_contained')}")
    print(f"null {s['null']}: rejected {s['rejected_on_null']} "
          f"({s.get('rejection_rate')}), errors {s['errors']}")
    print("wrote", str(stem) + ".json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
