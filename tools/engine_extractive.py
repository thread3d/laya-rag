#!/usr/bin/env python
"""The first end-to-end row whose every component is the engine's.

`SCORE-LIFT.md` §P12: `compose_extractive` emits only sentences that are verbatim substrings of a
retrieved passage, so an end-to-end score over its output has **no generator confound at all** —
the strongest form of the claim this project can make — and it did not exist, because every
end-to-end target was priced at the *LLM* path's ~36 s/query. Retrieval in the same pilot is
~3 s/query, so the extractive row is cheap where the generated one is not.

This runs the shipped engine over MultiHop-RAG's answerable records, composes an answer from the
passages it returned, and scores four things that are all attributable to the engine:

* **retrieval** — the benchmark's own Hits@10 / Hits@4 / MAP@10 / MRR@10, so the row can be placed
  beside the retrieval-only runs on the same queries;
* **attribution** — `verify_citations` over the composed answer, deterministically and with no
  judge. Every emitted sentence is a substring of a returned passage and carries its `[n]`, the
  path and the chunk id, so a citation score below 1.0 is a **bug**, not a result;
* **abstention** — how often nothing in the retrieved evidence qualified, on queries that *have*
  gold evidence. This is the number an extractive-by-construction system has to own: it can only
  answer with the corpus's own words;
* **answer nugget recall** — the fraction of each gold fact's programmatically-derived nuggets
  (numbers, dates, entities, key terms) that the composed answer actually contains. This is the
  **completeness** side, measured without a judge, and it is expected to be much lower than the
  evidence-side claim recall in `tools/nugget_recall.py`: a five-sentence extract of a retrieved
  passage is not a written answer, and that gap is the finding.

    make engine-extractive ENGINE_EXTRACTIVE_ARGS="--limit 400 --sample-seed 17"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit, jobkit  # noqa: E402
from engine_beir import write_corpus  # noqa: E402
from engine_multihop import (  # noqa: E402  (the retrieval row's own helpers, so the two agree)
    _ingest_settings,
    answerable_items,
    configure_engine,
    gold_facts,
    index_dir_for,
    select_items,
)
from engine_multihop import METRIC_KEYS  # noqa: E402  (the retrieval row's keys)
from multihop_official import official_metrics  # noqa: E402
from nugget_recall import Evidence, nuggets  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="engine_extractive.py")
    ap.add_argument("--out", default=str(ROOT_BENCH / "engine-extractive"))
    ap.add_argument("--embed-model", default=None,
                    help="bi-encoder for the engine's dense channel; omit for the shipped default")
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--sparse-model", default="opensearch-project/opensearch-neural-sparse-encoding-v1")
    ap.add_argument("--rerank-model", default=None)
    ap.add_argument("--context-prefix", action="store_true")
    ap.add_argument("--no-chunk-auto", action="store_true")
    ap.add_argument("--chunk-chars", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--retrieve-k", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=10,
                    help="passages the answer may quote from")
    ap.add_argument("--max-sentences", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop documents below this relevance before quoting")
    ap.add_argument("--index-dir", default=None,
                    help="reuse an existing engine index instead of deriving one from --out")
    ap.add_argument("--zero-thresholds", action="store_true",
                    help="zero the relevance and answerability gates, so the row is comparable "
                         "to the retrieval-only runs rather than to the shipped engine")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None, help="cap the QUERIES only")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="reproduce the harness's seeded subset at the same --limit")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--merge", type=int, default=0, metavar="N")
    return ap.parse_args(argv)


def score_one(payload: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    """Everything one query contributes, computed here so the report cannot re-derive it."""
    from laya_rag.evidence import group_passages_by_document
    from laya_rag.verify import verify_citations

    passages = payload.get("passages") or []
    # The composer merges the passages per document and numbers them in first-appearance order;
    # `verify_citations` has to see exactly that, or the `[n]` it checks is not the `[n]` emitted.
    grouped = group_passages_by_document(
        (passage.get("path") or "", str(passage.get("text") or "")) for passage in passages
    )
    citation = verify_citations(
        str(payload.get("answer") or ""), [text for _path, text in grouped],
        query=str(payload.get("query") or ""),
    )
    facts = gold_facts(item)
    wanted: set[tuple[str, str]] = set()
    for fact in facts:
        wanted |= nuggets(fact)
    # The answer is the evidence here: "how much of the gold fact is in what we said", not
    # "how much is in what we retrieved" — that is `tools/nugget_recall.py`'s question.
    answer_evidence = Evidence({"answer": str(payload.get("answer") or "")})
    found = {n for n in wanted if answer_evidence.covers("answer", n)}
    # The benchmark's own metric scores chunk *text* against the gold fact by normalised
    # substring, which is what `engine_multihop` feeds it; a document-id ranking would score zero
    # and look like a retrieval failure.
    texts = [str(passage.get("text") or "") for passage in passages[: 10]]
    metrics, _matches = official_metrics([texts], [facts])
    return {
        "answer": str(payload.get("answer") or ""),
        "abstained": bool(payload.get("abstained")),
        "sentences": len(payload.get("sentences") or []),
        "answerable": bool(payload.get("answerable")),
        "answerability": payload.get("answerability"),
        "citation_clean": bool(citation["clean"]),
        "citation_claims": int(citation["counts"]["claims"]),
        "citation_uncited": int(citation["counts"]["uncited"]),
        "citation_invalid": int(citation["counts"]["invalid_citations"]),
        "citation_unfound_quotes": int(citation["counts"]["quotes_not_found"]),
        "nugget_recall": round(len(found) / len(wanted), 4) if wanted else None,
        "nuggets": len(wanted),
        **{key: float(metrics[key]) for key in METRIC_KEYS},
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The macro row, over the queries that were scored.

    ``citation_clean_rate`` is over the **non-abstained** answers, and ``citation_clean_all`` over
    every query: an abstention emits the fixed sentence "The retrieved evidence does not contain a
    sentence that answers the question.", which the project's own checker counts as one uncited
    claim. That is a real interaction between the two surfaces and it is reported, not smoothed
    over — but attribution is a property of an *answer*, so the rate that measures it is the one
    over answers that were actually emitted.
    """
    total = len(rows)
    if not total:
        return {"queries": 0}
    nugget_rows = [row for row in rows if row["nugget_recall"] is not None]
    answered = [row for row in rows if not row["abstained"]]
    return {
        "queries": total,
        "answered": len(answered),
        "abstention_rate": round(sum(1 for r in rows if r["abstained"]) / total, 4),
        "citation_clean_rate": (
            round(sum(1 for r in answered if r["citation_clean"]) / len(answered), 4)
            if answered
            else None
        ),
        "citation_clean_all": round(sum(1 for r in rows if r["citation_clean"]) / total, 4),
        "uncited_sentences": sum(r["citation_uncited"] for r in rows),
        "claims": sum(r["citation_claims"] for r in rows),
        "answer_nugget_recall": (
            round(sum(r["nugget_recall"] for r in nugget_rows) / len(nugget_rows), 4)
            if nugget_rows
            else None
        ),
        **{
            key: round(sum(r[key] for r in rows) / total, 4) for key in METRIC_KEYS
        },
    }


def render(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    recall = summary["answer_nugget_recall"]
    recall_text = "n/a" if recall is None else f"{recall:.4f}"
    clean = summary["citation_clean_rate"]
    clean_text = "n/a" if clean is None else f"{clean:.4f}"
    lines = [
        "# The engine's extractive row, end to end",
        "",
        f"{summary['queries']} answerable MultiHop-RAG queries; every component below is the",
        "engine's — retrieval, the composer, the citation check and the nugget count. No model,",
        "no judge and no API are in the path.",
        "",
        "| metric | value | what it is |",
        "| --- | ---: | --- |",
        f"| Hits@10 | {summary['Hits@10']:.4f} | the benchmark's own retrieval metric |",
        f"| Hits@4 | {summary['Hits@4']:.4f} | |",
        f"| MAP@10 | {summary['MAP@10']:.4f} | |",
        f"| MRR@10 | {summary['MRR@10']:.4f} | |",
        f"| citation clean, answered | {clean_text} | fraction of emitted answers whose every "
        f"claim cites in-range evidence; **1.0 by construction — below is a bug** |",
        f"| citation clean, all | {summary['citation_clean_all']:.4f} | over every query; the "
        f"abstention sentence is an uncited claim to the checker |",
        f"| uncited sentences | {summary['uncited_sentences']} of {summary['claims']} | |",
        f"| abstention | {summary['abstention_rate']:.4f} | nothing in the retrieved evidence "
        f"qualified for quoting |",
        f"| answer nugget recall | {recall_text} | "
        f"share of each gold fact's nuggets the **answer** contains |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("engine_extractive.py")
    shard_index, shard_count = jobkit.parse_shard(args.shard)
    identity = jobkit.index_identity(
        _ingest_settings(args), [ROOT_BENCH / "corpus.json", ROOT_BENCH / "MultiHopRAG.json"]
    )
    fingerprint = jobkit.config_fingerprint(
        tool="engine_extractive",
        embed_model=args.embed_model or "engine-default",
        sparse=args.sparse,
        rerank_model=args.rerank_model,
        context_prefix=args.context_prefix,
        chunk_chars=args.chunk_chars,
        chunk_auto=not args.no_chunk_auto,
        top_k=args.top_k,
        max_sentences=args.max_sentences,
        min_score=args.min_score,
        limit=args.limit,
        sample_seed=args.sample_seed,
        **({"index": identity} if identity is not None else {}),
    )
    index_dir = index_dir_for(args)

    if args.merge:
        try:
            merged = jobkit.merge_shards(args.out, args.merge, fingerprint=fingerprint,
                                         build=_build_report)
        except (RuntimeError, jobkit.FingerprintMismatch) as exc:
            print(f"merge failed: {exc}", file=sys.stderr)
            return 2
        print(f"merged {args.merge} shard(s) -> {args.out}.json "
              f"({merged['_job']['completed']} queries)")
        return 0

    stem = jobkit.shard_stem(args.out, shard_index, shard_count)
    if args.fresh:
        for suffix in (".jsonl", ".manifest.json"):
            Path(str(stem) + suffix).unlink(missing_ok=True)
    run = jobkit.ResumableRun(str(stem) + ".json", fingerprint=fingerprint,
                              shard=(shard_index, shard_count))

    raw = json.load(open(ROOT_BENCH / "corpus.json"))
    items_all = json.load(open(ROOT_BENCH / "MultiHopRAG.json"))
    items = select_items(answerable_items(items_all), args.limit, args.sample_seed)

    configure_engine(args)
    if args.index_dir:
        # Reuse the index a retrieval row was measured against, so the end-to-end row is on the
        # same corpus rather than a second one nobody can compare it to.
        os.environ["LAYA_RAG_INDEX_DIR"] = str(Path(args.index_dir))
    if not args.zero_thresholds:
        # The end-to-end row is the engine **as deployed**: the relevance and answerability gates
        # decide, and an abstention because the gate emptied the evidence is part of the number.
        # `--zero-thresholds` is what the retrieval-only runs do, and makes the two comparable.
        for key in ("LAYA_RAG_RELEVANCE_THRESHOLD", "LAYA_RAG_ANSWERABILITY_THRESHOLD"):
            os.environ.pop(key, None)
    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    docs_dir = ROOT_BENCH / "engine-corpus"
    doc_ids = write_corpus({a["url"]: a["title"] + " " + a["body"] for a in raw}, docs_dir)
    item_by_index = {i: x for i, x in items}
    queries = {str(i): x["query"] for i, x in items}

    config = EngineConfig()
    engine = RagEngine(config)
    if not engine.has_index():
        print(f"no index at {index_dir}; run `make engine-multihop` first", file=sys.stderr)
        return 2
    effective = Path(os.environ["LAYA_RAG_INDEX_DIR"])
    print(f"using the index at {effective} ({len(engine.list_chunks())} chunks, "
          f"dense={config.resolved_dense_weight != 0.0}, sparse={config.sparse_enabled}, "
          f"gates {'zeroed' if args.zero_thresholds else 'as shipped'})", flush=True)

    if jobkit.META_KEY not in run:
        run.record(jobkit.META_KEY, {
            "_settings": {**vars(args), "index_dir": str(index_dir), "index_identity": identity},
            "queries": queries,
        })

    mine = jobkit.shard_items(list(queries), shard_index, shard_count)
    todo = run.pending(mine)
    print(f"  {len(todo)} of {len(mine)} queries to answer ({len(run) - 1} checkpointed)",
          flush=True)

    started = time.perf_counter()
    for position, key in enumerate(todo, 1):
        item = item_by_index[int(key)]
        payload = engine.query_extractive(
            queries[key],
            max_sentences=args.max_sentences,
            min_score=args.min_score,
            include_evidence=True,
            top_k=args.top_k,
            retrieve_k=args.retrieve_k,
            # The retrieval rows ask for the whole chunk: the default truncates a passage at
            # `max_passage_chars`, which can cut a gold fact out of the middle of the text and
            # make this row's Hits@10 look like a retrieval failure rather than a quoting artefact.
            max_passage_chars=None,
        )
        payload["query"] = queries[key]
        run.record(key, score_one(payload, item))
        if position % 25 == 0:
            run.touch(total=len(mine))
            print(f"  {position}/{len(todo)} in {time.perf_counter() - started:.0f}s", flush=True)
    seconds = time.perf_counter() - started

    mine_rows = [value for key, value in run.values().items() if key != jobkit.META_KEY]
    run.finish({
        "_settings": {**vars(args), "index_dir": str(index_dir), "index_identity": identity},
        "documents": len(raw),
        "queries": len(mine_rows),
        "seconds": round(seconds, 1),
        "summary": summarize(mine_rows),
        "per_query": {key: value for key, value in run.values().items()
                      if key != jobkit.META_KEY},
    }, total=len(mine))
    print(render(json.loads(Path(str(stem) + ".json").read_text(encoding="utf-8"))))
    print("wrote", str(stem) + ".json")
    if shard_count > 1:
        print(f"  run all {shard_count} shards, then: --merge {shard_count}")
    return 0


def _build_report(rows: dict) -> dict:
    if jobkit.META_KEY not in rows:
        raise RuntimeError("no shard checkpoints found — run the shards before --merge")
    meta = rows.pop(jobkit.META_KEY)
    per_query = dict(rows)
    return {
        "_settings": meta["_settings"],
        "queries": len(per_query),
        "summary": summarize(list(per_query.values())),
        "per_query": per_query,
    }


if __name__ == "__main__":
    sys.exit(main())
