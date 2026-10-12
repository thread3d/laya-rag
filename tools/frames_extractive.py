#!/usr/bin/env python
"""The engine's extractive row on FRAMES — §P12's second run.

MultiHop-RAG is multi-hop by construction: its evidence is a handful of news sentences and its gold
answers are long. FRAMES is the other regime — **824 questions whose gold answer is usually a few
words** ("2005", "the University of Tokyo") over **2,509 full Wikipedia articles**, so the same
five-sentence extract has a very different chance of containing the answer. The retrieval adapter
already exists (`tools/frames_eval.py`, gold coverage 0.9977 after redirect resolution) and every
question carries a gold `Answer`, so both halves are scorable without a judge or a model.

Scores, all of them the engine's:

* **retrieval** — recall@k / MRR@10 / nDCG@10 over the gold article set, the same metrics
  `frames_eval` reports, so the two rows are comparable;
* **attribution** — `verify_citations` over the composed answer, deterministic;
* **abstention** — how often nothing in the retrieved evidence qualified;
* **answer nugget recall** — the share of the gold `Answer`'s programmatic nuggets the composed
  answer contains. FRAMES answers are short, so this is a much harder test of an extract than
  MultiHop-RAG's, which is the point of running it.

    make frames-extractive FRAMES_EXTRACTIVE_ARGS="--limit 50"
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
from frames_eval import load_corpus, load_questions, load_title_map, score_ranking  # noqa: E402
from nugget_recall import Evidence, nuggets  # noqa: E402

FRAMES = ROOT / "eval" / "frames"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="frames_extractive.py")
    ap.add_argument("--out", default=str(ROOT / "eval" / "frames-extractive"))
    ap.add_argument("--tsv", default=str(FRAMES / "test.tsv"))
    ap.add_argument("--corpus", default=str(FRAMES / "corpus.jsonl"))
    ap.add_argument("--title-map", default=str(FRAMES / "title-map.json"))
    ap.add_argument("--index-dir", default=None,
                    help="reuse an existing engine index instead of deriving one from --out")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--sparse-model", default="opensearch-project/opensearch-neural-sparse-encoding-v1")
    ap.add_argument("--rerank-model", default=None)
    ap.add_argument("--context-prefix", action="store_true")
    ap.add_argument("--no-chunk-auto", action="store_true")
    ap.add_argument("--chunk-chars", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--retrieve-k", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-sentences", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--merge", type=int, default=0, metavar="N")
    return ap.parse_args(argv)


def score_one(payload: Mapping[str, Any], gold: set[str], answer: str,
              titles_by_position: Sequence[str]) -> dict[str, Any]:
    """One question: retrieval, attribution, abstention and answer nugget recall."""
    from laya_rag.evidence import group_passages_by_document
    from laya_rag.verify import verify_citations

    passages = payload.get("passages") or []
    # The corpus was written by position, so a passage path maps back to the article it came from
    # and then to the title `gold` is stated in.
    ranked: list[str] = []
    for passage in passages:
        stem = Path(str(passage.get("path") or "")).stem
        if stem.isdigit() and int(stem) < len(titles_by_position):
            title = titles_by_position[int(stem)]
            if title not in ranked:
                ranked.append(title)

    grouped = group_passages_by_document(
        (passage.get("path") or "", str(passage.get("text") or "")) for passage in passages
    )
    citation = verify_citations(
        str(payload.get("answer") or ""), [text for _path, text in grouped],
        query=str(payload.get("query") or ""),
    )
    wanted = nuggets(answer) if answer else set()
    answer_evidence = Evidence({"answer": str(payload.get("answer") or "")})
    found = {n for n in wanted if answer_evidence.covers("answer", n)}

    return {
        "answer": str(payload.get("answer") or ""),
        "abstained": bool(payload.get("abstained")),
        "sentences": len(payload.get("sentences") or []),
        "citation_clean": bool(citation["clean"]),
        "citation_claims": int(citation["counts"]["claims"]),
        "citation_uncited": int(citation["counts"]["uncited"]),
        "citation_unfound_quotes": int(citation["counts"]["quotes_not_found"]),
        "nugget_recall": round(len(found) / len(wanted), 4) if wanted else None,
        "nuggets": len(wanted),
        "gold_articles": len(gold),
        **score_ranking(ranked, gold),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The extraction metrics plus the retrieval ones, over the questions that were scored."""
    if not rows:
        return {"queries": 0}
    total = len(rows)
    answered = [row for row in rows if not row["abstained"]]
    nugget_rows = [row for row in rows if row["nugget_recall"] is not None]
    retrieval_keys = [key for key in rows[0] if key.startswith(("recall@", "mrr@", "ndcg@"))]
    return {
        "queries": total,
        "answered": len(answered),
        "abstention_rate": round(sum(1 for r in rows if r["abstained"]) / total, 4),
        "citation_clean_rate": (
            round(sum(1 for r in answered if r["citation_clean"]) / len(answered), 4)
            if answered
            else None
        ),
        "uncited_sentences": sum(r["citation_uncited"] for r in rows),
        "claims": sum(r["citation_claims"] for r in rows),
        "answer_nugget_recall": (
            round(sum(r["nugget_recall"] for r in nugget_rows) / len(nugget_rows), 4)
            if nugget_rows
            else None
        ),
        **{key: round(sum(r[key] for r in rows) / total, 4) for key in retrieval_keys},
    }


def render(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    recall = summary["answer_nugget_recall"]
    recall_text = "n/a" if recall is None else f"{recall:.4f}"
    clean = summary["citation_clean_rate"]
    clean_text = "n/a" if clean is None else f"{clean:.4f}"
    lines = [
        "# The engine's extractive row on FRAMES",
        "",
        f"{summary['queries']} questions over 2,509 Wikipedia articles. Every component is the",
        "engine's: retrieval, the generator-free composer, the deterministic citation check and a",
        "judge-free nugget count against the gold answer itself.",
        "",
        "| metric | value | what it is |",
        "| --- | ---: | --- |",
    ]
    for key in sorted(k for k in summary if k.startswith(("recall@", "mrr@", "ndcg@"))):
        lines.append(f"| {key} | {summary[key]:.4f} | over the gold article set |")
    lines += [
        f"| citation clean | {clean_text} | every claim of every emitted answer cites in range; "
        f"**1.0 by construction — below is a bug** |",
        f"| uncited sentences | {summary['uncited_sentences']} of {summary['claims']} | |",
        f"| abstention | {summary['abstention_rate']:.4f} | nothing in the evidence qualified |",
        f"| answer nugget recall | {recall_text} | share of the gold **answer**'s nuggets the "
        f"answer contains |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("frames_extractive.py")
    shard_index, shard_count = jobkit.parse_shard(args.shard)
    identity = jobkit.index_identity(
        {
            "embed_model": args.embed_model or "engine-default",
            "sparse": args.sparse,
            "sparse_model": args.sparse_model if args.sparse else None,
            "chunk_chars": args.chunk_chars,
            "chunk_overlap": args.chunk_overlap,
            "chunk_auto": not args.no_chunk_auto,
            "context_prefix": args.context_prefix,
        },
        [FRAMES / "corpus.jsonl"],
    )
    fingerprint = jobkit.config_fingerprint(
        tool="frames_extractive",
        embed_model=args.embed_model or "engine-default",
        sparse=args.sparse,
        top_k=args.top_k,
        max_sentences=args.max_sentences,
        min_score=args.min_score,
        limit=args.limit,
        index=identity,
    )
    if args.merge:
        try:
            merged = jobkit.merge_shards(args.out, args.merge, fingerprint=fingerprint,
                                         build=_build_report)
        except (RuntimeError, jobkit.FingerprintMismatch) as exc:
            print(f"merge failed: {exc}", file=sys.stderr)
            return 2
        print(f"merged {args.merge} shard(s) -> {args.out}.json "
              f"({merged['_job']['completed']} questions)")
        return 0

    stem = jobkit.shard_stem(args.out, shard_index, shard_count)
    if args.fresh:
        for suffix in (".jsonl", ".manifest.json"):
            Path(str(stem) + suffix).unlink(missing_ok=True)
    run = jobkit.ResumableRun(str(stem) + ".json", fingerprint=fingerprint,
                              shard=(shard_index, shard_count))

    title_map = load_title_map(Path(args.title_map))
    questions = load_questions(Path(args.tsv), title_map)
    if args.limit:
        questions = questions[: args.limit]
    corpus = load_corpus(Path(args.corpus))
    if not corpus:
        raise SystemExit(f"no corpus at {args.corpus}")
    print(f"{len(questions)} questions, {len(corpus)} articles", flush=True)

    from engine_multihop import configure_engine, index_dir_for

    configure_engine(args)
    if args.index_dir:
        os.environ["LAYA_RAG_INDEX_DIR"] = str(Path(args.index_dir))
    for key in ("LAYA_RAG_RELEVANCE_THRESHOLD", "LAYA_RAG_ANSWERABILITY_THRESHOLD"):
        os.environ.pop(key, None)

    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    index_dir = index_dir_for(args)
    docs_dir = Path(str(args.out)).parent / "frames-engine-corpus"
    titles_by_position = list(corpus)
    write_corpus({str(i): corpus[title] for i, title in enumerate(titles_by_position)}, docs_dir)

    config = EngineConfig()
    engine = RagEngine(config)
    if not engine.has_index():
        started = time.perf_counter()
        report = engine.index_paths([docs_dir], rebuild=True)
        print(f"indexed {report.documents} docs / {report.chunks} chunks in "
              f"{time.perf_counter() - started:.0f}s", flush=True)
    print(f"using the index at {os.environ['LAYA_RAG_INDEX_DIR']} "
          f"({len(engine.list_chunks())} chunks)", flush=True)

    if jobkit.META_KEY not in run:
        run.record(jobkit.META_KEY, {
            "_settings": {**vars(args), "index_identity": identity,
                          "chunks": len(engine.list_chunks())},
        })

    queries = {str(i): question for i, question in enumerate(questions)}
    mine = jobkit.shard_items(list(queries), shard_index, shard_count)
    todo = run.pending(mine)
    print(f"  {len(todo)} of {len(mine)} questions to answer ({len(run) - 1} checkpointed)",
          flush=True)

    started = time.perf_counter()
    for position, key in enumerate(todo, 1):
        question = queries[key]
        payload = engine.query_extractive(
            question["query"],
            max_sentences=args.max_sentences,
            min_score=args.min_score,
            include_evidence=True,
            top_k=args.top_k,
            retrieve_k=args.retrieve_k,
            max_passage_chars=None,
        )
        payload["query"] = question["query"]
        run.record(key, score_one(payload, question["gold"], question["answer"],
                                  titles_by_position))
        if position % 25 == 0:
            run.touch(total=len(mine))
            print(f"  {position}/{len(todo)} in {time.perf_counter() - started:.0f}s", flush=True)

    per_query = {key: value for key, value in run.values().items() if key != jobkit.META_KEY}
    payload = {
        "_settings": {**vars(args), "index_identity": identity},
        "questions": len(per_query),
        "seconds": round(time.perf_counter() - started, 1),
        "summary": summarize(list(per_query.values())),
        "per_query": per_query,
    }
    run.finish(payload, total=len(mine))
    print(render(payload))
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
        "questions": len(per_query),
        "summary": summarize(list(per_query.values())),
        "per_query": per_query,
    }


if __name__ == "__main__":
    sys.exit(main())
