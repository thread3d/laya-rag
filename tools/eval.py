#!/usr/bin/env python
"""Quality evaluation for the Laya RAG engine.

Answers "how good is it?" on a labelled query set, which the unit suite deliberately
cannot: candidate recall for each retrieval channel, the rank Laya's re-ranker gives
the gold passage, the relevance margin over the best negative, whether the shipped
thresholds are defensible, and whether the two calibrated decisions are calibrated.

    .venv/bin/python tools/eval.py --corpus planted
    .venv/bin/python tools/eval.py --corpus docs --baseline eval/baseline.json

Writes ``eval/<corpus>.json`` and ``eval/<corpus>.md``. The run needs the real Laya
checkpoints; ``--stub`` swaps in the hash embedder for a plumbing-only smoke run.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import evalkit  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.engine import RagEngine  # noqa: E402
from laya_rag.model import LayaModel  # noqa: E402
from laya_rag.rerank import ScoredPassage  # noqa: E402
from tests.support import StubModel  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval.py", description="Evaluate Laya RAG retrieval quality")
    parser.add_argument("--corpus", default="planted", help="planted, docs, or a directory path")
    parser.add_argument("--queries", default=None, help="JSONL query set to override the corpus default")
    parser.add_argument("--out", default=None, help="report prefix (default eval/<corpus>)")
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--rerank-k", type=int, default=5)
    parser.add_argument("--relevance-threshold", type=float, default=None)
    parser.add_argument("--answerability-threshold", type=float, default=None)
    parser.add_argument("--rebuild", action="store_true", help="re-embed the corpus even if cached")
    parser.add_argument("--stub", action="store_true", help="use the hash embedder (no checkpoints)")
    parser.add_argument("--baseline", default=None, help="report JSON to diff the quality metrics against")
    parser.add_argument("--device", default=None, help="override LAYA_RAG_DEVICE for this run")
    return parser.parse_args(argv)


def resolve_corpus(name: str) -> tuple[list[Path], list[evalkit.EvalQuery], str]:
    """Return ``(roots, queries, corpus_label)`` for a corpus name."""
    if name == "planted":
        directory = ROOT / "eval" / "corpus-planted"
        for relative, text in evalkit.planted_corpus().items():
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.read_text(encoding="utf-8") != text:
                target.write_text(text, encoding="utf-8")
        return [directory], evalkit.planted_queries(), "planted"
    if name == "docs":
        return [ROOT / "vendor" / "laya" / "docs"], evalkit.docs_queries(), "docs"
    path = Path(name)
    if not path.exists():
        raise SystemExit(f"corpus {name!r} is neither 'planted', 'docs' nor an existing path")
    return [path], [], path.name


def score_query(
    engine: RagEngine,
    query: evalkit.EvalQuery,
    retrieve_k: int,
    rerank_k: int,
    relevance_threshold: float,
    answerability_threshold: float,
) -> tuple[dict[str, Any], list[tuple[float, int]], list[tuple[float, int]]]:
    """Run one query through the real retrieval and decision stages.

    Returns the per-query row plus the ``(probability, label)`` pairs the two
    calibrated decisions produced, for the calibration and sweep aggregates.
    """
    index = engine.index
    lexical = engine.lexical

    query_vector = engine.model.embed_one(query.query)
    dense = index.search(query_vector, k=retrieve_k)
    lexical_hits = lexical.search(query.query, k=retrieve_k) if lexical else []
    hybrid = engine._fuse(index, dense, lexical_hits, limit=retrieve_k)

    dense_rank = evalkit.first_match_rank([hit.chunk.path for hit in dense], query.gold)
    hybrid_rank = evalkit.first_match_rank([hit.chunk.path for hit in hybrid], query.gold)

    texts = [hit.chunk.text for hit in hybrid]
    scores, _ = engine.reranker.score_relevance(query.query, texts)
    scored = [
        ScoredPassage(
            text=hit.chunk.text,
            path=hit.chunk.path,
            chunk_id=hit.chunk.chunk_id,
            recall_score=hit.score,
            relevance=score,
            index=hit.chunk.index,
            start_char=hit.chunk.start_char,
            end_char=hit.chunk.end_char,
            channels=hit.channels,
        )
        for hit, score in zip(hybrid, scores)
    ]
    scored.sort(key=lambda passage: (-passage.relevance, -passage.recall_score))

    relevance_pairs = [
        (passage.relevance, 1 if query.matches(passage.path) else 0) for passage in scored
    ]
    gold_scores = [passage.relevance for passage in scored if query.matches(passage.path)]
    negative_scores = [passage.relevance for passage in scored if not query.matches(passage.path)]
    best_gold = max(gold_scores) if gold_scores else None
    best_negative = max(negative_scores) if negative_scores else None
    margin = None if best_gold is None or best_negative is None else best_gold - best_negative

    # The same filter-and-gate `LayaReranker.rerank` performs, through its public parts.
    above = [passage for passage in scored if passage.relevance >= relevance_threshold]
    kept = above[:rerank_k]
    if kept:
        answerability, _ = engine.reranker.check_answerability(query.query, kept)
    else:
        answerability = 0.0

    row = {
        "id": query.id,
        "query": query.query,
        "category": query.category,
        "answerable_label": query.answerable,
        "dense_rank": dense_rank,
        "hybrid_rank": hybrid_rank,
        "rerank_rank": evalkit.first_match_rank([p.path for p in scored], query.gold),
        "best_gold_relevance": best_gold,
        "best_negative_relevance": best_negative,
        "margin": margin,
        "candidates": len(hybrid),
        "above_threshold": len(above),
        "kept": len(kept),
        "measured": len(kept) > 0,
        "answerability": answerability,
        "answerable": answerability >= answerability_threshold,
        "evidence_chars": sum(len(p.text) for p in kept),
    }
    return row, relevance_pairs, [(answerability, 1 if query.answerable else 0)]


def index_matches(engine: RagEngine, model: Any) -> bool:
    """Whether a cached index was built with the embedder this run will use.

    The stub embedder is a different width from the checkpoint, so reusing its cache
    for a real run fails deep inside search with a dimension error. Catch it here and
    rebuild instead.
    """
    try:
        existing = engine.index
        return len(existing) == 0 or existing.dim == int(model.dim)
    except Exception:  # noqa: BLE001 - any unreadable cache is a reason to rebuild
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots, queries, label = resolve_corpus(args.corpus)
    if args.queries:
        queries = evalkit.parse_queries(Path(args.queries).read_text(encoding="utf-8"))
    if not queries:
        raise SystemExit("no query set: add --queries <file.jsonl> for a custom corpus")

    config = EngineConfig(index_dir=ROOT / "eval" / f".index-{label}{'-stub' if args.stub else ''}")
    if args.device:
        config.device = args.device
    relevance_threshold = (
        config.relevance_threshold if args.relevance_threshold is None else args.relevance_threshold
    )
    answerability_threshold = (
        config.answerability_threshold
        if args.answerability_threshold is None
        else args.answerability_threshold
    )

    model: Any = StubModel() if args.stub else LayaModel(config)
    engine = RagEngine(config, model=model)

    started = time.perf_counter()
    if args.rebuild or not engine.has_index() or not index_matches(engine, model):
        report = engine.index_paths(roots, rebuild=True)
        indexed = report.as_dict()
    else:
        indexed = {"cached": True, **engine.index.stats()}
    index_seconds = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    relevance_pairs: list[tuple[float, int]] = []
    answerability_pairs: list[tuple[float, int]] = []
    for query in queries:
        row, rel_pairs, ans_pairs = score_query(
            engine, query, args.retrieve_k, args.rerank_k, relevance_threshold, answerability_threshold
        )
        rows.append(row)
        relevance_pairs.extend(rel_pairs)
        answerability_pairs.extend(ans_pairs)

    labelled = [row for row in rows if row["answerable_label"]]
    unlabelled = [row for row in rows if not row["answerable_label"]]

    categories = sorted({row["category"] for row in rows})
    per_category = [
        {
            "category": category,
            "queries": sum(1 for row in rows if row["category"] == category),
            **{
                f"hybrid_{key}": value
                for key, value in evalkit.rank_summary(
                    [row["hybrid_rank"] for row in rows if row["category"] == category], args.retrieve_k
                ).items()
                if key in {"recall", "mrr"}
            },
        }
        for category in categories
    ]

    answerable_rows = labelled
    unanswerable_rows = unlabelled
    # A query whose relevance gate kept nothing short-circuits to 0.0 without Laya ever
    # being asked the answerability question. Pooling those zeros with measured
    # probabilities would report the gate's design as a calibration failure, so the
    # headline ECE uses only the queries the gate actually reached. The all-in figure
    # is kept beside it, because that is what an end-to-end threshold sees.
    measured_rows = [row for row in rows if row["measured"]]
    measured_answerable = [row for row in measured_rows if row["answerable_label"]]
    measured_unanswerable = [row for row in measured_rows if not row["answerable_label"]]
    short_circuit_answerable = [
        row for row in answerable_rows if not row["measured"]
    ]
    summary = {
        "candidate_recall": {
            "dense": evalkit.rank_summary([row["dense_rank"] for row in labelled], args.retrieve_k),
            "hybrid": evalkit.rank_summary([row["hybrid_rank"] for row in labelled], args.retrieve_k),
        },
        "rerank": evalkit.rank_summary([row["rerank_rank"] for row in labelled], args.rerank_k),
        "per_category": per_category,
        "margin": evalkit.margin_summary([row["margin"] for row in labelled]),
        "relevance": {
            "pairs": len(relevance_pairs),
            "positives": sum(label for _, label in relevance_pairs),
            "ece": evalkit.expected_calibration_error(
                [p for p, _ in relevance_pairs], [y for _, y in relevance_pairs]
            ),
            "bins": evalkit.reliability_bins(
                [p for p, _ in relevance_pairs], [y for _, y in relevance_pairs]
            ),
            "sweep": evalkit.threshold_sweep(
                [p for p, _ in relevance_pairs], [y for _, y in relevance_pairs]
            ),
        },
        "answerability": {
            "measured": len(measured_rows),
            "short_circuited": len(rows) - len(measured_rows),
            "ece_measured": evalkit.expected_calibration_error(
                [row["answerability"] for row in measured_rows],
                [1 if row["answerable_label"] else 0 for row in measured_rows],
            ),
            "ece_all_queries": evalkit.expected_calibration_error(
                [p for p, _ in answerability_pairs], [y for _, y in answerability_pairs]
            ),
            "bins_measured": evalkit.reliability_bins(
                [row["answerability"] for row in measured_rows],
                [1 if row["answerable_label"] else 0 for row in measured_rows],
            ),
            "sweep_all_queries": evalkit.threshold_sweep(
                [p for p, _ in answerability_pairs], [y for _, y in answerability_pairs]
            ),
            "mean_on_answerable": _mean([row["answerability"] for row in measured_answerable]),
            "mean_on_unanswerable": _mean([row["answerability"] for row in measured_unanswerable]),
            "gate_measured_answerable": len(measured_answerable),
            "gate_measured_unanswerable": len(measured_unanswerable),
            "gate_refused_answerable": sum(1 for row in measured_answerable if not row["answerable"]),
            "abstained_correctly": sum(1 for row in unanswerable_rows if not row["answerable"]),
            "unanswerable_total": len(unanswerable_rows),
            "false_abstention": sum(1 for row in answerable_rows if not row["answerable"]),
            "false_abstention_from_relevance": len(short_circuit_answerable),
            "answerable_total": len(answerable_rows),
        },
    }

    payload = {
        "corpus": {
            "label": label,
            "roots": [str(root) for root in roots],
            "indexed": indexed,
            "index_seconds": round(index_seconds, 3),
        },
        "config": {
            "checkpoint": config.checkpoint,
            "device": config.device,
            "retrieve_k": args.retrieve_k,
            "rerank_k": args.rerank_k,
            "relevance_threshold": relevance_threshold,
            "answerability_threshold": answerability_threshold,
            "max_len": config.max_len,
            "chunk_chars": config.chunk_chars,
            "stub": args.stub,
        },
        "queries": rows,
        "summary": summary,
    }

    out_prefix = Path(args.out) if args.out else ROOT / "eval" / label
    evalkit.write_json(out_prefix.with_suffix(".json"), payload)
    evalkit.write_text(out_prefix.with_suffix(".md"), render(payload))

    print(render_console(payload))
    print(f"\nwrote {out_prefix.with_suffix('.json')} and {out_prefix.with_suffix('.md')}")

    if args.baseline:
        differences = evalkit.compare_reports(
            evalkit.read_json(args.baseline)["summary"], payload["summary"]
        )
        print(f"\n=== vs baseline {args.baseline} ===")
        if not differences:
            print("no metric changed")
        for difference in differences[:60]:
            print(
                f"  {difference['metric']}: {difference['baseline']} -> {difference['current']} "
                f"({difference['delta']:+})"
            )
    return 0


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def render_console(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rerank_k = payload["config"]["rerank_k"]
    rerank_precision = summary["rerank"][f"precision@{rerank_k}"]
    lines = [
        f"corpus      : {payload['corpus']['label']} "
        f"({payload['corpus']['indexed'].get('chunks', '?')} chunks)",
        f"checkpoint  : {payload['config']['checkpoint']} on {payload['config']['device']}",
        "",
        "candidate recall (labelled queries, k=%d)" % payload["config"]["retrieve_k"],
        f"  dense     : recall {summary['candidate_recall']['dense']['recall']}  "
        f"mrr {summary['candidate_recall']['dense']['mrr']}",
        f"  hybrid    : recall {summary['candidate_recall']['hybrid']['recall']}  "
        f"mrr {summary['candidate_recall']['hybrid']['mrr']}",
        f"Laya rerank : recall@k {rerank_precision}  "
        f"mrr {summary['rerank']['mrr']}  median rank {summary['rerank']['median_rank']}",
        "",
        f"margin gold-vs-best-negative : median {summary['margin']['median']}  "
        f"min {summary['margin']['min']}  negative {summary['margin']['negative']}",
        f"relevance ECE              : {summary['relevance']['ece']}",
        f"answerability ECE          : {summary['answerability']['ece_measured']} "
        f"(measured; {summary['answerability']['short_circuited']} queries short-circuited "
        f"before the gate)",
        f"mean P(answerable)         : "
        f"{summary['answerability']['mean_on_answerable']} answerable vs "
        f"{summary['answerability']['mean_on_unanswerable']} not (gate-reached only)",
        f"abstention                 : {summary['answerability']['abstained_correctly']}/"
        f"{summary['answerability']['unanswerable_total']} unanswerable refused, "
        f"{summary['answerability']['false_abstention']}/"
        f"{summary['answerability']['answerable_total']} answerable refused "
        f"({summary['answerability']['false_abstention_from_relevance']} of them a relevance miss)",
    ]
    misses = [row for row in payload["queries"] if row["answerable_label"] and row["rerank_rank"] is None]
    if misses:
        lines.append("")
        lines.append("labelled queries whose gold never reached the candidate set:")
        for row in misses:
            lines.append(f"  - {row['id']} ({row['category']}): dense={row['dense_rank']} hybrid={row['hybrid_rank']}")
    return "\n".join(lines)


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rerank_k = payload["config"]["rerank_k"]
    parts = [
        f"# Quality evaluation — {payload['corpus']['label']}",
        "",
        f"Checkpoint `{payload['config']['checkpoint']}` on `{payload['config']['device']}`; "
        f"retrieve_k={payload['config']['retrieve_k']}, rerank_k={rerank_k}, "
        f"relevance>={payload['config']['relevance_threshold']}, "
        f"answerability>={payload['config']['answerability_threshold']}. "
        f"Index: {payload['corpus']['indexed'].get('chunks', '?')} chunks.",
        "",
        "## Candidate recall",
        "",
        evalkit.markdown_table([
            {"channel": name, **values}
            for name, values in summary["candidate_recall"].items()
        ]),
        "",
        "## Laya re-rank",
        "",
        evalkit.markdown_table([{"stage": "after Laya relevance", **summary["rerank"]}]),
        "",
        "## By category",
        "",
        evalkit.markdown_table(summary["per_category"]),
        "",
        "## Relevance margin (gold minus best negative)",
        "",
        evalkit.markdown_table([summary["margin"]]),
        "",
        "## Calibration",
        "",
        evalkit.markdown_table([
            {"decision": "relevance", "ece": summary["relevance"]["ece"],
             "pairs": summary["relevance"]["pairs"], "positives": summary["relevance"]["positives"]},
            {"decision": "answerability (measured only)", "ece": summary["answerability"]["ece_measured"],
             "pairs": summary["answerability"]["measured"],
             "positives": summary["answerability"]["gate_measured_answerable"]},
            {"decision": "answerability (all queries, 0.0 when short-circuited)",
             "ece": summary["answerability"]["ece_all_queries"],
             "pairs": summary["answerability"]["answerable_total"]
                      + summary["answerability"]["unanswerable_total"],
             "positives": summary["answerability"]["answerable_total"]},
        ]),
        "",
        "## Answerability separation and abstention",
        "",
        evalkit.markdown_table([{
            "mean P(ans) answerable": summary["answerability"]["mean_on_answerable"],
            "mean P(ans) unanswerable": summary["answerability"]["mean_on_unanswerable"],
            "gate reached (answerable)": summary["answerability"]["gate_measured_answerable"],
            "gate reached (unanswerable)": summary["answerability"]["gate_measured_unanswerable"],
            "short-circuited": summary["answerability"]["short_circuited"],
            "unanswerable refused": f"{summary['answerability']['abstained_correctly']}/"
                                    f"{summary['answerability']['unanswerable_total']}",
            "answerable refused": f"{summary['answerability']['false_abstention']}/"
                                  f"{summary['answerability']['answerable_total']}",
            "…of which relevance miss": summary["answerability"]["false_abstention_from_relevance"],
        }]),
        "",
        f"### Relevance threshold sweep (candidate level, n={summary['relevance']['pairs']})",
        "",
        evalkit.markdown_table(summary["relevance"]["sweep"]),
        "",
        f"### Answerability threshold sweep (query level, short-circuited = 0.0)",
        "",
        evalkit.markdown_table(summary["answerability"]["sweep_all_queries"]),
        "",
        "## Per query",
        "",
        evalkit.markdown_table([
            {
                "id": row["id"],
                "category": row["category"],
                "dense": row["dense_rank"],
                "hybrid": row["hybrid_rank"],
                "rerank": row["rerank_rank"],
                "gold_rel": row["best_gold_relevance"],
                "best_neg": row["best_negative_relevance"],
                "margin": row["margin"],
                "kept": row["kept"],
                "measured": row["measured"],
                "P(ans)": row["answerability"],
                "verdict": "answerable" if row["answerable"] else "abstain",
            }
            for row in payload["queries"]
        ]),
        "",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    sys.exit(main())
