"""2WikiMultiHopQA: does the multi-hop picture survive a second external dataset?

The coverage objective wins on MultiHop-RAG and **falsified on MuSiQue** (−0.0109 MAP@10), which
left one question open: is that failure idiosyncratic to MuSiQue, or general? A third dataset is
the highest-value falsification still available, and 2WikiMultiHopQA is it — a distractor-setting
multi-hop set whose records carry their context paragraphs (like MuSiQue), 12,576 dev questions.

The mapping onto the existing framework is direct and mirrors ``tools/musique.py``:

* **corpus** — the union of the context paragraphs the records carry, deduplicated by
  ``(title, text)`` and keyed ``slug(title)#n`` so ``document_of`` groups paragraphs of one article;
* **gold** — the paragraph ids named by the record's ``supporting_facts``; a question needs all of
  them, which is the set-coverage problem MAP@10 measures;
* **metrics** — the benchmark's own definitions (Hits@k, MAP@10 with the true gold count), so the
  rows are comparable to MultiHop-RAG's and MuSiQue's.

This harness is deliberately retrieval-only and cheap: BM25, a bi-encoder and their RRF fusion, no
cross-encoder and no selection rules. It answers the channel question on a third corpus.

    curl -sSL -o eval/benchmarks/2wiki/dev.json \\
        https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json
    .venv/bin/python tools/twowiki.py --split dev --limit-queries 1000
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
import beir  # noqa: E402
import evalkit  # noqa: E402
from musique import gold_matches, slug  # noqa: E402
from multihop_coverage import atom_index, split_atoms  # noqa: E402
from multihop_rerank_experiment import (  # noqa: E402
    coverage_doc_quota_order,
    coverage_order,
    metrics_from_matches,
)

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "2wiki"
MODELS = beir.model_aliases(
    {"bge-small": "BAAI/bge-small-en-v1.5", "bge-base": "BAAI/bge-base-en-v1.5"}
)
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")


def _rrf_scores(
    rankings: Sequence[Mapping[str, Sequence[str]]],
    query: str,
    ids: Sequence[str],
    rrf_k: int,
) -> list[float]:
    """Reciprocal-rank fusion score for ``ids`` under ``rankings`` — the relevance term.

    The MultiHop/MuSiQue coverage runs use a cross-encoder logit as the relevance. This harness
    has no cross-encoder (it is retrieval-only), so the fused RRF score stands in: the same
    signal the baseline order is built from, which is the honest comparison for "does coverage
    beat the order it was given?".
    """
    scores = {cid: 0.0 for cid in ids}
    for ranking in rankings:
        for rank, cid in enumerate(ranking.get(query, []), start=1):
            if cid in scores:
                scores[cid] += 1.0 / (rrf_k + rank)
    return [scores[cid] for cid in ids]


def load_split(path: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Build ``(corpus, queries, golds, titles, hops, order)`` from a 2Wiki split file.

    Paragraphs are deduplicated across records by ``(title, text)`` exactly as in the MuSiQue
    adapter, so one fact cannot occupy several retrieval slots. Gold is assigned as positional
    indices into the finished corpus.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit:
        records = records[:limit]
    corpus: dict[str, str] = {}
    titles: dict[str, str] = {}
    seen: dict[tuple[str, str], str] = {}
    per_title: dict[str, int] = {}
    queries: dict[str, str] = {}
    gold_ids: dict[str, list[str]] = {}
    hops: dict[str, int] = {}

    for record in records:
        qid = str(record.get("_id") or record.get("id"))
        queries[qid] = str(record.get("question") or "")
        title_to_doc: dict[str, str] = {}
        for entry in record.get("context") or []:
            title = str(entry[0])
            sentences = entry[1] if len(entry) > 1 else []
            text = " ".join(str(sentence) for sentence in sentences).strip()
            if not text:
                continue
            key = (title, text)
            doc_id = seen.get(key)
            if doc_id is None:
                position = per_title.get(title, 0)
                per_title[title] = position + 1
                doc_id = f"{slug(title)}#{position}"
                seen[key] = doc_id
                corpus[doc_id] = f"{title}\n{text}" if title else text
                titles[doc_id] = title
            title_to_doc[title] = doc_id
        gold = [
            title_to_doc[title]
            for title, _sentence in (record.get("supporting_facts") or [])
            if title in title_to_doc
        ]
        # De-duplicate a title named twice among the supporting facts.
        gold_ids[qid] = list(dict.fromkeys(gold))
        hops[qid] = len(gold_ids[qid]) or 1

    order = {doc_id: position for position, doc_id in enumerate(corpus)}
    golds = {qid: [order[doc_id] for doc_id in ids if doc_id in order]
             for qid, ids in gold_ids.items()}
    return {
        "corpus": corpus, "queries": queries, "golds": golds, "titles": titles,
        "hops": hops, "order": order,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="twowiki.py")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--model", default="bge-small")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rrf-k", type=int, default=5)
    ap.add_argument("--embed-chunk", type=int, default=20000,
                    help="checkpoint the embedding pass every N documents, so a crash resumes "
                         "at the next chunk instead of re-embedding")
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--coverage-lambda", type=float, default=0.7,
                    help="coverage/relevance trade-off for the coverage arms")
    ap.add_argument("--min-docs", type=int, default=3,
                    help="distinct articles seeded before coverage fills, for the covquota arm")
    ap.add_argument("--out", default=str(ROOT_BENCH / "retrieval"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("twowiki.py")
    split_path = ROOT_BENCH / f"{args.split}.json"
    if not split_path.is_file():
        raise SystemExit(
            f"{split_path} is missing; download it with:\n"
            "  curl -sSL -o eval/benchmarks/2wiki/dev.json "
            "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json"
        )
    data = load_split(split_path, limit=args.limit_queries)
    corpus, queries, golds, titles, order = (
        data["corpus"], data["queries"], data["golds"], data["titles"], data["order"]
    )
    print(f"[{args.split}] {len(corpus)} paragraphs, {len(queries)} queries", flush=True)

    model_id = MODELS.get(args.model, args.model)
    encoder = beir.BiEncoder(model_id)
    cache = ROOT_BENCH / f"cache-{args.split}{'' if not args.limit_queries else '-limit'}-{encoder.name}"
    vectors = beir.cached_embed(encoder, corpus, cache, chunk=args.embed_chunk)
    print("  lexical ranking...", flush=True)
    lexical = beir.bm25_rankings(corpus, queries, args.depth, stem=True, drop_stopwords=True,
                                 titles=titles, title_boost=2)
    print("  dense ranking...", flush=True)
    dense = beir.dense_rankings(corpus, queries, encoder, vectors, args.depth, fixed_length=True)
    fused = beir.multi_channel_fusion([dense, lexical], [1.0, 1.0], args.depth, rrf_k=args.rrf_k)

    arms = {"bm25-engine": lexical, "biencoder": dense, "hybrid-biencoder": fused}
    per_query: dict[str, dict[str, dict[str, float]]] = {name: {} for name in arms}
    for name, ranked in arms.items():
        for q in queries:
            pool = list(ranked[q])[: args.depth]
            matches = gold_matches(pool, golds[q], order)
            per_query[name][q] = metrics_from_matches(
                list(range(len(pool))), matches, len(golds[q]), args.top_k
            )

    # ── the coverage objective, on the third corpus (the owed replication) ──
    # MultiHop-RAG: coverage wins. MuSiQue: it falsified (−0.0109 MAP@10). 2Wiki decides whether
    # that failure is idiosyncratic to MuSiQue. Same rule family, retrieval-only relevance.
    print("  coverage selection...", flush=True)
    corpus_ids = list(corpus)
    atom_ids, atom_weight, _atom_texts = atom_index(
        [split_atoms(corpus[cid]) for cid in corpus_ids]
    )
    atom_by_id = dict(zip(corpus_ids, atom_ids))
    lam, min_docs = args.coverage_lambda, args.min_docs
    coverage_arms = [f"coverage-{lam}", f"covquota-{min_docs}-{lam}"]
    for name in coverage_arms:
        per_query[name] = {}
    for q in queries:
        pool = list(fused[q])[: args.depth]
        matches = gold_matches(pool, golds[q], order)
        relevance = _rrf_scores([dense, lexical], q, pool, args.rrf_k)
        atoms = [atom_by_id[cid] for cid in pool]
        docs = [cid.rsplit("#", 1)[0] for cid in pool]
        ranked_indices = {
            f"coverage-{lam}": coverage_order(
                relevance, atoms, atom_weight, args.top_k, lam),
            f"covquota-{min_docs}-{lam}": coverage_doc_quota_order(
                relevance, atoms, atom_weight, docs, args.top_k, lam, min_docs),
        }
        for name, order_indices in ranked_indices.items():
            per_query[name][q] = metrics_from_matches(
                list(order_indices), matches, len(golds[q]), args.top_k
            )

    summary: dict[str, dict[str, float]] = {}
    for name, block in per_query.items():
        summary[name] = {
            metric: round(sum(row[metric] for row in block.values()) / max(1, len(block)), 4)
            for metric in METRICS
        }

    comparisons: dict[str, Any] = {}
    for name in ("biencoder", "hybrid-biencoder"):
        for metric in METRICS:
            baseline = {q: per_query["bm25-engine"][q][metric] for q in queries}
            candidate = {q: per_query[name][q][metric] for q in queries}
            comparisons[f"{name} vs bm25-engine {metric}"] = evalkit.paired_bootstrap(
                baseline, candidate, samples=args.samples
            )
    # The replication question: does the coverage rule beat the order it was given?
    for name in coverage_arms:
        for metric in ("MAP@10", "Hits@4"):
            baseline = {q: per_query["hybrid-biencoder"][q][metric] for q in queries}
            candidate = {q: per_query[name][q][metric] for q in queries}
            comparisons[f"{name} vs hybrid-biencoder {metric}"] = evalkit.paired_bootstrap(
                baseline, candidate, samples=args.samples
            )

    report = {
        "_settings": {
            "split": args.split, "model": model_id, "depth": args.depth,
            "rrf_k": args.rrf_k, "limit_queries": args.limit_queries,
            "paragraphs": len(corpus), "queries": len(queries),
            "coverage_lambda": args.coverage_lambda, "min_docs": args.min_docs,
        },
        "summary": summary,
        "comparisons": comparisons,
        "per_query": per_query,
    }
    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    evalkit.write_text(out.with_suffix(".md"), render_markdown(report))
    print(render_markdown(report))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 2WikiMultiHopQA retrieval",
        "",
        f"- {report['_settings']['queries']} queries over "
        f"{report['_settings']['paragraphs']} paragraphs",
        "",
        "| arm | Hits@10 | Hits@4 | MAP@10 | MRR@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in report["summary"].items():
        lines.append(
            f"| {name} | {row['Hits@10']:.4f} | {row['Hits@4']:.4f} | "
            f"{row['MAP@10']:.4f} | {row['MRR@10']:.4f} |"
        )
    lines += ["", "| comparison | delta | 95% CI | p |", "| --- | --- | --- | --- |"]
    for name, result in report["comparisons"].items():
        lines.append(
            f"| {name} | {result['delta']:+.4f} | "
            f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] | {result['p_value']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
