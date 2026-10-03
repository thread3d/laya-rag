"""MuSiQue: does the multi-hop coverage gain survive a second dataset?

Every multi-hop number in this repository is measured on MultiHop-RAG alone, and its 2,556
records are the whole dataset — so there is no train/test separation to appeal to. MuSiQue
supplies both things this study needs: a second, harder multi-hop dataset (its questions are
built so that naive lexical matching fails), and a real **train/validation split**.

The mapping onto the existing framework is direct:

* **corpus** — the union of the paragraphs the records carry, deduplicated, keyed by
  ``slug(title)#n`` so ``document_of`` groups paragraphs from one article;
* **gold** — the paragraphs flagged ``is_supporting``; a multi-hop question needs all of
  them, which is exactly the set-coverage problem MAP@10 measures;
* **metrics** — the same definitions as the MultiHop harness (Hits@k, MAP@10 with the true
  gold count, MRR@10), so the comparison is like for like.

Cross-encoder scores are cached, because they are the expensive part and every selection
rule is then pure NumPy over the same pool.

    .venv/bin/python tools/musique.py --split validation --limit-queries 200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir  # noqa: E402
import evalkit  # noqa: E402
from multihop_coverage import atom_index, split_atoms  # noqa: E402
from multihop_rerank_experiment import (  # noqa: E402
    blend_order,
    coverage_doc_quota_order,
    coverage_order,
    metrics_from_matches,
)

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "musique"
MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
}
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")


def slug(text: str) -> str:
    """A filesystem- and id-safe key for a paragraph title."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return cleaned or "untitled"


def load_split(path: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Build ``(corpus, queries, golds, titles, hops)`` from a MuSiQue split file.

    Paragraphs are deduplicated across records by ``(title, text)``: MuSiQue repeats the
    same context paragraphs in many records' candidate lists, and indexing each copy
    separately would let one fact occupy several retrieval slots.

    Gold is returned as **positional indices** into ``corpus``, which is the form the metric
    helpers shared with the MultiHop harness expect. The corpus is assembled first and the
    indices are assigned afterwards, because an index is only meaningful once the whole
    corpus order is known.
    """
    records = json.loads(path.read_text(encoding="utf-8"))
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
        qid = str(record["id"])
        queries[qid] = str(record["question"])
        supporting: list[str] = []
        for paragraph in record.get("paragraphs") or []:
            title = str(paragraph.get("title") or "")
            text = str(paragraph.get("paragraph_text") or "")
            if not text.strip():
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
            if paragraph.get("is_supporting"):
                supporting.append(doc_id)
        gold_ids[qid] = supporting
        hops[qid] = len(record.get("question_decomposition") or []) or 1

    order = {doc_id: position for position, doc_id in enumerate(corpus)}
    golds = {qid: [order[doc_id] for doc_id in ids if doc_id in order]
             for qid, ids in gold_ids.items()}
    return {
        "corpus": corpus, "queries": queries, "golds": golds, "titles": titles,
        "hops": hops, "order": order,
    }


def gold_matches(pool: Sequence[str], gold: Sequence[int], order: Mapping[str, int]) -> list[set]:
    """Per pool candidate, the set of gold-fact indices it carries (empty when it is not gold)."""
    wanted = set(gold)
    matches: list[set] = []
    for doc_id in pool:
        position = order.get(doc_id)
        matches.append({position} if position is not None and position in wanted else set())
    return matches


def evaluate(name: str, ranked: Sequence[int], matches, gold_count: int, top_k: int) -> dict[str, float]:
    return metrics_from_matches(ranked, matches, gold_count, top_k)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="musique.py")
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--model", default="bge-small")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rrf-k", type=int, default=5)
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--coverage-lambda", type=float, default=0.7)
    ap.add_argument("--min-docs", type=int, default=3)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "retrieval"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    split_path = ROOT_BENCH / f"{args.split}.json"
    if not split_path.is_file():
        raise SystemExit(f"{split_path} is missing; run tools/fetch_musique.py first")
    data = load_split(split_path, limit=args.limit_queries)
    corpus, queries, golds, titles = data["corpus"], data["queries"], data["golds"], data["titles"]
    order = data["order"]
    print(f"[{args.split}] {len(corpus)} paragraphs, {len(queries)} queries", flush=True)

    model_id = MODELS.get(args.model, args.model)
    encoder = beir.BiEncoder(model_id)
    cache = ROOT_BENCH / f"cache-{args.split}-{encoder.name}"
    vectors = beir.cached_embed(encoder, corpus, cache)
    print("  lexical ranking...", flush=True)
    lexical = beir.bm25_rankings(corpus, queries, args.depth, stem=True, drop_stopwords=True,
                                 titles=titles, title_boost=2)
    print("  dense ranking...", flush=True)
    dense = beir.dense_rankings(corpus, queries, encoder, vectors, args.depth, fixed_length=True)
    fused = beir.multi_channel_fusion([dense, lexical], [1.0, 1.0], args.depth, rrf_k=args.rrf_k)

    print("  atomising...", flush=True)
    atoms_by_doc = {doc_id: split_atoms(text) for doc_id, text in corpus.items()}
    atom_ids_of, atom_weight, _ = atom_index(list(atoms_by_doc.values()))
    atom_ids_by_doc = dict(zip(atoms_by_doc, atom_ids_of))

    ce_path = ROOT_BENCH / f"ce-scores-{args.split}{'' if not args.limit_queries else '-limit'}.json"
    if ce_path.is_file():
        payload = json.loads(ce_path.read_text(encoding="utf-8"))
        per_query = payload["queries"]
        print(f"  reusing {ce_path.name}", flush=True)
    else:
        cross_encoder = beir.CrossEncoder(args.rerank_model, batch_size=args.batch_size)
        print(f"  scoring with {args.rerank_model}...", flush=True)
        started = time.perf_counter()
        per_query = {}
        for n, (q, text) in enumerate(queries.items()):
            pool = list(fused[q])[: args.depth]
            scores = cross_encoder.score([(text, corpus[doc_id]) for doc_id in pool],
                                         fixed_length=True)
            per_query[q] = {"pool": pool,
                            "ce": [round(float(s), 4) for s in scores],
                            "atoms": [list(atom_ids_by_doc[doc_id]) for doc_id in pool]}
            if n and n % 50 == 0:
                print(f"    {n}/{len(queries)} ({time.perf_counter() - started:.0f}s)", flush=True)
        evalkit.write_json(ce_path, {
            "_settings": {"split": args.split, "model": model_id, "depth": args.depth,
                          "rrf_k": args.rrf_k, "rerank_model": args.rerank_model,
                          "atom_weight": atom_weight, "seconds": round(time.perf_counter() - started, 1)},
            "queries": per_query,
        })
        print(f"wrote {ce_path.name} in {time.perf_counter() - started:.0f}s", flush=True)

    per_query_metrics: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, dict[str, float]] = {}
    for q, payload in per_query.items():
        pool = payload["pool"]
        matches = gold_matches(pool, golds[q], order)
        gold_count = max(len(golds[q]), 1)
        docs = [doc_id.rsplit("#", 1)[0] for doc_id in pool]
        positions = list(range(len(pool)))
        strategies = {
            "fused": positions[: args.top_k],
            "ce": sorted(positions, key=lambda i: (-float(payload["ce"][i]), i))[: args.top_k],
            "blend-0.75": [int(i) for i in blend_order(positions, payload["ce"], 0.75, 60)][: args.top_k],
            f"coverage-{args.coverage_lambda}": coverage_order(
                payload["ce"], payload["atoms"], atom_weight, args.top_k, args.coverage_lambda),
            f"covquota-{args.min_docs}-{args.coverage_lambda}": coverage_doc_quota_order(
                payload["ce"], payload["atoms"], atom_weight, docs, args.top_k,
                args.coverage_lambda, args.min_docs),
        }
        for name, ranked in strategies.items():
            row = evaluate(name, ranked, matches, gold_count, args.top_k)
            bucket = per_query_metrics.setdefault(name, {metric: {} for metric in METRICS})
            for metric in METRICS:
                bucket[metric][q] = row[metric]
    for name, block in per_query_metrics.items():
        summary[name] = {metric: round(sum(block[metric].values()) / len(block[metric]), 4)
                         for metric in METRICS}

    comparisons: dict[str, Any] = {}
    for name in per_query_metrics:
        if name == "fused":
            continue
        for metric in ("MAP@10", "Hits@4"):
            comparisons[f"{name} vs fused {metric}"] = {
                "bootstrap": evalkit.paired_bootstrap(
                    per_query_metrics["fused"][metric], per_query_metrics[name][metric],
                    samples=args.samples),
                "permutation": evalkit.permutation_test(
                    per_query_metrics["fused"][metric], per_query_metrics[name][metric],
                    samples=args.samples),
            }
    result = {
        "_settings": {"split": args.split, "queries": len(per_query), "depth": args.depth,
                      "top_k": args.top_k, "model": model_id, "corpus": len(corpus),
                      "coverage_lambda": args.coverage_lambda, "min_docs": args.min_docs},
        "summary": summary,
        "comparisons": comparisons,
        "per_query": {name: block["MAP@10"] for name, block in per_query_metrics.items()},
    }
    out = Path(args.out)
    evalkit.write_json(Path(f"{out}.json"), result)
    evalkit.write_text(Path(f"{out}.md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", f"{out}.json and .md")
    return 0


def render_markdown(result: Mapping[str, Any]) -> str:
    settings = result["_settings"]
    lines = [
        "# MuSiQue retrieval — does the coverage gain replicate?",
        "",
        f"- split `{settings['split']}`, {settings['queries']} queries over "
        f"{settings['corpus']} paragraphs, depth {settings['depth']}, {settings['model']}",
        "",
        "| rule | Hits@10 | Hits@4 | MAP@10 | MRR@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in result["summary"].items():
        lines.append(f"| {name} | {row['Hits@10']:.4f} | {row['Hits@4']:.4f} | "
                     f"{row['MAP@10']:.4f} | {row['MRR@10']:.4f} |")
    lines += ["", "## Paired comparisons vs the fused first stage", ""]
    for key, value in sorted(result["comparisons"].items()):
        boot, perm = value["bootstrap"], value["permutation"]
        lines.append(f"- `{key}`: {boot['delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
                     f"p={perm['p_value']:.4f} ({'sig' if boot['significant'] else 'ns'})")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
