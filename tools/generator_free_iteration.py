"""Generator-free iteration, Stage 1: does entity-probe pool expansion raise gold coverage?

The audit's Bet 1 is the largest single claim in the programme — CoRAG's MuSiQue EM moves 17.4 -> 27.7
from one retrieval to six with the *same fixed retriever*, so iteration, not scorer capacity, is the
lever this engine does not use. Its first stage is cheap and falsifiable on its own:

    if a second retrieval pass, driven by entities the first pass found but the query did not name,
    raises the pool's gold-fact coverage, the mechanism is real; if it does not, the bet closes
    without a controller, a generator or an LLM.

This tool runs exactly that, with no model: BM25 (the shipped default's channel) retrieves a first
pool; from the top of that pool it mines **capitalised entity runs absent from the query**; each
probe retrieves again into the pool; coverage is the benchmark's own normalised whole-fact
substring test against the pool. A **shuffled-probe control** runs the identical procedure with
another query's probes, so a gain can be attributed to the bridge rather than to "more retrieval".
It also reports the **pool ceiling MAP@10** per round — an oracle ordering of the pool — so the
answer says whether the added coverage is headroom for ordering, not only a larger set.

    .venv/bin/python tools/generator_free_iteration.py --limit-queries 400
    .venv/bin/python tools/generator_free_iteration.py          # full answerable set

Scope, stated: lexical-only (the shipped default) and no answerability controller — this is the
*mechanism* test, not the end-to-end Bet 1 result. The fused version is the same procedure with the
dense channel added.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402
import nugget_recall as nugget  # noqa: E402
from laya_rag.chunking import Chunk  # noqa: E402
from laya_rag.lexical import BM25Index  # noqa: E402

BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
CORPUS = BENCH / "corpus.json"
ITEMS = BENCH / "MultiHopRAG.json"

#: Capitalised word runs of one to three tokens, the same unit the nugget protocol uses.
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\b")
_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub("", text).lower()


def build_chunks(corpus_path: Path = CORPUS, *, overlap: int = 200) -> dict[str, str]:
    return nugget.build_chunks(corpus_path, overlap=overlap)


def build_bm25(chunks: Mapping[str, str]) -> BM25Index:
    index = BM25Index(stem=True, drop_stopwords=True)
    index.build([
        Chunk(chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], path=chunk_id, text=text,
              index=0, start_char=0, end_char=len(text))
        for chunk_id, text in chunks.items()
    ])
    return index


def probes_from(
    texts: Sequence[str],
    query: str,
    *,
    max_probes: int,
    min_chars: int = 4,
) -> list[str]:
    """Entity runs frequent in ``texts`` and absent from ``query``, most frequent first.

    Frequency across the pool is the cheap bridge prior: an entity that recurs in the first
    pool's top passages is the thing those passages share, which is what a second hop should
    search for. Ties break alphabetically so the result is deterministic.
    """
    query_norm = _normalise(query)
    counts: dict[str, int] = {}
    for text in texts:
        for entity in _ENTITY.findall(text or ""):
            entity = entity.strip()
            if len(entity) < min_chars:
                continue
            if _normalise(entity) in query_norm:
                continue
            counts[entity] = counts.get(entity, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [entity for entity, _count in ranked[:max_probes]]


def fact_counts(
    pool: Sequence[str], facts: Sequence[str], evidence: nugget.Evidence
) -> tuple[int, int]:
    """``(gold facts present anywhere in the pool, gold facts total)``."""
    wanted = [fact for fact in facts if fact and fact.strip()]
    if not wanted:
        return 0, 0
    found = sum(
        1 for fact in wanted
        if any(evidence.covers_fact(chunk_id, fact) for chunk_id in pool)
    )
    return found, len(wanted)


def coverage(pool: Sequence[str], facts: Sequence[str], evidence: nugget.Evidence) -> float:
    """Fraction of a query's gold facts present anywhere in the pool."""
    found, total = fact_counts(pool, facts, evidence)
    return found / total if total else 0.0


def ceiling_map_at_10(covered: int, gold: int, top_k: int = 10) -> float:
    """The best MAP@10 an oracle ordering of this pool could reach.

    MAP@10 normalises by ``min(gold, top_k)``, and an oracle places candidates so that each
    rank reveals one new fact, so the ceiling is ``sum(1/r for r in 1..min(covered, top_k)) /
    min(gold, top_k)`` — an upper bound (a chunk covering several facts can only do better).
    Comparing it across rounds answers whether extra *coverage* is headroom for *ordering*.
    """
    if gold <= 0:
        return 0.0
    # You cannot reveal more distinct facts than exist, so the ranks cap at the gold count too.
    reveal = min(covered, gold, top_k)
    return sum(1.0 / rank for rank in range(1, reveal + 1)) / min(gold, top_k)


def load_queries(path: Path = ITEMS, *, limit: int | None = None) -> tuple[dict[str, str], dict[str, list[str]]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    items = [(i, record) for i, record in enumerate(records)
             if record.get("question_type") != "null_query"]
    if limit:
        items = items[:limit]
    queries = {str(i): str(record["query"]) for i, record in items}
    golds = {
        str(i): [str(entry.get("fact") or "") for entry in record.get("evidence_list") or []]
        for i, record in items
    }
    return queries, golds


def load_question_types(path: Path = ITEMS, *, limit: int | None = None) -> dict[str, str]:
    """The benchmark's question type per query id, for the per-type breakdown."""
    records = json.loads(path.read_text(encoding="utf-8"))
    items = [(i, record) for i, record in enumerate(records)
             if record.get("question_type") != "null_query"]
    if limit:
        items = items[:limit]
    return {str(i): str(record.get("question_type") or "unknown") for i, record in items}


def run(
    queries: Mapping[str, str],
    golds: Mapping[str, list[str]],
    index: BM25Index,
    evidence: nugget.Evidence,
    chunks: Mapping[str, str],
    *,
    depth: int,
    iterations: int,
    max_probes: int,
    top_texts: int,
    shuffle_seed: int = 17,
    types: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the real and shuffled-probe arms for every query, iteration by iteration."""
    order = list(queries)
    # A deterministic derangement for the control: query i gets query (i+1)'s probes.
    shifted = {query_id: order[(position + 1) % len(order)]
               for position, query_id in enumerate(order)}
    per_iteration: list[dict[str, float]] = []
    real_pools: dict[str, set[str]] = {query_id: set() for query_id in order}
    shuffled_pools: dict[str, set[str]] = {query_id: set() for query_id in order}
    real_coverage: dict[str, dict[str, float]] = {query_id: {} for query_id in order}
    shuffled_coverage: dict[str, dict[str, float]] = {query_id: {} for query_id in order}
    real_ceiling: dict[str, dict[str, float]] = {query_id: {} for query_id in order}
    shuffled_ceiling: dict[str, dict[str, float]] = {query_id: {} for query_id in order}
    real_probes: dict[str, list[str]] = {}
    shuffled_probes: dict[str, list[str]] = {}

    for query_id in order:
        hits = index.search(queries[query_id], k=depth)
        real_pools[query_id] = {hit.chunk_id for hit in hits}
        shuffled_pools[query_id] = set(real_pools[query_id])

    for iteration in range(iterations + 1):
        sizes_real = sizes_shuffled = 0
        complete = 0
        real_values: dict[str, float] = {}
        shuffled_values: dict[str, float] = {}
        real_ceilings: list[float] = []
        shuffled_ceilings: list[float] = []
        for query_id in order:
            real_found, total = fact_counts(sorted(real_pools[query_id]), golds[query_id], evidence)
            shuffled_found, _total = fact_counts(
                sorted(shuffled_pools[query_id]), golds[query_id], evidence
            )
            real = real_found / total if total else 0.0
            shuffled = shuffled_found / total if total else 0.0
            real_coverage[query_id][str(iteration)] = round(real, 4)
            shuffled_coverage[query_id][str(iteration)] = round(shuffled, 4)
            real_values[query_id] = real
            shuffled_values[query_id] = shuffled
            real_ceiling_value = ceiling_map_at_10(real_found, total)
            shuffled_ceiling_value = ceiling_map_at_10(shuffled_found, total)
            real_ceiling[query_id][str(iteration)] = round(real_ceiling_value, 4)
            shuffled_ceiling[query_id][str(iteration)] = round(shuffled_ceiling_value, 4)
            real_ceilings.append(real_ceiling_value)
            shuffled_ceilings.append(shuffled_ceiling_value)
            complete += 1 if real >= 1.0 else 0
            sizes_real += len(real_pools[query_id])
            sizes_shuffled += len(shuffled_pools[query_id])
        per_iteration.append({
            "iteration": iteration,
            "real_coverage": round(sum(real_values.values()) / len(order), 4),
            "shuffled_coverage": round(sum(shuffled_values.values()) / len(order), 4),
            "real_ceiling_map@10": round(sum(real_ceilings) / len(order), 4),
            "shuffled_ceiling_map@10": round(sum(shuffled_ceilings) / len(order), 4),
            "complete_fraction": round(complete / len(order), 4),
            "mean_pool_size": round(sizes_real / len(order), 2),
        })
        if iteration == iterations:
            break
        for query_id in order:
            text_ids = [hit.chunk_id for hit in index.search(queries[query_id], k=top_texts)]
            texts = [chunks.get(chunk_id, "") for chunk_id in text_ids]
            if query_id not in real_probes:
                real_probes[query_id] = probes_from(texts, queries[query_id], max_probes=max_probes)
                other = shifted[query_id]
                other_ids = [hit.chunk_id for hit in index.search(queries[other], k=top_texts)]
                other_texts = [chunks.get(chunk_id, "") for chunk_id in other_ids]
                shuffled_probes[query_id] = probes_from(
                    other_texts, queries[query_id], max_probes=max_probes
                )
            for probe in real_probes[query_id]:
                real_pools[query_id].update(hit.chunk_id for hit in index.search(probe, k=depth))
            for probe in shuffled_probes[query_id]:
                shuffled_pools[query_id].update(
                    hit.chunk_id for hit in index.search(probe, k=depth)
                )

    paired_real = evalkit.paired_bootstrap(
        {q: real_coverage[q]["0"] for q in order},
        {q: real_coverage[q][str(iterations)] for q in order},
    )
    paired_control = evalkit.paired_bootstrap(
        {q: shuffled_coverage[q][str(iterations)] for q in order},
        {q: real_coverage[q][str(iterations)] for q in order},
    )
    paired_ceiling = evalkit.paired_bootstrap(
        {q: real_ceiling[q]["0"] for q in order},
        {q: real_ceiling[q][str(iterations)] for q in order},
    )
    paired_ceiling_control = evalkit.paired_bootstrap(
        {q: shuffled_ceiling[q][str(iterations)] for q in order},
        {q: real_ceiling[q][str(iterations)] for q in order},
    )
    return {
        "iterations": per_iteration,
        "paired": {
            "real_expansion_vs_baseline": paired_real,
            "real_vs_shuffled_at_final": paired_control,
            "ceiling_expansion_vs_baseline": paired_ceiling,
            "ceiling_real_vs_shuffled_at_final": paired_ceiling_control,
        },
        "by_question_type": _by_question_type(
            order, types, real_coverage, real_ceiling, iterations
        ),
        "per_query_real": real_coverage,
        "per_query_shuffled": shuffled_coverage,
        "per_query_real_ceiling": real_ceiling,
        "per_query_shuffled_ceiling": shuffled_ceiling,
    }


def _by_question_type(
    order: Sequence[str],
    types: Mapping[str, str] | None,
    coverage: Mapping[str, Mapping[str, float]],
    ceiling: Mapping[str, Mapping[str, float]],
    iterations: int,
) -> dict[str, dict[str, Any]]:
    """Mean coverage and ceiling before/after, grouped by the benchmark's question type."""
    if not types:
        return {}
    buckets: dict[str, dict[str, float]] = {}
    for query_id in order:
        kind = types.get(query_id, "unknown")
        row = buckets.setdefault(kind, {"queries": 0.0, "baseline": 0.0, "final": 0.0,
                                        "ceiling_baseline": 0.0, "ceiling_final": 0.0})
        row["queries"] += 1
        row["baseline"] += coverage[query_id]["0"]
        row["final"] += coverage[query_id][str(iterations)]
        row["ceiling_baseline"] += ceiling[query_id]["0"]
        row["ceiling_final"] += ceiling[query_id][str(iterations)]
    out: dict[str, dict[str, Any]] = {}
    for kind, row in sorted(buckets.items()):
        count = row["queries"] or 1
        out[kind] = {
            "queries": int(row["queries"]),
            "coverage_baseline": round(row["baseline"] / count, 4),
            "coverage_final": round(row["final"] / count, 4),
            "ceiling_baseline": round(row["ceiling_baseline"] / count, 4),
            "ceiling_final": round(row["ceiling_final"] / count, 4),
        }
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="generator_free_iteration.py")
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--max-probes", type=int, default=4)
    ap.add_argument("--top-texts", type=int, default=20)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--out", default=str(BENCH / "generator-free-iteration"))
    return ap.parse_args(argv)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Generator-free iteration, Stage 1 (lexical, no model)",
        "",
        f"- {report['_settings']['queries']} queries, depth {report['_settings']['depth']}, "
        f"{report['_settings']['iterations']} expansion round(s), "
        f"<= {report['_settings']['max_probes']} entity probes per round",
        "",
        "| iteration | real coverage | shuffled-probe coverage | real ceiling MAP@10 | "
        "shuffled ceiling MAP@10 | complete | mean pool |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["iterations"]:
        lines.append(
            f"| {row['iteration']} | {row['real_coverage']:.4f} | {row['shuffled_coverage']:.4f} | "
            f"{row['real_ceiling_map@10']:.4f} | {row['shuffled_ceiling_map@10']:.4f} | "
            f"{row['complete_fraction']:.4f} | {row['mean_pool_size']:.1f} |"
        )
    paired = report["paired"]
    lines += ["", "| comparison | delta | 95% CI | p |", "| --- | --- | --- | --- |"]
    for name, result in paired.items():
        lines.append(
            f"| {name} | {result['delta']:+.4f} | "
            f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] | {result['p_value']:.4f} |"
        )
    by_type = report.get("by_question_type") or {}
    if by_type:
        lines += [
            "",
            "| question type | queries | coverage 0 → final | ceiling 0 → final |",
            "| --- | --- | --- | --- |",
        ]
        for kind, row in by_type.items():
            lines.append(
                f"| {kind} | {row['queries']} | {row['coverage_baseline']:.4f} → "
                f"{row['coverage_final']:.4f} | {row['ceiling_baseline']:.4f} → "
                f"{row['ceiling_final']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("generator_free_iteration.py")
    if not CORPUS.is_file() or not ITEMS.is_file():
        print("generator-free-iteration: SKIPPED — MultiHop corpus/items not found")
        return 0
    queries, golds = load_queries(limit=args.limit_queries)
    types = load_question_types(limit=args.limit_queries)
    chunks = build_chunks()
    index = build_bm25(chunks)
    evidence = nugget.Evidence(chunks)
    result = run(
        queries, golds, index, evidence, chunks,
        depth=args.depth, iterations=args.iterations,
        max_probes=args.max_probes, top_texts=args.top_texts, types=types,
    )
    report = {
        "_settings": {
            "queries": len(queries), "chunks": len(chunks), "depth": args.depth,
            "iterations": args.iterations, "max_probes": args.max_probes,
            "top_texts": args.top_texts, "limit_queries": args.limit_queries,
        },
        **result,
    }
    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    evalkit.write_text(out.with_suffix(".md"), render_markdown(report))
    print(render_markdown(report))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
