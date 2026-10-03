"""Depth-stratified pool ceiling and second-hop retention on MultiHop-RAG.

This is the Tier 0.1 diagnostic from `RESEARCH.md` §3.1: it measures, on this repo's
own retrieval, whether the multi-hop MAP@10 gap is a *recall* problem (the evidence is
not in the pool) or a *selection* problem (it is in the pool and the ordering spends
its top-10 on the wrong candidates) — and whether the second hop specifically is taxed
by the rank budget.

Three quantities per query, for each depth D in ``--depths``:

* ``pool_fact_recall@D`` — fraction of distinct gold facts present anywhere in the
  fused top-D. This is the candidate-pool recall ceiling.
* ``pool_ceiling_MAP@10@D`` — the best MAP@10 *any* reordering of the fused top-D could
  reach (:func:`multihop_coverage.pool_ceiling_map`). Comparing D=100 with D=500 answers
  whether widening the pool can help at all; comparing the ceiling with the achieved
  MAP@10 measures the selection headroom.
* ``hop1/hop2_fact_recall@D`` and ``hop2_retention@D`` — per gold fact, its *grounding*
  is the maximum stopword-filtered stemmed query-term overlap over the pool candidates
  that contain it. Facts at the strict minimum grounding are the **second hop** (linked
  to the query only through a bridge entity; the definition of "The Second-Hop Tax",
  redhop 2025); the rest are the first hop. Retention is ``hop2_fact_recall@D`` divided
  by ``hop2_fact_recall@max_depth``, i.e. how much of the reachable second hop the
  top-D budget keeps. Queries whose present facts all share one grounding value have no
  relevance gap and are excluded from the hop statistics.

Everything is deterministic and offline once the corpus is chunked and embedded; the
cached ``multihop-chunks<overlap>.<model>.cls.npy`` sidecar is reused when present.

    .venv/bin/python tools/multihop_diagnostics.py --limit-queries 200
    .venv/bin/python tools/multihop_diagnostics.py --depths 10,100,500 --out eval/benchmarks/multihop-rag/diagnostics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir, evalkit, jobkit  # noqa: E402
from multihop_coverage import pool_ceiling_map  # noqa: E402

from laya_rag.lexical import STOPWORDS, reciprocal_rank_fusion, tokenize  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}


# ── pure helpers (unit-tested offline) ───────────────────────────────────

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """256-token windows with the benchmark's splitter convention."""
    toks = text.split()
    step = max(1, size - overlap)
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i : i + size]))
        if i + size >= len(toks):
            break
        i += step
    return out


def normalize_fact(fact: str) -> str:
    """The benchmark's own gold-match normalization: strip spaces and newlines."""
    return fact.replace(" ", "").replace("\n", "")


def normalize_corpus(corpus) -> dict[str, str]:
    """Normalize every chunk once; the per-query match loop reuses it."""
    return {doc_id: normalize_fact(text) for doc_id, text in corpus.items()}


def chunk_token_sets(corpus) -> dict[str, set[str]]:
    """Stemmed, stopword-filtered token set per chunk, computed once.

    The overlap is asked for the same chunks on every query, so tokenizing inside the
    per-query loop is quadratic in a way that dominates the run; this is the single
    precomputation that keeps it linear.
    """
    return {
        doc_id: {t for t in tokenize(text, stem=True) if t not in STOPWORDS}
        for doc_id, text in corpus.items()
    }


def gold_matches(corpus, pool, gold_facts, *, norm: dict[str, str] | None = None) -> list[set[int]]:
    """``gold_matches[i]`` = ids of gold facts substrings of ``corpus[pool[i]]``."""
    if norm is None:
        norm = normalize_corpus(corpus)
    facts = [normalize_fact(g) for g in gold_facts]
    matches: list[set[int]] = []
    for doc_id in pool:
        text = norm[doc_id]
        matches.append({j for j, g in enumerate(facts) if g and g in text})
    return matches


def candidate_overlap(
    corpus, pool, query_terms, *, token_sets: dict[str, set[str]] | None = None
) -> list[float]:
    """Per-candidate query grounding: share of query terms present in the candidate."""
    if not query_terms:
        return [0.0] * len(pool)
    if token_sets is None:
        token_sets = chunk_token_sets(corpus)
    return [len(token_sets[doc_id] & query_terms) / len(query_terms) for doc_id in pool]


def fact_grounding(matches, overlap) -> list[float]:
    """Max grounding over the pool candidates containing each gold fact.

    A fact absent from the pool has grounding ``0.0`` *and* ``present=False``; the two
    are distinguished by :func:`split_hops` using the match sets directly, so a genuine
    zero-overlap (but present) fact is not confused with an absent one.
    """
    if not matches:
        return []
    total = max((max(m) for m in matches if m), default=-1) + 1
    grounding = [0.0] * total
    for row, score in zip(matches, overlap):
        for fact_id in row:
            if score > grounding[fact_id]:
                grounding[fact_id] = score
    return grounding


def split_hops(matches, overlap) -> tuple[set[int], set[int]]:
    """Partition present gold facts into (first hop, second hop).

    The second hop is every fact at the *strict* minimum grounding among the facts that
    appear in the pool; the first hop is the rest. Returns empty sets when fewer than two
    facts are present or all present facts share one grounding value (no relevance gap,
    so the tax does not apply — the same exclusion redhop makes).
    """
    if not matches:
        return set(), set()
    grounding = fact_grounding(matches, overlap)
    present = {fact_id for row in matches for fact_id in row}
    if len(present) < 2:
        return set(), set()
    values = {fact_id: grounding[fact_id] for fact_id in present}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return set(), set()
    second = {fact_id for fact_id, value in values.items() if value == low}
    first = present - second
    return first, second


def facts_present(matches) -> set[int]:
    return {fact_id for row in matches for fact_id in row}


def recall_at_depth(
    matches,
    depth: int,
    wanted: set[int] | None = None,
    *,
    gold_count: int | None = None,
) -> float | None:
    """Fraction of ``wanted`` facts matched within the first ``depth`` candidates.

    ``None`` when ``wanted`` is empty, because "recall of nothing" is undefined rather
    than zero.

    Two normalisations, and the difference matters. With ``wanted`` given the denominator
    is that set (used for the hop-1/hop-2 splits, which are themselves derived from the
    pool's grounding). With ``wanted`` None and ``gold_count`` given the denominator is
    the query's **true** number of gold facts, which is the official metrics'
    normalisation and is stable under ``--max-depth``. With neither, the denominator is
    the number of facts the *deepest* pool matched — correct only while that pool is
    complete, and the same shape of error that inflated ``pool_ceiling_map`` by 0.12
    before it was fixed. Callers measuring recall should pass ``gold_count``.
    """
    if wanted is None and gold_count is not None:
        if gold_count <= 0:
            return None
        return len(facts_present(matches[:depth])) / gold_count
    if wanted is None:
        wanted = facts_present(matches)
    if not wanted:
        return None
    found = facts_present(matches[:depth])
    return len(found & wanted) / len(wanted)


def hop_grounding_gap(matches, overlap) -> float:
    """Max minus min grounding over present facts; 0.0 means no relevance gap."""
    grounding = fact_grounding(matches, overlap)
    present = facts_present(matches)
    if not present:
        return 0.0
    values = [grounding[f] for f in present]
    return max(values) - min(values)


# ── report helpers ───────────────────────────────────────────────────────

def _mean_ci(values: list[float], samples: int = 10_000, seed: int = 20210617):
    """Query-level bootstrap mean interval (pairs with the repo's existing machinery)."""
    import numpy as np

    if not values:
        return None, None, None
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, len(arr), size=(samples, len(arr)))].mean(axis=1)
    return (
        round(float(arr.mean()), 4),
        round(float(np.percentile(draws, 2.5)), 4),
        round(float(np.percentile(draws, 97.5)), 4),
    )


def summarize(per_query: list[dict], depths: list[int]) -> dict:
    """Aggregate per-query rows into the report summary, with query-level intervals."""
    summary: dict = {"queries": len(per_query)}

    def collect(key):
        return [row[key] for row in per_query if row.get(key) is not None]

    summary["fused_MAP@10"] = _mean_ci(collect("fused_MAP@10"))[0]
    summary["fused_Hits@10"] = _mean_ci(collect("fused_Hits@10"))[0]
    summary["fused_Hits@4"] = _mean_ci(collect("fused_Hits@4"))[0]

    per_depth: dict[str, dict] = {}
    for depth in depths:
        entry: dict = {}
        for key in ("pool_fact_recall", "pool_ceiling_MAP@10"):
            mean, low, high = _mean_ci(collect(f"{key}@{depth}"))
            entry[key] = {"mean": mean, "ci_low": low, "ci_high": high}
        for key in ("hop1_fact_recall", "hop2_fact_recall", "hop2_retention"):
            values = collect(f"{key}@{depth}")
            mean, low, high = _mean_ci(values)
            entry[key] = {"mean": mean, "ci_low": low, "ci_high": high, "n": len(values)}
        per_depth[str(depth)] = entry
    summary["depth"] = per_depth

    # Paired comparisons over the same queries: does widening the pool raise the ceiling,
    # and does the budget actually tax the second hop?
    pairs = {}
    for lower, higher in zip(depths, depths[1:]):
        for key in ("pool_ceiling_MAP@10", "pool_fact_recall", "hop2_fact_recall"):
            base = {str(r["query"]): r[f"{key}@{lower}"] for r in per_query
                    if r.get(f"{key}@{lower}") is not None and r.get(f"{key}@{higher}") is not None}
            cand = {str(r["query"]): r[f"{key}@{higher}"] for r in per_query
                    if r.get(f"{key}@{lower}") is not None and r.get(f"{key}@{higher}") is not None}
            if base:
                pairs[f"{key}:{higher}-{lower}"] = evalkit.paired_bootstrap(base, cand, samples=5000)
    summary["paired"] = pairs

    hop_queries = [r for r in per_query if r.get("hop2_size")]
    summary["hop_queries"] = len(hop_queries)
    return summary


def render_markdown(report: dict) -> str:
    s = report["summary"]
    depths = report["_settings"]["depths"]
    lines = [
        "# MultiHop-RAG retrieval diagnostics",
        "",
        "Depth-stratified candidate-pool ceiling and second-hop retention. Offline and "
        "deterministic; see `RESEARCH.md` §3.1.",
        "",
        f"- queries: **{s['queries']}** (of which **{s.get('hop_queries', 0)}** have a "
        "first/second-hop relevance gap)",
        f"- fused MAP@10: **{s['fused_MAP@10']}**, Hits@10 **{s['fused_Hits@10']}**, "
        f"Hits@4 **{s['fused_Hits@4']}**",
        "",
        "| depth | pool fact recall | ceiling MAP@10 | hops w/ gap | hop1 recall | hop2 recall | hop2 retention |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for depth in depths:
        d = s["depth"][str(depth)]
        def cell(key):
            v = d[key]
            if v["mean"] is None:
                return "—"
            return f"{v['mean']:.4f}"
        lines.append(
            f"| {depth} | {cell('pool_fact_recall')} | {cell('pool_ceiling_MAP@10')} | "
            f"{d['hop2_fact_recall']['n']} | {cell('hop1_fact_recall')} | "
            f"{cell('hop2_fact_recall')} | {cell('hop2_retention')} |"
        )
    lines += ["", "## Paired comparisons", ""]
    if not s["paired"]:
        lines.append("_none_")
    for name, row in s["paired"].items():
        lines.append(
            f"- `{name}`: {row['delta']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
            f"({'significant' if row['significant'] else 'not significant'}, n={row['queries']})"
        )
    lines.append("")
    return "\n".join(lines)


# ── experiment ───────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_diagnostics.py")
    ap.add_argument("--model", default="bge-small",
                    help="short name (bge-small/base/large) or a Hub id")
    ap.add_argument("--overlap", type=int, default=200, help="chunk overlap, official is 200")
    ap.add_argument("--max-depth", type=int, default=500, help="fused pool depth to retrieve")
    ap.add_argument("--depths", default="10,100,500", help="depths to stratify over")
    ap.add_argument("--top-k", type=int, default=10, help="MAP@k; the official metric is 10")
    ap.add_argument("--rrf-k", type=int, default=5)
    ap.add_argument("--limit-queries", type=int, default=None, help="smoke runs only")
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "diagnostics"))
    ap.add_argument("--shard", default="0/1", help="run one contiguous slice, 'i/n'")
    ap.add_argument("--fresh", action="store_true", help="discard any existing checkpoint")
    ap.add_argument("--merge", type=int, default=0, metavar="N",
                    help="only fold N finished shards into <out>.json, then exit")
    return ap.parse_args(argv)


def _fingerprint(args, model_id: str, depths: list[int]) -> str:
    return jobkit.config_fingerprint(
        tool="multihop_diagnostics",
        model=model_id,
        overlap=args.overlap,
        max_depth=args.max_depth,
        depths=depths,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        limit_queries=args.limit_queries,
    )


def _build_report(rows: dict) -> dict:
    """Rebuild the report shape from merged rows, using the reserved shared-state row."""
    if jobkit.META_KEY not in rows:
        raise RuntimeError("no shard checkpoints found — run the shards before --merge")
    meta = rows.pop(jobkit.META_KEY)
    per_query = [rows[key] for key in meta["query_order"] if key in rows]
    return {
        "_settings": meta["_settings"],
        "summary": summarize(per_query, meta["_settings"]["depths"]),
        "per_query": per_query,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    shard_index, shard_count = jobkit.parse_shard(args.shard)
    depths = sorted({int(v) for v in str(args.depths).split(",") if v.strip()})
    if not depths or depths[0] < 1:
        raise SystemExit("--depths must be positive integers")
    if depths[-1] > args.max_depth:
        args.max_depth = depths[-1]
    model_id = MODELS.get(args.model, args.model)
    fingerprint = _fingerprint(args, model_id, depths)

    if args.merge:
        try:
            merged = jobkit.merge_shards(args.out, args.merge, fingerprint=fingerprint,
                                         build=_build_report)
        except (RuntimeError, jobkit.FingerprintMismatch) as exc:
            print(f"merge failed: {exc}", file=sys.stderr)
            return 2
        out = Path(args.out)
        evalkit.write_text(Path(str(out) + ".md"), render_markdown(merged))
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
    items = [(i, x) for i, x in enumerate(items_all) if x["question_type"] != "null_query"]
    if args.limit_queries:
        items = items[: args.limit_queries]
    golds_by_index = {i: [e["fact"] for e in x["evidence_list"]] for i, x in items}

    corpus: dict[str, str] = {}
    titles: dict[str, str] = {}
    for a in raw:
        for j, c in enumerate(chunk_text(a["title"] + " " + a["body"], 256, args.overlap)):
            cid = f"{a['url']}#{j}"
            corpus[cid] = c
            titles[cid] = a["title"]
    queries = {str(i): x["query"] for i, x in items}
    print(f"[overlap={args.overlap}] {len(corpus)} chunks, {len(queries)} queries", flush=True)
    # Normalized text and stemmed token sets are per-chunk, not per-query: compute once.
    norm_corpus = normalize_corpus(corpus)
    tokens_of = chunk_token_sets(corpus)

    enc = beir.BiEncoder(model_id)
    vectors = beir.cached_embed(
        enc, corpus, ROOT_BENCH / "cache" / f"multihop-chunks{args.overlap}.{enc.name}"
    )
    print("  lexical ranking...", flush=True)
    lexical = beir.bm25_rankings(
        corpus, queries, args.max_depth, stem=True, drop_stopwords=True,
        titles=titles, title_boost=2,
    )
    print("  dense ranking...", flush=True)
    # Fixed-length padding: dynamic padding gives every batch a different tensor shape and
    # the MPS backend recompiles per shape, which dominates the run on a few thousand queries.
    dense = beir.dense_rankings(
        corpus, queries, enc, vectors, args.max_depth, fixed_length=True
    )
    print("  retrieval done; scoring candidates...", flush=True)
    from multihop_official import official_metrics  # noqa: E402  (circular-safe import)

    if jobkit.META_KEY not in run:
        run.record(jobkit.META_KEY, {
            "_settings": {
                "model": model_id, "overlap": args.overlap, "max_depth": args.max_depth,
                "depths": depths, "top_k": args.top_k, "limit_queries": args.limit_queries,
                "corpus_chunks": len(corpus), "rrf_k": args.rrf_k,
            },
            "query_order": list(queries),
        })

    mine = jobkit.shard_items(list(queries), shard_index, shard_count)
    todo = run.pending(mine)
    print(f"  {len(todo)} of {len(mine)} queries to score ({len(run) - 1} checkpointed)",
          flush=True)

    started = time.perf_counter()
    for n, q in enumerate(todo, 1):
        text = queries[q]
        gold = golds_by_index[int(q)]
        fused = reciprocal_rank_fusion(
            [dense[q], lexical[q]], k=args.rrf_k, limit=args.max_depth, weights=[1.0, 1.0]
        )
        pool = [doc_id for doc_id, _ in fused]
        matches = gold_matches(corpus, pool, gold, norm=norm_corpus)
        qterms = {t for t in tokenize(text, stem=True) if t not in STOPWORDS}
        overlap = candidate_overlap(corpus, pool, qterms, token_sets=tokens_of)
        first, second = split_hops(matches, overlap)
        present = facts_present(matches)

        row: dict = {"query": q}
        top = [corpus[d] for d in pool[: args.top_k]]
        metrics, _ = official_metrics([top], [gold])
        row.update({f"fused_{k}": v for k, v in metrics.items()})
        row["hop2_size"] = len(second)
        row["hop_gap"] = round(hop_grounding_gap(matches, overlap), 4)
        row["facts_present"] = len(present)

        max_depth = max(depths)
        full_second = second
        for depth in depths:
            row[f"pool_fact_recall@{depth}"] = recall_at_depth(
                matches, depth, gold_count=len(gold)
            )
            row[f"pool_ceiling_MAP@10@{depth}"] = pool_ceiling_map(
                overlap[: min(depth, len(pool))],
                matches[:depth],
                args.top_k,
                # The true gold count, not the facts this depth happens to match: the
                # denominator has to match official_metrics or the ceiling is inflated for
                # the very queries whose pool is incomplete (see pool_ceiling_map).
                gold_count=len(gold),
            )
            row[f"hop1_fact_recall@{depth}"] = recall_at_depth(matches, depth, first)
            row[f"hop2_fact_recall@{depth}"] = recall_at_depth(matches, depth, second)
            base = recall_at_depth(matches, max_depth, full_second)
            here = recall_at_depth(matches, depth, full_second)
            row[f"hop2_retention@{depth}"] = (
                None if base in (None, 0.0) or here is None else round(here / base, 4)
            )
        run.record(q, row)
        if n % 500 == 0:
            run.touch(total=len(mine))
            print(f"  {n}/{len(todo)}", flush=True)

    shared = run.get(jobkit.META_KEY)
    per_query = [run.get(q) for q in shared["query_order"] if run.get(q) is not None]
    report = {
        "_settings": {**shared["_settings"], "seconds": round(time.perf_counter() - started, 1)},
        "summary": summarize(per_query, depths),
        "per_query": per_query,
    }
    run.finish(report, total=len(mine))
    # `finish` already wrote `<stem>.json` atomically; the markdown is a rendering of it, and
    # both carry the shard stem so a sharded run never clobbers the merged report path.
    evalkit.write_text(Path(str(stem) + ".md"), render_markdown(report))
    print("\n" + render_markdown(report))
    print("wrote", str(stem) + ".json", "and", str(stem) + ".md")
    if shard_count > 1:
        print(f"  run all {shard_count} shards, then: --merge {shard_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
