"""Hop-2 expansion: does bridging raise candidate-pool recall?

The corrected diagnostics say a third of the remaining headroom is recall: 36% of depth-100
pools are missing at least one gold fact, and second-hop facts are retained at 0.719 against
0.950 for first-hop ones. The bridge-conditioned decision is a *validated* signal
(+0.0293 AUC over a shuffled bridge) that failed as a ranker and as a filter — this is its
recall use, which is what the mechanism is actually for.

The cheap version, measured first because it needs no model: mine the rare terms of the
best-ranked candidates that the query itself does not contain (a bridge-entity proxy), issue
them as extra BM25 probes, and measure **pool fact recall** before and after. If recall does
not rise, the entity proxy is the problem and there is no point scoring the new candidates
with a cross-encoder; if it does, the expanded pool is worth the cross-encoder.

    .venv/bin/python tools/multihop_hop2.py --base-depth 100 --expand-depth 100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir  # noqa: E402
import evalkit  # noqa: E402
from multihop_official import chunk_text  # noqa: E402
from multihop_rerank_experiment import gold_matches_for_pool  # noqa: E402

from laya_rag.lexical import STOPWORDS, tokenize  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"


def load_corpus(overlap: int) -> dict[str, str]:
    raw = json.loads((ROOT_BENCH / "corpus.json").read_text(encoding="utf-8"))
    corpus: dict[str, str] = {}
    for article in raw:
        body = article["title"] + " " + article["body"]
        for j, text in enumerate(chunk_text(body, 256, overlap)):
            corpus[f"{article['url']}#{j}"] = text
    return corpus


def document_frequency(corpus: Mapping[str, str]) -> Counter:
    """How many chunks each stemmed term appears in — the IDF source for term mining."""
    df: Counter = Counter()
    for text in corpus.values():
        df.update({t for t in tokenize(text, stem=True) if t not in STOPWORDS})
    return df


def bridge_terms(
    query: str,
    pool: Sequence[str],
    corpus: Mapping[str, str],
    df: Mapping[str, int],
    *,
    top_candidates: int = 1,
    max_terms: int = 5,
) -> list[str]:
    """Rare terms of the best-ranked candidates that the query does not already contain.

    A bridge entity is what hop 1 resolved and hop 2 depends on, so it is *absent from the
    query by construction*. Ranking candidates' new terms by ascending document frequency is
    the model-free proxy for "distinctive entity" — a term shared by half the corpus is not
    a bridge, it is a stopword in disguise.
    """
    query_terms = {t for t in tokenize(query, stem=True) if t not in STOPWORDS}
    seen: set[str] = set()
    mined: list[str] = []
    for doc_id in list(pool)[:top_candidates]:
        for term in tokenize(corpus.get(doc_id, ""), stem=True):
            if term in STOPWORDS or term in query_terms or term in seen:
                continue
            seen.add(term)
            mined.append(term)
    mined.sort(key=lambda term: (df.get(term, 10**9), term))
    return mined[:max_terms]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_hop2.py")
    ap.add_argument("--ce-cache", default=str(ROOT_BENCH / "ce-scores-full.json"))
    ap.add_argument("--base-depth", type=int, default=100)
    ap.add_argument("--expand-depth", type=int, default=100)
    ap.add_argument("--top-candidates", type=int, default=1)
    ap.add_argument("--max-terms", type=int, default=5)
    ap.add_argument("--rrf-k", type=int, default=5)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "hop2"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cache = json.loads(Path(args.ce_cache).read_text(encoding="utf-8"))
    overlap = int(cache["_settings"]["overlap"])
    corpus = load_corpus(overlap)
    print(f"{len(corpus)} chunks, {len(cache['queries'])} queries", flush=True)
    print("  document frequencies...", flush=True)
    df = document_frequency(corpus)

    queries = _query_texts()

    expansion: dict[str, str] = {}
    for q, payload in cache["queries"].items():
        terms = bridge_terms(
            queries[q], payload["pool"][: args.base_depth], corpus, df,
            top_candidates=args.top_candidates, max_terms=args.max_terms,
        )
        expansion[q] = " ".join(terms)

    print("  expanding with BM25 over the mined terms...", flush=True)
    expanded = beir.bm25_rankings(
        corpus, expansion, args.expand_depth, stem=True, drop_stopwords=True
    )

    base_recall: dict[str, float] = {}
    expanded_recall: dict[str, float] = {}
    added_facts = 0
    for q, payload in cache["queries"].items():
        gold = cache["golds"][q]
        base_pool = list(payload["pool"])[: args.base_depth]
        base_matches = gold_matches_for_pool(corpus, base_pool, gold)
        base_found = {j for row in base_matches for j in row}
        # Keep the base pool first so its ordering (and any tie) is untouched; append only
        # genuinely new candidates.
        seen = set(base_pool)
        extra = [doc for doc in expanded.get(q, []) if doc not in seen]
        expanded_matches = gold_matches_for_pool(corpus, base_pool + extra, gold)
        expanded_found = {j for row in expanded_matches for j in row}
        gold_count = max(len(gold), 1)
        base_recall[q] = len(base_found) / gold_count
        expanded_recall[q] = len(expanded_found) / gold_count
        added_facts += len(expanded_found) - len(base_found)

    summary = {
        "base_pool_fact_recall": round(sum(base_recall.values()) / len(base_recall), 4),
        "expanded_pool_fact_recall": round(sum(expanded_recall.values()) / len(expanded_recall), 4),
        "queries_improved": sum(1 for q in base_recall if expanded_recall[q] > base_recall[q]),
        "queries_worsened": sum(1 for q in base_recall if expanded_recall[q] < base_recall[q]),
        "gold_facts_added": added_facts,
    }
    comparison = {
        "bootstrap": evalkit.paired_bootstrap(base_recall, expanded_recall, samples=args.samples),
        "permutation": evalkit.permutation_test(base_recall, expanded_recall, samples=args.samples),
    }
    result = {
        "_settings": {
            "ce_cache": Path(args.ce_cache).name,
            "queries": len(base_recall),
            "base_depth": args.base_depth,
            "expand_depth": args.expand_depth,
            "top_candidates": args.top_candidates,
            "max_terms": args.max_terms,
        },
        "summary": summary,
        "comparison": comparison,
        "per_query": {"base": base_recall, "expanded": expanded_recall},
    }
    out = Path(args.out)
    evalkit.write_json(Path(f"{out}.json"), result)
    evalkit.write_text(Path(f"{out}.md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", f"{out}.json and .md")
    return 0


def _query_texts() -> dict[str, str]:
    items = json.loads((ROOT_BENCH / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return {str(i): x["query"] for i, x in enumerate(items) if x.get("question_type") != "null_query"}


def render_markdown(result: Mapping[str, Any]) -> str:
    s = result["summary"]
    boot = result["comparison"]["bootstrap"]
    perm = result["comparison"]["permutation"]
    return "\n".join([
        "# Hop-2 expansion — does it raise candidate-pool recall?",
        "",
        f"- {result['_settings']['queries']} queries; base depth {result['_settings']['base_depth']}, "
        f"expansion depth {result['_settings']['expand_depth']}, "
        f"{result['_settings']['max_terms']} mined terms from the top "
        f"{result['_settings']['top_candidates']} candidate(s)",
        "",
        "| pool | gold-fact recall |",
        "| --- | --- |",
        f"| base fused | {s['base_pool_fact_recall']:.4f} |",
        f"| base + hop-2 expansion | {s['expanded_pool_fact_recall']:.4f} |",
        "",
        f"- queries improved: **{s['queries_improved']}**, worsened: **{s['queries_worsened']}**",
        f"- gold facts added to pools: **{s['gold_facts_added']}**",
        "",
        "## Paired comparison",
        "",
        f"- recall delta: {boot['delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
        f"p={perm['p_value']:.4f} ({'sig' if boot['significant'] else 'ns'})",
        "",
    ])


if __name__ == "__main__":
    sys.exit(main())
