"""Second-stage selection experiments on cached MultiHop-RAG cross-encoder scores.

The published comparison this project lost was MAP@10, and the diagnostics
(`tools/multihop_diagnostics.py`) show the loss is selection, not recall. This sweep is
where the candidate second stages are compared on the *same* pool, so the only thing that
differs is the selection rule:

* ``fused`` — the first-stage order (the baseline);
* ``ce`` — full cross-encoder re-sort, the historical (and partly damaging) behaviour;
* ``blend-b`` — RRF over the first-stage order and the cross-encoder order, weight ``b``;
* ``guardrail-n`` — the cross-encoder order with the first ``n`` first-stage candidates
  pinned, so a second stage can improve the tail without demoting a confident rank-1;
* ``coverage-l`` — sentence-atom max-coverage selection using the cross-encoder score as
  the relevance term, so the top-10 spends its slots on distinct facts.

It reads the cache written by ``tools/multihop_ce_cache.py`` (no model, no network) and
reports paired bootstrap intervals and permutation p-values against both the fused
baseline and the full cross-encoder.

    .venv/bin/python tools/multihop_ce_cache.py --limit-queries 400
    .venv/bin/python tools/multihop_rerank_experiment.py
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
import evalkit  # noqa: E402
from multihop_official import chunk_text  # noqa: E402

from laya_rag.lexical import STOPWORDS, reciprocal_rank_fusion, tokenize  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
DEFAULT_CACHE = ROOT_BENCH / "ce-scores"
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")


# ── pure selection rules (unit-tested offline) ───────────────────────────

def ce_order(pool: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Pool ids in descending cross-encoder score; ties break to the pool order."""
    return [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-float(scores[i]), i))]


def blend_order(
    pool: Sequence[str], scores: Sequence[float], blend: float, rrf_k: int = 60
) -> list[str]:
    """RRF over the first-stage order and the cross-encoder order.

    ``blend`` in ``[0, 1]`` is the weight on the cross-encoder, so 0 keeps the first
    stage and 1 replaces it. This is the same shape ``beir.crossencoder_rankings`` uses
    when a second stage nudges rather than replaces.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend!r}")
    if blend >= 1.0:
        return ce_order(pool, scores)
    if blend <= 0.0:
        return list(pool)
    merged = reciprocal_rank_fusion(
        [list(pool), ce_order(pool, scores)],
        k=rrf_k,
        limit=len(pool),
        weights=[1.0 - blend, blend],
    )
    return [doc_id for doc_id, _ in merged]


def guardrail_order(
    pool: Sequence[str], scores: Sequence[float], pin: int = 1
) -> list[str]:
    """The cross-encoder order with the first ``pin`` first-stage ids pinned in place.

    A second stage that may not touch a confident rank-1 keeps most of its upside on the
    tail while bounding the top-heavy flips that cost the most nDCG on BEIR.
    """
    order = list(range(len(pool)))
    order.sort(key=lambda i: (-float(scores[i]), i))
    for position in range(max(0, min(pin, len(pool)))):
        current = order.index(position)
        if current != position:
            order.pop(current)
            order.insert(position, position)
    return [pool[i] for i in order]


def coverage_order(
    scores: Sequence[float],
    atoms: Sequence[Sequence[int]],
    atom_weight: Sequence[float],
    top_k: int,
    lam: float = 0.5,
) -> list[int]:
    """Greedy max-coverage over sentence atoms, with the CE score as relevance.

    Mirrors ``multihop_coverage.facility_location_select`` but without the dense-gram
    redundancy term: the atom coverage already prevents re-selecting the same fact, and
    the gram matrix is not in the cache. Marginal gain is normalised by the best available
    gain at each step so it is comparable to the normalized relevance.
    """
    rel = [float(value) for value in scores]
    lo, hi = (min(rel), max(rel)) if rel else (0.0, 0.0)
    normalized = [(value - lo) / (hi - lo) if hi > lo else 0.0 for value in rel]
    selected: list[int] = []
    covered: set[int] = set()
    candidates = list(range(len(rel)))
    while candidates and len(selected) < top_k:
        gains = {
            i: float(sum(atom_weight[a] for a in atoms[i] if a not in covered))
            for i in candidates
        }
        scale = max(gains.values()) if gains else 0.0
        scale = scale if scale > 0.0 else 1.0
        best_index, best_score = None, -1e18
        for i in candidates:
            score = lam * normalized[i] + (1.0 - lam) * (gains[i] / scale)
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        covered.update(atoms[best_index])
        candidates.remove(best_index)
    return selected


def rescue_order(
    scores: Sequence[float],
    tokens: Sequence[set],
    top_k: int,
    seed_k: int = 3,
    beta: float = 0.5,
) -> list[int]:
    """Bridge-linked second-hop rescue (RESEARCH.md §3.1).

    Keep the strongest ``seed_k`` cross-encoder candidates, then fill the remaining slots
    with candidates that share *rare* terms with an already-selected seed — the bridge
    entity — trading some relevance for the low-relevance hop the relevance order would
    discard. A term's weight is ``1 / df`` within the pool, so a shared generic term
    counts for almost nothing and a shared name or number counts a lot.

    At each step the score is ``beta * normalized_bridge + (1 - beta) * normalized_ce``,
    both normalised over the candidates still available. ``beta=0`` is the CE order and
    ``beta=1`` is pure bridge linkage.
    """
    from collections import Counter

    n = len(scores)
    if n == 0 or top_k <= 0:
        return []
    seed_k = max(1, min(seed_k, top_k, n))
    ce = [i for i in sorted(range(n), key=lambda i: (-float(scores[i]), i))]
    selected = list(ce[:seed_k])
    df: Counter = Counter()
    for candidate_tokens in tokens:
        df.update(candidate_tokens)
    seed_tokens: set = set().union(*(tokens[i] for i in selected))
    remaining = [i for i in range(n) if i not in selected]
    while len(selected) < top_k and remaining:
        values = [float(scores[i]) for i in remaining]
        lo, hi = min(values), max(values)
        bridges = {
            i: sum(1.0 / df[term] for term in tokens[i] if term in seed_tokens and df[term] > 0)
            for i in remaining
        }
        bridge_max = max(bridges.values()) if bridges else 0.0
        best_index, best_score = None, -1e18
        for i in remaining:
            ce_norm = (float(scores[i]) - lo) / (hi - lo) if hi > lo else 0.0
            bridge_norm = bridges[i] / bridge_max if bridge_max > 0.0 else 0.0
            score = beta * bridge_norm + (1.0 - beta) * ce_norm
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        remaining.remove(best_index)
        seed_tokens |= tokens[best_index]
    return selected


def rescue_rankings(
    cache: Mapping[str, Any],
    tokens_by_query: Mapping[str, Sequence[set]],
    top_k: int,
    seed_k: int = 3,
) -> dict[str, dict[str, list[int]]]:
    """Rescue strategies at three bridge weights, keyed by strategy then query id."""
    names = [f"rescue-{beta}" for beta in (0.3, 0.5, 0.7)]
    out: dict[str, dict[str, list[int]]] = {name: {} for name in names}
    for q, payload in cache["queries"].items():
        tokens = tokens_by_query.get(q, [])
        for beta in (0.3, 0.5, 0.7):
            out[f"rescue-{beta}"][q] = rescue_order(payload["ce"], tokens, top_k, seed_k, beta)
    return out


def document_of(chunk_id: str) -> str:
    """A MultiHop-RAG chunk id is ``<url>#<position>``, so the article is the prefix."""
    return chunk_id.rsplit("#", 1)[0]


def doc_quota_order(
    scores: Sequence[float], docs: Sequence[str], top_k: int, min_docs: int
) -> list[int]:
    """CE order, but force the first ``min_docs`` slots onto distinct articles.

    The multi-hop failure mode is one strongly-matching article occupying the top-10 with
    overlapping chunks. A document quota spends early slots on breadth, then fills from
    the untouched candidates in CE order. Deterministic; ties break to the lower index.
    """
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    selected: list[int] = []
    seen: set[str] = set()
    for i in order:
        if len(selected) >= top_k or len(seen) >= min_docs:
            break
        if docs[i] in seen:
            continue
        seen.add(docs[i])
        selected.append(i)
    taken = set(selected)
    for i in order:
        if len(selected) >= top_k:
            break
        if i in taken:
            continue
        selected.append(i)
        taken.add(i)
    return selected


def doc_best_order(scores: Sequence[float], docs: Sequence[str], top_k: int) -> list[int]:
    """Round-robin across articles, best article first.

    Articles are ordered by their best candidate, then the selection takes one chunk from
    each article in turn, so a second article's best chunk is emitted before the first
    article's second-best. This is the "document-level aggregation" reading: the top-k
    should represent the *set* of articles the query needs, not one article's overlapping
    windows. Deterministic; ties break by document id then index.
    """
    groups: dict[str, list[int]] = {}
    for i, doc in enumerate(docs):
        groups.setdefault(doc, []).append(i)
    for indices in groups.values():
        indices.sort(key=lambda i: (-float(scores[i]), i))
    doc_order = sorted(groups, key=lambda doc: (-float(scores[groups[doc][0]]), doc))
    out: list[int] = []
    position = {doc: 0 for doc in groups}
    while len(out) < top_k:
        progressed = False
        for doc in doc_order:
            if position[doc] < len(groups[doc]):
                out.append(groups[doc][position[doc]])
                position[doc] += 1
                progressed = True
                if len(out) >= top_k:
                    break
        if not progressed:
            break
    return out


def coverage_doc_quota_order(
    scores: Sequence[float],
    atoms: Sequence[Sequence[int]],
    atom_weight: Sequence[float],
    docs: Sequence[str],
    top_k: int,
    lam: float = 0.7,
    min_docs: int = 3,
) -> list[int]:
    """Coverage selection seeded with the best candidate from distinct articles.

    Two measured wins are combined: a document quota (``docquota``) stops one article
    occupying the top-10 with overlapping chunks, and atom coverage (``coverage``) stops
    a slot being spent on a repeat of a fact already present. Phase 1 takes the best
    cross-encoder candidate of each of the top ``min_docs`` articles; phase 2 fills the
    rest greedily by ``lam * normalized_ce + (1 - lam) * marginal_atom_gain``.
    """
    n = len(scores)
    if n == 0 or top_k <= 0:
        return []
    rel = [float(value) for value in scores]
    lo, hi = min(rel), max(rel)
    normalized = [(value - lo) / (hi - lo) if hi > lo else 0.0 for value in rel]
    best: dict[str, float] = {}
    for i, doc in enumerate(docs):
        best[doc] = max(best.get(doc, -1e18), rel[i])
    doc_order = sorted(set(docs), key=lambda doc: (-best[doc], doc))

    selected: list[int] = []
    covered: set[int] = set()
    remaining = list(range(n))
    represented: set[str] = set()
    for doc in doc_order:
        if len(selected) >= top_k or len(represented) >= min_docs:
            break
        candidates = [i for i in remaining if docs[i] == doc]
        if not candidates:
            continue
        pick = max(candidates, key=lambda i: (rel[i], -i))
        selected.append(pick)
        covered.update(atoms[pick])
        remaining.remove(pick)
        represented.add(doc)

    while remaining and len(selected) < top_k:
        gains = {
            i: float(sum(atom_weight[a] for a in atoms[i] if a not in covered))
            for i in remaining
        }
        scale = max(gains.values()) if gains else 0.0
        scale = scale if scale > 0.0 else 1.0
        best_index, best_score = None, -1e18
        for i in remaining:
            score = lam * normalized[i] + (1.0 - lam) * (gains[i] / scale)
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        covered.update(atoms[best_index])
        remaining.remove(best_index)
    return selected


def doc_rankings(cache: Mapping[str, Any], top_k: int) -> dict[str, dict[str, list[int]]]:
    """Document-aware strategies, keyed by strategy then query id."""
    atom_weight = cache["_settings"]["atom_weight"]
    names = ["docquota-3", "docquota-4", "docbest", "covquota-3-0.5", "covquota-3-0.7"]
    out: dict[str, dict[str, list[int]]] = {name: {} for name in names}
    for q, payload in cache["queries"].items():
        scores = payload["ce"]
        atoms = payload["atoms"]
        docs = [document_of(chunk_id) for chunk_id in payload["pool"]]
        out["docquota-3"][q] = doc_quota_order(scores, docs, top_k, 3)
        out["docquota-4"][q] = doc_quota_order(scores, docs, top_k, 4)
        out["docbest"][q] = doc_best_order(scores, docs, top_k)
        for lam in (0.5, 0.7):
            out[f"covquota-3-{lam}"][q] = coverage_doc_quota_order(
                scores, atoms, atom_weight, docs, top_k, lam, 3
            )
    return out


def strategy_rankings(cache: Mapping[str, Any], top_k: int, rrf_k: int) -> dict[str, dict[str, list[int]]]:
    """Every strategy's top-k **pool indices**, keyed by strategy then query id.

    Strategies work over positions ``0..n-1`` rather than document ids so the result can
    be scored against the cached per-candidate gold matches without rebuilding the corpus.
    """
    atom_weight = cache["_settings"]["atom_weight"]
    names = ["fused", "ce", "blend-0.25", "blend-0.5", "blend-0.75",
             "guardrail-1", "guardrail-2",
             "coverage-0.3", "coverage-0.5", "coverage-0.7"]
    out: dict[str, dict[str, list[int]]] = {name: {} for name in names}
    for q, payload in cache["queries"].items():
        scores = payload["ce"]
        atoms = payload["atoms"]
        positions = list(range(len(payload["pool"])))
        out["fused"][q] = positions[:top_k]
        out["ce"][q] = [int(i) for i in ce_order(positions, scores)[:top_k]]
        for blend in (0.25, 0.5, 0.75):
            out[f"blend-{blend}"][q] = [
                int(i) for i in blend_order(positions, scores, blend, rrf_k)[:top_k]
            ]
        for pin in (1, 2):
            out[f"guardrail-{pin}"][q] = [
                int(i) for i in guardrail_order(positions, scores, pin)[:top_k]
            ]
        for lam in (0.3, 0.5, 0.7):
            out[f"coverage-{lam}"][q] = coverage_order(scores, atoms, atom_weight, top_k, lam)
    return out


def metrics_from_matches(
    ranked: Sequence[int],
    matches: Sequence[set],
    gold_count: int,
    top_k: int = 10,
) -> dict[str, float]:
    """The benchmark's own metric definitions, computed from per-candidate gold matches.

    Mirrors ``multihop_official.official_metrics``: a fact found at rank r contributes
    ``1/r`` to AP the first time it is found among the top-k, and MAP normalizes by
    ``min(len(gold), top_k)`` whether or not every fact is present in the pool.
    """
    hits10 = hits4 = False
    first: int | None = None
    found: set = set()
    ap = 0.0
    for rank, index in enumerate(ranked[:top_k], start=1):
        new = matches[index] - found
        if not new:
            continue
        if rank <= top_k:
            hits10 = True
        if rank <= 4:
            hits4 = True
        if first is None:
            first = rank
        ap += len(new) / rank
        found |= new
    denominator = min(max(gold_count, 1), top_k)
    return {
        "Hits@10": float(hits10),
        "Hits@4": float(hits4),
        "MAP@10": ap / denominator,
        "MRR@10": 1.0 / first if first else 0.0,
    }


def gold_matches_for_pool(
    corpus: Mapping[str, str], pool: Sequence[str], gold_facts: Sequence[str]
) -> list[set]:
    """Gold-fact index sets per pool candidate, using the benchmark's normalization."""
    normalized = [f.replace(" ", "").replace("\n", "") for f in gold_facts]
    matches: list[set] = []
    for doc_id in pool:
        text = corpus[doc_id].replace(" ", "").replace("\n", "")
        matches.append({j for j, fact in enumerate(normalized) if fact and fact in text})
    return matches


def holdout_selection(
    per_query: Mapping[str, Mapping[str, float]], seed: int
) -> dict[str, Any]:
    """Pick the best strategy on one half of the queries, score it on the other.

    Comparing ~20 rules on the same queries and quoting the winner's score is how a
    benchmark result turns into an artefact. This splits the queries, chooses by MAP@10 on
    half A only, and reports every strategy on the untouched half B, so the number that
    matters is the held-out one for the strategy that selection would have shipped.
    """
    import random

    names = list(per_query)
    if not names:
        return {}
    qids = sorted(per_query[names[0]])
    shuffled = list(qids)
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    set_a, set_b = set(shuffled[:half]), set(shuffled[half:])

    def mean(name: str, subset: set[str]) -> float:
        values = [per_query[name][q] for q in subset if q in per_query[name]]
        return round(sum(values) / len(values), 4) if values else 0.0

    means_a = {name: mean(name, set_a) for name in names}
    best = max(means_a, key=lambda name: means_a[name])
    held_out = {name: mean(name, set_b) for name in names}
    return {
        "seed": seed,
        "queries_a": len(set_a),
        "queries_b": len(set_b),
        "selected_on_a": best,
        "selected_a_score": means_a[best],
        "selected_b_score": held_out[best],
        "baseline_b": {
            name: held_out[name]
            for name in ("ce", "coverage-0.7", "covquota-3-0.7")
            if name in held_out
        },
    }


def coverage_query_order(
    scores: Sequence[float],
    atoms: Sequence[Sequence[int]],
    atom_weight: Sequence[float],
    atom_tokens: Sequence[set],
    query_terms: set,
    top_k: int,
    lam: float = 0.7,
    qboost: float = 1.0,
) -> list[int]:
    """Coverage selection with atoms that address the query weighted more.

    Plain coverage weights an atom by its length, so a long generic sentence counts as
    much as a short sentence that names the thing asked about. This multiplies each
    atom's weight by ``1 + qboost * overlap`` with the query's surface terms, so the
    marginal-gain term prefers *new query-relevant* facts. It reuses the same greedy rule
    via :func:`coverage_order`.
    """
    adjusted = list(atom_weight)
    if query_terms:
        touched: set[int] = set()
        for index, tokens in enumerate(atom_tokens):
            shared = len(tokens & query_terms)
            if shared:
                touched.add(index)
                adjusted[index] = atom_weight[index] * (1.0 + qboost * shared / len(query_terms))
        if not touched:
            adjusted = list(atom_weight)
    return coverage_order(scores, atoms, adjusted, top_k, lam)


def coverage_query_rankings(
    cache: Mapping[str, Any],
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    top_k: int,
    *,
    lam: float = 0.7,
    qboosts: Sequence[float] = (1.0, 3.0),
) -> dict[str, dict[str, list[int]]]:
    """Query-aware coverage strategies, rebuilding the atom index the cache points at."""
    from multihop_coverage import atom_index, split_atoms

    from laya_rag.lexical import STOPWORDS, tokenize

    atoms_by_doc = {doc_id: split_atoms(text) for doc_id, text in corpus.items()}
    atom_ids_of, atom_weight, atom_texts = atom_index(list(atoms_by_doc.values()))
    atom_ids_by_doc = dict(zip(atoms_by_doc, atom_ids_of))
    atom_tokens = [set(text.split()) for text in atom_texts]

    names = [f"coverage-q-{boost}" for boost in qboosts]
    out: dict[str, dict[str, list[int]]] = {name: {} for name in names}
    for q, payload in cache["queries"].items():
        query_terms = {
            t for t in tokenize(queries.get(q, ""), stem=False) if t not in STOPWORDS
        }
        for boost in qboosts:
            out[f"coverage-q-{boost}"][q] = coverage_query_order(
                payload["ce"], payload["atoms"], atom_weight, atom_tokens,
                query_terms, top_k, lam, boost,
            )
    return out


def truncate_pools(cache: Mapping[str, Any], limit: int | None) -> Mapping[str, Any]:
    """Truncate every cached pool to its first ``limit`` candidates, in place.

    The cache stores a fixed depth-100 pool; truncating models a narrower ``RETRIEVE_K``
    without re-running the cross-encoder, which is how the cost of a shallow first stage is
    measured here.
    """
    if not limit:
        return cache
    for payload in cache["queries"].values():
        payload["pool"] = payload["pool"][:limit]
        payload["ce"] = payload["ce"][:limit]
        payload["atoms"] = payload["atoms"][:limit]
    return cache


def filter_pools(cache: Mapping[str, Any], threshold: float | None) -> Mapping[str, Any]:
    """Drop cached candidates whose cross-encoder score is below ``threshold``, in place.

    The engine historically applied its Laya-calibrated ``RELEVANCE_THRESHOLD`` (0.5) to a
    substituted cross-encoder's sigmoid, which on this pool removes far too much; the cached
    cross-encoder scores here are raw logits, so ``threshold=0.0`` models a sigmoid cut of
    0.5 and ``None`` models the experiments' no-threshold behaviour.
    """
    if threshold is None:
        return cache
    for payload in cache["queries"].values():
        keep = [i for i, score in enumerate(payload["ce"]) if score >= threshold]
        payload["pool"] = [payload["pool"][i] for i in keep]
        payload["atoms"] = [payload["atoms"][i] for i in keep]
        payload["ce"] = [payload["ce"][i] for i in keep]
    return cache


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_rerank_experiment.py")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--seed-k", type=int, default=3, help="CE seeds kept before rescue fills")
    ap.add_argument(
        "--depth-limit", type=int, default=None,
        help="truncate each cached pool to this many candidates before scoring — models a "
             "narrower RETRIEVE_K without re-running the cross-encoder",
    )
    ap.add_argument(
        "--score-threshold", type=float, default=None,
        help="drop candidates whose cached cross-encoder logit is below this; 0.0 models "
             "the engine's historical sigmoid>=0.5 cut, omitting it is the experiments' "
             "no-threshold behaviour",
    )
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--holdout-seed", type=int, default=17,
                    help="split for choose-on-A / report-on-B selection")
    ap.add_argument("--out", default=str(ROOT_BENCH / "rerank-experiment"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cache_path = Path(str(args.cache) + ".json")
    if not cache_path.is_file():
        raise SystemExit(
            f"{cache_path} is missing; run tools/multihop_ce_cache.py first"
        )
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    truncate_pools(cache, args.depth_limit)
    filter_pools(cache, args.score_threshold)
    golds = cache["golds"]
    order = list(cache["queries"])
    overlap = cache["_settings"]["overlap"]

    # The cache stores pool ids; rebuild the corpus so a candidate's gold matches can be
    # recovered, then score every strategy against the same matches. No model, no network.
    raw = json.loads((ROOT_BENCH / "corpus.json").read_text(encoding="utf-8"))
    corpus: dict[str, str] = {}
    for article in raw:
        for j, text in enumerate(chunk_text(article["title"] + " " + article["body"], 256, overlap)):
            corpus[f"{article['url']}#{j}"] = text
    matches_by_query = {
        q: gold_matches_for_pool(corpus, payload["pool"], golds[q])
        for q, payload in cache["queries"].items()
    }
    # Bridge tokens are per chunk, so stem each chunk once and reuse across queries.
    tokens_of = {
        doc_id: {t for t in tokenize(text, stem=True) if t not in STOPWORDS}
        for doc_id, text in corpus.items()
    }
    tokens_by_query = {
        q: [tokens_of[cid] for cid in payload["pool"]]
        for q, payload in cache["queries"].items()
    }

    rankings = strategy_rankings(cache, args.top_k, args.rrf_k)
    rankings.update(rescue_rankings(cache, tokens_by_query, args.top_k, args.seed_k))
    rankings.update(doc_rankings(cache, args.top_k))
    item_queries = {
        str(i): x["query"]
        for i, x in enumerate(json.loads((ROOT_BENCH / "MultiHopRAG.json").read_text(encoding="utf-8")))
        if x["question_type"] != "null_query"
    }
    rankings.update(coverage_query_rankings(cache, corpus, item_queries, args.top_k))
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, dict[str, float]] = {}
    for name, ranking in rankings.items():
        rows: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
        for q in order:
            metrics = metrics_from_matches(
                ranking[q], matches_by_query[q], len(golds[q]), args.top_k
            )
            for metric in METRICS:
                rows[metric][q] = metrics[metric]
        per_query[name] = rows
        summary[name] = {m: round(sum(rows[m].values()) / len(rows[m]), 4) for m in METRICS}

    comparisons: dict[str, dict[str, Any]] = {}
    baselines = [name for name in ("fused", "ce", "coverage-0.7") if name in rankings]
    for baseline in baselines:
        for name in rankings:
            if name == baseline:
                continue
            for metric in ("MAP@10", "Hits@4"):
                key = f"{name} vs {baseline} {metric}"
                comparisons[key] = {
                    "bootstrap": evalkit.paired_bootstrap(
                        per_query[baseline][metric], per_query[name][metric], samples=args.samples
                    ),
                    "permutation": evalkit.permutation_test(
                        per_query[baseline][metric], per_query[name][metric], samples=args.samples
                    ),
                }

    result = {
        "_settings": {**cache["_settings"], "top_k": args.top_k, "rrf_k": args.rrf_k},
        "summary": summary,
        "comparisons": comparisons,
        "holdout": holdout_selection(
            {name: per_query[name]["MAP@10"] for name in rankings}, args.holdout_seed
        ),
        "per_query": {name: per_query[name]["MAP@10"] for name in rankings},
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), result)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Second-stage selection experiment (MultiHop-RAG subset)",
        "",
        f"- cache: {result['_settings']['limit_queries']} queries x "
        f"depth {result['_settings']['depth']}, reranker {result['_settings']['rerank_model']}",
        "",
        "| strategy | Hits@10 | Hits@4 | MAP@10 | MRR@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in result["summary"].items():
        lines.append(
            f"| {name} | {row['Hits@10']:.4f} | {row['Hits@4']:.4f} | "
            f"{row['MAP@10']:.4f} | {row['MRR@10']:.4f} |"
        )
    lines += ["", "## Paired comparisons (bootstrap 95% CI, permutation p)", ""]
    for key, value in sorted(result["comparisons"].items()):
        boot = value["bootstrap"]
        perm = value["permutation"]
        lines.append(
            f"- `{key}`: {boot['delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
            f"p={perm['p_value']:.4f} ({'sig' if boot['significant'] else 'ns'})"
        )
    lines += ["", "## Choose on half, report on the other", ""]
    holdout = result.get("holdout") or {}
    if holdout:
        lines.append(
            f"- selected on {holdout['queries_a']} queries: `{holdout['selected_on_a']}` "
            f"(MAP@10 {holdout['selected_a_score']:.4f})"
        )
        lines.append(
            f"- its held-out MAP@10 on the other {holdout['queries_b']}: "
            f"**{holdout['selected_b_score']:.4f}**"
        )
        for name, value in holdout.get("baseline_b", {}).items():
            lines.append(f"- held-out `{name}`: {value:.4f}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
