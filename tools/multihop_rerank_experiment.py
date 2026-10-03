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

With ``--tri-cache`` (from ``tools/multihop_tri_cache.py``) it adds the bridge-conditioned
second stage: Laya's relevance decision asked as ``s(q, b, c)`` rather than ``s(q, c)``, where
``b`` is the passage hop 1 found. That is the one piece of BridgeRAG (arXiv:2604.03384) this
engine can run without a generator, and the shuffled-bridge arm is the control that says
whether a gain is conditioning or just a longer instruction. ``pit-<name>-ce`` fuses the two
channels by percentile rank, which is the fusion BridgeRAG prefers over min-max.

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


def pit_order(
    score_lists: Sequence[Sequence[float]],
    weights: Sequence[float],
    top_k: int,
) -> list[int]:
    """Fuse channels by percentile-rank (PIT), not by their raw scores.

    Each channel is mapped to the empirical percentile of each candidate — best candidate
    1.0, worst 0.0, ties sharing their average rank — and the weighted sum of percentiles
    orders the pool. BridgeRAG reports percentile-rank fusion as directionally more robust
    than min-max normalisation on multi-hop benchmarks (arXiv:2604.03384, citing Bacellar
    2026), which matters here because the relevance decision, the cross-encoder logit and
    the fused recall score live on three different scales: min-max lets one outlier set the
    scale, percentile-rank cannot.
    """
    if not score_lists:
        return []
    if len(score_lists) != len(weights):
        raise ValueError(
            f"got {len(score_lists)} channels for {len(weights)} weights"
        )
    n = len(score_lists[0])
    if any(len(scores) != n for scores in score_lists):
        raise ValueError("every channel must score the same number of candidates")
    if n == 0 or top_k <= 0:
        return []
    fused = [0.0] * n
    for scores, weight in zip(score_lists, weights):
        values = [float(s) for s in scores]
        order = sorted(range(n), key=lambda i: (-values[i], i))
        # Average rank for ties, then a percentile in [0, 1] with the best at 1.0.
        percentile = [0.0] * n
        position = 0
        while position < n:
            end = position
            while end + 1 < n and values[order[end + 1]] == values[order[position]]:
                end += 1
            average = (position + end) / 2.0
            share = 1.0 - (average / (n - 1)) if n > 1 else 1.0
            for k in range(position, end + 1):
                percentile[order[k]] = share
            position = end + 1
        for i in range(n):
            fused[i] += float(weight) * percentile[i]
    return sorted(range(n), key=lambda i: (-fused[i], i))[:top_k]


def bridge_filter_order(
    scores: Sequence[float],
    top_k: int,
    threshold: float,
) -> list[int]:
    """Keep candidates whose bridge-conditioned score clears ``threshold``, in pool order.

    This is the *filter* use of the decision, and it is a different hypothesis from ranking by
    the same number. The repository's own measured prior is that Laya's relevance decision is
    a stronger filter than an orderer — reordering BM25's candidates by it lowered nDCG@10 on
    both BEIR datasets (SciFact 0.654 -> 0.470), while thresholding by it is what the engine
    ships. Asking the conditional question is the same decision asked a question that is not
    structurally blind to the bridge, so if the score carries signal at all, this is where the
    prior says it should show up.

    Pool positions are already in fused order, so returning them ascending preserves the
    first-stage ranking among the survivors.
    """
    return [i for i, score in enumerate(scores) if float(score) >= threshold][:top_k]


def tri_rankings(
    cache: Mapping[str, Any],
    tri_scores: Mapping[str, Sequence[float]],
    top_k: int,
    rrf_k: int,
    name: str,
) -> dict[str, dict[str, list[int]]]:
    """Selection rules built on the bridge-conditioned score, keyed by strategy.

    ``<name>`` orders by the conditional score alone; the blended, coverage and
    coverage-plus-quota variants mirror the cross-encoder strategies so the two second
    stages are compared under identical rules. ``pit-<name>-ce`` is BridgeRAG's
    percentile-rank fusion of the two channels, which is the variant that does not have to
    choose between them.
    """
    atom_weight = cache["_settings"]["atom_weight"]
    names = [
        name,
        f"{name}-coverage-0.5",
        f"{name}-coverage-0.7",
        f"{name}-covquota-3-0.5",
        f"{name}-covquota-3-0.7",
        f"{name}-filter-0.3",
        f"{name}-filter-0.5",
        f"ce-{name}-filter-0.3",
        f"ce-{name}-filter-0.5",
        f"ce-{name}-top40",
        f"pit-{name}-ce",
    ]
    out: dict[str, dict[str, list[int]]] = {key: {} for key in names}
    for q, payload in cache["queries"].items():
        scores = [float(s) for s in tri_scores[q]]
        if len(scores) != len(payload["pool"]):
            raise ValueError(
                f"query {q}: {len(scores)} bridge scores for "
                f"{len(payload['pool'])} pool candidates — the caches disagree"
            )
        atoms = payload["atoms"]
        docs = [document_of(chunk_id) for chunk_id in payload["pool"]]
        positions = list(range(len(scores)))
        ce_positions = [int(i) for i in ce_order(positions, payload["ce"])]
        out[name][q] = [int(i) for i in ce_order(positions, scores)[:top_k]]
        for lam in (0.5, 0.7):
            out[f"{name}-coverage-{lam}"][q] = coverage_order(
                scores, atoms, atom_weight, top_k, lam
            )
            out[f"{name}-covquota-3-{lam}"][q] = coverage_doc_quota_order(
                scores, atoms, atom_weight, docs, top_k, lam, 3
            )
        for threshold in (0.3, 0.5):
            # Fused order among the conditional survivors, then the cross-encoder order among
            # them — the engine's real filter shape is the second one, and it is a different
            # hypothesis from "the decision orders the list".
            out[f"{name}-filter-{threshold}"][q] = bridge_filter_order(
                scores, top_k, threshold
            )
            out[f"ce-{name}-filter-{threshold}"][q] = [
                i for i in ce_positions if scores[i] >= threshold
            ][:top_k]
        # A *relative* cut, because an absolute calibrated probability is exactly the thing the
        # repo measured as not transferring across domains.
        top40 = set(sorted(positions, key=lambda i: (-scores[i], i))[:40])
        out[f"ce-{name}-top40"][q] = [i for i in ce_positions if i in top40][:top_k]
        out[f"pit-{name}-ce"][q] = pit_order(
            [scores, payload["ce"]], [1.0, 1.0], top_k
        )
    return out


def atom_weight_rankings(
    cache: Mapping[str, Any],
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    top_k: int,
    *,
    lam: float = 0.7,
    beam: bool = False,
    beam_width: int = 4,
) -> dict[str, dict[str, list[int]]]:
    """Selection rules over a better atom weighting, and optionally beam search.

    ``atom_index`` weights an atom by its character length. Two replacements are measured
    here: the IDF of its terms (a gold fact is specific, so its terms are rare), either as a
    mean over distinct terms (length-invariant) or a sum (favours more distinct terms). Beam
    search is the other half of the estimator question — greedy commits to one marginal gain
    at a time and never reconsiders — and it is opt-in because it costs ``beam_width`` times
    the greedy work.
    """
    from multihop_coverage import (
        atom_index,
        beam_facility_location_select,
        coverage_beam_order,
        idf_atom_weight,
        split_atoms,
    )

    atoms_by_doc = {doc_id: split_atoms(text) for doc_id, text in corpus.items()}
    atom_ids_of, char_weight, atom_texts = atom_index(list(atoms_by_doc.values()))
    idf_mean = idf_atom_weight(atom_texts, reduction="mean")
    idf_sum = idf_atom_weight(atom_texts, reduction="sum")

    names = [
        f"idf-mean-{lam}", f"idf-sum-{lam}",
        f"idf-mean-covquota-3-{lam}", f"idf-sum-covquota-3-{lam}",
    ]
    if beam:
        # Two beam variants: the plain global set objective, and beam search over the
        # *same* greedy objective (beam_width=1 == greedy), so the lookahead is attributable.
        names += [f"beam-{lam}", f"beam-idf-{lam}", f"beam-greedy-{lam}", f"beam-greedy-idf-{lam}"]
    out: dict[str, dict[str, list[int]]] = {name: {} for name in names}
    for q, payload in cache["queries"].items():
        scores = payload["ce"]
        atoms = payload["atoms"]
        docs = [document_of(chunk_id) for chunk_id in payload["pool"]]
        out[f"idf-mean-{lam}"][q] = coverage_order(scores, atoms, idf_mean, top_k, lam)
        out[f"idf-sum-{lam}"][q] = coverage_order(scores, atoms, idf_sum, top_k, lam)
        out[f"idf-mean-covquota-3-{lam}"][q] = coverage_doc_quota_order(
            scores, atoms, idf_mean, docs, top_k, lam, 3
        )
        out[f"idf-sum-covquota-3-{lam}"][q] = coverage_doc_quota_order(
            scores, atoms, idf_sum, docs, top_k, lam, 3
        )
        if beam:
            out[f"beam-{lam}"][q] = beam_facility_location_select(
                scores, atoms, char_weight, top_k, lam, beam_width=beam_width
            )
            out[f"beam-idf-{lam}"][q] = beam_facility_location_select(
                scores, atoms, idf_mean, top_k, lam, beam_width=beam_width
            )
            out[f"beam-greedy-{lam}"][q] = coverage_beam_order(
                scores, atoms, char_weight, top_k, lam, beam_width=beam_width
            )
            out[f"beam-greedy-idf-{lam}"][q] = coverage_beam_order(
                scores, atoms, idf_sum, top_k, lam, beam_width=beam_width
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
    ap.add_argument("--atom-lam", type=float, default=0.7,
                    help="coverage weight for the IDF / beam selector variants")
    ap.add_argument("--beam", action="store_true",
                    help="also run beam search over the coverage objective (costs "
                         "beam_width x the greedy work; slow on the full protocol)")
    ap.add_argument("--beam-width", type=int, default=4)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--holdout-seed", type=int, default=17,
                    help="split for choose-on-A / report-on-B selection")
    ap.add_argument(
        "--tri-cache", action="append", default=[], metavar="PATH",
        help="bridge-conditioned score cache from tools/multihop_tri_cache.py; repeatable. "
             "Each file's settings name its arm and wording, and adds that arm's rules",
    )
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
    for path in args.tri_cache:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        settings = payload.get("_settings") or {}
        arm = str(settings.get("arm") or "real")
        wording = str(settings.get("wording") or "bridge")
        name = "tri" if arm == "real" else f"tri-{arm}"
        scores = payload["scores"]
        missing = [q for q in order if q not in scores]
        if missing:
            raise SystemExit(
                f"{path}: {len(missing)} of {len(order)} queries have no bridge score "
                f"(first missing: {sorted(missing)[:3]}); rescore with matching "
                "--limit-queries, or narrow this experiment with --depth-limit"
            )
        print(f"  {name}: bridge scores for {len(scores)} queries (wording {wording})")
        rankings.update(tri_rankings(cache, scores, args.top_k, args.rrf_k, name))
    item_queries = {
        str(i): x["query"]
        for i, x in enumerate(json.loads((ROOT_BENCH / "MultiHopRAG.json").read_text(encoding="utf-8")))
        if x["question_type"] != "null_query"
    }
    rankings.update(coverage_query_rankings(cache, corpus, item_queries, args.top_k))
    rankings.update(atom_weight_rankings(
        cache, corpus, item_queries, args.top_k,
        lam=args.atom_lam, beam=args.beam, beam_width=args.beam_width,
    ))
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
    baselines = [
        name
        for name in ("fused", "ce", "coverage-0.7", "covquota-3-0.7")
        if name in rankings
    ]
    # The conditional arms are baselines too: `tri` asks whether any rule beats the treatment,
    # and `tri-shuffled` is the control that says whether the treatment is the bridge or just
    # the longer instruction. Without it in the list the report cannot answer its own question.
    baselines += [name for name in ("tri", "tri-shuffled") if name in rankings]
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
        "_settings": {
            **cache["_settings"],
            "top_k": args.top_k,
            "rrf_k": args.rrf_k,
            "tri_caches": [Path(p).name for p in args.tri_cache],
        },
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
