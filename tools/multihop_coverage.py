"""MultiHop-RAG MAP@10: can a coverage objective beat per-document rank?

The hybrid's weakness under the benchmark's own metric is MAP@10 — 0.251 against a
published 0.479 — because it finds *an* evidence passage, not *the set*. The failure mode
is a lexical channel locking onto one strongly-matching article and spending all ten slots
on it, precisely when MultiHop-RAG spreads its 2–4 gold facts across different articles.

This attacks that with diversification and coverage selection over the fused top-100,
rather than with a bigger model:

* ``mmr``           — Maximal Marginal Relevance on chunk embeddings: relevance from the
                      fused RRF score, redundancy from cosine similarity to what is chosen.
* ``doc-mmr``       — MMR whose redundancy term is "is this the same article", forcing the
                      list across the 609 documents.
* ``cluster-dedup`` — greedy fused order that skips a candidate too similar to one already
                      chosen (a near-duplicate / overlap chunk).
* ``query-coverage``— greedy marginal coverage of the query's own content terms.

Hyperparameters are **not** tuned on the evaluated queries: each family picks its setting on
a calibration half and is scored on the held-out half, with the full-set number shown only
as the optimistic bound it is.

    .venv/bin/python tools/multihop_coverage.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir, evalkit  # noqa: E402
from laya_rag.lexical import STOPWORDS, reciprocal_rank_fusion, tokenize  # noqa: E402

BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"

MMR_LAMBDAS = (0.3, 0.5, 0.7, 0.9)
DEDUP_THRESHOLDS = (0.8, 0.9, 0.95)
COVERAGE_LAMBDAS = (0.5, 0.8, 1.0)
DOC_QUOTA_MIN_DOCS = (2, 3, 4)
FACILITY_LAMBDAS = (0.3, 0.5, 0.7)
ATOM_MIN_CHARS = 25


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    toks = text.split()
    step = max(1, size - overlap)
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i : i + size]))
        if i + size >= len(toks):
            break
        i += step
    return out


def official_metrics_one(retrieved: list[str], gold: list[str]) -> dict[str, float]:
    gold = [g.replace(" ", "").replace("\n", "") for g in gold]
    retrieved = [r.replace(" ", "").replace("\n", "") for r in retrieved]
    f10 = f4 = False
    ap = 0.0
    first = None
    found: list[str] = []
    for rank, item in enumerate(retrieved[:11], start=1):
        if any(g in item for g in gold):
            if rank <= 10:
                f10 = True
                if first is None:
                    first = rank
                if rank <= 4:
                    f4 = True
                c = 0
                for g in gold:
                    if g in item and g not in found:
                        c += 1
                        found.append(g)
                ap += c / rank
    return {
        "Hits@10": float(f10),
        "Hits@4": float(f4),
        "MAP@10": ap / min(len(gold), 10),
        "MRR@10": 1 / first if first else 0.0,
    }


def summarize(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    return {key: round(float(np.mean(list(values.values()))), 4) for key, values in rows.items()}


# ── selection strategies over the fused candidate list ──────────────────

def mmr_select(ids, rel, gram, same_doc, top_k, lam, mode="embed"):
    """Greedy MMR. ``gram[i, j]`` is cosine(i, j); ``same_doc[i]`` the article bitmask free form."""
    selected: list[int] = []
    candidates = list(range(len(ids)))
    while candidates and len(selected) < top_k:
        best_index, best_score = None, -1e9
        for i in candidates:
            if selected:
                if mode == "doc":
                    redundancy = 1.0 if same_doc[i, selected].any() else 0.0
                else:
                    redundancy = float(np.max(gram[i, selected]))
            else:
                redundancy = 0.0
            score = lam * rel[i] - (1.0 - lam) * redundancy
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        candidates.remove(best_index)
    return selected


def cluster_dedup_select(gram, top_k, threshold):
    selected: list[int] = []
    for i in range(gram.shape[0]):
        if selected and float(np.max(gram[i, selected])) >= threshold:
            continue
        selected.append(i)
        if len(selected) >= top_k:
            break
    return selected


def query_coverage_select(rel, cand_tokens, query_terms, top_k, lam):
    if not query_terms:
        return list(range(top_k))
    covered: set[str] = set()
    selected: list[int] = []
    candidates = list(range(len(rel)))
    while candidates and len(selected) < top_k:
        best_index, best_score = None, -1e9
        for i in candidates:
            marginal = len(cand_tokens[i] - covered) / len(query_terms)
            score = lam * rel[i] + (1.0 - lam) * marginal
            if score > best_score:
                best_index, best_score = i, score
        covered |= cand_tokens[best_index]
        selected.append(best_index)
        candidates.remove(best_index)
    return selected


# ── evidence-set coverage: document quota and atom-level facility location ──

def doc_quota_select(rel: np.ndarray, docs: np.ndarray, top_k: int, min_docs: int) -> list[int]:
    """Greedy document-quota selection over the fused pool.

    The first pass walks candidates in descending relevance and takes the best-ranked
    candidate for each *new* document id until ``min_docs`` distinct documents are
    represented (or candidates run out). The second pass fills the remaining slots up
    to ``top_k`` from the untouched candidates, again in descending relevance order.
    Deterministic; ties are broken by lower index.
    """
    order = sorted(range(len(rel)), key=lambda i: (-float(rel[i]), i))
    selected: list[int] = []
    taken: set[int] = set()
    seen_docs: set = set()
    for i in order:
        if len(selected) >= top_k or len(seen_docs) >= min_docs:
            break
        doc = docs[i]
        if doc in seen_docs:
            continue
        seen_docs.add(doc)
        selected.append(i)
        taken.add(i)
    if len(selected) < top_k:
        for i in order:
            if len(selected) >= top_k:
                break
            if i in taken:
                continue
            selected.append(i)
            taken.add(i)
    return selected


def split_atoms(text: str, min_chars: int = ATOM_MIN_CHARS) -> list[str]:
    """Split a chunk into sentence-like "fact atoms" on ``[.!?]`` and newline boundaries."""
    atoms = []
    for part in re.split(r"(?<=[.!?])|\n", text):
        part = part.strip()
        if len(part) >= min_chars:
            atoms.append(part)
    return atoms


def _norm_atom(atom: str) -> str:
    return " ".join(atom.lower().split())


def atom_index(atom_lists: Sequence[Sequence[str]]) -> tuple[list[list[int]], list[float], list[str]]:
    """Global atom index: identical atoms across overlapping chunks share one id.

    Returns ``(per_candidate_atom_ids, atom_weight, unique_atom_texts)`` where each
    candidate's ids are unique and in first-seen order, and ``atom_weight`` (>= 0) is
    the normalized atom length so longer / more specific atoms count more. Deterministic
    and independent of any query or gold labels.
    """
    per_candidate: list[list[int]] = []
    weights: list[float] = []
    texts: list[str] = []
    id_of: dict[str, int] = {}
    for atoms in atom_lists:
        ids: list[int] = []
        seen: set[int] = set()
        for atom in atoms:
            norm = _norm_atom(atom)
            if not norm:
                continue
            atom_id = id_of.get(norm)
            if atom_id is None:
                atom_id = len(texts)
                id_of[norm] = atom_id
                texts.append(norm)
                weights.append(float(len(norm)))
            if atom_id not in seen:
                seen.add(atom_id)
                ids.append(atom_id)
        per_candidate.append(ids)
    return per_candidate, weights, texts


def facility_location_select(
    rel: np.ndarray,
    candidate_atom_ids: Sequence[Sequence[int]],
    atom_weight: Sequence[float],
    gram: np.ndarray,
    top_k: int,
    lam: float = 0.5,
) -> list[int]:
    """Greedy submodular facility-location / max-coverage selection over the pool.

    Marginal gain is the total weight of a candidate's atoms not yet covered. The raw
    gain is normalised by the largest available gain *at that step*, so it is in
    ``[0, 1]`` and comparable to the normalised relevance — without that, raw atom
    weights (character lengths) swamped relevance and the objective selected long,
    off-topic chunks. The score is ``lam * normalized_rel[i] +
    (1 - lam) * marginal_gain`` minus a redundancy penalty ``(1 - lam) *
    max(gram[i, selected])`` once anything is selected. ``rel`` is min-max normalised
    internally (all-zeros when constant). Deterministic; ties break to the lower index.
    """
    rel = np.asarray(rel, dtype=float)
    lo, hi = float(rel.min()), float(rel.max())
    normalized = (rel - lo) / (hi - lo) if hi > lo else np.zeros_like(rel)
    selected: list[int] = []
    covered: set[int] = set()
    candidates = list(range(len(rel)))
    while candidates and len(selected) < top_k:
        gains = {
            i: float(sum(atom_weight[a] for a in candidate_atom_ids[i] if a not in covered))
            for i in candidates
        }
        scale = max(gains.values()) if gains else 0.0
        scale = scale if scale > 0.0 else 1.0
        best_index, best_score = None, -1e18
        for i in candidates:
            marginal = gains[i] / scale
            redundancy = float(np.max(gram[i, selected])) if selected else 0.0
            score = lam * normalized[i] + (1.0 - lam) * marginal - (1.0 - lam) * redundancy
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        covered.update(candidate_atom_ids[best_index])
        candidates.remove(best_index)
    return selected


def pool_ceiling_map(rel: np.ndarray, gold_matches, top_k: int) -> float:
    """Best MAP@10 attainable by reordering the fused pool: the reorder ceiling.

    ``gold_matches[i]`` is the set of gold-fact ids matched by pool candidate ``i``.
    Greedily place the candidate with the most not-yet-found matches first, then the
    next, and so on; AP sums ``newly_found / rank`` over the selected ranks, divided by
    ``min(total distinct gold matches, top_k)``. Returns 0.0 when the pool matches
    nothing.
    """
    matches = [set(g) for g in gold_matches]
    total = len(set().union(*matches)) if matches else 0
    if total == 0 or top_k <= 0:
        return 0.0
    picked: list[int] = []
    found: set = set()
    candidates = list(range(len(matches)))
    while candidates and len(picked) < top_k:
        best_index, best_new = None, -1
        for i in candidates:
            new = len(matches[i] - found)
            if new > best_new:
                best_index, best_new = i, new
        picked.append(best_index)
        found |= matches[best_index]
        candidates.remove(best_index)
    ap = 0.0
    found = set()
    for rank, i in enumerate(picked, start=1):
        newly = matches[i] - found
        ap += len(newly) / rank
        found |= newly
    return ap / min(total, top_k)


# ── experiment ──────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_coverage.py")
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--candidates", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rrf-k", type=int, default=5)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit-queries", type=int, default=None, help="smoke runs only")
    ap.add_argument("--out", default=str(BENCH / "coverage"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    raw = json.load(open(BENCH / "corpus.json"))
    items_all = json.load(open(BENCH / "MultiHopRAG.json"))
    items = [(i, x) for i, x in enumerate(items_all) if x["question_type"] != "null_query"]
    if args.limit_queries:
        items = items[: args.limit_queries]
    golds = [[e["fact"] for e in x["evidence_list"]] for _, x in items]

    corpus: dict[str, str] = {}
    titles: dict[str, str] = {}
    for a in raw:
        for j, c in enumerate(chunk_text(a["title"] + " " + a["body"], 256, args.overlap)):
            cid = f"{a['url']}#{j}"
            corpus[cid] = c
            titles[cid] = a["title"]
    ids_order = list(corpus)
    row_of = {doc_id: i for i, doc_id in enumerate(ids_order)}
    doc_of = {doc_id: doc_id.rsplit("#", 1)[0] for doc_id in ids_order}
    queries = {str(i): x["query"] for i, x in items}
    print(f"[overlap={args.overlap}] {len(corpus)} chunks, {len(queries)} queries", flush=True)

    enc = beir.BiEncoder(args.model)
    vectors_all = beir.cached_embed(enc, corpus, BENCH / "cache" / f"multihop-chunks{args.overlap}.{enc.name}")
    lexical = beir.bm25_rankings(
        corpus, queries, args.candidates, stem=True, drop_stopwords=True, titles=titles, title_boost=2
    )
    dense = beir.dense_rankings(corpus, queries, enc, vectors_all, args.candidates)
    # Chunks recur across queries, so stem each one once instead of once per candidate.
    tokens_of = {doc_id: {t for t in tokenize(text, stem=True) if t not in STOPWORDS}
                 for doc_id, text in corpus.items()}
    print(f"  tokenised {len(tokens_of)} chunks", flush=True)
    # Evidence-set coverage: split each chunk into fact atoms once and index them
    # globally, so the same fact in two overlapping chunks is one atom id.
    atoms_by_doc = {doc_id: split_atoms(text) for doc_id, text in corpus.items()}
    atom_ids_of, atom_weight, _ = atom_index(list(atoms_by_doc.values()))
    atom_ids_by_doc = dict(zip(atoms_by_doc, atom_ids_of))
    print(f"  atomised {len(atoms_by_doc)} chunks -> {len(atom_weight)} atoms", flush=True)

    norm_gold = {
        str(i): [g.replace(" ", "").replace("\n", "") for g in gold]
        for (i, _), gold in zip(items, golds)
    }

    names = ["fused-baseline"]
    names += [f"mmr-{lam}" for lam in MMR_LAMBDAS]
    names += [f"doc-mmr-{lam}" for lam in MMR_LAMBDAS]
    names += [f"cluster-dedup-{t}" for t in DEDUP_THRESHOLDS]
    names += [f"query-coverage-{lam}" for lam in COVERAGE_LAMBDAS]
    names += [f"doc-quota-{n}" for n in DOC_QUOTA_MIN_DOCS]
    names += [f"facility-{lam}" for lam in FACILITY_LAMBDAS]
    rankings: dict[str, dict[str, list[str]]] = {name: {} for name in names}
    pool_ceiling: dict[str, float] = {}

    total = len(queries)
    for n, (q, text) in enumerate(queries.items()):
        fused = reciprocal_rank_fusion([dense[q], lexical[q]], k=args.rrf_k, limit=args.candidates, weights=[1.0, 1.0])
        cand = [d for d, _ in fused]
        scores = np.array([s for _, s in fused], dtype=float)
        lo, hi = float(scores.min()), float(scores.max())
        rel = (scores - lo) / (hi - lo) if hi > lo else np.ones_like(scores)
        vecs = vectors_all[[row_of[d] for d in cand]]
        gram = vecs @ vecs.T
        docs = np.array([doc_of[d] for d in cand])
        same_doc = docs[:, None] == docs[None, :]
        terms = {t for t in tokenize(text, stem=True) if t not in STOPWORDS}
        cand_tokens = [tokens_of[d] & terms for d in cand]
        cand_atom_ids = [atom_ids_by_doc[d] for d in cand]

        rankings["fused-baseline"][q] = cand[: args.top_k]
        for lam in MMR_LAMBDAS:
            sel = mmr_select(cand, rel, gram, same_doc, args.top_k, lam, "embed")
            rankings[f"mmr-{lam}"][q] = [cand[i] for i in sel]
            sel = mmr_select(cand, rel, gram, same_doc, args.top_k, lam, "doc")
            rankings[f"doc-mmr-{lam}"][q] = [cand[i] for i in sel]
        for threshold in DEDUP_THRESHOLDS:
            sel = cluster_dedup_select(gram, args.top_k, threshold)
            rankings[f"cluster-dedup-{threshold}"][q] = [cand[i] for i in sel]
        for lam in COVERAGE_LAMBDAS:
            sel = query_coverage_select(rel, cand_tokens, terms, args.top_k, lam)
            rankings[f"query-coverage-{lam}"][q] = [cand[i] for i in sel]
        for n in DOC_QUOTA_MIN_DOCS:
            sel = doc_quota_select(rel, docs, args.top_k, n)
            rankings[f"doc-quota-{n}"][q] = [cand[i] for i in sel]
        for lam in FACILITY_LAMBDAS:
            sel = facility_location_select(rel, cand_atom_ids, atom_weight, gram, args.top_k, lam)
            rankings[f"facility-{lam}"][q] = [cand[i] for i in sel]

        # Pool ceiling diagnostic: the best MAP@10 any reordering of the fused pool could
        # reach, using the same normalized-substring gold match as official_metrics_one.
        gold_matches = []
        for d in cand:
            text_norm = corpus[d].replace(" ", "").replace("\n", "")
            gold_matches.append({j for j, g in enumerate(norm_gold[q]) if g in text_norm})
        pool_ceiling[q] = pool_ceiling_map(rel, gold_matches, args.top_k)
        if n and n % 500 == 0:
            print(f"  {n}/{total}", flush=True)

    per_query: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, dict[str, float]] = {}
    for name, ranking in rankings.items():
        rows: dict[str, dict[str, float]] = {k: {} for k in ("Hits@10", "Hits@4", "MAP@10", "MRR@10")}
        for i, gold in zip((i for i, _ in items), golds):
            qid = str(i)
            row = official_metrics_one([corpus[d] for d in ranking[qid]], gold)
            for key, value in row.items():
                rows[key][qid] = value
        per_query[name] = rows
        summary[name] = summarize(rows)

    base_map = per_query["fused-baseline"]["MAP@10"]
    comparisons = {}
    for name in names:
        if name == "fused-baseline":
            continue
        comparisons[name] = {
            "MAP@10": evalkit.paired_bootstrap(base_map, per_query[name]["MAP@10"], samples=args.bootstrap_samples),
            "Hits@10": evalkit.paired_bootstrap(
                per_query["fused-baseline"]["Hits@10"], per_query[name]["Hits@10"], samples=args.bootstrap_samples
            ),
        }

    # Held-out tuning: pick each family's setting on a calibration half by MAP@10, score
    # the other half; average the two directions so every query is scored out-of-sample.
    order = list(queries)
    random.Random(args.seed).shuffle(order)
    half = len(order) // 2
    halves = (order[:half], order[half:])
    families = {
        "mmr": [f"mmr-{lam}" for lam in MMR_LAMBDAS],
        "doc-mmr": [f"doc-mmr-{lam}" for lam in MMR_LAMBDAS],
        "cluster-dedup": [f"cluster-dedup-{t}" for t in DEDUP_THRESHOLDS],
        "query-coverage": [f"query-coverage-{lam}" for lam in COVERAGE_LAMBDAS],
        "doc-quota": [f"doc-quota-{n}" for n in DOC_QUOTA_MIN_DOCS],
        "facility": [f"facility-{lam}" for lam in FACILITY_LAMBDAS],
    }
    transfer = {}
    for family, variants in families.items():
        pooled, base_pooled, chosen = [], [], []
        for calib, test in (halves, halves[::-1]):
            scores = {v: float(np.mean([per_query[v]["MAP@10"][q] for q in calib])) for v in variants}
            pick = max(scores, key=scores.get)
            chosen.append(pick)
            pooled.extend(per_query[pick]["MAP@10"][q] for q in test)
            base_pooled.extend(per_query["fused-baseline"]["MAP@10"][q] for q in test)
        transfer[family] = {
            "chosen": chosen,
            "held_out_MAP@10": round(float(np.mean(pooled)), 4),
            "baseline_MAP@10": round(float(np.mean(base_pooled)), 4),
            "delta": round(float(np.mean(pooled) - np.mean(base_pooled)), 4),
            "paired_ci": evalkit.paired_bootstrap(
                {str(i): v for i, v in enumerate(base_pooled)},
                {str(i): v for i, v in enumerate(pooled)},
                samples=args.bootstrap_samples,
            ),
        }

    report = {
        "settings": vars(args),
        "summary": summary,
        "comparisons_vs_fused": comparisons,
        "held_out_tuning": transfer,
        "per_query_MAP@10": {name: rows["MAP@10"] for name, rows in per_query.items()},
        "pool_ceiling_MAP@10": round(float(np.mean(list(pool_ceiling.values()))), 4),
    }
    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    print("\nfull-set summary (MAP@10):")
    for name, row in sorted(summary.items(), key=lambda kv: -kv[1]["MAP@10"]):
        print(f"  {name:22s} MAP@10={row['MAP@10']:.4f} Hits@10={row['Hits@10']:.4f} Hits@4={row['Hits@4']:.4f}")
    print("\nheld-out tuned (MAP@10):")
    for family, row in transfer.items():
        print(f"  {family:16s} {row['chosen']} -> {row['held_out_MAP@10']:.4f} "
              f"(baseline {row['baseline_MAP@10']:.4f}, delta {row['delta']:+.4f})")
    print(f"\npool ceiling MAP@10 (best any reorder of the fused pool): {report['pool_ceiling_MAP@10']:.4f}")
    print("\nwrote", out.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
