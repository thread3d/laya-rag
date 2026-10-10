"""Comparison charts: this engine against other retrieval and RAG systems.

Every figure here is a *comparison across systems* rather than an internal ablation, and every one
is generated from something already on disk or pinned with a source:

* this repository's own reports under `eval/` (per-dataset nDCG@10, the MultiHop-RAG protocol, the
  cross-encoder arms), read through `tools/consistency_check.py`'s registries so the coverage
  membership is the gate's, not a hand-typed list;
* the published BEIR-18 table in `tools/published_beir18.py`, parsed from the paper's HTML;
* the survey of open-source systems in `../.research/repo-metadata.json` (stars, language, last
  push, archived), classified by an explicit table in this module.

Three rules the figures follow, because the surrounding documents are held to them:

1. **Membership is stated.** A system's macro-average is drawn with the dataset set it is over, and
   anything of ours measured on a different set is annotated as not comparable rather than placed
   on the same axis silently.
2. **A missing report is a gap, not a zero.** Cells and points we have not measured are drawn as
   gaps; a chart that skips silently is the failure mode `RESEARCH.md` §8 exists to prevent.
3. **No number is typed into a chart.** Values are read from reports or from the pinned table, and
   `--check` pins the table's own arithmetic (the published average equals the published rows).

    .venv/bin/python tools/charts_compare.py --charts docs/charts
    .venv/bin/python tools/charts_compare.py --check      # no files, no matplotlib
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import compare as cmp                    # noqa: E402  (style, published 4-set constants)
import consistency_check as cc           # noqa: E402  (the coverage registries)
import published_beir18 as published     # noqa: E402  (the parsed BEIR-18 table)

BEIR_DIR = ROOT / "eval" / "beir"
MH_DIR = ROOT / "eval" / "benchmarks" / "multihop-rag"
OSS_METADATA = ROOT.parent / ".research" / "repo-metadata.json"

OURS, OUR_ALT, PUB, PUB_DARK, ACCENT = (
    cmp.OURS, cmp.OURS_ALT, cmp.PUBLISHED, cmp.PUBLISHED_DARK, cmp.ACCENT
)

#: BEIR-18 in the order the benchmark's own overview table lists them.
BEIR_ORDER: tuple[str, ...] = (
    "trec-covid", "bioasq", "nfcorpus", "nq", "hotpotqa", "fiqa", "signal1m", "trec-news",
    "robust04", "arguana", "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity",
    "scidocs", "fever", "climate-fever", "scifact",
)

LABEL = {
    "trec-covid": "TREC-COVID", "bioasq": "BioASQ", "nfcorpus": "NFCorpus", "nq": "NQ",
    "hotpotqa": "HotpotQA", "fiqa": "FiQA-2018", "signal1m": "Signal-1M",
    "trec-news": "TREC-NEWS", "robust04": "Robust04", "arguana": "ArguAna",
    "webis-touche2020": "Touché-2020", "cqadupstack": "CQADupStack", "quora": "Quora",
    "dbpedia-entity": "DBPedia", "scidocs": "SCIDOCS", "fever": "FEVER",
    "climate-fever": "Climate-FEVER", "scifact": "SciFact",
}

#: Why a canonical member has no report here. Licensing and gating are stated per dataset, not
#: summarised, because the four blocks have three different causes.
STATE_REASON = {
    "robust04": "blocked (licensing)", "trec-news": "blocked (licensing)",
    "signal1m": "blocked (licensing)", "bioasq": "blocked (gated)",
    "cqadupstack": "public, needs an adapter",
}

#: Retrievers that mean a dataset has a dense (or learned-sparse) arm, not only lexical.
DENSE_RETRIEVERS = frozenset({
    "biencoder", "dense", "hybrid", "hybrid-biencoder", "dense-ensemble", "hybrid-ensemble",
    "splade", "hybrid-splade",
})

#: BRIGHT's published leader scores ~21 nDCG@10 macro over its 12 reasoning-intensive subsets
#: (arXiv:2407.12883). The comparison on this page is deliberately not like-for-like: ours is a
#: 33M encoder with no reranker on 6 of the 12 subsets, and the repository quotes it as such.
BRIGHT_PUBLISHED_BEST = 0.21
BRIGHT_REPORTS = ("eval/bright/report.json", "eval/bright/report-small.json")
#: The adapted Laya encoder's BRIGHT run. Merged in as an extra arm when present; a checkout
#: without it gets exactly the previous two-report figure.
BRIGHT_ADAPTED_REPORT = "eval/bright/report-laya.json"

#: Published encoder sizes in millions of parameters, from the paper's model sections and the
#: model cards. BM25 has none; the two `bge-small` rows of ours are 33M.
PARAMS = {"BM25": None, "uniCOIL": 110, "SPLADE": 110, "TAS-B": 66, "Contriever": 110}
OURS_PARAMS = 33

#: A curated classification of the surveyed open-source field, keyed by the exact `full_name` in
#: `repo-metadata.json`. It is explicit so it can be argued with: a name absent from it is charted
#: as `other` rather than guessed at, and `--check` reports how many rows fell through.
CATEGORY = {
    # vector index / ANN
    "facebookresearch/faiss": "vector index",
    "nmslib/hnswlib": "vector index",
    "unum-cloud/USearch": "vector index",
    "microsoft/DiskANN": "vector index",
    # search engine / vector database
    "qdrant/qdrant": "search engine / vector DB",
    "milvus-io/milvus": "search engine / vector DB",
    "weaviate/weaviate": "search engine / vector DB",
    "chroma-core/chroma": "search engine / vector DB",
    "lancedb/lancedb": "search engine / vector DB",
    "lance-format/lance": "search engine / vector DB",
    "vespa-engine/vespa": "search engine / vector DB",
    "marqo-ai/marqo": "search engine / vector DB",
    "typesense/typesense": "search engine / vector DB",
    "meilisearch/meilisearch": "search engine / vector DB",
    "zincsearch/zincsearch": "search engine / vector DB",
    "elastic/elasticsearch": "search engine / vector DB",
    "opensearch-project/OpenSearch": "search engine / vector DB",
    "paradedb/paradedb": "search engine / vector DB",
    "neuml/txtai": "search engine / vector DB",
    # lexical retrieval
    "apache/lucene": "lexical retrieval",
    "quickwit-oss/tantivy": "lexical retrieval",
    "quickwit-oss/tantivy-py": "lexical retrieval",
    "castorini/anserini": "lexical retrieval",
    "castorini/pyserini": "lexical retrieval",
    "pisa-engine/pisa": "lexical retrieval",
    "xhluca/bm25s": "lexical retrieval",
    # learned retrieval (dense, sparse, late interaction) and their benchmarks
    "naver/splade": "learned retrieval",
    "stanford-futuredata/ColBERT": "learned retrieval",
    "lightonai/pylate": "learned retrieval",
    "AnswerDotAI/RAGatouille": "learned retrieval",
    "rasyosef/splade-index": "learned retrieval",
    "bizreach-inc/light-splade": "learned retrieval",
    "huggingface/sentence-transformers": "learned retrieval",
    "FlagOpen/FlagEmbedding": "learned retrieval",
    "beir-cellar/beir": "learned retrieval",
    # rerankers
    "AnswerDotAI/rerankers": "reranker",
    "PrithivirajDamodaran/FlashRank": "reranker",
    # RAG frameworks and orchestration
    "run-llama/llama_index": "RAG framework",
    "deepset-ai/haystack": "RAG framework",
    "stanfordnlp/dspy": "RAG framework",
    "infiniflow/ragflow": "RAG framework",
    "IntelLabs/fastRAG": "RAG framework",
    "neuml/paperetl": "RAG framework",
    # graph RAG
    "microsoft/graphrag": "graph RAG",
    "HKUDS/LightRAG": "graph RAG",
    "gusye1234/nano-graphrag": "graph RAG",
    "OSU-NLP-Group/HippoRAG": "graph RAG",
    "OpenSPG/KAG": "graph RAG",
    "circlemind-ai/fast-graphrag": "graph RAG",
    "parthsarthi03/raptor": "graph RAG",
}

#: Case-insensitive lookup, because the metadata's spelling is GitHub's, not ours.
_CATEGORY_BY_LOWER = {name.lower(): category for name, category in CATEGORY.items()}


def category_of(name: str) -> str:
    """The curated category for a repository, or ``other`` when it is not in the table."""
    return _CATEGORY_BY_LOWER.get(str(name).lower(), "other")


CATEGORY_ORDER = ("learned retrieval", "search engine / vector DB", "RAG framework", "graph RAG",
                  "lexical retrieval", "vector index", "reranker", "other")


# ── loaders ───────────────────────────────────────────────────────────────

def _report_scored(report: Mapping[str, Any]) -> int | None:
    """How many queries a report scored, when it says so."""
    scored = report.get("scored_queries")
    return int(scored) if isinstance(scored, int) else None


def our_rows(directory: Path | None = None) -> dict[str, dict[str, Any]]:
    """Per dataset: the best nDCG@10 row and the shipped lexical row, from **full** reports.

    Three ways a row can be disqualified, because each has actually happened here:

    * a superseded report, skipped through the same registry `make consistency-check` uses;
    * a report whose *name* says it is a subset (`hotpotqa-subset`), or that scored fewer queries
      than another report for the same dataset — the HotpotQA 600-query estimate reads 0.6365
      against the full 7,405-query run's 0.6352, so a naive maximum charts an estimate as a result;
    * a seeded subset is used only when it is all there is, and then ``from_subset`` says so.
    """
    beir = BEIR_DIR if directory is None else Path(directory)
    entries: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(beir.glob("*.json")):
        if path.stem in cc.SUPERSEDED or path.stem in {"coverage", "lexical-control"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        datasets = payload.get("datasets")
        if not isinstance(datasets, Mapping):
            continue
        for name, report in datasets.items():
            if not isinstance(report, Mapping):
                continue
            retr = report.get("retrievers")
            if not isinstance(retr, Mapping):
                continue
            rows = {
                key: float(row["nDCG@10"])
                for key, row in retr.items()
                if isinstance(row, Mapping) and isinstance(row.get("nDCG@10"), (int, float))
            }
            if rows:
                entries.setdefault(name, []).append({
                    "report": path.stem,
                    "rows": rows,
                    "scored": _report_scored(report),
                    "named_subset": "subset" in path.stem,
                })

    out: dict[str, dict[str, Any]] = {}
    for name, group in entries.items():
        most_scored = max((item["scored"] for item in group if item["scored"] is not None), default=None)

        def usable(item: Mapping[str, Any]) -> bool:
            if item["named_subset"]:
                return False
            if most_scored is not None and item["scored"] is not None and item["scored"] < most_scored:
                return False
            return True

        chosen = [item for item in group if usable(item)]
        from_subset = not chosen
        chosen = chosen or group
        merged: dict[str, float] = {}
        for item in chosen:
            for key, value in item["rows"].items():
                if key not in merged or value > merged[key]:
                    merged[key] = value
        best = max(merged.items(), key=lambda pair: pair[1]) if merged else None
        out[name] = {
            "best": best,
            "lexical": ("bm25-engine", merged["bm25-engine"]) if "bm25-engine" in merged else None,
            "retrievers": sorted(merged),
            "has_dense": bool(DENSE_RETRIEVERS & set(merged)),
            "from_subset": from_subset,
            "reports": sorted(item["report"] for item in chosen),
        }
    return {name: out[name] for name in sorted(out)}


def coverage_states(
    measured: Sequence[str],
    extras: Sequence[str],
    dense: Sequence[str],
) -> list[dict[str, str]]:
    """One row per canonical dataset plus the extras, with how this repository stands on it.

    Pure: the caller passes the computed sets, so the classification is testable without a
    checkout's reports, and the chart cannot disagree with `make consistency-check` about which
    datasets are measured.
    """
    measured_set, dense_set = set(measured), set(dense)
    rows: list[dict[str, str]] = []
    for name in BEIR_ORDER:
        if name in measured_set:
            reason = "measured with a dense arm" if name in dense_set else "measured, lexical only"
            kind = "measured-dense" if name in dense_set else "measured-lexical"
        else:
            reason = STATE_REASON.get(name, "no report")
            kind = "needs-adapter" if name == "cqadupstack" else "blocked"
        rows.append({"dataset": name, "kind": kind, "reason": reason})
    for name in extras:
        rows.append({"dataset": name, "kind": "extra",
                     "reason": "measured, but not one of the 18 (the training collection)"})
    return rows


def macro(values: Sequence[float]) -> dict[str, float]:
    """A plain macro average with its count — the count is the membership and travels with it."""
    numbers = [float(value) for value in values]
    if not numbers:
        return {"mean": 0.0, "n": 0}
    return {"mean": round(sum(numbers) / len(numbers), 4), "n": len(numbers)}


def our_macro(
    rows: Mapping[str, Mapping[str, Any]],
    measured: Sequence[str] | None = None,
) -> dict[str, float]:
    """This engine's macro over **BEIR-18 members only**.

    ``our_rows`` also carries corpora that are not part of the benchmark (MS MARCO is measured as
    an extra), and averaging one of those into a "of 18" figure is exactly the membership slip the
    coverage gate exists to catch. The member list comes from `make consistency-check` unless the
    caller passes one, so the figure cannot disagree with the gate.
    """
    names = list(cc.measured_beir()["measured"]) if measured is None else list(measured)
    values = [rows[name]["best"][1] for name in names if rows.get(name, {}).get("best")]
    return macro(values)


def lexical_gap(rows: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Our lexical control minus the published BM25 (multi-field) row, per dataset.

    Both sides are BM25, but not the same BM25: the published column indexes title and body as
    separate fields and this repository's control is flat. The comparison is therefore of *rank
    quality under two BM25 conventions*, which is what the repository's lexical-control discussion
    is about — it is not a like-for-like reading of one retriever.
    """
    out: list[dict[str, Any]] = []
    for source, canonical in published.CANONICAL.items():
        lexical = (rows.get(canonical) or {}).get("lexical")
        anchor = published.NDCG10[source]["BM25"]
        if not lexical:
            continue
        out.append({"dataset": canonical, "ours": lexical[1], "published": anchor,
                    "gap": round(lexical[1] - anchor, 4)})
    return sorted(out, key=lambda row: row["gap"])


def ce_deltas(stage2: Path | None = None) -> list[dict[str, Any]]:
    """The cross-encoder's nDCG@10 change per corpus, from the reports that measured it.

    BEIR rows come from `stage2-rerank.json` (the fused stage against the full re-sort), and
    MultiHop-RAG from the rerank experiment (fused against the cross-encoder over one pool).
    """
    out: list[dict[str, Any]] = []
    path = (BEIR_DIR / "stage2-rerank.json") if stage2 is None else Path(stage2)
    if path.is_file():
        datasets = (json.loads(path.read_text(encoding="utf-8")).get("datasets") or {})
        for name, report in datasets.items():
            retr = (report or {}).get("retrievers") or {}
            base, ce = retr.get("hybrid-biencoder"), retr.get("crossencoder")
            if isinstance(base, Mapping) and isinstance(ce, Mapping):
                out.append({"corpus": name, "base": float(base["nDCG@10"]),
                            "ce": float(ce["nDCG@10"]),
                            "delta": round(float(ce["nDCG@10"]) - float(base["nDCG@10"]), 4)})
    experiment = MH_DIR / "rerank-experiment-full.json"
    if experiment.is_file():
        summary = json.loads(experiment.read_text(encoding="utf-8")).get("summary") or {}
        fused, ce = summary.get("fused"), summary.get("ce")
        if isinstance(fused, Mapping) and isinstance(ce, Mapping):
            out.append({"corpus": "multihop-rag", "base": float(fused["MAP@10"]),
                        "ce": float(ce["MAP@10"]),
                        "delta": round(float(ce["MAP@10"]) - float(fused["MAP@10"]), 4)})
    return out


def oss_rows(path: Path | None = None, now: datetime | None = None) -> list[dict[str, Any]]:
    """The surveyed open-source systems, classified and dated, one row per usable entry.

    Entries whose fetch did not complete are dropped rather than charted at zero stars: the
    metadata file is regenerated by `.research/repo-metadata.sh`, and a failed fetch writes the
    string `None`, which would otherwise draw a cluster of fake points at the left edge.
    """
    source = OSS_METADATA if path is None else Path(path)
    if not source.is_file():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    moment = now or datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for item in payload:
        name = str(item.get("full_name") or "")
        stars = item.get("stars")
        if not name or name == "None" or not isinstance(stars, int):
            continue
        pushed = str(item.get("pushed_at") or "")
        try:
            when = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
            days = max(0, (moment - when).days)
        except ValueError:
            days = None
        rows.append({
            "name": name,
            "stars": int(stars),
            "language": str(item.get("language") or "—"),
            "days_since_push": days,
            "archived": bool(item.get("archived")),
            "category": category_of(name),
        })
    return rows


def oss_totals(path: Path | None = None) -> dict[str, int]:
    """Usable entries against entries surveyed, so a chart can say how much of the field it drew."""
    source = OSS_METADATA if path is None else Path(path)
    if not source.is_file():
        return {"usable": 0, "surveyed": 0}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {"usable": len(oss_rows(path)), "surveyed": len(payload)}


def category_totals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Projects and stars per category, largest star total first."""
    counts: Counter[str] = Counter()
    stars: Counter[str] = Counter()
    for row in rows:
        category = str(row.get("category") or "other")
        counts[category] += 1
        stars[category] += int(row.get("stars") or 0)
    order = sorted(stars, key=lambda name: (-stars[name], name))
    return [{"category": name, "projects": counts[name], "stars": stars[name]} for name in order]


def multihop_shipped_row(official: Mapping[str, Any]) -> Mapping[str, Any]:
    """Our shipped MultiHop-RAG hybrid row, preferring the adapted encoder's when measured."""
    candidates = [key for key in official
                  if key.endswith(" hybrid@ov200")
                  and ("adapted" in key.lower() or "laya" in key.lower())]
    candidates.append("hybrid (ours)@ov200")
    for key in candidates:
        if official.get(key):
            return official[key]
    return {}


def multihop_gaps(official: Mapping[str, Any], published_rows: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    """Our MultiHop-RAG row minus each published row, per metric."""
    metrics = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")
    ours = multihop_shipped_row(official)
    out: list[dict[str, Any]] = []
    for name, h10, h4, mp, mrr in published_rows:
        theirs = {"Hits@10": h10, "Hits@4": h4, "MAP@10": mp, "MRR@10": mrr}
        out.append({
            "system": name,
            "gaps": {metric: round(float(ours.get(metric, 0.0)) - float(theirs[metric]), 4)
                     for metric in metrics},
        })
    return out


# ── figures ───────────────────────────────────────────────────────────────

def chart_beir_matrix(outdir: Path, plt: Any) -> bool:
    """Six systems across all 18 BEIR datasets.

    Every one of the 18 is measured here, so a dash in *our* row would be a bug, not a gap —
    `make consistency-check` is the gate for that. A dash in a published system's row means that
    system does not report that dataset; the five published systems between them cover 18, but
    not each of them 18.
    """
    rows = our_rows()
    datasets = list(BEIR_ORDER)
    names = ["this engine (best measured row)"] + [f"{s} (published)" for s in published.SYSTEMS]
    values = []
    ours = [float((rows.get(name, {}).get("best") or (None, float("nan")))[1]) if rows.get(name, {}).get("best")
            else float("nan") for name in datasets]
    values.append(ours)
    for system in published.SYSTEMS:
        values.append([
            published.NDCG10[source][system]
            for source in (next(s for s, c in published.CANONICAL.items() if c == name) for name in datasets)
        ])
    import numpy as np

    grid = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(13.2, 4.4))
    cmap = plt.get_cmap("YlGnBu").with_extremes(bad="#f2f2f2")
    masked = np.ma.masked_invalid(grid)
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.10, vmax=0.90)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = grid[row, col]
            if np.isnan(value):
                ax.text(col, row, "—", ha="center", va="center", fontsize=7, color="#999999")
                continue
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if value > 0.60 else "#222222",
                    fontweight="bold" if row == 0 else "normal")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([LABEL[name] for name in datasets], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("BEIR nDCG@10 by system and dataset — this engine on all 18; "
                 "a dash is a cell the published system does not report")
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.85, label="nDCG@10")
    cmp._save(fig, outdir, "cmp_beir18_matrix.png", plt)
    return True


def chart_beir_macro(outdir: Path, plt: Any) -> bool:
    """Macro averages with their membership stated: a 13-dataset mean is not an 18-dataset mean."""
    import numpy as np

    rows = our_rows()
    ours13 = our_macro(rows)
    systems = list(published.SYSTEMS)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    left, right = axes

    bars = [(name, published.AVERAGE[name]) for name in systems]
    bars.sort(key=lambda item: -item[1])
    positions = np.arange(len(bars))
    left.bar(positions, [value for _, value in bars], 0.62, color=PUB, label="published, 18 datasets")
    left.bar([len(bars) + 0.4], [ours13["mean"]], 0.62, color=OURS,
             label=f"this engine, {ours13['n']} datasets")
    left.set_xticks(list(positions) + [len(bars) + 0.4])
    left.set_xticklabels([name for name, _ in bars] + [f"this engine\n({ours13['n']} of 18)"],
                         fontsize=8, rotation=30, ha="right")
    left.set_ylim(0.0, 0.60)
    left.set_ylabel("mean nDCG@10")
    left.set_title(f"{published.SOURCE.split(',')[0]}: 18-dataset macro")
    left.legend(fontsize=8, frameon=False)
    left.annotate("different membership — not comparable",
                  xy=(len(bars) + 0.4, ours13["mean"]), xytext=(len(bars) - 2.6, 0.55),
                  fontsize=8, color=ACCENT,
                  arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))

    try:
        four_best = [float(np.mean(cmp.per_query(cmp.BEST_PER_DATASET[d][0], d,
                                                  cmp.BEST_PER_DATASET[d][1])))
                     for d in cmp.FOUR]
        four_ens = [float(np.mean(cmp.per_query_ensemble(d))) for d in cmp.FOUR]
        ours4 = [(f"this engine — best per dataset", macro(four_best)["mean"]),
                 (f"this engine — ensemble + BM25", macro(four_ens)["mean"])]
    except Exception:                                     # reports absent: leave the panel empty
        ours4 = []
    published4 = [(f"{name} ({params})", value) for name, params, value in cmp.PUBLISHED_4]
    combined = ours4 + published4
    combined.sort(key=lambda item: -item[1])
    positions = np.arange(len(combined))
    colours = [OURS if name.startswith("this engine") else PUB for name, _ in combined]
    right.bar(positions, [value for _, value in combined], 0.62, color=colours)
    right.set_xticks(positions)
    right.set_xticklabels([name for name, _ in combined], fontsize=7.5, rotation=40, ha="right")
    right.set_ylim(0.0, 0.58)
    right.set_title("SciFact, NFCorpus, ArguAna, SCIDOCS — 4-dataset macro")
    right.set_ylabel("mean nDCG@10")
    fig.suptitle("Macro averages are only comparable inside one dataset set", fontsize=11,
                 fontweight="bold")
    cmp._save(fig, outdir, "cmp_beir18_macro.png", plt)
    return True


def bright_rows(reports: Sequence[Path] | None = None) -> list[dict[str, Any]]:
    """BRIGHT subsets this repository has run, with the three arms' nDCG@10.

    The two reports cover six of the benchmark's twelve subsets between them; a missing report is
    skipped, so the figure shrinks rather than inventing a subset.
    """
    paths = [ROOT / name for name in BRIGHT_REPORTS] if reports is None else list(reports)
    out: list[dict[str, Any]] = []
    for path in paths:
        if not Path(path).is_file():
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for subset, report in (payload.get("subsets") or {}).items():
            arms = {
                arm: float(values["nDCG@10"])
                for arm, values in (report.get("summary") or {}).items()
                if isinstance(values, Mapping) and isinstance(values.get("nDCG@10"), (int, float))
            }
            if arms:
                out.append({"subset": subset, "documents": report.get("documents"),
                            "queries": report.get("queries"), "arms": arms})
    # The adapted Laya encoder's dense arm is merged in beside the bge one rather than
    # replacing it: on BRIGHT the adapted channel is far below BM25, so a figure that showed
    # only it — or only the bge arm — would misstate the trade. An explicit `reports` list is
    # honoured exactly, so a caller that pins its own files gets no surprise arm.
    adapted_path = ROOT / BRIGHT_ADAPTED_REPORT
    if reports is None and adapted_path.is_file():
        adapted = json.loads(adapted_path.read_text(encoding="utf-8")).get("subsets") or {}
        for row in out:
            report = adapted.get(row["subset"]) or {}
            dense = (report.get("summary") or {}).get("dense", {})
            if isinstance(dense.get("nDCG@10"), (int, float)):
                row["arms"]["dense (Laya adapted)"] = float(dense["nDCG@10"])
    return out


def chart_bright(outdir: Path, plt: Any) -> bool:
    """BRIGHT by subset and arm, against the published leader — with the incomparability stated."""
    import numpy as np

    rows = bright_rows()
    if not rows:
        print("charts-compare: skip cmp_bright.png (no BRIGHT reports)")
        return False
    rows.sort(key=lambda row: row["subset"])
    wanted = ("bm25", "dense", "dense (Laya adapted)", "hybrid")
    arms = [arm for arm in wanted if all(arm in row["arms"] for row in rows)] or \
           sorted({arm for row in rows for arm in row["arms"]})
    positions = np.arange(len(rows))
    width = 0.8 / len(arms)
    colours = {"bm25": OUR_ALT, "dense": "#f0c674", "dense (Laya adapted)": "#e08b3a",
               "hybrid": OURS}
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for index, arm in enumerate(arms):
        ax.bar(positions + (index - (len(arms) - 1) / 2) * width,
               [row["arms"].get(arm, 0.0) for row in rows], width, label=arm,
               color=colours.get(arm, PUB_DARK))
    ax.axhline(BRIGHT_PUBLISHED_BEST, color=ACCENT, ls="--", lw=1.1)
    ax.annotate(f"published leader ≈ {BRIGHT_PUBLISHED_BEST:.2f} (12 subsets, much larger models)",
                xy=(len(rows) - 1.2, BRIGHT_PUBLISHED_BEST + 0.006), fontsize=8, color=ACCENT)
    macro_hybrid = sum(row["arms"].get("hybrid", 0.0) for row in rows) / len(rows)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{row['subset']}\n({row['queries']} q)" for row in rows], fontsize=8)
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0.0, 0.30)
    ax.set_title(f"BRIGHT on {len(rows)} of 12 subsets — hybrid macro {macro_hybrid:.4f} "
                 f"with a 33M encoder and no reranker")
    ax.legend(fontsize=8, frameon=False)
    cmp._save(fig, outdir, "cmp_bright.png", plt)
    return True


def chart_lexical_gap(outdir: Path, plt: Any) -> bool:
    """Our flat BM25 minus the paper's multi-field BM25 — a convention gap, not the control.

    This is deliberately **not** the repository's lexical control, which compares our flat `bm25`
    against the flat Pyserini anchors pinned in `tools/compare.py`. Here both sides are BM25 but
    under different indexing conventions, and one row is a different collection release
    (Touché-2020 v2), so the spread is a measure of how much the convention costs rather than a
    statement about the retriever. `lexical_vs_published.png` is the control.
    """
    rows = lexical_gap(our_rows())
    if not rows:
        print("charts-compare: skip cmp_lexical_gap.png (no lexical rows)")
        return False
    import numpy as np

    positions = np.arange(len(rows))
    colours = [ACCENT if row["gap"] < 0 else OURS for row in rows]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.barh(positions, [row["gap"] for row in rows], 0.62, color=colours)
    for position, row in zip(positions, rows):
        offset = -0.004 if row["gap"] < 0 else 0.004
        ax.text(row["gap"] + offset, position, f"{row['gap']:+.3f}",
                va="center", ha="right" if row["gap"] < 0 else "left", fontsize=8)
    ax.axvline(0.0, color="#444444", lw=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels([LABEL.get(row["dataset"], row["dataset"]) for row in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.155, 0.105)
    within = sum(1 for row in rows if abs(row["gap"]) <= 0.02)
    ax.set_xlabel("our BM25 (flat) − published BM25 (multi-field), nDCG@10")
    ax.set_title(f"Two BM25 conventions, not two systems: {within} of {len(rows)} corpora inside "
                 f"±0.02 — the control is lexical_vs_published.png")
    touche = next((row for row in rows if row["dataset"] == "webis-touche2020"), None)
    if touche is not None:
        ax.annotate("Touché here is the v2 release,\nnot the anchor this repo quotes",
                    xy=(touche["gap"], positions[rows.index(touche)]), xytext=(-0.14, 8.6),
                    fontsize=7.5, color=ACCENT,
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    cmp._save(fig, outdir, "cmp_lexical_gap.png", plt)
    return True


def chart_parameters(outdir: Path, plt: Any) -> bool:
    """Quality against encoder size: the published 18-set systems, and where this engine sits."""
    import numpy as np

    rows = our_rows()
    ours = our_macro(rows)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for name in published.SYSTEMS:
        params = PARAMS.get(name)
        if params is None:
            continue
        ax.scatter([params], [published.AVERAGE[name]], s=54, color=PUB_DARK, zorder=4)
        ax.annotate(f"{name} ({params}M)", (params, published.AVERAGE[name]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.scatter([OURS_PARAMS], [ours["mean"]], s=110, color=OURS, marker="*", zorder=5)
    ax.annotate(f"this engine, bge-small {OURS_PARAMS}M\n({ours['n']} datasets, not comparable)",
                (OURS_PARAMS, ours["mean"]), textcoords="offset points", xytext=(8, -18),
                fontsize=8, color=OURS, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("encoder parameters, millions (log scale)")
    ax.set_ylabel("mean nDCG@10")
    ax.set_title("Quality against encoder size — published systems on all 18, this engine on its own set")
    ax.set_ylim(0.38, 0.56)
    cmp._save(fig, outdir, "cmp_parameters.png", plt)
    return True


def chart_coverage(outdir: Path, plt: Any) -> bool:
    """What of the benchmark is measured, and why the rest is not."""
    measured = cc.measured_beir()
    rows = our_rows()
    dense = [name for name in measured["measured"] if rows.get(name, {}).get("has_dense")]
    states = coverage_states(measured["measured"], measured["extras"], dense)
    colours = {"measured-dense": OURS, "measured-lexical": OUR_ALT, "blocked": "#d9d9d9",
               "needs-adapter": "#f0c674", "extra": "#8fbf8f"}
    legend = {"measured-dense": "measured, with a dense arm",
              "measured-lexical": "measured, lexical only",
              "blocked": "blocked (licensing or gated)",
              "needs-adapter": "public, needs an adapter",
              "extra": "measured extra (not one of the 18)"}
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    labels, kinds = [row["dataset"] for row in states], [row["kind"] for row in states]
    positions = range(len(states))
    ax.barh(list(positions), [1.0] * len(states), color=[colours[kind] for kind in kinds], height=0.72)
    for position, row in zip(positions, states):
        ax.text(1.02, position, row["reason"], va="center", fontsize=7.5, color="#333333")
    ax.set_yticks(list(positions))
    ax.set_yticklabels([LABEL.get(name, name) for name in labels], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.05)
    ax.set_xticks([])
    ax.set_title(f"BEIR-18 coverage, computed by make consistency-check "
                 f"({measured['count']} of {measured['total']} measured)")
    # Only the states actually present: the legend is read as a key, so a colour with no bar under
    # it (an adapter-needing member that is now measured, say) is noise rather than information.
    present = [kind for kind in legend if kind in set(kinds)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[kind]) for kind in present]
    ax.legend(handles, [legend[kind] for kind in present],
              fontsize=7.5, frameon=False, loc="lower right")
    cmp._save(fig, outdir, "cmp_coverage.png", plt)
    return True


def chart_multihop_delta(outdir: Path, plt: Any) -> bool:
    """Our MultiHop-RAG row minus each published row: where the published systems are still ahead."""
    import numpy as np

    path = MH_DIR / "official.json"
    if not path.is_file():
        print("charts-compare: skip cmp_multihop_delta.png (no official.json)")
        return False
    official = json.loads(path.read_text(encoding="utf-8"))
    gaps = multihop_gaps(official, cmp.PUBLISHED_MH)
    metrics = ["Hits@10", "Hits@4", "MAP@10", "MRR@10"]
    positions = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    colours = [OURS, OUR_ALT, "#6f9bc0"]
    for index, row in enumerate(gaps):
        deltas = [row["gaps"][metric] for metric in metrics]
        ax.bar(positions + (index - 1) * width, deltas, width, label=row["system"], color=colours[index])
    ax.axhline(0.0, color="#444444", lw=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("our row − published row")
    ax.set_ylim(-0.35, 0.22)
    ax.set_title("MultiHop-RAG against the paper's Table 5 — we win the top-k, they win MAP@10")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.annotate("the measured gap", xy=(2.0, -0.17), xytext=(1.15, -0.30), fontsize=8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    cmp._save(fig, outdir, "cmp_multihop_delta.png", plt)
    return True


def chart_reranker(outdir: Path, plt: Any) -> bool:
    """The cross-encoder's effect per corpus: a loss on every BEIR set, a gain on multi-hop."""
    import numpy as np

    rows = ce_deltas()
    if not rows:
        print("charts-compare: skip cmp_reranker.png (no stage2-rerank.json)")
        return False
    rows.sort(key=lambda row: row["delta"])
    positions = np.arange(len(rows))
    colours = [ACCENT if row["delta"] < 0 else OURS for row in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.bar(positions, [row["delta"] for row in rows], 0.6, color=colours)
    for position, row in zip(positions, rows):
        ax.text(position, row["delta"] + (0.004 if row["delta"] >= 0 else -0.010),
                f"{row['delta']:+.3f}", ha="center", fontsize=8,
                color="#333333")
    ax.axhline(0.0, color="#444444", lw=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABEL.get(row["corpus"], row["corpus"]) for row in rows],
                       fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("cross-encoder Δ nDCG@10 (MAP@10 on MultiHop)")
    ax.set_title("A second stage is a per-corpus instrument: it loses on every BEIR set and wins on multi-hop")
    ax.set_ylim(-0.36, 0.07)
    cmp._save(fig, outdir, "cmp_reranker.png", plt)
    return True


def chart_oss_landscape(outdir: Path, plt: Any) -> bool:
    """The surveyed open-source field: size against activity, by category.

    Recency is clipped: seven projects have not been pushed in years and would otherwise stretch
    the axis until every live project sat on one line. Clipped points are drawn as triangles and
    the threshold is stated on the axis, so nothing is hidden — only bounded.
    """
    import numpy as np

    stale_days = 540
    rows = [row for row in oss_rows() if row["days_since_push"] is not None]
    totals = oss_totals()
    if not rows:
        print("charts-compare: skip cmp_oss_landscape.png (no .research/repo-metadata.json)")
        return False
    cmap = plt.get_cmap("tab10")
    colours = {name: cmap(index) for index, name in enumerate(CATEGORY_ORDER)}
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    for category in CATEGORY_ORDER:
        group = [row for row in rows if row["category"] == category]
        if not group:
            continue
        fresh = [row for row in group if row["days_since_push"] <= stale_days]
        stale = [row for row in group if row["days_since_push"] > stale_days]
        if fresh:
            ax.scatter([max(1, row["stars"]) for row in fresh],
                       [row["days_since_push"] for row in fresh],
                       s=44, color=colours[category], label=f"{category} ({len(group)})",
                       alpha=0.85, edgecolor="white", linewidth=0.6)
        if stale:
            ax.scatter([max(1, row["stars"]) for row in stale], [stale_days] * len(stale),
                       s=52, color=colours[category], marker="^", alpha=0.9,
                       edgecolor="white", linewidth=0.6)
    for row in rows:
        if row["archived"]:
            ax.scatter([max(1, row["stars"])], [min(row["days_since_push"], stale_days)],
                       marker="x", s=46, color="#333333", zorder=4)
    # Labels are placed by hand: the biggest projects are also the freshest, so an automatic
    # "top N" annotation stacks them on one pixel row. Each entry names its own offset.
    label_offsets = {
        "infiniflow/ragflow": ((8, 8), "left"),
        "elastic/elasticsearch": ((-8, -2), "right"),
        "meilisearch/meilisearch": ((6, -14), "left"),
        "run-llama/llama_index": ((-6, 12), "right"),
        "naver/splade": ((6, -4), "left"),
        "parthsarthi03/raptor": ((6, 6), "left"),
    }
    by_name = {row["name"]: row for row in rows}
    for name, (offset, align) in label_offsets.items():
        row = by_name.get(name)
        if row is None:
            continue
        ax.annotate(name, (max(1, row["stars"]), min(row["days_since_push"], stale_days)),
                    textcoords="offset points", xytext=offset, fontsize=7, ha=align)
    ax.set_xscale("log")
    ax.set_ylim(stale_days + 60, -60)
    ax.set_yticks([0, 90, 180, 270, 360, 450, stale_days])
    ax.set_yticklabels(["0", "3 mo", "6 mo", "9 mo", "12 mo", "15 mo", "18 mo+"])
    ax.set_xlabel("GitHub stars (log scale)")
    ax.set_ylabel("time since the last push")
    ax.set_title(f"The surveyed open-source field — {totals['usable']} of {totals['surveyed']} "
                 f"entries carried metadata; triangles are 18 months or staler")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    cmp._save(fig, outdir, "cmp_oss_landscape.png", plt)
    return True


def chart_oss_categories(outdir: Path, plt: Any) -> bool:
    """Stars and project counts per category, so the field's centre of mass is visible."""
    import numpy as np

    rows = oss_rows()
    totals = [row for row in category_totals(rows) if row["category"] != "other"]
    if not totals:
        print("charts-compare: skip cmp_oss_categories.png (no .research/repo-metadata.json)")
        return False
    positions = np.arange(len(totals))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.barh(positions, [row["stars"] for row in totals], color=OUR_ALT)
    for position, row in zip(positions, totals):
        ax.text(row["stars"] + 4000, position, f"{row['projects']} projects", va="center", fontsize=8)
    ax.set_yticks(positions)
    ax.set_yticklabels([row["category"] for row in totals], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("total GitHub stars across surveyed projects")
    ax.set_title("Where the open-source retrieval field's attention sits")
    ax.set_xlim(0, max(row["stars"] for row in totals) * 1.28)
    cmp._save(fig, outdir, "cmp_oss_categories.png", plt)
    return True


# ── offline invariants ─────────────────────────────────────────────────────

def self_check() -> list[str]:
    """Invariants that need no checkout and no matplotlib.

    The published table is the load-bearing input: if its own arithmetic does not reproduce, or a
    dataset loses its canonical name, every figure drawn from it is wrong in a way no chart shows.
    """
    problems: list[str] = []
    datasets = [name for name in published.NDCG10 if name != "Avg."]
    if len(datasets) != 18:
        problems.append(f"the published table has {len(datasets)} datasets, expected 18")
    if set(published.CANONICAL) != set(datasets):
        problems.append("CANONICAL does not map exactly the published dataset names")
    if set(published.CANONICAL.values()) != set(BEIR_ORDER):
        problems.append("CANONICAL's targets are not the canonical BEIR-18 set")
    for system in published.SYSTEMS:
        values = [published.NDCG10[name][system] for name in datasets]
        if any(not 0.0 <= value <= 1.0 for value in values):
            problems.append(f"{system} has a value outside [0, 1]")
        mean = sum(values) / len(values)
        if abs(mean - published.AVERAGE[system]) > 0.002:
            problems.append(
                f"{system}: the paper's average {published.AVERAGE[system]} is not the mean of its "
                f"18 rows ({mean:.4f}) beyond rounding"
            )
    states = coverage_states(["scifact"], ["msmarco"], ["scifact"])
    kinds = {row["dataset"]: row["kind"] for row in states}
    if kinds.get("scifact") != "measured-dense":
        problems.append("a measured dataset with a dense arm was not classified as such")
    if kinds.get("bioasq") != "blocked" or kinds.get("cqadupstack") != "needs-adapter":
        problems.append("the unimplemented members are not classified by their reason")
    if kinds.get("msmarco") != "extra":
        problems.append("an extra dataset was not classified as an extra")
    if len(states) != 19:
        problems.append(f"coverage has {len(states)} rows, expected 18 members plus the extra")
    if macro([0.2, 0.4]) != {"mean": 0.3, "n": 2}:
        problems.append("macro() does not average with its count")
    if macro([]) != {"mean": 0.0, "n": 0}:
        problems.append("macro() of nothing is not the empty result")
    member_rows = {"scifact": {"best": ("x", 0.70)}, "msmarco": {"best": ("y", 0.10)}}
    if our_macro(member_rows, ["scifact"]) != {"mean": 0.7, "n": 1}:
        problems.append("our_macro() let a non-member corpus into a BEIR-18 average")
    totals = category_totals([
        {"category": "graph RAG", "stars": 10},
        {"category": "graph RAG", "stars": 5},
        {"category": "lexical retrieval", "stars": 1},
    ])
    if totals[0] != {"category": "graph RAG", "projects": 2, "stars": 15}:
        problems.append("category_totals does not roll up counts and stars")
    gaps = multihop_gaps({"hybrid (ours)@ov200": {"Hits@10": 0.80, "Hits@4": 0.60,
                                                  "MAP@10": 0.25, "MRR@10": 0.50}},
                         [("published", 0.70, 0.60, 0.40, 0.45)])
    if abs(gaps[0]["gaps"]["MAP@10"] + 0.15) > 1e-9 or gaps[0]["gaps"]["Hits@10"] != 0.1:
        problems.append("multihop_gaps does not subtract per metric")
    return problems


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="charts_compare.py")
    ap.add_argument("--charts", default=str(ROOT / "docs" / "charts"))
    ap.add_argument("--check", action="store_true",
                    help="run the offline invariants and exit; reads no reports, needs no matplotlib")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        problems = self_check()
        if problems:
            print("charts-compare --check: FAILED")
            for line in problems:
                print(f"  - {line}")
            return 1
        print("ok: charts-compare invariants hold (18 datasets, arithmetic, classification)")
        return 0

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "eval" / ".mplconfig"))
    try:
        plt = cmp._plt_mod()
    except Exception as exc:                          # pragma: no cover - optional dependency
        print(f"charts-compare: matplotlib unavailable ({exc}); skipping")
        return 0
    outdir = Path(args.charts)
    outdir.mkdir(parents=True, exist_ok=True)
    wrote = False
    for render in (chart_beir_matrix, chart_beir_macro, chart_parameters, chart_coverage,
                   chart_lexical_gap, chart_bright, chart_multihop_delta, chart_reranker,
                   chart_oss_landscape, chart_oss_categories):
        wrote = render(outdir, plt) or wrote
    if not wrote:
        print("charts-compare: nothing to render")
    return 0


if __name__ == "__main__":
    sys.exit(main())
