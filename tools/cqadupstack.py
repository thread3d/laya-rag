"""CQADupStack: twelve sub-forums behind one BEIR name.

`tools/beir.py` models **one corpus per dataset**, and CQADupStack is not one corpus. The BEIR
release ships it as twelve independent collections — `android`, `english`, `gaming`, `gis`,
`mathematica`, `physics`, `programmers`, `stats`, `tex`, `unix`, `webmasters`, `wordpress` — each with
its own `corpus.jsonl`, `queries.jsonl` and `qrels/test.tsv`, and the benchmark's published number
for the member is the **macro-average of the twelve sub-forum scores**, not a run over a pooled
corpus. That is why MTEB carries a single aggregated `CQADupstackRetrieval` task rather than twelve
leaderboard rows.

Pooling the twelve would not be a shortcut but a *different measurement*: a query from `tex` could
then retrieve `physics` documents, which the per-forum qrels never judge. So each sub-forum is run
independently through the same ranking and evaluation code `beir.py` uses, and the twelve are
averaged.

Two things this is **not**. It is not a licensing problem — the zip is public on the same UKP mirror
as every other dataset, and `BeIR/cqadupstack` and `ir_datasets beir/cqadupstack/*` carry it too. And
it is not a format problem — unlike the corpora `docs/findings-beir-recovered.md` recovers, these
qrels are already three-column tab-separated with the header row `load_qrels` skips. It is purely an
aggregation shape the one-corpus harness does not have.

    .venv/bin/python tools/cqadupstack.py --retrievers bm25,bm25-engine
    .venv/bin/python tools/cqadupstack.py --no-download     # once the zip is extracted

The report is written in `beir.py`'s schema under `datasets.cqadupstack`, so `make consistency-check`
counts it as measured and `make charts-compare` draws it without either tool knowing about the
sub-forums; the twelve rows ride along under `subforums` for the reader who wants to see which forum
carries a gap.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import beir          # noqa: E402  (the ranking and evaluation code this shares)
import evalkit       # noqa: E402
import published_beir18 as published  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402

#: The twelve sub-collections, in the order the release ships them.
SUBFORUMS: tuple[str, ...] = (
    "android", "english", "gaming", "gis", "mathematica", "physics",
    "programmers", "stats", "tex", "unix", "webmasters", "wordpress",
)

#: The member's published BM25 nDCG@10, from the table the rest of the repository quotes.
#: Read rather than typed: it is the number the macro below is comparable to.
PUBLISHED_BM25_CQADUPSTACK: float = published.NDCG10["CQADupStack"]["BM25"]

#: The retrievers this tool can run. Every other corpus row in this repository that was measured
#: lexical-only is lexical-only for the same reason — a dense arm needs its own embedding pass over
#: the whole corpus — so the second stage is deliberately absent rather than quietly unsupported.
LEXICAL_RETRIEVERS: tuple[str, ...] = ("bm25", "bm25-engine")


def ensure_subforums(data_root: Path, *, download: bool = True) -> Path:
    """The extracted ``cqadupstack/`` tree, fetching and unzipping the one archive if needed.

    The zip holds all twelve sub-collections, so a single missing forum re-fetches all of them.
    Extraction mirrors `beir.ensure_dataset`, but returns the shared root: the per-sub-forum
    directories are what the loader reads, and this is exactly the level `ensure_dataset` cannot
    return, because it asserts a `corpus.jsonl` at the dataset level that CQADupStack does not have.
    """
    root = data_root / "cqadupstack"
    missing = [name for name in SUBFORUMS if not (root / name / "corpus.jsonl").is_file()]
    if not missing:
        return root
    if not download:
        raise FileNotFoundError(
            f"{root} is missing {len(missing)} sub-forum(s) ({', '.join(missing)}) "
            f"and --no-download was given"
        )
    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / "cqadupstack.zip"
    if not archive.is_file():
        url = beir.BEIR_URL.format(name="cqadupstack")
        print(f"  downloading {url}", flush=True)
        urllib.request.urlretrieve(url, archive)
    print(f"  extracting {archive.name}", flush=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(data_root)
    still = [name for name in SUBFORUMS if not (root / name / "corpus.jsonl").is_file()]
    if still:
        raise FileNotFoundError(f"{archive} did not contain {root.name}/<forum>/corpus.jsonl: {still}")
    return root


def rank_subforum(
    directory: Path,
    retrievers: Sequence[str],
    *,
    top_k: int,
    k1: float,
    b: float,
    stem: bool,
    include_titles: bool,
    ignore_identical_ids: bool,
) -> dict[str, Any]:
    """Run each retriever over one sub-forum and score it exactly as `beir.py` scores a dataset."""
    corpus = beir.load_corpus(directory, include_titles=include_titles)
    all_queries = beir.load_queries(directory)
    qrels = beir.load_qrels(directory)
    # Only rank the queries that have qrels; that is what BEIR scores.
    queries = {qid: text for qid, text in all_queries.items() if qid in qrels}

    results: dict[str, Any] = {}
    per_query: dict[str, dict[str, float]] = {}
    for retriever in retrievers:
        started = time.perf_counter()
        if retriever == "bm25":
            ranking = beir.bm25_rankings(corpus, queries, top_k, k1=k1, b=b)
        else:  # bm25-engine
            ranking = beir.bm25_rankings(
                corpus, queries, top_k, k1=k1, b=b, drop_stopwords=True, stem=stem
            )
        evaluation = evalkit.evaluate_rankings(
            ranking, qrels, k_values=(10, 100), ignore_identical_ids=ignore_identical_ids
        )
        results[retriever] = dict(evaluation["summary"])
        results[retriever]["seconds"] = round(time.perf_counter() - started, 2)
        results[retriever]["scored_queries"] = evaluation["queries"]
        per_query[retriever] = evalkit.per_query_metric(evaluation["per_query"], "nDCG@10")
    return {
        "documents": len(corpus),
        "queries": len(all_queries),
        "scored_queries": len(queries),
        "retrievers": results,
        "per_query_nDCG@10": per_query,
    }


def aggregate(per_forum: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average the sub-forums — the member's own convention, not an average of averages of raw.

    Every metric is averaged across the twelve sub-forum scores, which is what BEIR reports for
    CQADupStack. The per-query series are pooled with the forum name as a prefix: query ids are
    per-forum integers and collide across the twelve (`19399` exists in more than one), so without
    the prefix a paired bootstrap would silently pair unrelated queries.
    """
    retriever_names = sorted(
        {name for forum in per_forum.values() for name in forum["retrievers"]}
    )
    retrievers: dict[str, Any] = {}
    for retriever in retriever_names:
        scored = [
            forum["retrievers"][retriever]
            for forum in per_forum.values()
            if retriever in forum["retrievers"] and "nDCG@10" in forum["retrievers"][retriever]
        ]
        if not scored:
            continue
        metrics = sorted(
            {key for row in scored for key in row} - {"seconds", "scored_queries", "note", "model"}
        )
        row: dict[str, Any] = {
            metric: round(statistics.fmean(float(item[metric]) for item in scored), 4)
            for metric in metrics
        }
        row["seconds"] = round(sum(float(item.get("seconds") or 0.0) for item in scored), 2)
        row["scored_queries"] = sum(int(item["scored_queries"]) for item in scored)
        row["note"] = (
            f"macro over {len(scored)} sub-forums"
            + (f"; engine settings: stopwords dropped, stem={_STEM[0]}" if retriever == "bm25-engine"
               else "; published settings")
        )
        retrievers[retriever] = row

    per_query = {
        retriever: {
            f"{forum}:{qid}": value
            for forum, detail in per_forum.items()
            for qid, value in detail["per_query_nDCG@10"].get(retriever, {}).items()
        }
        for retriever in retriever_names
    }
    return {
        "documents": sum(int(forum["documents"]) for forum in per_forum.values()),
        "queries": sum(int(forum["queries"]) for forum in per_forum.values()),
        "scored_queries": sum(int(forum["scored_queries"]) for forum in per_forum.values()),
        "retrievers": retrievers,
        "per_query_nDCG@10": per_query,
        "subforums": {
            name: {
                "documents": forum["documents"],
                "queries": forum["queries"],
                "scored_queries": forum["scored_queries"],
                "retrievers": forum["retrievers"],
            }
            for name, forum in per_forum.items()
        },
    }


#: The stem setting the aggregate note reports. The ranking happens per sub-forum, where the flag is
#: live; this only carries it into the note, and is set once in `main`.
_STEM: list[bool] = [True]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cqadupstack.py", description="CQADupStack, per sub-forum")
    parser.add_argument("--retrievers", default="bm25,bm25-engine")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=beir.BEIR_K1)
    parser.add_argument("--b", type=float, default=beir.BEIR_B)
    parser.add_argument("--no-stem", dest="stem", action="store_false")
    parser.add_argument("--no-titles", action="store_true")
    parser.add_argument("--keep-identical-ids", action="store_true")
    parser.add_argument("--baseline", default="bm25-engine")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "beir" / "cqadupstack"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--subforums", default="", help="a comma-separated subset (smoke runs only)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("cqadupstack.py")
    wanted = [name.strip() for name in args.retrievers.split(",") if name.strip()]
    unsupported = [name for name in wanted if name not in LEXICAL_RETRIEVERS]
    if unsupported:
        raise SystemExit(
            f"cqadupstack.py runs lexical rows only ({', '.join(LEXICAL_RETRIEVERS)}); "
            f"got {unsupported}. A dense or fused row needs its own embedding pass over all "
            f"twelve sub-forums — see the module docstring."
        )
    stem = args.stem
    _STEM[0] = stem
    requested = [name.strip() for name in args.subforums.split(",") if name.strip()]
    forums = tuple(requested) or SUBFORUMS
    unknown = [name for name in forums if name not in SUBFORUMS]
    if unknown:
        raise SystemExit(f"not a CQADupStack sub-forum: {unknown}; known: {list(SUBFORUMS)}")
    if requested:
        print(f"  SUBSET RUN: {len(forums)} of {len(SUBFORUMS)} sub-forums — not the benchmark row",
              flush=True)

    data_root = Path(args.data_root)
    root = ensure_subforums(data_root, download=not args.no_download)

    per_forum: dict[str, Any] = {}
    for name in forums:
        directory = root / name
        print(f"[cqadupstack/{name}]", flush=True)
        detail = rank_subforum(
            directory, wanted, top_k=args.top_k, k1=args.k1, b=args.b, stem=stem,
            include_titles=not args.no_titles,
            ignore_identical_ids=not args.keep_identical_ids,
        )
        per_forum[name] = detail
        engine = detail["retrievers"].get("bm25-engine", {})
        print(f"  {detail['documents']} documents, {detail['scored_queries']} scored queries"
              + (f", bm25-engine nDCG@10 {engine['nDCG@10']:.4f}" if "nDCG@10" in engine else ""),
              flush=True)

    dataset = aggregate(per_forum)
    baseline = dataset["per_query_nDCG@10"].get(args.baseline)
    comparisons: dict[str, Any] = {}
    if baseline:
        for retriever, scores in dataset["per_query_nDCG@10"].items():
            if retriever == args.baseline:
                continue
            try:
                comparisons[retriever] = evalkit.paired_bootstrap(
                    baseline, scores, samples=args.bootstrap_samples
                )
            except ValueError:  # no query scored by both
                continue
    dataset["comparisons_vs"] = args.baseline
    dataset["comparisons"] = comparisons

    control = dataset["retrievers"].get("bm25", {})
    anchor = (
        evalkit.anchor_deviations(
            {"cqadupstack": control["nDCG@10"]},
            {"cqadupstack": PUBLISHED_BM25_CQADUPSTACK},
            tolerance=beir.ANCHOR_TOLERANCE,
        )
        if "nDCG@10" in control
        else []
    )

    report: dict[str, Any] = {
        "settings": {
            "top_k": args.top_k,
            "bm25_k1": args.k1,
            "bm25_b": args.b,
            "drop_stopwords": True,
            "stem": stem,
            "include_titles": not args.no_titles,
            "ignore_identical_ids": not args.keep_identical_ids,
            "checkpoint": EngineConfig().checkpoint,
            "baseline": args.baseline,
            "bootstrap_samples": args.bootstrap_samples,
            "subforums": list(forums),
            "aggregation": "macro-average over sub-forums",
        },
        "references": beir.REFERENCES,
        "anchors": {"cqadupstack": anchor[0] if anchor else None},
        "datasets": {"cqadupstack": dataset},
    }

    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    evalkit.write_text(out.with_suffix(".md"), render_markdown(report))
    print()
    print(render_console(report))
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


def _subforum_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset = report["datasets"]["cqadupstack"]
    rows: list[dict[str, Any]] = []
    for name, forum in dataset.get("subforums", {}).items():
        for retriever, result in forum["retrievers"].items():
            rows.append({
                "sub-forum": name,
                "documents": forum["documents"],
                "retriever": retriever,
                **{key: result.get(key) for key in ("nDCG@10", "Recall@100", "MAP@100")},
            })
    macro = dataset["retrievers"]
    label = f"MACRO ({len(dataset.get('subforums', {}))})"
    for retriever, result in macro.items():
        rows.append({
            "sub-forum": label,
            "documents": dataset["documents"],
            "retriever": retriever,
            **{key: result.get(key) for key in ("nDCG@10", "Recall@100", "MAP@100")},
        })
    return rows


def render_console(report: Mapping[str, Any]) -> str:
    rows = _subforum_rows(report)
    header = f"{'sub-forum':<14} {'documents':>9} {'retriever':<12} {'nDCG@10':>9} {'Recall@100':>11} {'MAP@100':>9}"
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = " ".join(
            f"{row[key]:>9.4f}" if isinstance(row[key], float) else f"{str(row[key]):>9}"
            for key in ("nDCG@10", "Recall@100", "MAP@100")
        )
        lines.append(
            f"{row['sub-forum']:<14} {row['documents']:>9} {row['retriever']:<12} {cells}"
        )
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any]) -> str:
    dataset = report["datasets"]["cqadupstack"]
    published_anchor = published.NDCG10["CQADupStack"]["BM25"]
    engine = dataset["retrievers"].get("bm25-engine", {})
    flat = dataset["retrievers"].get("bm25", {})
    return "\n".join([
        "# CQADupStack, per sub-forum",
        "",
        f"{len(SUBFORUMS)} sub-forums, {dataset['documents']:,} documents, "
        f"{dataset['scored_queries']:,} scored queries. Each sub-forum is retrieved and scored "
        "against its **own** corpus and qrels — the twelve are independent collections, not one "
        "pool — and the member's score is the macro-average of the twelve, which is the number "
        "BEIR publishes.",
        "",
        f"Published BM25 nDCG@10 for the member: **{published_anchor}** "
        "(`tools/published_beir18.py`). This run's flat `bm25` control reads "
        f"**{flat.get('nDCG@10')}**, and the engine row **{engine.get('nDCG@10')}**.",
        "",
        "## Per sub-forum",
        "",
        evalkit.markdown_table(_subforum_rows(report)),
        "",
        "## How to read this",
        "",
        beir.REFERENCES["note"],
        "",
        f"- BEIR paper: {beir.REFERENCES['BEIR paper']}",
        f"- BEIR code: {beir.REFERENCES['BEIR code']}",
        "",
    ])


if __name__ == "__main__":
    sys.exit(main())
