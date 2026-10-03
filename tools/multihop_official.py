"""MultiHop-RAG retrieval under the benchmark's own protocol.

reimplemented from yixuantt/MultiHop-RAG `simple_retrieval.py` + `retrieval_evaluate.py`:
256-token SentenceSplitter chunks, and Hits@10 / Hits@4 / MAP@10 / MRR@10 computed by
substring-matching each gold evidence *fact* against each retrieved chunk.

Defaults reproduce the original `bge-small` run exactly. `--models` adds rows for other
bi-encoders (the published comparison row is `bge-large`, which this harness had never
run), and `--rerank` adds a cross-encoder second stage over the fused / dense top-100
(the published "bge-large + rerank" row). Rows are merged into an existing report, so a
new model never discards the measurements already on disk.

    .venv/bin/python tools/multihop_official.py --models bge-small,bge-large
    .venv/bin/python tools/multihop_official.py --models bge-large \
        --rerank --rerank-model BAAI/bge-reranker-base --rerank-depth 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir, evalkit  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"

# Short names accepted on the command line, mapped to the Hub ids the harness embeds.
MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}


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


def official_metrics(retrieved_lists, gold_lists):
    """Aggregate metrics plus the per-query rows they average, so they can be bracketed.

    Retrieval here is deterministic — re-running reproduces the same number exactly — so the
    only meaningful error bar is over *queries*, which is what the per-query rows are for.
    """
    rows = []
    for retrieved, gold in zip(retrieved_lists, gold_lists):
        gold = [g.replace(" ", "").replace("\n", "") for g in gold]
        retrieved = [r.replace(" ", "").replace("\n", "") for r in retrieved]
        f10 = f4 = False
        ap = 0.0
        first = None
        found = []
        for rank, item in enumerate(retrieved[:10], start=1):
            # Exactly the metric's top-k: slicing 11 examined a rank the guard then
            # discarded, which is a trap for anyone who edits the guard.
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
        rows.append(
            {
                "Hits@10": float(f10),
                "Hits@4": float(f4),
                # A query with no gold facts would divide by zero and abort the whole run; the
                # denominator is floored at 1 (which scores it 0.0), matching
                # metrics_from_matches in tools/multihop_rerank_experiment.py.
                "MAP@10": ap / min(max(len(gold), 1), 10),
                "MRR@10": 1 / first if first else 0.0,
            }
        )
    n = len(rows)
    summary = {k: sum(r[k] for r in rows) / n for k in rows[0]}
    return summary, rows


def resolve_models(raw: str) -> list[tuple[str, str]]:
    out = []
    for name in [v.strip() for v in raw.split(",") if v.strip()]:
        out.append((name, MODELS.get(name, name)))
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_official.py")
    ap.add_argument("--models", default="bge-small",
                    help="comma list of short names (bge-small, bge-base, bge-large) or Hub ids")
    ap.add_argument("--overlaps", default="200,0", help="chunk overlaps to evaluate")
    ap.add_argument("--rerank", action="store_true",
                    help="also rerank the dense and fused top-depth lists with a cross-encoder")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    ap.add_argument("--rerank-depth", type=int, default=100,
                    help="candidates scored by the cross-encoder (dense and fused each get one)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT_BENCH / "official.json"))
    ap.add_argument("--no-merge", action="store_true",
                    help="write only this run's rows instead of merging into the report")
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="score a deterministic sample of this many answerable queries; a "
                         "subset signal (the report records the limit), not the full protocol")
    ap.add_argument("--sample-seed", type=int, default=17)
    ap.add_argument("--rerank-targets", default="dense,hybrid",
                    help="which first stages the cross-encoder reranks: dense, hybrid, or both — "
                         "reranking one halves the scoring cost")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    raw = json.load(open(ROOT_BENCH / "corpus.json"))
    items = json.load(open(ROOT_BENCH / "MultiHopRAG.json"))
    models = resolve_models(args.models)
    overlaps = [int(v) for v in str(args.overlaps).split(",") if v.strip()]
    depth = max(10, args.rerank_depth) if args.rerank else 10

    ev_items = [(i, x) for i, x in enumerate(items) if x["question_type"] != "null_query"]
    total_answerable = len(ev_items)
    if args.limit_queries and args.limit_queries < total_answerable:
        sampled = sorted(
            random.Random(args.sample_seed).sample(range(total_answerable), args.limit_queries)
        )
        ev_items = [ev_items[i] for i in sampled]
        print(f"subset: {len(ev_items)} of {total_answerable} answerable queries "
              f"(seed {args.sample_seed})", flush=True)
    golds = [[e["fact"] for e in x["evidence_list"]] for _, x in ev_items]

    results: dict = {}
    out_path = Path(args.out)
    if out_path.is_file() and not args.no_merge:
        results = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"merging into {out_path} ({len(results)} existing rows)")

    cross_encoder = None
    if args.rerank:
        print(f"loading cross-encoder {args.rerank_model}", flush=True)
        cross_encoder = beir.CrossEncoder(args.rerank_model, batch_size=args.batch_size)

    for overlap in overlaps:
        corpus: dict[str, str] = {}
        titles: dict[str, str] = {}
        for a in raw:
            cs = chunk_text(a["title"] + " " + a["body"], 256, overlap)
            for j, c in enumerate(cs):
                cid = f"{a['url']}#{j}"
                corpus[cid] = c
                titles[cid] = a["title"]
        queries = {str(i): x["query"] for i, x in ev_items}
        print(f"\n[overlap={overlap}] {len(corpus)} chunks from {len(raw)} articles", flush=True)

        # The lexical channel does not depend on the dense model, so it is built once per
        # overlap and shared by every model's fused row.
        lexical = beir.bm25_rankings(
            corpus, queries, depth, stem=True, drop_stopwords=True, titles=titles, title_boost=2
        )
        plain = beir.bm25_rankings(corpus, queries, 10, stem=False, drop_stopwords=False)
        for name, ranking in (("bm25", plain), ("bm25-engine", lexical)):
            ret = [[corpus[d] for d in ranking[str(i)][:10]] for i, _ in ev_items]
            m, rows = official_metrics(ret, golds)
            results[f"{name}@ov{overlap}"] = {**m, "_per_query": rows}
            print(f"  {name:22s} Hits@10={m['Hits@10']:.4f} Hits@4={m['Hits@4']:.4f} "
                  f"MAP@10={m['MAP@10']:.4f} MRR@10={m['MRR@10']:.4f}", flush=True)

        for short, model_id in models:
            enc = beir.BiEncoder(model_id, batch_size=args.batch_size)
            cache = ROOT_BENCH / "cache" / f"multihop-chunks{overlap}.{enc.name}"
            vec = beir.cached_embed(enc, corpus, cache)
            dense_full = beir.dense_rankings(corpus, queries, enc, vec, depth)
            fused_full = beir.multi_channel_fusion(
                [dense_full, lexical], [1.0, 1.0], depth, rrf_k=5
            )

            rows: dict[str, dict[str, list[str]]] = {
                f"{short} dense": {q: v[:10] for q, v in dense_full.items()},
                f"{short} hybrid": {q: v[:10] for q, v in fused_full.items()},
            }
            if cross_encoder is not None:
                targets = {t.strip() for t in str(args.rerank_targets).split(",") if t.strip()}
                options = []
                if "dense" in targets:
                    options.append((f"{short} dense+rerank", dense_full))
                if "hybrid" in targets:
                    options.append((f"{short} hybrid+rerank", fused_full))
                for label, base in options:
                    started = time.perf_counter()
                    reranked, _ = beir.crossencoder_rankings(
                        corpus, queries, base, cross_encoder, depth, 10
                    )
                    rows[label] = reranked
                    print(f"    reranked {label} in {time.perf_counter() - started:.0f}s", flush=True)

            for name, ranking in rows.items():
                ret = [[corpus[d] for d in ranking[str(i)]] for i, _ in ev_items]
                m, per_query = official_metrics(ret, golds)
                results[f"{name}@ov{overlap}"] = {**m, "_per_query": per_query}
                print(f"  {name:22s} Hits@10={m['Hits@10']:.4f} Hits@4={m['Hits@4']:.4f} "
                      f"MAP@10={m['MAP@10']:.4f} MRR@10={m['MRR@10']:.4f}", flush=True)

    results["_settings"] = vars(args)
    evalkit.write_json(out_path, results)
    print("\nwrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
