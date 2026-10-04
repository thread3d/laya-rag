"""A relational (stance) reranker for ArguAna — ROADMAP Phase 2.3.

ArguAna is the one BEIR dataset where a pointwise ranker collapses: its gold is the
**counter-argument**, so an *agreeing* passage is topically near-identical and a pointwise
MS MARCO cross-encoder scores it high. The measured collapse is severe — `bge-reranker-base` over
the fused top-100 took ArguAna from 0.568 to **0.250**. The diagnosis in `RESEARCH.md` §5.2 is that
the pointwise model has no supervision for the *relation* between two texts.

This tests the prescribed fix: score by **contradiction minus entailment** with an NLI
cross-encoder, so opposition is rewarded rather than similarity. Both arms reorder the *same*
candidate pool, so any difference is ordering alone — the same discipline as Phase 2.1.

    .venv/bin/python tools/arguana_stance.py --limit 40        # smoke
    .venv/bin/python tools/arguana_stance.py                   # 1406 queries
    .venv/bin/python tools/arguana_stance.py --compare
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
# Project-local model cache, following tools/beir.py and tools/bright.py. Set before any import
# that might read HF_HOME.
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "nli-hf-cache"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir, evalkit, jobkit  # noqa: E402

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
DATASET = "arguana"
ARMS = ("dense", "stance")


def rank_positions(scores: Sequence[float]) -> list[int]:
    """Positions by descending score, ties broken by position. Mirrors the ordering harness."""
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def order_doc_ids(pool: Sequence[str], scores: Sequence[float]) -> list[str]:
    return [pool[i] for i in rank_positions(scores)]


def stance_label_indices(id2label: Mapping[Any, str]) -> tuple[int, int]:
    """`(contradiction, entailment)` column indices from a checkpoint's `id2label`.

    Read from the checkpoint rather than assumed: `cross-encoder/nli-deberta-v3-base` happens to
    be contradiction/entailment/neutral, but that is a property of the model, not of NLI, and a
    silently swapped pair would invert the score.
    """
    labels = {int(i): str(v).lower() for i, v in id2label.items()}
    try:
        contradiction = next(i for i, v in labels.items() if "contradiction" in v)
        entailment = next(i for i, v in labels.items() if "entailment" in v)
    except StopIteration as exc:
        raise ValueError(f"no contradiction/entailment labels in {id2label!r}") from exc
    return contradiction, entailment


def _batched(items: Sequence[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


class StanceScorer:
    """`P(contradiction) - P(entailment)` for a (premise, hypothesis) pair.

    The label order is read from the checkpoint rather than assumed: `cross-encoder/nli-deberta-v3-base`
    is contradiction/entailment/neutral, but that is a property of the model, not of NLI.
    """

    def __init__(self, model_id: str = NLI_MODEL, *, device: str | None = None,
                 batch_size: int = 16, max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        self.contradiction, self.entailment = stance_label_indices(self.model.config.id2label)
        self.batch_size = batch_size
        self.max_length = max_length

    def scores(self, premise: str, hypotheses: Sequence[str]) -> list[float]:
        out: list[float] = []
        for batch in _batched(list(hypotheses), self.batch_size):
            enc = self.tokenizer([premise] * len(batch), list(batch), padding=True,
                                 truncation=True, max_length=self.max_length, return_tensors="pt")
            enc = {key: value.to(self.device) for key, value in enc.items()}
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1)
            diff = probs[:, self.contradiction] - probs[:, self.entailment]
            out.extend(diff.float().cpu().tolist())
        return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="arguana_stance.py")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--model", default=NLI_MODEL)
    ap.add_argument("--pool", type=int, default=20, help="dense candidates reranked per query")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "eval" / "beir" / "arguana-stance"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--merge", type=int, default=0, metavar="N")
    ap.add_argument("--compare", action="store_true",
                    help="re-render the paired comparison from a finished report")
    return ap.parse_args(argv)


def _fingerprint(args: argparse.Namespace) -> str:
    return jobkit.config_fingerprint(
        tool="arguana_stance",
        dataset=args.dataset,
        embed_model=args.embed_model,
        nli_model=args.model,
        pool=args.pool,
        top_k=args.top_k,
        max_length=args.max_length,
        limit=args.limit,
    )


def pool_and_rankings(
    data_root: Path, dataset: str, embed_model: str, pool: int, limit: int | None
) -> tuple[dict[str, str], dict[str, str], Mapping[str, Any], dict[str, list[str]]]:
    """`(corpus, queries, qrels, dense rankings)`, dense over text only.

    ArguAna's "titles" are shared topic keywords, which is why the project's documented dense
    number is text-only; the cache stem carries a `.notitles` tag so this cannot overwrite the
    with-title embeddings another report depends on.
    """
    dataset_dir = beir.ensure_dataset(dataset, data_root, download=False)
    corpus = beir.load_corpus(dataset_dir, include_titles=False)
    queries = beir.load_queries(dataset_dir)
    qrels = beir.load_qrels(dataset_dir, split="test")
    test_queries = {qid: text for qid, text in queries.items() if qid in qrels}
    if limit:
        test_queries = dict(list(test_queries.items())[:limit])
    encoder = beir.BiEncoder(embed_model)
    cache = data_root / "cache" / f"{dataset}.{encoder.name}.notitles"
    vectors = beir.cached_embed(encoder, corpus, cache)
    dense = beir.dense_rankings(corpus, test_queries, encoder, vectors, pool, fixed_length=True)
    return corpus, test_queries, qrels, dense


def render_markdown(report: Mapping[str, Any]) -> str:
    s = report["summary"]
    top_k = report["_settings"].get("top_k", 10)
    lines = [
        "# A relational (stance) reranker on ArguAna",
        "",
        f"- {report['_settings'].get('queries')} queries, dense pool {report['_settings'].get('pool')}, "
        f"nDCG@{top_k}, model `{report['_settings'].get('model')}`",
        "",
        f"| arm | nDCG@{top_k} | MRR@{top_k} |",
        "| --- | --- | --- |",
    ]
    for arm in ARMS:
        row = s.get(arm, {})
        lines.append(f"| {arm} | {row.get(f'nDCG@{top_k}', 0.0):.4f} | {row.get(f'MRR@{top_k}', 0.0):.4f} |")
    lines += ["", "## Paired comparison (stance vs dense, same pool)", ""]
    for metric, row in report["comparison"].items():
        sig = "" if row["significant"] else " (ns)"
        lines.append(f"- {metric}: {row['delta']:+.4f} "
                     f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_value']:.4f}{sig}")
    lines.append("")
    return "\n".join(lines)


def _build_report(rows: dict) -> dict:
    if jobkit.META_KEY not in rows:
        raise RuntimeError("no shard checkpoints found — run the shards before --merge")
    meta = rows.pop(jobkit.META_KEY)
    return {**meta, "stance": rows}


def _compare(args: argparse.Namespace) -> int:
    path = Path(str(args.out) + ".json")
    if not path.is_file():
        print(f"compare: {path} not found — run the measurement first", file=sys.stderr)
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    print(render_markdown(report))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare:
        return _compare(args)
    shard_index, shard_count = jobkit.parse_shard(args.shard)
    fingerprint = _fingerprint(args)

    if args.merge:
        try:
            merged = jobkit.merge_shards(args.out, args.merge, fingerprint=fingerprint,
                                         build=_build_report)
        except (RuntimeError, jobkit.FingerprintMismatch) as exc:
            print(f"merge failed: {exc}", file=sys.stderr)
            return 2
        print(f"merged {args.merge} shard(s) -> {args.out}.json")
        return 0

    corpus, queries, qrels, dense = pool_and_rankings(
        Path(args.data_root), args.dataset, args.embed_model, args.pool, args.limit
    )
    print(f"[{args.dataset}] {len(corpus)} docs, {len(queries)} queries, pool {args.pool}", flush=True)

    stem = jobkit.shard_stem(args.out, shard_index, shard_count)
    if args.fresh:
        for suffix in (".jsonl", ".manifest.json"):
            Path(str(stem) + suffix).unlink(missing_ok=True)
    run = jobkit.ResumableRun(str(stem) + ".json", fingerprint=fingerprint,
                              shard=(shard_index, shard_count))
    if jobkit.META_KEY not in run:
        run.record(jobkit.META_KEY, {
            "_settings": {**vars(args), "queries": len(queries), "documents": len(corpus)},
            "query_order": list(queries),
        })

    mine = jobkit.shard_items(list(queries), shard_index, shard_count)
    todo = run.pending(mine)
    print(f"  {len(todo)} of {len(mine)} queries to rerank ({len(run) - 1} checkpointed)",
          flush=True)

    scorer = StanceScorer(args.model, device=args.device, batch_size=args.batch_size,
                          max_length=args.max_length) if todo else None
    started = time.perf_counter()
    for position, qid in enumerate(todo, 1):
        pool = list(dense[qid])[: args.pool]
        scores = scorer.scores(queries[qid], [corpus[doc_id] for doc_id in pool])
        run.record(qid, {"pool": pool, "stance": [round(float(s), 6) for s in scores]})
        if position % 25 == 0:
            run.touch(total=len(mine))
            print(f"    {position}/{len(todo)} in {time.perf_counter() - started:.0f}s", flush=True)

    shared = run.get(jobkit.META_KEY)
    rows = {q: run.get(q) for q in shared["query_order"] if run.get(q) is not None}
    dense_rankings = {q: list(rows[q]["pool"]) for q in rows}
    stance_rankings = {q: order_doc_ids(rows[q]["pool"], rows[q]["stance"]) for q in rows}
    # `evaluate_rankings` averages over every query in the qrels, so a `--limit` run must score
    # against only the queries it ranked — otherwise 8 ranked queries are diluted by the ~1400
    # unranked ones and every number collapses to ~0.
    scored_qrels = {q: qrels[q] for q in rows if q in qrels}
    dense_eval = evalkit.evaluate_rankings(dense_rankings, scored_qrels, k_values=(args.top_k,),
                                           ignore_identical_ids=True)
    stance_eval = evalkit.evaluate_rankings(stance_rankings, scored_qrels, k_values=(args.top_k,),
                                            ignore_identical_ids=True)
    metric = f"nDCG@{args.top_k}"
    mrr = f"MRR@{args.top_k}"
    comparison = {
        metric: evalkit.paired_bootstrap(
            evalkit.per_query_metric(dense_eval["per_query"], metric),
            evalkit.per_query_metric(stance_eval["per_query"], metric), samples=10_000),
        mrr: evalkit.paired_bootstrap(
            evalkit.per_query_metric(dense_eval["per_query"], mrr),
            evalkit.per_query_metric(stance_eval["per_query"], mrr), samples=10_000),
    }
    report = {
        "_settings": {**shared["_settings"], "seconds": round(time.perf_counter() - started, 1)},
        "summary": {"dense": dense_eval["summary"], "stance": stance_eval["summary"]},
        "comparison": comparison,
        "pool": dense_rankings,
        "stance": {q: rows[q]["stance"] for q in rows},
    }
    run.finish(report, total=len(mine))
    evalkit.write_text(Path(str(stem) + ".md"), render_markdown(report))
    print("\n" + render_markdown(report))
    print("wrote", str(stem) + ".json", "and", str(stem) + ".md")
    if shard_count > 1:
        print(f"  run all {shard_count} shards, then: --merge {shard_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
