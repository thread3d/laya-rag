"""Evaluate a distilled student against its zero-shot self and its teachers.

`tools/finetune.py --teacher-models ...` trains a student on the teachers' fused top
documents. This scores all three on the dataset's **test** split with the same evaluator:
the zero-shot student, the teacher ensemble, and the adapted student — with a paired
bootstrap so the distillation delta is bracketed, not asserted.

    .venv/bin/python tools/distill_eval.py --dataset nfcorpus \
        --adapted eval/distill-nfcorpus-model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir, evalkit  # noqa: E402
import finetune  # noqa: E402

DEFAULT_TEACHERS = "BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="distill_eval.py")
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    ap.add_argument("--student", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--teachers", default=DEFAULT_TEACHERS)
    ap.add_argument("--adapted", default=str(ROOT / "eval" / "distill-nfcorpus-model"))
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--out", default=str(ROOT / "eval" / "distill-eval"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = beir.ensure_dataset(args.dataset, Path(args.data_root), download=False)
    corpus = beir.load_corpus(dataset_dir)
    queries = beir.load_queries(dataset_dir)
    test_qrels = beir.load_qrels(dataset_dir, split="test")
    test_queries = {qid: text for qid, text in queries.items() if qid in test_qrels}

    def rankings_for(model_id: str) -> dict[str, list[str]]:
        enc = beir.BiEncoder(model_id)
        cache = ROOT / "eval" / "beir" / "cache" / f"{args.dataset}.{enc.name}"
        vectors = beir.cached_embed(enc, corpus, cache)
        return beir.dense_rankings(corpus, test_queries, enc, vectors, args.top_k, fixed_length=True)

    rankings: dict[str, dict[str, list[str]]] = {
        "zero_shot": rankings_for(args.student),
        "adapted": finetune.rank_with_encoder(
            _sentence_transformer(args.adapted), corpus, test_queries, args.top_k
        ),
    }
    teacher_names = [name.strip() for name in args.teachers.split(",") if name.strip()]
    if teacher_names:
        teacher_rankings = [rankings_for(name) for name in teacher_names]
        rankings["teacher_ensemble"] = beir.multi_channel_fusion(
            teacher_rankings, [1.0] * len(teacher_rankings), args.top_k, rrf_k=5
        )

    summary: dict[str, dict[str, float]] = {}
    per_query: dict[str, dict[str, float]] = {}
    for name, ranking in rankings.items():
        evaluation = evalkit.evaluate_rankings(ranking, test_qrels, k_values=(10,), ignore_identical_ids=True)
        summary[name] = evaluation["summary"]
        per_query[name] = evalkit.per_query_metric(evaluation["per_query"], "nDCG@10")
        print(f"  {name:16s} nDCG@10 {summary[name]['nDCG@10']:.4f}", flush=True)

    comparisons = {}
    for name in rankings:
        if name == "zero_shot":
            continue
        comparisons[name] = {
            "bootstrap": evalkit.paired_bootstrap(
                per_query["zero_shot"], per_query[name], samples=args.samples
            ),
            "permutation": evalkit.permutation_test(
                per_query["zero_shot"], per_query[name], samples=args.samples
            ),
        }
        row = comparisons[name]
        print(f"  {name} vs zero_shot: {row['bootstrap']['delta']:+.4f} "
              f"[{row['bootstrap']['ci_low']:+.4f}, {row['bootstrap']['ci_high']:+.4f}] "
              f"p={row['permutation']['p_value']:.4f}", flush=True)

    payload = {
        "_settings": vars(args),
        "queries": len(test_queries),
        "summary": summary,
        "comparisons": comparisons,
        "per_query": per_query,
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), payload)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(payload))
    print(render_markdown(payload))
    print("wrote", str(out) + ".json and .md")
    return 0


def _sentence_transformer(path: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(path)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Distillation: student vs zero-shot vs teacher ensemble",
        "",
        f"- dataset: {payload['_settings']['dataset']}, {payload['queries']} test queries",
        "",
        "| model | nDCG@10 |",
        "| --- | --- |",
    ]
    for name, row in payload["summary"].items():
        lines.append(f"| {name} | {row['nDCG@10']:.4f} |")
    lines += ["", "## Paired comparison vs zero-shot (nDCG@10)", ""]
    for name, value in payload["comparisons"].items():
        boot = value["bootstrap"]
        lines.append(
            f"- {name}: {boot['delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
            f"({'significant' if boot['significant'] else 'not significant'})"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
