"""Is a late-interaction (MaxSim) channel worth building? The DISCo feasibility probe.

`docs/research/08-frontier-2026.md` names the blocking dependency for the highest-ceiling
item in the programme: the repo's dense channel is mean-pooled and single-vector, while
DISCo's collective-coverage index and every ColBERT-style reranker need *token-level*
representations. Before committing weeks, this measures three things on real queries:

1. **Are token states available at all?** `laya.embed_fn_from_agent` reads
   `agent.model.encoder(...).last_hidden_state`, so the shipped checkpoint already exposes
   them — no new weights, no download. This probe asserts that rather than assuming it.
2. **Does MaxSim order the pool better than the fused first stage?** It scores each cached
   candidate by the ColBERT late-interaction sum over query tokens, and reports MAP@10 with
   a paired bootstrap interval against the fused baseline. Laya's encoder is a decision
   model, not a retrieval-trained one, so the honest prior is that this is weak — the point
   is to find out cheaply instead of inferring it from the mean-pooled cosine.
3. **What would it cost?** Token vectors per chunk and the resident size they imply, which
   is the budget the DISCo path has to fit in.

    .venv/bin/python tools/laya_maxsim_probe.py --limit-queries 200

Writes `<out>.json` and `<out>.md`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402
import numpy as np  # noqa: E402
from multihop_official import chunk_text  # noqa: E402
from multihop_rerank_experiment import (  # noqa: E402
    gold_matches_for_pool,
    metrics_from_matches,
    pit_order,
)

from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.model import LayaModel  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")


def query_texts() -> dict[str, str]:
    items = json.loads((ROOT_BENCH / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return {
        str(i): x["query"]
        for i, x in enumerate(items)
        if x.get("question_type") != "null_query"
    }


def load_corpus(overlap: int) -> dict[str, str]:
    raw = json.loads((ROOT_BENCH / "corpus.json").read_text(encoding="utf-8"))
    corpus: dict[str, str] = {}
    for article in raw:
        for j, text in enumerate(chunk_text(article["title"] + " " + article["body"], 256, overlap)):
            corpus[f"{article['url']}#{j}"] = text
    return corpus


def _normalized_tokens(encoder, tokenizer, texts: Sequence[str], device, max_length: int, batch_size: int):
    """Per-text L2-normalised token matrices, padding excluded.

    Returns a list of ``(tokens, dim)`` float32 arrays. Normalising per token is what makes
    MaxSim a cosine-like maximum rather than a magnitude contest.
    """
    import torch

    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            hidden = encoder(input_ids=input_ids, attention_mask=mask).last_hidden_state
            hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
        for row in range(hidden.shape[0]):
            length = int(mask[row].sum().item())
            out.append(hidden[row, :length].cpu().numpy())
    return out


def maxsim(query_tokens: np.ndarray, document_tokens: np.ndarray) -> float:
    """ColBERT late interaction: sum over query tokens of the best document match."""
    if query_tokens.size == 0 or document_tokens.size == 0:
        return 0.0
    return float((query_tokens @ document_tokens.T).max(axis=1).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="laya_maxsim_probe.py")
    ap.add_argument("--ce-cache", default=str(ROOT_BENCH / "ce-scores.json"))
    ap.add_argument("--limit-queries", type=int, default=200)
    ap.add_argument("--sample-seed", type=int, default=17)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "maxsim-probe"))
    args = ap.parse_args(argv)

    cache = json.loads(Path(str(args.ce_cache)).read_text(encoding="utf-8"))
    overlap = int(cache["_settings"]["overlap"])
    depth = int(cache["_settings"]["depth"])
    names = sorted(cache["queries"], key=lambda v: (len(v), v))
    if args.limit_queries and args.limit_queries < len(names):
        names = sorted(random.Random(args.sample_seed).sample(names, args.limit_queries),
                       key=lambda v: (len(v), v))
    texts = query_texts()
    corpus = load_corpus(overlap)
    print(f"{len(names)} queries, depth {depth}, {len(corpus)} chunks", flush=True)

    model = LayaModel(EngineConfig())
    agent = model.ensure_loaded()
    encoder = agent.model.encoder
    tokenizer = agent.tok
    device = agent.device
    hidden_size = int(getattr(encoder.config, "hidden_size", 0))
    print(f"encoder hidden size: {hidden_size}, device: {device}", flush=True)

    started = time.perf_counter()
    per_query: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    baseline: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    maxsim_scores: dict[str, list[float]] = {}
    token_counts: list[int] = []
    total_tokens = 0
    total_docs = 0
    for n, q in enumerate(names, start=1):
        payload = cache["queries"][q]
        pool = list(payload["pool"])[:depth]
        matches = gold_matches_for_pool(corpus, pool, cache["golds"][q])
        query_tokens = _normalized_tokens(
            encoder, tokenizer, [texts[q]], device, args.max_length, args.batch_size
        )[0]
        document_tokens = _normalized_tokens(
            encoder, tokenizer, [corpus[doc] for doc in pool], device, args.max_length,
            args.batch_size,
        )
        scores = [maxsim(query_tokens, tokens) for tokens in document_tokens]
        maxsim_scores[q] = [round(float(s), 6) for s in scores]
        token_counts.append(int(np.mean([t.shape[0] for t in document_tokens])))
        total_tokens += int(sum(t.shape[0] for t in document_tokens))
        total_docs += len(document_tokens)

        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        metrics = metrics_from_matches(ranked, matches, len(cache["golds"][q]), 10)
        fused = metrics_from_matches(list(range(len(pool))), matches, len(cache["golds"][q]), 10)
        for metric in METRICS:
            per_query[metric][q] = metrics[metric]
            baseline[metric][q] = fused[metric]
        if n % 25 == 0 or n == len(names):
            print(f"  {n}/{len(names)} ({time.perf_counter() - started:.0f}s)", flush=True)

    summary = {
        metric: {
            "maxsim": round(float(np.mean(list(per_query[metric].values()))), 4),
            "fused": round(float(np.mean(list(baseline[metric].values()))), 4),
        }
        for metric in METRICS
    }
    comparisons = {
        f"maxsim vs fused {metric}": {
            "bootstrap": evalkit.paired_bootstrap(
                baseline[metric], per_query[metric], samples=args.samples
            ),
            "permutation": evalkit.permutation_test(
                baseline[metric], per_query[metric], samples=args.samples
            ),
        }
        for metric in ("MAP@10", "Hits@4")
    }
    mean_tokens = float(np.mean(token_counts)) if token_counts else 0.0
    cost = {
        "mean_tokens_per_chunk": round(mean_tokens, 1),
        "tokens_per_chunk_p95": int(np.percentile(token_counts, 95)) if token_counts else 0,
        "resident_bytes_per_chunk_fp32": int(mean_tokens * hidden_size * 4),
        "projected_gib_at_1m_chunks": round(mean_tokens * hidden_size * 4 * 1_000_000 / 2**30, 2),
        "projected_gib_at_1m_chunks_int8": round(mean_tokens * hidden_size * 1 * 1_000_000 / 2**30, 2),
        "scored_documents": total_docs,
        "scored_tokens": total_tokens,
    }
    result = {
        "_settings": {
            "ce_cache": Path(str(args.ce_cache)).name,
            "limit_queries": len(names),
            "sample_seed": args.sample_seed,
            "depth": depth,
            "overlap": overlap,
            "max_length": args.max_length,
            "hidden_size": hidden_size,
            "seconds": round(time.perf_counter() - started, 1),
        },
        "summary": summary,
        "comparisons": comparisons,
        "cost": cost,
        "per_query": {metric: per_query[metric] for metric in METRICS},
        "maxsim_scores": maxsim_scores,
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), result)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(result: Mapping[str, Any]) -> str:
    settings = result["_settings"]
    lines = [
        "# Late-interaction (MaxSim) feasibility probe",
        "",
        f"- {settings['limit_queries']} queries x depth {settings['depth']} from "
        f"`{settings['ce_cache']}`; Laya encoder, max_length {settings['max_length']}, "
        f"hidden {settings['hidden_size']}",
        "",
        "| metric | fixed-order pool (index 0..n) | MaxSim order | delta |",
        "| --- | --- | --- | --- |",
    ]
    for metric, row in result["summary"].items():
        lines.append(
            f"| {metric} | {row['fused']:.4f} | {row['maxsim']:.4f} | "
            f"{row['maxsim'] - row['fused']:+.4f} |"
        )
    lines += ["", "## Paired comparisons (bootstrap 95% CI, permutation p)", ""]
    for key, value in result["comparisons"].items():
        boot = value["bootstrap"]
        perm = value["permutation"]
        lines.append(
            f"- `{key}`: {boot['delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
            f"p={perm['p_value']:.4f} ({'sig' if boot['significant'] else 'ns'})"
        )
    cost = result["cost"]
    lines += [
        "",
        "## What a multi-vector index would cost",
        "",
        f"- mean tokens per chunk: **{cost['mean_tokens_per_chunk']}** "
        f"(p95 {cost['tokens_per_chunk_p95']})",
        f"- resident per chunk at fp32: **{cost['resident_bytes_per_chunk_fp32'] / 1024:.1f} KiB**",
        f"- projected at 1M chunks: **{cost['projected_gib_at_1m_chunks']} GiB** fp32, "
        f"**{cost['projected_gib_at_1m_chunks_int8']} GiB** int8",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
