"""A late-interaction ordering head over the MultiHop-RAG candidate pool.

This is ROADMAP Phase 2.1. `ceiling-decomp.md` measures the headroom as mostly *ordering*
(**+0.1254** to reorder the cross-encoder's own top-10), and every model-free optimiser over the
cached features is already null, so the next movement has to come from new information in the
scorer. This adds one: ColBERT-style late interaction, where different query tokens can match
different evidence tokens — the mechanism the project's earlier probe could not test, because it
ran MaxSim over Laya's *untrained* token states and failed for that reason
(`make laya-maxsim-probe`).

`answerdotai/answerai-colbert-small-v1` is 33M parameters with a 96-dimensional token output and
no `MultiVectorEncoder` in the pinned `sentence-transformers` 4.1.0, so the head is built
directly: `BertModel` -> the checkpoint's `linear` projection -> L2 normalisation -> masked
MaxSim. A sanity check against the model card's own example is a test
(`test_multihop_ordering.py`): the absolute scale differs (the card's `similarity` normalises
differently), but the *ordering* is identical, and ordering is what a reranker uses.

Arms, all scored from the same cached pool with the same metric function so they are directly
comparable: the fused order, the cross-encoder order, coverage-plus-document-quota, and ColBERT.

Resumable per query through jobkit — the checkpoint stores one query's MaxSim scores, not token
embeddings, which keeps it small and makes a kill cost a single query.

    .venv/bin/python tools/multihop_ordering.py --limit 40          # smoke
    .venv/bin/python tools/multihop_ordering.py --out eval/benchmarks/multihop-rag/ordering
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
# Project-local model cache, following tools/beir.py and tools/bright.py.
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "colbert-hf-cache"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit, jobkit  # noqa: E402
from multihop_official import chunk_text  # noqa: E402
from multihop_rerank_experiment import (  # noqa: E402
    coverage_doc_quota_order,
    document_of,
    gold_matches_for_pool,
    metrics_from_matches,
)

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
MODEL_ID = "answerdotai/answerai-colbert-small-v1"
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")
ARMS = ("fused", "ce", "covquota", "colbert", "colbert-ce10")


# ── pure helpers ─────────────────────────────────────────────────────────

def rank_positions(scores: Sequence[float]) -> list[int]:
    """Positions ordered by descending score, ties broken by position for determinism."""
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def maxsim_scores(
    query: Any, docs: Sequence[Any], masks: Sequence[Any] | None = None
) -> list[float]:
    """ColBERT MaxSim: for each document, the sum over query tokens of the best match.

    `query` is `(tq, dim)`; each doc is `(td, dim)`. With `masks`, entries that are False —
    padding and special tokens — are excluded from the max, so a padded zero vector cannot win a
    query token's max against a genuinely negative dot product.
    """
    import numpy as np

    q = np.asarray(query, dtype=np.float32)
    out: list[float] = []
    for index, doc in enumerate(docs):
        d = np.asarray(doc, dtype=np.float32)
        sim = q @ d.T  # (tq, td)
        if masks is not None:
            keep = np.asarray(masks[index], dtype=bool)
            if not keep.any():
                out.append(float("-inf"))
                continue
            sim = sim[:, keep]
        out.append(float(sim.max(axis=1).sum()))
    return out


def pad_scores(scores: Sequence[float], length: int, fill: float = float("-inf")) -> list[float]:
    """Align a candidate-subset score list to the full pool, so positions stay comparable."""
    out = [float(s) for s in scores[:length]]
    return out + [fill] * (length - len(out))


def build_corpus(raw: Sequence[Mapping[str, Any]], overlap: int, size: int = 256) -> dict[str, str]:
    """The benchmark's chunking, exactly as the harness and the CE cache built it."""
    corpus: dict[str, str] = {}
    for article in raw:
        for j, chunk in enumerate(chunk_text(article["title"] + " " + article["body"], size, overlap)):
            corpus[f"{article['url']}#{j}"] = chunk
    return corpus


def covquota_order(
    pool: Sequence[str],
    ce_scores: Sequence[float],
    atoms: Sequence[Sequence[int]],
    atom_weight: Sequence[float],
    top_k: int,
    lam: float,
    min_docs: int,
) -> list[int]:
    """The project's coverage-plus-document-quota rule, over pool positions.

    The quota is per *article*: `document_of` maps a chunk id back to the article it came from.
    Passing chunk ids instead makes every chunk its own document, so the quota never fires and
    the arm silently scores lower — a 0.017 MAP@10 difference on the 400-query subset.
    """
    docs = [document_of(chunk_id) for chunk_id in pool]
    return coverage_doc_quota_order(ce_scores, atoms, atom_weight, docs, top_k,
                                    lam=lam, min_docs=min_docs)


def arm_metrics(
    pool: Sequence[str],
    ce_scores: Sequence[float],
    atoms: Sequence[Sequence[int]],
    atom_weight: Sequence[float],
    matches: Sequence[set],
    gold_count: int,
    colbert: Sequence[float],
    *,
    top_k: int,
    lam: float,
    min_docs: int,
) -> dict[str, dict[str, float]]:
    """One metric row per arm, each selecting/reordering from the same pool."""
    ce_ranked = rank_positions(ce_scores)[:top_k]
    colbert_ranked = rank_positions(colbert)[:top_k]
    covquota = covquota_order(pool, ce_scores, atoms, atom_weight, top_k, lam, min_docs)[:top_k]
    # The pure ordering-headroom test: reorder the cross-encoder's *own* top-k by MaxSim. Its
    # recall is fixed by the CE selection, so any change is ordering alone — which is what
    # `ceiling-decomp.md` measures as +0.1254 available from reordering the same top-10.
    colbert_ce10 = sorted(ce_ranked, key=lambda i: (-float(colbert[i]), i))
    return {
        "fused": metrics_from_matches(list(range(min(top_k, len(pool)))), matches, gold_count, top_k),
        "ce": metrics_from_matches(ce_ranked, matches, gold_count, top_k),
        "covquota": metrics_from_matches(covquota, matches, gold_count, top_k),
        "colbert": metrics_from_matches(colbert_ranked, matches, gold_count, top_k),
        "colbert-ce10": metrics_from_matches(colbert_ce10, matches, gold_count, top_k),
    }


def summarize(per_query: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        rows = [per_query[q][arm] for q in per_query if arm in per_query[q]]
        out[arm] = {m: sum(r[m] for r in rows) / len(rows) for m in METRICS} if rows else {m: 0.0 for m in METRICS}
    return out


def compare_to(
    per_query: Mapping[str, Mapping[str, Mapping[str, float]]],
    baseline: str,
    candidate: str,
    *,
    samples: int = 10_000,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap of `candidate` against `baseline` over the queries both scored."""
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        base = {q: per_query[q][baseline][metric] for q in per_query if baseline in per_query[q]}
        cand = {q: per_query[q][candidate][metric] for q in base}
        out[metric] = evalkit.paired_bootstrap(base, cand, samples=samples)
    return out


# ── the model ────────────────────────────────────────────────────────────

def _batched(items: Sequence[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


class ColBERT:
    """Masked MaxSim over `BertModel` + the checkpoint's linear projection."""

    def __init__(self, model_id: str = MODEL_ID, *, query_max_len: int = 32,
                 doc_max_len: int = 256, batch_size: int = 32, device: str | None = None) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import BertModel, BertTokenizerFast

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        state = load_file(hf_hub_download(model_id, "model.safetensors"))
        weight = state["linear.weight"]
        self.tokenizer = BertTokenizerFast.from_pretrained(model_id)
        self.bert = BertModel.from_pretrained(model_id).to(self.device).eval()
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            linear.weight.copy_(weight)
        self.linear = linear.to(self.device).eval()
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len
        self.batch_size = batch_size
        self._special = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                         self.tokenizer.pad_token_id}

    def _encode(self, texts: Sequence[str], max_len: int):
        torch = self.torch
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=max_len,
                             return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            hidden = self.bert(input_ids=input_ids, attention_mask=attention).last_hidden_state
            vectors = torch.nn.functional.normalize(self.linear(hidden), dim=-1)
        mask = attention.bool().clone()
        for special in self._special:
            mask &= input_ids != special
        return vectors.float().cpu().numpy(), mask.cpu().numpy()

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        query_vectors, _ = self._encode([query], self.query_max_len)
        out: list[float] = []
        for batch in _batched(list(docs), self.batch_size):
            doc_vectors, doc_mask = self._encode(batch, self.doc_max_len)
            out.extend(maxsim_scores(query_vectors[0], doc_vectors, doc_mask))
        return out


# ── run ──────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_ordering.py")
    ap.add_argument("--cache", default=str(ROOT_BENCH / "ce-scores.json"),
                    help="the CE cache that defines the pools, atoms and golds")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", default=str(ROOT_BENCH / "ordering"))
    ap.add_argument("--candidates", type=int, default=100,
                    help="pool prefix to score with MaxSim (the cache depth is 100)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.7, help="coverage weight for the covquota arm")
    ap.add_argument("--min-docs", type=int, default=3)
    ap.add_argument("--query-max-len", type=int, default=32)
    ap.add_argument("--doc-max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke runs only")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--merge", type=int, default=0, metavar="N")
    return ap.parse_args(argv)


def _fingerprint(args: argparse.Namespace) -> str:
    return jobkit.config_fingerprint(
        tool="multihop_ordering",
        model=args.model,
        cache=Path(args.cache).name,
        candidates=args.candidates,
        top_k=args.top_k,
        lam=args.lam,
        min_docs=args.min_docs,
        query_max_len=args.query_max_len,
        doc_max_len=args.doc_max_len,
        limit=args.limit,
    )


def _build_report(rows: dict) -> dict:
    if jobkit.META_KEY not in rows:
        raise RuntimeError("no shard checkpoints found — run the shards before --merge")
    meta = rows.pop(jobkit.META_KEY)
    per_query = {k: v for k, v in rows.items()}
    return {
        "_settings": meta["_settings"],
        "summary": summarize(per_query),
        "comparisons": {f"{c}_vs_{b}": compare_to(per_query, b, c)
                        for b, c in (("fused", "ce"), ("fused", "covquota"),
                                     ("covquota", "colbert"), ("ce", "colbert"), ("ce", "colbert-ce10"))},
        "per_query": per_query,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Late interaction as an ordering head on MultiHop-RAG",
        "",
        f"- {report['_settings'].get('queries')} queries, candidates "
        f"{report['_settings'].get('candidates')}, top-k {report['_settings'].get('top_k')}, "
        f"model `{report['_settings'].get('model')}`",
        "",
        "| arm | Hits@10 | Hits@4 | MAP@10 | MRR@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        row = s.get(arm, {})
        lines.append(f"| {arm} | " + " | ".join(f"{row.get(m, 0.0):.4f}" for m in METRICS) + " |")
    lines += ["", "## Paired comparisons", ""]
    for name, block in report["comparisons"].items():
        lines.append(f"**{name}**")
        for metric, row in block.items():
            sig = "" if row["significant"] else " (ns)"
            lines.append(f"- {metric}: {row['delta']:+.4f} "
                         f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_value']:.4f}{sig}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    shard_index, shard_count = jobkit.parse_shard(args.shard)
    fingerprint = _fingerprint(args)

    if args.merge:
        try:
            merged = jobkit.merge_shards(args.out, args.merge, fingerprint=fingerprint,
                                         build=_build_report)
        except (RuntimeError, jobkit.FingerprintMismatch) as exc:
            print(f"merge failed: {exc}", file=sys.stderr)
            return 2
        evalkit.write_text(Path(str(args.out) + ".md"), render_markdown(merged))
        print(f"merged {args.merge} shard(s) -> {args.out}.json")
        return 0

    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    settings = cache.get("_settings", {})
    overlap = int(settings.get("overlap", 200))
    raw = json.load(open(ROOT_BENCH / "corpus.json"))
    corpus = build_corpus(raw, overlap)
    items_all = json.load(open(ROOT_BENCH / "MultiHopRAG.json"))
    queries = dict(cache["queries"])
    golds = cache["golds"]
    # The cache stores pools/scores/golds but not query text; it is keyed by the record's
    # index in MultiHopRAG.json, which is where the text comes from.
    query_text = {key: items_all[int(key)]["query"] for key in queries}
    atom_weight = settings.get("atom_weight")
    if not atom_weight:
        highest = max((a for row in queries.values() for atom in row["atoms"] for a in atom),
                      default=-1)
        atom_weight = [1.0] * (highest + 1)
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])
    print(f"[overlap={overlap}] {len(corpus)} chunks, {len(queries)} queries, "
          f"candidates {args.candidates}", flush=True)

    stem = jobkit.shard_stem(args.out, shard_index, shard_count)
    if args.fresh:
        for suffix in (".jsonl", ".manifest.json"):
            Path(str(stem) + suffix).unlink(missing_ok=True)
    run = jobkit.ResumableRun(str(stem) + ".json", fingerprint=fingerprint,
                              shard=(shard_index, shard_count))

    if jobkit.META_KEY not in run:
        run.record(jobkit.META_KEY, {
            "_settings": {**vars(args), "queries": len(queries), "corpus_chunks": len(corpus)},
            "query_order": list(queries),
        })

    mine = jobkit.shard_items(list(queries), shard_index, shard_count)
    todo = run.pending(mine)
    print(f"  {len(todo)} of {len(mine)} queries to score ({len(run) - 1} checkpointed)",
          flush=True)

    encoder = ColBERT(args.model, query_max_len=args.query_max_len,
                      doc_max_len=args.doc_max_len, batch_size=args.batch_size,
                      device=args.device) if todo else None
    started = time.perf_counter()
    for position, key in enumerate(todo, 1):
        pool = list(queries[key]["pool"])
        docs = [corpus[doc_id] for doc_id in pool[: args.candidates]]
        scores = encoder.score(query_text[key], docs)
        run.record(key, {"maxsim": pad_scores(scores, len(pool))})
        if position % 25 == 0:
            run.touch(total=len(mine))
            print(f"    {position}/{len(todo)} in {time.perf_counter() - started:.0f}s", flush=True)

    shared = run.get(jobkit.META_KEY)
    per_query: dict[str, dict] = {}
    for key in shared["query_order"]:
        row = run.get(key)
        if row is None:
            continue
        pool = list(queries[key]["pool"])
        ce = list(queries[key]["ce"])
        atoms = [list(a) for a in queries[key]["atoms"]]
        matches = gold_matches_for_pool(corpus, pool, golds[key])
        per_query[key] = arm_metrics(
            pool, ce, atoms, atom_weight, matches, max(len(golds[key]), 1), row["maxsim"],
            top_k=args.top_k, lam=args.lam, min_docs=args.min_docs,
        )
    report = {
        "_settings": {**shared["_settings"], "seconds": round(time.perf_counter() - started, 1)},
        "summary": summarize(per_query),
        "comparisons": {f"{c}_vs_{b}": compare_to(per_query, b, c)
                        for b, c in (("fused", "ce"), ("fused", "covquota"),
                                     ("covquota", "colbert"), ("ce", "colbert"), ("ce", "colbert-ce10"))},
        "per_query": per_query,
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
