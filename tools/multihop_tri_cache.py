"""Cache bridge-conditioned Laya scores for the MultiHop-RAG candidate pool.

The two-way relevance question cannot separate the second hop from a wrong chain that
shares the query's surface entity, because a second-hop passage is only weakly aligned
with the *query* and strongly aligned with the entity hop 1 resolved. BridgeRAG
(arXiv:2604.03384) buys the fix with a 70B LLM judge; this engine already owns a
calibrated decision head, so the fix here is a third state key (``bridge``) and a question
(:data:`laya_rag.rerank.BRIDGE_RELEVANCE_WORDINGS`).

Scoring the pool is the expensive part, so it is cached once and every selection rule is
then compared over the same numbers, exactly as ``tools/multihop_ce_cache.py`` does for the
cross-encoder. The pool comes from the CE cache, so a candidate's index means the same
thing in both files and no re-retrieval is needed.

Two arms, because a conditional score can improve for the wrong reason:

* ``real`` — the bridge is the top fused candidate's text (the passage hop 1 found);
* ``shuffled`` — the bridge is *another query's* top candidate. The instruction text and
  the state keys are identical; only the conditioning content is wrong. A gain that
  survives only in ``real`` is conditioning; a gain in both is the longer instruction.

    .venv/bin/python tools/multihop_tri_cache.py --arm both --limit-queries 400

Writes ``<out>-<arm>-<wording>.json`` (gitignored). Resumable: an interrupted run is
continued from the queries already scored.
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
from multihop_official import chunk_text  # noqa: E402

from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.model import LayaModel  # noqa: E402
from laya_rag.rerank import BRIDGE_RELEVANCE_WORDINGS, LayaReranker  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"


def query_texts() -> dict[str, str]:
    """``{cache query id: question}`` for every non-null MultiHop-RAG record.

    The id is the record's index in ``MultiHopRAG.json``, which is what
    ``tools/multihop_ce_cache.py`` keys its cache by, so the two files agree by
    construction rather than by convention.
    """
    items = json.loads((ROOT_BENCH / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return {
        str(i): x["query"]
        for i, x in enumerate(items)
        if x.get("question_type") != "null_query"
    }


def load_corpus(overlap: int) -> dict[str, str]:
    """Rebuild the chunked corpus exactly as the CE cache did, so ids resolve to text."""
    raw = json.loads((ROOT_BENCH / "corpus.json").read_text(encoding="utf-8"))
    corpus: dict[str, str] = {}
    for article in raw:
        body = article["title"] + " " + article["body"]
        for j, text in enumerate(chunk_text(body, 256, overlap)):
            corpus[f"{article['url']}#{j}"] = text
    return corpus


def bridge_for(fused_pool: Sequence[str], corpus: Mapping[str, str], bridge_k: int = 1) -> str:
    """The bridge text: the top fused candidate, or the top ``bridge_k`` joined.

    BridgeRAG conditions on the single top-1 hop-1 passage. ``bridge_k > 1`` is the
    multi-bridge variant — unmeasured here, exposed so it can be swept rather than assumed.
    """
    if bridge_k < 1:
        raise ValueError(f"bridge_k must be >= 1, got {bridge_k!r}")
    chosen = [corpus[doc_id] for doc_id in list(fused_pool)[:bridge_k] if doc_id in corpus]
    return "\n\n".join(chosen)


def shuffled_bridges(
    query_ids: Sequence[str],
    bridge_text: Mapping[str, str],
    seed: int,
) -> dict[str, str]:
    """Each query's bridge replaced by a different query's bridge.

    A derangement where possible: no query keeps its own bridge, so the control can never
    accidentally be the treatment. Deterministic in ``seed`` and in the (sorted) query
    order, so the control is reproducible.
    """
    ordered = sorted(query_ids, key=lambda value: (len(value), value))
    texts = [bridge_text[q] for q in ordered]
    if len(ordered) < 2:
        return {q: "" for q in ordered}
    shifted = list(texts)
    for _ in range(100):
        random.Random(seed).shuffle(shifted)
        if all(a != b for a, b in zip(texts, shifted)):
            break
    else:  # pragma: no cover - a shuffle that always collided on >= 2 distinct texts
        shifted = texts[1:] + texts[:1]
    return dict(zip(ordered, shifted))


def score_arm(
    reranker: LayaReranker,
    queries: Mapping[str, str],
    pools: Mapping[str, Sequence[str]],
    corpus: Mapping[str, str],
    bridges: Mapping[str, str],
    wording: str,
    batch_size: int,
    started: float,
    done: dict[str, Any],
) -> dict[str, Any]:
    """Score every unresolved query for one arm, returning the extended result map."""
    pending = [q for q in sorted(queries, key=lambda v: (len(v), v)) if q not in done]
    for n, q in enumerate(pending, start=1):
        texts = [corpus[doc_id] for doc_id in pools[q]]
        scores, _usage = reranker.score_relevance_conditioned(
            queries[q], bridges[q], texts, batch_size=batch_size, wording=wording
        )
        done[q] = [round(float(s), 6) for s in scores]
        if n % 25 == 0 or n == len(pending):
            print(
                f"    {n}/{len(pending)} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    return done


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_tri_cache.py")
    ap.add_argument("--ce-cache", default=str(ROOT_BENCH / "ce-scores.json"))
    ap.add_argument("--wording", default="bridge", choices=sorted(BRIDGE_RELEVANCE_WORDINGS))
    ap.add_argument("--arm", default="both", choices=["real", "shuffled", "both"])
    ap.add_argument("--bridge-k", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shuffle-seed", type=int, default=17)
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="score only this many queries (they are taken in cache order)")
    ap.add_argument("--out", default=str(ROOT_BENCH / "tri-scores"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ce_path = Path(str(args.ce_cache))
    if not ce_path.is_file():
        raise SystemExit(f"{ce_path} is missing; run tools/multihop_ce_cache.py first")
    cache = json.loads(ce_path.read_text(encoding="utf-8"))
    settings = cache["_settings"]
    overlap = int(settings["overlap"])
    depth = int(settings["depth"])

    texts = query_texts()
    corpus = load_corpus(overlap)
    names = sorted(cache["queries"], key=lambda v: (len(v), v))
    if args.limit_queries:
        names = names[: args.limit_queries]
    queries = {q: texts[q] for q in names}
    pools = {q: list(cache["queries"][q]["pool"])[:depth] for q in names}
    bridge_text = {q: bridge_for(pools[q], corpus, args.bridge_k) for q in names}
    shuffled = shuffled_bridges(names, bridge_text, args.shuffle_seed)
    print(
        f"bridge-k={args.bridge_k} wording={args.wording} depth={depth} "
        f"queries={len(queries)} chunks={len(corpus)}",
        flush=True,
    )

    reranker = LayaReranker(LayaModel(EngineConfig()))
    arms = ["real", "shuffled"] if args.arm == "both" else [args.arm]
    for arm in arms:
        out = Path(f"{args.out}-{arm}-{args.wording}.json")
        done: dict[str, Any] = {}
        if out.is_file():
            existing = json.loads(out.read_text(encoding="utf-8"))
            done = dict(existing.get("scores") or {})
            print(f"  {arm}: resuming with {len(done)} queries already scored", flush=True)
        bridges = bridge_text if arm == "real" else shuffled
        started = time.perf_counter()
        print(f"  {arm}: scoring {len(queries) - len(done)} queries...", flush=True)
        scores = score_arm(
            reranker, queries, pools, corpus, bridges, args.wording,
            args.batch_size, started, done,
        )
        payload = {
            "_settings": {
                "arm": arm,
                "wording": args.wording,
                "bridge_k": args.bridge_k,
                "batch_size": args.batch_size,
                "shuffle_seed": args.shuffle_seed,
                "ce_cache": ce_path.name,
                "model": settings.get("model"),
                "overlap": overlap,
                "depth": depth,
                "limit_queries": len(queries),
                "seconds": round(time.perf_counter() - started, 1),
            },
            "scores": scores,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out} in {payload['_settings']['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
