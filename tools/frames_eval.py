"""FRAMES retrieval evaluation — ROADMAP 3.1.

824 questions whose evidence is English Wikipedia articles, so this is a *retrieval* benchmark with a
document-level gold set rather than a labelled span set: the question's `wiki_links` name the articles
that contain what is needed. The plan's rule applies — if the generator dominates the variance, report
retrieval only — and retrieval is what can be measured offline and deterministically.

    .venv/bin/python tools/frames_eval.py                 # lexical + dense arms
    .venv/bin/python tools/frames_eval.py --lexical-only  # no weights needed

Two honest limits are *reported rather than hidden*. Gold titles come from the dataset's URLs while the
corpus is keyed by the API's canonical titles, so a redirect (`USA` → `United States`) can miss; the
report therefore carries **gold coverage**, the share of gold titles that resolve to a corpus article,
and recall is only interpretable alongside it. And a question may be answerable from articles it does
not link, which no retrieval metric can see.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "beir" / "hf-cache"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402

FRAMES = ROOT / "eval" / "frames"
LINK_RE = re.compile(r"https?://en\.wikipedia\.org/wiki/([^\s'\"\]|]+)")
K_VALUES = (1, 5, 10, 20)


def normalise_title(title: str) -> str:
    """Fold a Wikipedia title so a dataset URL and an API title can be compared.

    Redirects are *not* resolved here — that needs the API — so callers report coverage alongside any
    recall figure rather than assuming the two spellings are the same article.
    """
    text = urllib.parse.unquote(str(title)).replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold().rstrip(".,;")


def load_title_map(path: Path) -> dict[str, str]:
    """Dataset title -> canonical article title, both normalised, from `frames_corpus --resolve-map`.

    Absent means the dataset's spellings are used as-is, which understates recall by the redirect gap;
    the report's `gold_coverage` is what makes that visible instead of silent.
    """
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {normalise_title(key): normalise_title(value) for key, value in raw.items()}


def gold_titles(row: Mapping[str, str], title_map: Mapping[str, str] | None = None) -> set[str]:
    """The articles a question's links name, normalised and resolved to canonical titles."""
    found = set()
    for value in row.values():
        if isinstance(value, str):
            found.update(normalise_title(match) for match in LINK_RE.findall(value))
    title_map = title_map or {}
    return {title_map.get(title, title) for title in found if title}


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return 1.0 if any(item in gold for item in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: set[str], k: int) -> float:
    for position, item in enumerate(ranked[:k], start=1):
        if item in gold:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """Binary-gain nDCG@k; the ideal ranking puts the gold articles first."""
    dcg = sum(1.0 / math.log2(position + 1)
              for position, item in enumerate(ranked[:k], start=1) if item in gold)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def score_ranking(ranked: Sequence[str], gold: set[str]) -> dict[str, float]:
    out = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in K_VALUES}
    out["mrr@10"] = reciprocal_rank(ranked, gold, 10)
    out["ndcg@10"] = ndcg_at_k(ranked, gold, 10)
    return out


def load_questions(path: Path, title_map: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    questions = []
    for row in rows:
        prompt = (row.get("Prompt") or "").strip()
        gold = gold_titles(row, title_map)
        if prompt and gold:
            questions.append({"query": prompt, "gold": gold, "answer": row.get("Answer", ""),
                              "types": row.get("reasoning_types", "")})
    return questions


def load_corpus(path: Path) -> dict[str, str]:
    """The corpus keyed by normalised title, which is what the rankings are matched against."""
    corpus: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            corpus[normalise_title(row["title"])] = row["text"]
    return corpus


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 5) -> list[str]:
    """Reciprocal rank fusion of top-k runs, the combiner this project uses everywhere else."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: -pair[1])]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize(per_query: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = [key for key in per_query[0] if key.startswith(("recall@", "mrr@", "ndcg@"))]
    summary = {key: mean(row[key] for row in per_query) for key in keys}
    summary["questions"] = len(per_query)
    summary["gold_coverage"] = mean(row["gold_coverage"] for row in per_query)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="frames_eval.py")
    ap.add_argument("--tsv", default=str(FRAMES / "test.tsv"))
    ap.add_argument("--corpus", default=str(FRAMES / "corpus.jsonl"))
    ap.add_argument("--title-map", default=str(FRAMES / "title-map.json"),
                    help="dataset-title -> canonical-title map; without it gold coverage is understated")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="encoder batch size; the 307M Laya-adapted encoder needs 8 on MPS")
    ap.add_argument("--max-length", type=int, default=512,
                    help="encoder sequence budget; FRAMES articles are long and ModernBERT "
                         "advertises 8192, which allocates an 8 GB attention buffer at batch 32")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ensemble-models", default=None,
                    help="comma-separated encoders to average into an ensemble arm "
                         "(e.g. bge-small,bge-base,bge-large), plus an RRF row against BM25")
    ap.add_argument("--rrf-k", type=int, default=5)
    ap.add_argument("--lexical-only", action="store_true", help="skip the dense arm (no weights)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "frames-eval"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("frames_eval.py")
    title_map = load_title_map(Path(args.title_map))
    if title_map:
        print(f"resolved {len(title_map)} dataset titles to canonical ones", flush=True)
    questions = load_questions(Path(args.tsv), title_map)
    if args.limit:
        questions = questions[: args.limit]
    corpus = load_corpus(Path(args.corpus))
    if not corpus:
        raise SystemExit(f"no corpus at {args.corpus}; run tools/frames_corpus.py first")
    print(f"{len(questions)} questions, {len(corpus)} articles", flush=True)

    from laya_rag.lexical import BM25Index

    started = time.perf_counter()
    index = BM25Index(drop_stopwords=True, stem=True)
    index.build(_Doc(title, text) for title, text in corpus.items())
    print(f"  lexical index built in {time.perf_counter() - started:.1f}s", flush=True)

    # `titles` is the corpus order every encoder shares, so a ranking index means one article.
    titles = list(corpus)
    texts = [corpus[title] for title in titles]
    dense = None
    ensemble: list[tuple[str, Any, Any]] = []
    arms: dict[str, list[dict[str, Any]]] = {"lexical": []}

    if not args.lexical_only:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model)
        if args.max_length:
            model.max_seq_length = args.max_length
        started = time.perf_counter()
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=args.batch_size,
                               show_progress_bar=False)
        print(f"  embedded {len(titles)} articles in {time.perf_counter() - started:.1f}s", flush=True)
        dense = (titles, vectors, model)
        arms["dense"] = []

        if args.ensemble_models:
            for name in (part.strip() for part in args.ensemble_models.split(",")):
                if not name:
                    continue
                member = SentenceTransformer(name)
                if args.max_length:
                    member.max_seq_length = args.max_length
                started = time.perf_counter()
                member_vectors = member.encode(texts, normalize_embeddings=True,
                                               batch_size=args.batch_size,
                                               show_progress_bar=False)
                print(f"  embedded with {name} in {time.perf_counter() - started:.1f}s", flush=True)
                ensemble.append((name, member, member_vectors))
            if ensemble:
                arms["ensemble"] = []
                arms["ensemble+bm25"] = []

    def rank_with(vectors, model, query: str) -> list[str]:
        query_vector = model.encode([query], normalize_embeddings=True)[0]
        order = (vectors @ query_vector).argsort()[::-1][: args.top_k]
        return [titles[i] for i in order]

    for question in questions:
        gold = question["gold"]
        coverage = len(gold & set(corpus)) / len(gold) if gold else 0.0
        ranked = [hit.chunk_id for hit in index.search(question["query"], k=args.top_k)]
        row = {"query": question["query"], "gold_coverage": coverage, **score_ranking(ranked, gold)}
        arms["lexical"].append(row)
        if dense is not None:
            arms["dense"].append({**row, **score_ranking(rank_with(dense[1], dense[2],
                                                                   question["query"]), gold)})
        if ensemble:
            # The mean of the members' similarities is what the BEIR ensemble does; the scores are
            # summed rather than averaged, which cannot change the order.
            total = None
            for _name, member, member_vectors in ensemble:
                scores = member_vectors @ member.encode([question["query"]],
                                                        normalize_embeddings=True)[0]
                total = scores if total is None else total + scores
            fused_ranking = [titles[i] for i in total.argsort()[::-1][: args.top_k]]
            arms["ensemble"].append({**row, **score_ranking(fused_ranking, gold)})
            arms["ensemble+bm25"].append({**row, **score_ranking(
                rrf_fuse([fused_ranking, ranked], k=args.rrf_k), gold)})

    report = {
        "_settings": vars(args),
        "articles": len(corpus),
        "summary": {arm: summarize(rows) for arm, rows in arms.items() if rows},
        "per_query": {arm: rows for arm, rows in arms.items() if rows},
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), report)
    print()
    for arm, summary in report["summary"].items():
        print(f"{arm:<8} recall@1 {summary['recall@1']:.4f}  recall@10 {summary['recall@10']:.4f}  "
              f"recall@20 {summary['recall@20']:.4f}  MRR@10 {summary['mrr@10']:.4f}  "
              f"nDCG@10 {summary['ndcg@10']:.4f}")
    print(f"gold coverage {report['summary']['lexical']['gold_coverage']:.4f} "
          f"over {report['summary']['lexical']['questions']} questions")
    print("wrote", str(out) + ".json")
    return 0


class _Doc:
    """One article as BM25 sees it: an id, and the text to index."""

    __slots__ = ("chunk_id", "index_text", "text")

    def __init__(self, chunk_id: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.index_text = text


if __name__ == "__main__":
    sys.exit(main())
