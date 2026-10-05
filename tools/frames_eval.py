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


def gold_titles(row: Mapping[str, str]) -> set[str]:
    """The articles a question's links name, normalised."""
    found = set()
    for value in row.values():
        if isinstance(value, str):
            found.update(normalise_title(match) for match in LINK_RE.findall(value))
    return {title for title in found if title}


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


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    questions = []
    for row in rows:
        prompt = (row.get("Prompt") or "").strip()
        gold = gold_titles(row)
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
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lexical-only", action="store_true", help="skip the dense arm (no weights)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "frames-eval"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    questions = load_questions(Path(args.tsv))
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

    dense = None
    if not args.lexical_only:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model)
        titles = list(corpus)
        started = time.perf_counter()
        vectors = model.encode([corpus[title] for title in titles], normalize_embeddings=True,
                               batch_size=32, show_progress_bar=False)
        print(f"  embedded {len(titles)} articles in {time.perf_counter() - started:.1f}s", flush=True)
        dense = (titles, vectors)

    arms: dict[str, list[dict[str, Any]]] = {"lexical": [], "dense": []}
    for question in questions:
        gold = question["gold"]
        coverage = len(gold & set(corpus)) / len(gold) if gold else 0.0
        ranked = [hit.chunk_id for hit in index.search(question["query"], k=args.top_k)]
        row = {"query": question["query"], "gold_coverage": coverage, **score_ranking(ranked, gold)}
        arms["lexical"].append(row)
        if dense is not None:
            titles, vectors = dense
            query = model.encode([question["query"]], normalize_embeddings=True)[0]
            order = (vectors @ query).argsort()[::-1][: args.top_k]
            arms["dense"].append({**row, **score_ranking([titles[i] for i in order], gold)})

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
