"""An LLM-free nugget / claim-recall protocol.

The evaluation brief (`docs/research/07` §5) asks for a nugget protocol with no API and no
generator: nuggets derived programmatically from the benchmark's gold support sentences, then
measured against retrieved evidence. It matters because the benchmark's headline MAP@10 is a
*single-gold-set* metric — it does not say how much of each gold fact a ranking actually contains —
and the repository already found that its coverage objective raises Hits@4 but not MAP@10. Claim
recall is the missing half of that story, and it needs no judge.

Two arms are computable from what is already on disk: the **fused** pool order and the **cross-
encoder** order over the same pool (`ce-scores-full.json`). Nuggets are extracted without a model:

* **numbers/dates** — digit runs and month names;
* **entities** — capitalised word runs (one to three tokens);
* **key terms** — content words of four or more characters.

A nugget is covered when it appears in the retrieved chunks — numbers/entities as the benchmark's
own whitespace-normalised substring, key terms as a token. Recall is averaged over the gold facts
of a query, then over queries.

    .venv/bin/python tools/nugget_recall.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402

BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
CORPUS = BENCH / "corpus.json"
SCORES = BENCH / "ce-scores-full.json"

#: The benchmark's 256-token window with the report's overlap.
CHUNK_TOKENS = 256
_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_MONTH = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b"
)
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\b")
_TERM = re.compile(r"[a-zA-Z][a-zA-Z\-]{3,}")
_STOPWORDS = frozenset(
    """about above after again against all also am an and any are because been before being below
    between both but by can could did does doing down during each few for from further had has have
    having he her here hers him his how into is it its itself just more most my no nor not now of off
    on once only or other our out over own same she should so some such than that the their them then
    there these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your""".split()
)


def chunk_text(text: str, size: int = CHUNK_TOKENS, overlap: int = 200) -> list[str]:
    """The benchmark's own 256-token window convention."""
    tokens = text.split()
    step = max(1, size - overlap)
    out: list[str] = []
    index = 0
    while index < len(tokens):
        out.append(" ".join(tokens[index : index + size]))
        if index + size >= len(tokens):
            break
        index += step
    return out


def build_chunks(corpus_path: Path = CORPUS, overlap: int = 200) -> dict[str, str]:
    """``url#index`` -> chunk text, exactly as the harness chunks the corpus."""
    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    chunks: dict[str, str] = {}
    for article in raw:
        for position, text in enumerate(
            chunk_text(f"{article['title']} {article['body']}", overlap=overlap)
        ):
            chunks[f"{article['url']}#{position}"] = text
    return chunks


def normalize(text: str) -> str:
    """The benchmark's gold-match normalisation: drop whitespace, casefold."""
    return _WHITESPACE.sub("", text).lower()


def nuggets(fact: str) -> set[tuple[str, str]]:
    """Programmatic nuggets of one gold fact: numbers, dates, entities and key terms."""
    out: set[tuple[str, str]] = set()
    for match in _NUMBER.findall(fact):
        out.add(("number", match))
    for match in _MONTH.findall(fact):
        out.add(("date", match))
    for match in _ENTITY.findall(fact):
        out.add(("entity", match))
    for word in _TERM.findall(fact):
        lowered = word.lower()
        if lowered not in _STOPWORDS:
            out.add(("term", lowered))
    return out


class Evidence:
    """Coverage lookups over the retrieved chunks, precomputed once per chunk."""

    def __init__(self, chunks: Mapping[str, str]) -> None:
        self._normalized: dict[str, str] = {}
        self._tokens: dict[str, set[str]] = {}
        for chunk_id, text in chunks.items():
            self._normalized[chunk_id] = normalize(text)
            self._tokens[chunk_id] = {t.lower() for t in _TERM.findall(text)}

    def covers(self, chunk_id: str, nugget: tuple[str, str]) -> bool:
        kind, value = nugget
        if kind == "term":
            return value in self._tokens.get(chunk_id, set())
        return normalize(value) in self._normalized.get(chunk_id, "")


def claim_recall(
    rankings: Mapping[str, Iterable[str]],
    golds: Mapping[str, Sequence[str]],
    evidence: Evidence,
    *,
    k: int,
) -> dict[str, Any]:
    """Macro claim-recall@k for each arm, plus the per-query values.

    Recall for one query is the fraction of its gold facts' nuggets found anywhere in the arm's
    top ``k`` chunks; the macro value is the mean over queries that have at least one nugget.
    """
    per_arm: dict[str, dict[str, float]] = {}
    for arm, by_query in rankings.items():
        per_query: dict[str, float] = {}
        for query_id, facts in golds.items():
            top = list(by_query.get(query_id, []))[:k]
            wanted: set[tuple[str, str]] = set()
            for fact in facts:
                wanted |= nuggets(fact)
            if not wanted:
                continue
            found = set()
            for chunk_id in top:
                for nugget in wanted - found:
                    if evidence.covers(chunk_id, nugget):
                        found.add(nugget)
                if len(found) == len(wanted):
                    break
            per_query[query_id] = round(len(found) / len(wanted), 4)
        values = list(per_query.values())
        per_arm[arm] = {
            "mean": round(sum(values) / len(values), 4) if values else 0.0,
            "queries": len(values),
            "per_query": per_query,
        }
    return per_arm


def build_arm_rankings(path: Path = SCORES) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """``(fused, ce)`` rankings per query from the cached pool and cross-encoder scores.

    The pool is already in fused order; the cross-encoder arm is the same pool sorted by the
    cached score, which is what the CE rerank does.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    fused: dict[str, list[str]] = {}
    ce: dict[str, list[str]] = {}
    for query_id, row in payload["queries"].items():
        pool = list(row["pool"])
        scores = list(row["ce"])
        fused[query_id] = pool
        ce[query_id] = [
            chunk_id
            for chunk_id, _score in sorted(zip(pool, scores), key=lambda pair: -pair[1])
        ]
    return fused, ce


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="nugget_recall.py")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--scores", default=str(SCORES))
    ap.add_argument("--out", default=str(ROOT / "eval" / "nugget-recall"))
    ap.add_argument("--depths", default="10,100")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scores_path = Path(args.scores)
    if not scores_path.is_file() or not Path(args.corpus).is_file():
        print("nugget-recall: SKIPPED — corpus/ce-scores reports not found")
        return 0
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    overlap = int(payload.get("_settings", {}).get("overlap") or 200)
    golds = {qid: facts for qid, facts in payload["golds"].items()}
    chunks = build_chunks(Path(args.corpus), overlap=overlap)
    evidence = Evidence(chunks)
    fused, ce = build_arm_rankings(scores_path)

    report: dict[str, Any] = {
        "_settings": {"overlap": overlap, "chunks": len(chunks), "queries": len(golds)},
        "depths": {},
    }
    for depth in [int(value) for value in args.depths.split(",")]:
        report["depths"][str(depth)] = claim_recall(
            {"fused": fused, "ce": ce}, golds, evidence, k=depth
        )
    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    evalkit.write_text(out.with_suffix(".md"), render_markdown(report))
    print(render_markdown(report))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# LLM-free nugget / claim recall",
        "",
        f"- {report['_settings']['queries']} labelled queries, "
        f"{report['_settings']['chunks']} chunks (overlap {report['_settings']['overlap']})",
        "- nuggets: numbers, dates, entities, key terms; coverage is the benchmark's normalised",
        "  substring for numbers/entities/dates and a token for key terms",
        "",
        "| depth | fused claim-recall | cross-encoder claim-recall |",
        "| --- | --- | --- |",
    ]
    for depth, arms in report["depths"].items():
        lines.append(f"| @{depth} | {arms['fused']['mean']:.4f} | {arms['ce']['mean']:.4f} |")
    lines += [
        "",
        "Neither arm is reordered beyond the pool for the fused row; the CE row sorts the same",
        "pool by the cached cross-encoder score, so the comparison isolates the second stage.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
