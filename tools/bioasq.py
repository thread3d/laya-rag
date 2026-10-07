"""Ingest the gated BioASQ download into the layout `tools/beir.py` reads.

BioASQ is the one BEIR member with no public corpus. Its two inputs come from a **free BioASQ
account**, and BEIR's own reproduction instructions say exactly what to do with them
([`examples/dataset`](https://github.com/beir-cellar/beir/blob/main/examples/dataset/README.md)):

1. the task 9a training set v2020 -- `allMeSH_2020.json`, ~14,913,939 articles, served as one JSON
   object whose `articles` array holds one article per line;
2. the 8B golden sets -- question `id`, `body`, and `documents`, where **every** listed document is
   relevant (BEIR's binary rule);
3. plus a handful of documents the qrels reference that the corpus lacks, listed in a public
   spreadsheet and merged in first.

This tool does those three things and then **refuses to finish unless the arithmetic lines up**: the
corpus must contain every document the 500 test queries judge, and its size must match what the
recipe predicts. That check is the point. `docs/findings-bioasq.md` records a public
mirror that is 813,939 documents short of the real corpus yet still holds 1,533 of the 1,549 gold
documents — a run on it would land near the published anchor and look right. A silent shortfall here
would be the same failure, so it is an assertion rather than a warning.

    # corpus.jsonl + queries.jsonl + qrels/test.tsv, from the downloads
    .venv/bin/python tools/bioasq.py \\
        --source eval/bioasq-src/allMeSH_2020.json \\
        --additional eval/bioasq-src/additional_passages.tsv \\
        --golden eval/bioasq-src/Task8BGoldenEnriched \\
        --out eval/beir/bioasq
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]

#: Article **lines** in the dump this repository ingested, verified by counting the file. The BioASQ
#: participants area advertises 14,913,939 for Training v.2020 and BEIR's reproduction note repeats it
#: with a "~"; the downloaded `allMeSH_2020.json` holds **one fewer**, 14,913,938. The gap is a single
#: document out of 14.9M, well inside the source's own approximation, and it is not a truncation —
#: the failure this constant exists to catch is the 813,939-document shortfall of the public mirror
#: (`docs/findings-bioasq.md`), which this rejects by a mile.
#:
#: The assertion is on lines parsed, not on distinct pmids: the dump repeats 84 pmids, and the
#: reference preprocessing drops the later copy, so the corpus it builds is 14,913,854 articles plus
#: the manual-add rows. Counting the file is the check that the *download* is whole; the dedup is a
#: processing detail and is reported separately.
EXPECTED_ARTICLES = 14_913_938

#: The member's test split: five golden batches of 100, which the recipe's binary rule turns into
#: 2,359 judgements over 2,298 distinct documents.
EXPECTED_QUERIES = 500


def _iter_source_lines(source: Path) -> Iterator[str]:
    """Yield raw lines from `allMeSH_2020.json`, whether given as the JSON or the zip that holds it.

    Decoded as **latin-1**, which is what the reference preprocessing uses and is why every byte
    still maps to a character: the dump is not valid UTF-8, so reading it as UTF-8 with replacement
    would silently turn accented characters in the abstracts into `U+FFFD`.
    """
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            member = next(
                (n for n in archive.namelist() if n.endswith(".json") and not n.startswith("__")),
                None,
            )
            if member is None:
                raise FileNotFoundError(f"{source} holds no .json member")
            with archive.open(member) as raw:
                yield from io.TextIOWrapper(raw, encoding="latin-1", errors="replace")
    else:
        with source.open(encoding="latin-1", errors="replace") as handle:
            yield from handle


def _parse_article_line(stripped: str) -> dict[str, Any] | None:
    """One line of the dump as an article, or None for the array's own closers.

    The array's closing `]}` is **appended to the last article's line** rather than sitting on its
    own, which is why a plain `json.loads` per line reports "Extra data" at the very end of a
    perfectly good download. That is a trap worth naming: it fails on the final record only, after
    fifteen million successful lines, so it reads like a corrupt file when it is an off-by-one.
    """
    if not stripped or stripped in {"]}", "]"}:
        return None
    payload = stripped
    if payload.endswith("]}"):
        payload = payload[:-2].rstrip()
    elif payload.endswith("]"):
        payload = payload[:-1].rstrip()
    payload = payload.rstrip(",")
    if not payload:
        return None
    return json.loads(payload)


def _clean(value: Any) -> str:
    """The recipe's text hygiene: a missing field is empty, and tabs/newlines become spaces."""
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("\t", " ").strip()


def _article_text(article: dict[str, Any]) -> tuple[str, str]:
    """`(title, abstract)` with the recipe's rule that neither may be the only one dropped."""
    title = _clean(article.get("title"))
    abstract = _clean(article.get("abstractText"))
    if not title and not abstract:
        raise ValueError(f"article {article.get('pmid')!r} has neither title nor abstract")
    return title, abstract


def golden_questions(directory: Path) -> list[dict[str, Any]]:
    """Every question across the golden batches, in filename order — `8B1` through `8B5`."""
    questions: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(directory / "*.json"))):
        questions.extend(json.loads(Path(path).read_text(encoding="utf-8"))["questions"])
    return questions


def read_additional(path: Path) -> list[tuple[str, str, str]]:
    """The manual-add rows: `pmid, title, text`, tab separated, first occurrence of an id wins."""
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        pmid = parts[0].strip()
        title = parts[1] if len(parts) > 1 else ""
        text = parts[2] if len(parts) > 2 else ""
        if pmid in seen:
            continue
        seen.add(pmid)
        rows.append((pmid, title, text))
    return rows


def build(
    *,
    source: Path,
    additional: Path,
    golden: Path,
    out: Path,
    expect_articles: int = EXPECTED_ARTICLES,
    expect_queries: int = EXPECTED_QUERIES,
) -> dict[str, Any]:
    """Write the corpus, queries and qrels, and assert the membership before returning a count."""
    questions = golden_questions(golden)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qrels").mkdir(exist_ok=True)

    written_queries = 0
    judgements = 0
    wanted: set[str] = set()
    with (out / "queries.jsonl").open("w", encoding="utf-8") as queries_out, \
            (out / "qrels" / "test.tsv").open("w", encoding="utf-8") as qrels_out:
        for question in questions:
            queries_out.write(json.dumps({
                "_id": str(question["id"]),
                "text": " ".join(str(question["body"]).split()),
            }) + "\n")
            written_queries += 1
            for url in question["documents"]:
                pmid = str(url).rstrip("/").split("/")[-1]
                qrels_out.write(f"{question['id']}\t{pmid}\t1\n")
                judgements += 1
                wanted.add(pmid)

    seen: set[str] = set()
    missing: set[str] = set(wanted)
    parsed = 0      # every well-formed article line in the dump
    articles = 0    # distinct pmids written, which is what the corpus holds
    started = time.perf_counter()
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as corpus_out:
        # The manual-add documents go in first and win any collision, which is the order the
        # reference preprocessing uses.
        for pmid, title, text in read_additional(additional):
            if pmid in seen:
                continue
            seen.add(pmid)
            missing.discard(pmid)
            corpus_out.write(json.dumps({"_id": pmid, "title": title, "text": text}) + "\n")

        header_seen = False
        for line in _iter_source_lines(source):
            stripped = line.strip()
            if not header_seen:
                if stripped.startswith('{"articles"'):
                    header_seen = True
                    continue
                raise ValueError(f"{source}: expected the '{{\"articles\":[' header, got {stripped[:60]!r}")
            try:
                article = _parse_article_line(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}: unparsable article line {stripped[:80]!r}: {exc}"
                ) from exc
            if article is None:
                continue
            pmid = str(article.get("pmid") or "").strip()
            if not pmid:
                continue
            parsed += 1
            if pmid in seen:
                continue
            title, abstract = _article_text(article)
            seen.add(pmid)
            missing.discard(pmid)
            corpus_out.write(json.dumps({"_id": pmid, "title": title, "text": abstract}) + "\n")
            articles += 1

    corpus_size = len(seen)
    # A judged document the corpus cannot supply is the more serious defect, so it is reported
    # first: it means the manual-add list is incomplete, whatever the article count says.
    if missing:
        raise AssertionError(
            f"{len(missing)} judged documents are absent from the corpus, e.g. "
            f"{sorted(missing)[:5]} — the manual-add list is incomplete"
        )
    if parsed != expect_articles:
        raise AssertionError(
            f"the dump parsed to {parsed} articles, expected {expect_articles} — short by "
            f"{expect_articles - parsed}. (BioASQ's own table says 14,913,939 for this set and the "
            f"dump here holds 14,913,938 lines.) Do not score a truncated corpus: the qrels are "
            f"fixed, so a short corpus scores a defect rather than a retriever "
            f"(docs/findings-bioasq.md)."
        )
    if written_queries != expect_queries:
        raise AssertionError(
            f"{written_queries} queries written, expected the member's {expect_queries}"
        )

    return {
        "parsed": parsed,
        "duplicates": parsed - articles,
        "articles": articles,
        "additional": corpus_size - articles,
        "documents": corpus_size,
        "queries": written_queries,
        "judgements": judgements,
        "distinct_judged": len(wanted),
        "seconds": round(time.perf_counter() - started, 1),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bioasq.py", description="Ingest the gated BioASQ download")
    parser.add_argument("--source", default=str(ROOT / "eval" / "bioasq-src" / "allMeSH_2020.json"))
    parser.add_argument("--additional",
                        default=str(ROOT / "eval" / "bioasq-src" / "additional_passages.tsv"))
    parser.add_argument("--golden",
                        default=str(ROOT / "eval" / "bioasq-src" / "Task8BGoldenEnriched"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "beir" / "bioasq"))
    parser.add_argument("--expect-articles", type=int, default=EXPECTED_ARTICLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build(
        source=Path(args.source),
        additional=Path(args.additional),
        golden=Path(args.golden),
        out=Path(args.out),
        expect_articles=args.expect_articles,
    )
    print(
        f"corpus {summary['documents']:,} documents "
        f"({summary['articles']:,} distinct pmids from {summary['parsed']:,} article lines, "
        f"{summary['duplicates']} duplicates dropped; {summary['additional']} manual-add), "
        f"{summary['queries']} queries, {summary['judgements']} judgements over "
        f"{summary['distinct_judged']} distinct documents, in {summary['seconds']}s"
    )
    print(f"wrote {Path(args.out) / 'corpus.jsonl'}, queries.jsonl and qrels/test.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
