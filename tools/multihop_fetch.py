#!/usr/bin/env python
"""Fetch and verify the MultiHop-RAG corpus, so the evidence protocol can be reproduced elsewhere.

`docs/findings-evidence-protocol.md` rests on two files this repository does not ship: the benchmark's
records and its 609-document corpus. Until now they arrived by hand, which makes the four-axis report
unreproducible — the exact test R4 set for itself ("if an external reviewer cannot reproduce it from a
fresh clone in under an hour, it is a tool and not a protocol").

This is the fetch, and it **asserts the membership before reporting**, the same discipline
`tools/bioasq.py` applies to a 26 GB dump and `tools/beir.py` to a BEIR mirror:

* 2,556 records, of which exactly **301 are `null_query`** — the negative class the abstention axis is
  scored on, and the number the whole R3 result depends on;
* 609 corpus documents;
* every record carrying a `query` and a `question_type`, every document a `title` and a `body`.

A truncated or re-versioned download fails here rather than producing a smaller number that looks like a
result. `--check` re-verifies an existing copy without touching the network.

    python tools/multihop_fetch.py            # download and verify
    python tools/multihop_fetch.py --check     # verify what is on disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
BASE_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"

#: The dataset's own shape, asserted rather than assumed. `RECORDS` and `NULL_QUERIES` are the two
#: numbers every downstream measurement is keyed on: 2,556 records, 301 of them unanswerable.
RECORDS = 2556
NULL_QUERIES = 301
DOCUMENTS = 609


def verify(records: Sequence[Any], corpus: Sequence[Any]) -> list[str]:
    """Every way the download can be wrong, as a list of complaints (empty means it is right)."""
    problems: list[str] = []
    if len(records) != RECORDS:
        problems.append(f"expected {RECORDS} records, found {len(records)}")
    if len(corpus) != DOCUMENTS:
        problems.append(f"expected {DOCUMENTS} documents, found {len(corpus)}")

    nulls = sum(
        1
        for record in records
        if isinstance(record, dict) and str(record.get("question_type") or "") == "null_query"
    )
    if nulls != NULL_QUERIES:
        problems.append(
            f"expected {NULL_QUERIES} null_query records (the abstention axis is scored on them), "
            f"found {nulls}"
        )

    for index, record in enumerate(records[:50]):
        if not isinstance(record, dict) or not str(record.get("query") or "").strip():
            problems.append(f"record {index} has no query")
            break
    for index, document in enumerate(corpus[:50]):
        if not isinstance(document, dict) or not str(document.get("body") or "").strip():
            problems.append(f"document {index} has no body")
            break
    return problems


def fetch(name: str, target: Path, *, timeout: int = 300) -> Path:
    url = f"{BASE_URL}/{name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"  downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=timeout) as response, open(partial, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    partial.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
    print(f"  {target.name}: {target.stat().st_size} bytes, sha256 {digest}…", flush=True)
    return target


def load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_fetch.py", description=__doc__)
    ap.add_argument("--out", default=str(ROOT_BENCH), help="directory to fetch into")
    ap.add_argument("--check", action="store_true",
                    help="verify an existing copy and exit; never touches the network")
    ap.add_argument("--force", action="store_true", help="re-download even if the files exist")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    directory = Path(args.out)
    records_path = directory / "MultiHopRAG.json"
    corpus_path = directory / "corpus.json"

    if args.check:
        for path in (records_path, corpus_path):
            if not path.is_file():
                raise SystemExit(f"{path} is missing; run `python tools/multihop_fetch.py` first")
        problems = verify(load(records_path), load(corpus_path))
        if problems:
            for problem in problems:
                print(f"  FAIL {problem}", file=sys.stderr)
            return 1
        print(f"ok: {RECORDS} records ({NULL_QUERIES} null_query), {DOCUMENTS} documents")
        return 0

    for path, name in ((records_path, "MultiHopRAG.json"), (corpus_path, "corpus.json")):
        if path.is_file() and not args.force:
            print(f"  {path.name} already present ({path.stat().st_size} bytes)", flush=True)
        else:
            fetch(name, path)

    problems = verify(load(records_path), load(corpus_path))
    if problems:
        for problem in problems:
            print(f"  FAIL {problem}", file=sys.stderr)
        raise SystemExit(
            "the download does not match the dataset this repository measured; "
            "delete the files and re-run, and re-check any number derived from them"
        )
    print(f"verified: {RECORDS} records ({NULL_QUERIES} null_query), {DOCUMENTS} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
