"""Where the on-disk lexical build's memory actually goes — evidence for ROADMAP 4.1.

`OnDiskBM25.build` bounds each *block*, and the plan's token cap fixed the per-document
`term -> count` dicts that were the 2.4 GB peak on TREC-COVID. What remains unbounded is the state
that outlives every block: the vocabulary (`term -> id`), its document frequencies, and the document
id and length lists — all Python objects, all growing with the corpus. That is the wall that keeps
BioASQ (14.9M documents), FEVER (5.4M) and NQ (2.7M) unbuildable, and therefore what keeps the BEIR
claim at 6 of 18 datasets.

This measures peak build RSS against two axes *independently*: **documents** at fixed vocabulary,
and **vocabulary** at fixed documents. If the peak tracks documents it is the id/length lists; if it
tracks vocabulary it is the dictionary. Sizes that fix one axis and sweep the other are the only way
to tell those apart.

Each point runs in its own **subprocess**, because `ru_maxrss` is a high-water mark: measuring
several builds in one process would report the largest peak for every point.

    .venv/bin/python tools/ondisk_scale.py                 # the default sweep
    .venv/bin/python tools/ondisk_scale.py --docs 200000 --vocab 200000 --tokens 100

This is a *measurement*: it changes no index format. The bounded external merge that would fix the
wall is a separate change, and this is the target it has to beat.
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402


def synthetic_corpus(docs: int, vocab: int, tokens: int, seed: int = 13) -> Iterator[tuple[str, str]]:
    """`docs` documents of `tokens` tokens drawn from exactly `vocab` distinct terms.

    The vocabulary is a *parameter* rather than a byproduct: a Zipfian or natural corpus couples
    the two axes, and a sweep that cannot hold one fixed says nothing about which one costs.
    """
    rng = random.Random(seed)
    for index in range(docs):
        yield f"doc-{index}", " ".join(f"t{rng.randrange(vocab)}" for _ in range(tokens))


def peak_rss_mb() -> float:
    """Peak RSS in MiB; `ru_maxrss` is bytes on macOS and kilobytes on Linux."""
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return round((usage if sys.platform == "darwin" else usage * 1024) / (1024 * 1024), 1)


def build_once(docs: int, vocab: int, tokens: int, block_tokens: int = 1_000_000,
               keep: bool = False) -> dict:
    """Build one synthetic index and report its cost. Runs in the child process."""
    from laya_rag.ondisk import OnDiskBM25

    directory = Path(tempfile.mkdtemp(prefix="ondisk-scale-"))
    try:
        index = OnDiskBM25(directory)
        started = time.perf_counter()
        index.build(synthetic_corpus(docs, vocab, tokens), block_tokens=block_tokens)
        seconds = time.perf_counter() - started
        on_disk = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        return {
            "docs": docs,
            "vocab": vocab,
            "tokens": tokens,
            "terms": len(index.terms),
            "indexed_docs": index.n_docs,
            "build_s": round(seconds, 1),
            "index_mb": round(on_disk / 1e6, 1),
            "peak_rss_mb": peak_rss_mb(),
        }
    finally:
        if not keep:
            shutil.rmtree(directory, ignore_errors=True)


def run_point(docs: int, vocab: int, tokens: int, block_tokens: int) -> dict:
    """One point, in its own process, so the peak belongs to that build alone."""
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child",
         "--docs", str(docs), "--vocab", str(vocab), "--tokens", str(tokens),
         "--block-tokens", str(block_tokens)],
        capture_output=True, text=True, check=True,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"child produced no report: {completed.stdout[-400:]} {completed.stderr[-400:]}")


def default_sweep() -> list[tuple[int, int]]:
    """(docs, vocab) points: one axis held fixed while the other moves."""
    return [(25_000, 50_000), (50_000, 50_000), (100_000, 50_000),   # documents axis
            (50_000, 10_000), (50_000, 100_000), (50_000, 400_000)]  # vocabulary axis


def render_markdown(rows: Sequence[dict], tokens: int) -> str:
    lines = [
        "# On-disk lexical build: where the memory goes",
        "",
        f"- {tokens} tokens per document, one subprocess per point",
        "",
        "| axis | docs | vocab | terms | build s | index MB | peak RSS MB |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    base_docs = rows[0]["docs"] if rows else 0
    for row in rows:
        axis = "documents" if row["vocab"] == rows[0]["vocab"] else "vocabulary"
        lines.append(f"| {axis} | {row['docs']} | {row['vocab']} | {row['terms']} | "
                     f"{row['build_s']} | {row['index_mb']} | **{row['peak_rss_mb']}** |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="ondisk_scale.py")
    ap.add_argument("--child", action="store_true", help="run one point and print it")
    ap.add_argument("--docs", type=int, default=50_000)
    ap.add_argument("--vocab", type=int, default=50_000)
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--block-tokens", type=int, default=1_000_000)
    ap.add_argument("--sweep", action="store_true", help="run the default two-axis sweep")
    ap.add_argument("--out", default=str(ROOT / "eval" / "ondisk-scale"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.child:
        print(json.dumps(build_once(args.docs, args.vocab, args.tokens, args.block_tokens)))
        return 0

    points = default_sweep() if args.sweep else [(args.docs, args.vocab)]
    rows: list[dict] = []
    for docs, vocab in points:
        row = run_point(docs, vocab, args.tokens, args.block_tokens)
        rows.append(row)
        print(f"  docs={docs} vocab={vocab} terms={row['terms']} "
              f"peak RSS {row['peak_rss_mb']} MB in {row['build_s']}s", flush=True)

    report = {"settings": vars(args), "rows": rows}
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), report)
    markdown = render_markdown(rows, args.tokens)
    evalkit.write_text(Path(str(out) + ".md"), markdown)
    print()
    print(markdown)
    print("wrote", str(out) + ".json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
