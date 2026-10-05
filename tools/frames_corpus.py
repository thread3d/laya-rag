"""Fetch the FRAMES evidence corpus — ROADMAP 3.1.

FRAMES ships 824 questions whose evidence is English Wikipedia articles: 2,646 links covering 2,517
distinct articles. That is small enough to fetch in full rather than crawl, which is what makes this
benchmark tractable where the plan assumed otherwise.

The fetch is **resumable**: each batch is appended to the corpus file, so an interrupted run (or a
rate limit) costs only the batch in flight. Wikimedia requires a descriptive User-Agent and paces
anonymous callers, so the batches are sequential and small.

    .venv/bin/python tools/frames_corpus.py
    .venv/bin/python tools/frames_corpus.py --report eval/frames/corpus.jsonl --limit 200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSV = ROOT / "eval" / "frames" / "test.tsv"
DEFAULT_OUT = ROOT / "eval" / "frames" / "corpus.jsonl"
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "laya-rag-eval/1.0 (offline research harness; contact: local)"
#: `prop=revisions` accepts up to 50 titles per request, which is why the corpus is fetchable.
BATCH = 50
LINK_RE = re.compile(r"https?://en\.wikipedia\.org/wiki/([^\s'\"\]|]+)")


def article_titles(rows: Iterable[dict]) -> list[str]:
    """Distinct article titles across every link column, in a stable order.

    Titles arrive percent-encoded and sometimes carry trailing punctuation from the surrounding
    prose; both are normalised so two references to one article fetch once.
    """
    seen: dict[str, None] = {}
    for row in rows:
        for value in row.values():
            if not isinstance(value, str):
                continue
            for match in LINK_RE.findall(value):
                title = urllib.parse.unquote(match).replace("_", " ").strip().rstrip(".,;")
                if title:
                    seen.setdefault(title, None)
    return list(seen)


def read_rows(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def already_fetched(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["title"])
    return done


def resolve_titles(titles: Sequence[str], pause: float = 2.0, timeout: int = 30,
                   seed: Mapping[str, str] | None = None) -> dict[str, str]:
    """Map each title to the canonical article it resolves to, following normalisation and redirects.

    FRAMES' links are URLs — ``USA``, ``Harriet_Lane`` — while a corpus built from the API is keyed by
    canonical titles, so 12% of the gold set matched nothing until the redirects were followed. Both
    hops come back in one response: ``normalized`` (underscores, case, percent-encoding) and
    ``redirects`` (the article the title actually points at).

    ``seed`` holds titles already resolved, so a run interrupted by a rate limit resumes rather than
    restarting — and a 429 is retried with backoff rather than skipped, which is how four batches went
    missing the first time this ran at 1 s pacing.
    """
    mapping: dict[str, str] = dict(seed or {})
    pending = [title for title in titles if title not in mapping]
    for start in range(0, len(pending), BATCH):
        batch = pending[start : start + BATCH]
        block: dict | None = None
        for attempt in range(5):
            query = urllib.parse.urlencode({
                "action": "query", "format": "json", "redirects": 1, "titles": "|".join(batch),
            })
            request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    block = (json.loads(response.read().decode("utf-8")).get("query") or {})
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    wait = 15.0 * (2**attempt)
                    print(f"  rate limited (429); waiting {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                print(f"  resolve batch {start // BATCH}: HTTP {exc.code}", flush=True)
                break
            except Exception as exc:
                print(f"  resolve batch {start // BATCH}: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(5.0)
                break
        if block is None:
            continue
        normalised = {row["from"]: row["to"] for row in block.get("normalized", [])}
        redirected = {row["from"]: row["to"] for row in block.get("redirects", [])}
        for title in batch:
            step = normalised.get(title, title)
            mapping[title] = redirected.get(step, step)
        print(f"  resolved {min(start + BATCH, len(pending))}/{len(pending)}", flush=True)
        time.sleep(pause)
    return mapping


def fetch_batch(titles: Sequence[str], timeout: int = 30) -> dict[str, str]:
    """Article text for up to ``BATCH`` titles, following redirects.

    Not ``prop=extracts``: the TextExtracts extension returns only **one** full extract per request
    however ``exlimit`` is set — measured, three titles in and three pages back, exactly one with
    text — so a 20-title batch stores one article and the corpus would need a request per article.
    ``prop=revisions&rvprop=content`` honours the whole batch instead, at the cost of wikitext rather
    than prose. For retrieval that is acceptable, since the entities and numbers a question asks
    about are in the markup, and `strip_wikitext` removes the obvious scaffolding.
    """
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "revisions", "rvprop": "content",
        "rvslots": "main", "redirects": 1, "titles": "|".join(titles),
    })
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    out: dict[str, str] = {}
    for page in (payload.get("query") or {}).get("pages", {}).values():
        title = page.get("title")
        revisions = page.get("revisions") or []
        if not title or not revisions:
            continue
        slots = revisions[0].get("slots") or {}
        content = (slots.get("main") or {}).get("*")
        if content:
            out[title] = strip_wikitext(content)
    return out


#: Cheap scaffolding removal, not a parser: tables and refs are left in place, because for retrieval a
#: noisy document that contains the answer beats a clean one that does not.
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_TAG = re.compile(r"<[^>]{1,200}>")


def strip_wikitext(text: str) -> str:
    """Lose the markup that carries no terms; keep everything that might."""
    text = _TAG.sub(" ", text)
    for _ in range(4):                       # collapse the innermost templates a few levels deep
        text = _TEMPLATE.sub(" ", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="frames_corpus.py")
    ap.add_argument("--tsv", default=str(DEFAULT_TSV))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="stop after this many articles")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between batches; Wikimedia throttles bulk extraction")
    ap.add_argument("--resolve-map", default=None,
                    help="write a dataset-title -> canonical-title map here and exit")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tsv, out = Path(args.tsv), Path(args.out)
    if not tsv.is_file():
        raise SystemExit(f"{tsv} is missing; fetch it with the command in ROADMAP 3.1")
    titles = article_titles(read_rows(tsv))
    if args.resolve_map:
        target = Path(args.resolve_map)
        seed = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
        if seed:
            print(f"resuming from {len(seed)} resolved titles", flush=True)
        mapping = resolve_titles(titles, pause=args.pause, seed=seed)
        target = Path(args.resolve_map)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(mapping, indent=1, sort_keys=True), encoding="utf-8")
        print(f"resolved {len(mapping)} titles -> {target}")
        return 0
    done = already_fetched(out)
    pending = [title for title in titles if title not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(titles)} distinct articles, {len(done)} cached, {len(pending)} to fetch", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    fetched = 0
    with out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), BATCH):
            batch = pending[start : start + BATCH]
            pages = None
            for attempt in range(5):
                try:
                    pages = fetch_batch(batch)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 429:
                        # Wikimedia throttles bulk extraction hard, and answers a fast caller with
                        # 429 rather than partial data. Back off multiplicatively instead of
                        # hammering: a resumed run loses only the batch in flight.
                        wait = 15.0 * (2**attempt)
                        print(f"  rate limited (429); waiting {wait:.0f}s", flush=True)
                        time.sleep(wait)
                        continue
                    print(f"  batch {start // BATCH} failed: HTTP {exc.code}", flush=True)
                    break
                except Exception as exc:  # a transient failure costs one batch
                    print(f"  batch {start // BATCH} failed: {type(exc).__name__}: {exc}", flush=True)
                    time.sleep(5.0)
                    break
            if pages is None:
                continue
            for title, text in pages.items():
                handle.write(json.dumps({"title": title, "text": text}) + "\n")
                fetched += 1
            handle.flush()
            print(f"  {min(start + BATCH, len(pending))}/{len(pending)} requested, "
                  f"{fetched} stored", flush=True)
            time.sleep(args.pause)
    total = len(already_fetched(out))
    print(f"corpus now holds {total} articles at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
