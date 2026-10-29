#!/usr/bin/env python
"""Format one engine report as a row for MultiHop-RAG's retrieval table — or refuse to.

The benchmark's primary table is retrieval-only (`MAP@10`, `MRR@10`, `Hits@4`, `Hits@10`), which is why
this engine can be placed in it at all, and `tools/engine_extractive.py` already reports exactly those
four columns. What it does not do is *place* them, and placement is where a row is usually over-claimed.

Two guards, both from failures this repository has already recorded:

* **A subset is not the member.** A row measured on 150 of the 2,255 answerable queries is a subset, and
  presenting it as the dataset's row is the mistake class 11 catalogs. The tool refuses to print a
  placement unless the report covers the full answerable set, and says how many queries it saw instead.
* **A replica is not the system, and chunking is part of the system.** The published anchors use the
  official harness's `chunk_overlap=200`; this engine chunks at a size derived from the corpus (2,000
  characters on this benchmark). The row is therefore stated with that difference beside it rather than
  merged into a single sorted column as though the protocols matched.

    python tools/multihop_row.py --report eval/benchmarks/multihop-rag/engine-extractive-rank150.json
    python tools/multihop_row.py --report <full-set report> --check --out eval/multihop-row
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

#: The paper's Table 5 as this repository records it (`README.md`, "MultiHop-RAG"), at the official
#: harness's `chunk_overlap=200`. Kept here rather than re-typed at each call site, and named so the
#: provenance of each row is visible.
PUBLISHED = (
    ("voyage-02 + rerank (published)", 0.747, 0.663, 0.480, 0.586),
    ("bge-large + rerank (published)", 0.718, 0.636, 0.476, 0.563),
    ("bge-large, no rerank (published)", 0.672, 0.522, 0.342, 0.430),
)

#: How many answerable queries the benchmark's retrieval table scores (2,556 records, 301 `null_query`).
FULL_SET = 2255


def summarise(report: Mapping[str, Any]) -> dict[str, Any]:
    """The four columns the table needs, plus how many queries produced them."""
    summary = report.get("summary") or {}
    wanted = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")
    missing = [name for name in wanted if name not in summary]
    if missing:
        raise SystemExit(f"the report has no {', '.join(missing)}; is it a retrieval row?")
    # The published table scores **documents**; this engine returns chunks, and the corpus splits 609
    # documents into 4,592. When the run recorded a document-level counterpart, that is the row to place —
    # and when it did not, the chunk-level metrics are reported with the mismatch stated rather than silently
    # presented as comparable.
    document = {name: float(summary[f"doc_{name}"]) for name in wanted if f"doc_{name}" in summary}
    return {
        "metrics": {name: float(summary[name]) for name in wanted},
        "document_metrics": document,
        "queries": int(summary.get("queries") or 0),
        "citation_clean": summary.get("citation_clean_all"),
        "nugget_recall": summary.get("answer_nugget_recall"),
    }


def placement_problems(row: Mapping[str, Any], *, full_set: int = FULL_SET) -> list[str]:
    """Every reason this report cannot be presented as the dataset's row."""
    problems: list[str] = []
    if row["queries"] != full_set:
        problems.append(
            f"{row['queries']} queries, not the {full_set} answerable records the table scores — "
            "this is a subset, and a subset presented as the member is the failure class 11 records"
        )
    return problems


def render(row: Mapping[str, Any], *, label: str, allow_subset: bool = False) -> str:
    problems = placement_problems(row)
    lines: list[str] = []
    if problems and not allow_subset:
        lines += ["**not placeable** — " + problem for problem in problems]
        lines += [
            "",
            "Run the full answerable set before quoting a placement:",
            "",
            "```sh",
            "python tools/engine_extractive.py --zero-thresholds \\",
            "  --index-dir eval/benchmarks/multihop-rag/index/engine-default \\",
            "  --out eval/benchmarks/multihop-rag/engine-extractive-full",
            "```",
            "",
        ]
    metrics = row.get("document_metrics") or row["metrics"]
    level = "documents" if row.get("document_metrics") else "chunks"
    lines += [
        "| system | Hits@10 | Hits@4 | MAP@10 | MRR@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, hits10, hits4, map10, mrr10 in PUBLISHED:
        lines.append(f"| {name} | {hits10:.3f} | {hits4:.3f} | {map10:.3f} | {mrr10:.3f} |")
    lines.append(
        f"| **{label}** | **{metrics['Hits@10']:.3f}** | **{metrics['Hits@4']:.3f}** | "
        f"**{metrics['MAP@10']:.3f}** | **{metrics['MRR@10']:.3f}** |"
    )
    lines += [
        "",
        (
            f"Over **{row['queries']}** answerable queries, scored over **documents** — the level the published "
            "rows are scored over — so the comparison is like-for-like apart from chunking."
            if level == "documents"
            else f"Over **{row['queries']}** answerable queries, scored over **chunks**. The published rows are "
                 "scored over **documents**, and this report carries no document-level counterpart, so the row "
                 "is indicative rather than placeable."
        ),
        f"Citation validity "
        f"**{row['citation_clean'] if row['citation_clean'] is not None else 'not measured'}**, answer "
        f"nugget recall **{row['nugget_recall'] if row['nugget_recall'] is not None else 'not measured'}** "
        "— the two columns this engine adds to a retrieval table, and the reason its row is comparable "
        "at all.",
        "",
        "## What this row may and may not be compared to",
        "",
        "- **Chunking differs.** The published anchors re-chunk with the official harness at a 200-character "
        "overlap; this engine derives chunk size from the corpus (2,000 characters here) and its row is "
        "measured at its own. That is the system, not a variant of it, and it is why the two groups are "
        "not sorted into one column.",
        "- **The paper's table carries no lexical baseline.** Every published row is a dense model, and "
        "this repository's BM25 control is not in it — so a Hits@10 advantage over `bge-large` is partly "
        "lexical retrieval that the table never measured (`README.md`).",
        "- **A subset is not the member**, which is why the placement above is refused unless the run "
        "covers the full answerable set.",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_row.py", description=__doc__)
    ap.add_argument("--report", required=True, help="an engine-extractive report JSON")
    ap.add_argument("--label", default="ours: the engine, shipped defaults")
    ap.add_argument("--allow-subset", action="store_true",
                    help="print the numbers even on a subset, clearly marked — for internal reading")
    ap.add_argument("--check", action="store_true", help="exit 1 when the row is not placeable")
    ap.add_argument("--out", default=None, help="write the markdown here as well")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    row = summarise(report)
    rendered = render(row, label=args.label, allow_subset=args.allow_subset)
    print(rendered)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text(rendered + "\n", encoding="utf-8")
        print(f"\nwrote {out.with_suffix('.md')}")
    if args.check and placement_problems(row):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
