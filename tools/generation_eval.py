#!/usr/bin/env python
"""Scaffold for the generation/placement workstream — no benchmark, no network, no model.

This module is deliberately inert. It exists so the *preparation* for four external
targets (MTEB, KILT, the TREC RAG track, Vectara HHEM) is recorded in a form that can be
imported, listed and checked without downloading anything or contacting an endpoint. It
does **not** run a benchmark, and importing it loads no model: the only Laya code it
touches is deterministic citation checking (:mod:`laya_rag.verify`), which is model-free
by construction.

The one rule the tool enforces in code is the labelling rule from
``docs/generation-placement.md``: a generation-side number can never be reported without
the model it belongs to. :func:`generation_side_metric` refuses an empty generator, so a
faithfulness or answer-quality score cannot be stripped of its ``engine + <model>``
attribution on the way to a table.

    .venv/bin/python tools/generation_eval.py --list
    .venv/bin/python tools/generation_eval.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from laya_rag.evidence import group_passages_by_document  # noqa: E402
from laya_rag.verify import verify_citations  # noqa: E402

#: The label every generation-side number must carry. Kept as a formatter so tables and
#: reports cannot spell the attribution four different ways.
LABEL_TEMPLATE = "engine + {generator}"


@dataclass
class GenerationResult:
    """One composed answer and everything known about it.

    ``generator`` is the model id that wrote ``answer``; it is mandatory-in-spirit even
    though the dataclass cannot make it so without breaking ``None`` defaults, so
    :func:`generation_side_metric` enforces it at the point a number is emitted.
    ``citations_ok`` and ``support`` are ``None`` until something has actually checked
    them — ``None`` means *not measured*, never *false* or *zero*.
    """

    query: str
    answer: str
    generator: str
    evidence: list[tuple[str, str]] = field(default_factory=list)
    citations_ok: bool | None = None
    support: float | None = None
    notes: str = ""

    @property
    def label(self) -> str:
        """How every number derived from this result must be attributed."""
        return LABEL_TEMPLATE.format(generator=self.generator or "<unknown>")


@dataclass(frozen=True)
class BenchmarkTarget:
    """A placement target and exactly what standing in it would require.

    ``requires`` is the tuple of external things the repo does not have; ``blocked_by``
    is the concrete reason nothing can be submitted today, or ``None`` for a target that
    is genuinely runnable end-to-end. ``adapter`` names the in-repo file that maps to the
    target's mechanics, or ``None`` when nothing does.
    """

    name: str
    scores: str
    requires: tuple[str, ...]
    adapter: str | None = None
    blocked_by: str | None = None
    notes: str = ""


#: The four targets named in the workstream. ``requires`` and ``blocked_by`` are the
#: point of the scaffold: they are what ``--check`` validates and what the doc expands.
TARGETS: tuple[BenchmarkTarget, ...] = (
    BenchmarkTarget(
        name="MTEB",
        scores=(
            "retrieval quality of a submitted encoder (nDCG@10 across the MTEB retrieval "
            "tasks). The leaderboard scores a MODEL, not the engine around it."
        ),
        requires=(
            "a single SentenceTransformer-compatible encoder packaged for submission",
            "the distilled encoder produced from the three-model dense ensemble",
            "a packaged pooling/projection configuration served by a loadable model",
            "the `mteb` harness plus network access to the MTEB task set",
            "a Hugging Face model repository and a model card",
        ),
        adapter=(
            "tools/beir.py (retrieval metrics and embedding caches); "
            "tools/projection_head.py (ridge distillation of Laya document embeddings "
            "onto a retrieval teacher, saved as a JSON affine map)"
        ),
        blocked_by=(
            "model-submission only: the projected encoder is an in-repo JSON affine map, "
            "not a packaged SentenceTransformer/Hub model, so there is no MTEB model "
            "submission yet"
        ),
        notes=(
            "MTEB's retrieval tasks are largely BEIR's, so tools/beir.py already produces "
            "comparable nDCG@10 locally, and tools/projection_head.py distils a map onto "
            "the teacher's space. What cannot be done today is a leaderboard entry, which "
            "requires submitting a packaged model rather than describing the engine."
        ),
    ),
    BenchmarkTarget(
        name="KILT",
        scores=(
            "entity linking + retrieval + seq2seq generation as one leaderboard score; the "
            "generation half means the downstream score belongs to the generator."
        ),
        requires=(
            "an entity linker (BLINK or equivalent) over the KILT Wikipedia snapshot",
            "a seq2seq generator, i.e. the compose endpoint",
            "the KILT benchmark download and network access",
            "a retrieval index over the KILT snapshot",
            "the submission format the KILT leaderboard expects (a full stack)",
        ),
        adapter=(
            "src/laya_rag/compose.py (OpenAI-compatible/Ollama generator endpoint); "
            "src/laya_rag/engine.py (retriever)"
        ),
        blocked_by=(
            "no entity linker in the repo, and the leaderboard expects a full stack; the "
            "score would be the generator's, not the engine's"
        ),
        notes=(
            "compose.py is the generator KILT would use, so any KILT number is "
            "`engine + <compose model>` and must never be shown as the engine's."
        ),
    ),
    BenchmarkTarget(
        name="TREC-RAG",
        scores=(
            "nugget recall/precision via the official AutoNuggetizer assessor, plus "
            "citation support assessment."
        ),
        requires=(
            "NIST access to the TREC RAG corpus",
            "gold nuggets from NIST assessors",
            "the official AutoNuggetizer scorer",
            "a retrieval run over the TREC corpus",
            "a generator to compose cited answers",
        ),
        adapter=(
            "src/laya_rag/verify.py — the deterministic structural match to the support "
            "half of the track (does the prose cite real spans)"
        ),
        blocked_by=(
            "NIST access to the corpus and gold nuggets is required, and scoring needs the "
            "official assessor; there is nothing to submit it to without that access"
        ),
        notes=(
            "verify.py checks citation mechanics, which is the same question as the support "
            "half. It is not a nugget-recall scorer and cannot replace AutoNuggetizer."
        ),
    ),
    BenchmarkTarget(
        name="HHEM",
        scores=(
            "ranks hallucination-detection MODELS by agreement with human labels; it does "
            "not score a RAG system."
        ),
        requires=(
            "a hallucination-detection model to submit",
            "the HHEM benchmark data and evaluation harness",
            "claim/premise pairs produced from generated prose",
        ),
        adapter=(
            "src/laya_rag/verify.py — citation mechanics only; explicitly NOT hallucination "
            "detection"
        ),
        blocked_by=(
            "nothing in the engine is a hallucination detector, so there is no honest "
            "leaderboard row; the defensible use is an optional faithfulness filter over "
            "composed prose"
        ),
        notes=(
            "The honest placement is a filter: run verify.py over composed answers and "
            "surface uncited claims, never a HHEM ranking. BENCHMARKS.md already records "
            "'no ranking'."
        ),
    ),
)


def describe_targets() -> list[dict[str, Any]]:
    """Return :data:`TARGETS` as plain, JSON-serialisable dictionaries.

    ``requires`` stays a tuple in the dict so callers can distinguish it from the other
    string fields; tuples serialise to JSON arrays, which is what the tests and any
    report writer want.
    """
    return [
        {
            "name": target.name,
            "scores": target.scores,
            "requires": target.requires,
            "adapter": target.adapter,
            "blocked_by": target.blocked_by,
            "notes": target.notes,
        }
        for target in TARGETS
    ]


def generation_side_metric(*, generator: str, metric: str, value: float) -> dict[str, Any]:
    """A generation-side number that always travels with the model it belongs to.

    Raises :class:`ValueError` when ``generator`` is empty or whitespace: a faithfulness
    or answer-quality number without its model is not a weaker result, it is a different
    (and unearned) claim about the engine.
    """
    name = (generator or "").strip()
    if not name:
        raise ValueError(
            "a generation-side metric must name its generator; every such number is "
            "'engine + <model>' and cannot be reported without the model"
        )
    metric_name = (metric or "").strip()
    if not metric_name:
        raise ValueError("a generation-side metric must name the metric it reports")
    return {
        "generator": name,
        "label": LABEL_TEMPLATE.format(generator=name),
        "metric": metric_name,
        "value": value,
        "side": "generation",
    }


def citation_report(answer: str, passages: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Run the deterministic citation check and label it as mechanics, not truth.

    ``passages`` are ``(path, text)`` pairs. They are grouped by document exactly as
    :func:`laya_rag.compose.build_messages` groups them, so a ``[n]`` in ``answer`` means
    the same evidence here as it did to the composer. The returned mapping is
    :func:`laya_rag.verify.verify_citations`'s report plus explicit provenance fields:
    ``generator`` is ``None`` (this check involves no model) and ``truth_claim`` is
    ``False``. It is importable and safe to call offline.
    """
    grouped = group_passages_by_document(passages)
    report = dict(verify_citations(answer, [text for _path, text in grouped]))
    report["generator"] = None
    report["truth_claim"] = False
    report["mechanics_only"] = True
    report["scored_by"] = "engine"
    return report


def _format_table(targets: Sequence[BenchmarkTarget]) -> str:
    """A readable target table for ``--list`` — no benchmark is run to produce it."""
    lines: list[str] = []
    for target in targets:
        lines.append(f"{target.name}")
        lines.append(f"  scores      : {target.scores}")
        lines.append(f"  requires    :")
        for item in target.requires:
            lines.append(f"    - {item}")
        lines.append(f"  adapter     : {target.adapter or '(none in repo)'}")
        lines.append(f"  blocked_by  : {target.blocked_by or '(runnable)'}")
        if target.notes:
            lines.append(f"  notes       : {target.notes}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _check() -> int:
    """Validate the scaffold and its target table. Returns 0 when healthy.

    Checks that the module imported, that every target carries a non-empty ``requires``
    and ``blocked_by``, and that names are unique. It runs no benchmark and opens no
    socket.
    """
    problems: list[str] = []
    if not TARGETS:
        problems.append("TARGETS is empty")

    seen: set[str] = set()
    for target in TARGETS:
        if not target.name.strip():
            problems.append("a target has an empty name")
        elif target.name in seen:
            problems.append(f"duplicate target name: {target.name}")
        seen.add(target.name)
        if not target.requires:
            problems.append(f"{target.name}: requires is empty")
        if not (target.blocked_by or "").strip():
            problems.append(f"{target.name}: blocked_by is empty (use None only if runnable)")
        if not (target.scores or "").strip():
            problems.append(f"{target.name}: scores is empty")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1

    print(f"ok: scaffold imports; {len(TARGETS)} targets with requires and blocked_by")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``--list`` (default) prints the targets; ``--check`` validates them.

    Neither mode downloads data, contacts an endpoint, or runs a benchmark.
    """
    parser = argparse.ArgumentParser(
        prog="generation_eval",
        description=(
            "Preparation scaffold for generation/placement targets. Lists what each "
            "external target requires and validates the scaffold. Runs no benchmark."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print the target table (default)")
    mode.add_argument("--check", action="store_true", help="validate the scaffold and targets")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check:
        return _check()
    print(_format_table(TARGETS))
    print()
    print(json.dumps(describe_targets(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
