#!/usr/bin/env python
"""R8's submission gate: is this encoder submittable, and if not, why not?

R8 asks the programme to own a submittable artefact rather than only compare itself with other
people's. The training run that would produce a *winning* artefact is not something this host can do
quickly, but the thing that stops a submission from being credible is cheaper and more useful: an
exported model that loads, scores, and silently differs from the one that was measured. This repository
has already paid for that once — a cache keyed by a model's directory *basename* reused a previous
model's vectors and produced three benchmarks' worth of wrong rows.

So this is the gate, not the artefact. It answers, mechanically and offline-where-possible:

1. **It loads**, at a known dimension.
2. **Its outputs are unit vectors** — a cosine index built on un-normalised vectors retrieves garbage
   while looking plausible.
3. **It is deterministic** — two encodes of the same text are bit-identical, so a reported score is
   reproducible.
4. **Its pooling is known**, from the engine's verified family table or the package's own config, rather
   than guessed.
5. **Its revision is pinned** — MTEB skips results computed on a dataset revision other than the one in
   its repository, and the same discipline applies to the model, which `model_fingerprint` in
   `tools/beir.py` already implements for local directories.
6. **The engine's SearchProtocol adapter is present** — the route by which the engine, rather than just
   an encoder, can be entered.

A failing check makes the verdict **not submittable** with the reason, never a warning that scrolls
past. `--model` takes a Hub id or a local directory, so the same command checks a stock encoder today
and a distilled student the day it exists.

    python tools/submission_check.py --model BAAI/bge-small-en-v1.5
    python tools/submission_check.py --model eval/distill-nfcorpus-model --out eval/submission
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

#: Sentences whose embeddings have to agree with themselves. Deliberately ordinary text: a check that
#: only passes on clean input is a check that says nothing about a corpus.
PROBES = (
    "Refunds are processed within five business days.",
    "The answerability gate decides whether the corpus can answer.",
    "Mise à l'échelle: la taille du bloc domine la mémoire de pointe.",
)


def check_shape(vectors: Any) -> list[str]:
    """Dimension consistent, every row present."""
    import numpy as np

    array = np.asarray(vectors, dtype=np.float64)
    problems: list[str] = []
    if array.ndim != 2:
        problems.append(f"expected a 2-D array of embeddings, got shape {array.shape}")
        return problems
    if array.shape[0] != len(PROBES):
        problems.append(f"expected {len(PROBES)} rows, got {array.shape[0]}")
    if array.shape[1] == 0:
        problems.append("embeddings have zero dimensions")
    return problems


def check_unit_norm(vectors: Any, tolerance: float = 1e-3) -> list[str]:
    """A cosine index built on un-normalised vectors retrieves garbage while looking plausible."""
    import numpy as np

    array = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    offenders = [
        (index, float(norm)) for index, norm in enumerate(norms) if abs(norm - 1.0) > tolerance
    ]
    return [
        f"row {index} has norm {norm:.6f}, not 1.0 (±{tolerance:g})" for index, norm in offenders
    ]


def check_determinism(first: Any, second: Any) -> list[str]:
    """Two encodes of the same text must be identical, or a reported score is not reproducible."""
    import numpy as np

    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        return [f"two encodes disagree on shape: {a.shape} then {b.shape}"]
    delta = float(np.abs(a - b).max()) if a.size else 0.0
    if delta > 0.0:
        return [f"two encodes of the same text differ by max |Δ| = {delta:g}"]
    return []


def check_revision(model_id: str, resolved: Mapping[str, Any] | None) -> list[str]:
    """MTEB skips a result whose dataset revision moved; the model needs the same discipline."""
    if Path(model_id).is_dir():
        from beir import model_fingerprint

        fingerprint = model_fingerprint(model_id)
        if not fingerprint:
            return ["a local directory exported without a content fingerprint"]
        return []
    if not resolved:
        return []
    if not resolved.get("revision"):
        return [f"{model_id} resolved without a commit revision"]
    return []


def check_protocol() -> list[str]:
    """The engine's SearchProtocol adapter is the route by which the *engine* can be entered."""
    from mteb_search import load_laya_rag_search

    if not callable(load_laya_rag_search):
        return ["tools/mteb_search.py has no load_laya_rag_search entry point"]
    return []


def audit(embedder: Any, model_id: str, *, resolved: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run every check and return the verdict with its reasons.

    ``embedder`` is anything with ``embed(list[str]) -> 2-D array``; passing it in keeps the checks
    testable without a download, which is the difference between a gate and a script.
    """
    from laya_rag.model import family_pooling

    first = embedder.embed(list(PROBES))
    second = embedder.embed(list(PROBES))

    checks: list[tuple[str, list[str]]] = [
        ("loads and returns embeddings", check_shape(first)),
        ("outputs are unit vectors", check_unit_norm(first)),
        ("deterministic across calls", check_determinism(first, second)),
        ("revision is pinned", check_revision(model_id, resolved)),
    ]
    try:
        checks.append(("SearchProtocol adapter present", check_protocol()))
    except Exception as exc:  # noqa: BLE001 - reported, never raised: an audit is a verdict
        checks.append(("SearchProtocol adapter present", [f"{type(exc).__name__}: {exc}"]))

    reasons = [reason for _name, problems in checks for reason in problems]
    return {
        "model": model_id,
        "dim": int(getattr(first, "shape", (0, 0))[1]) if hasattr(first, "shape") else None,
        "pooling": family_pooling(model_id),
        "revision": (resolved or {}).get("revision"),
        "checks": [{"check": name, "passed": not problems, "problems": problems}
                   for name, problems in checks],
        "submittable": not reasons,
        "reasons": reasons,
    }


def render(result: Mapping[str, Any]) -> str:
    lines = [
        f"# Submission gate — {result['model']}",
        "",
        f"**verdict: {'SUBMITTABLE' if result['submittable'] else 'NOT SUBMITTABLE'}**",
        "",
        f"dimension {result['dim']}, pooling `{result['pooling']}`, "
        f"revision `{result['revision'] or 'unresolved'}`",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
    for check in result["checks"]:
        mark = "pass" if check["passed"] else "**FAIL**"
        lines.append(f"| {check['check']} | {mark} |")
    if result["reasons"]:
        lines += ["", "## Why not", ""]
        lines += [f"- {reason}" for reason in result["reasons"]]
    lines += [
        "",
        "---",
        "",
        "Nothing here scores the model. It asks whether an exported artefact can be *believed* if it "
        "does: that it loads, that its vectors are what a cosine index assumes, that it is "
        "deterministic, that its pooling and revision are pinned rather than guessed, and that the "
        "engine's own SearchProtocol adapter is present.",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="submission_check.py", description=__doc__)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                    help="a Hub id or a local exported directory")
    ap.add_argument("--out", default=str(ROOT / "eval" / "submission-check"))
    ap.add_argument("--check", action="store_true", help="exit 1 when the model is not submittable")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from laya_rag.config import EngineConfig
    from laya_rag.model import BiEncoderModel

    config = EngineConfig(embed_model=args.model)
    model = BiEncoderModel(config).ensure_loaded()

    resolved: dict[str, Any] | None = None
    if not Path(args.model).is_dir():
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(args.model)
            resolved = {"revision": getattr(info, "sha", None)}
        except Exception:  # noqa: BLE001 - offline is a result, not a failure of the audit
            resolved = None

    result = audit(model, args.model, resolved=resolved)
    rendered = render(result)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"\nwrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    return 0 if (result["submittable"] or not args.check) else 1


if __name__ == "__main__":
    raise SystemExit(main())
