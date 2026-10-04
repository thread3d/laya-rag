"""Verify the distilled encoder's export — ROADMAP 4.3's packaging half.

Phase 4.3 produced a distilled encoder and exported it as a SentenceTransformer directory so MTEB
could load it. The open question was whether that export is *faithful*: an exported package that
loads but silently differs from the checkpoint it was built from would make every downstream number
wrong in a way no metric would flag.

This loads both, encodes the same probe sentences, and compares — exactly, because the export should
be a copy rather than a re-derivation.

    .venv/bin/python tools/distill_verify.py --check

Artifacts live under `eval/`, which is regenerable and not committed, so a missing package is
reported as **absent**, never as a pass.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "beir" / "hf-cache"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402

PACKAGE = ROOT / "eval" / "distill-nfcorpus-package"
SOURCE = ROOT / "eval" / "distill-nfcorpus-model"
PROBES = (
    "Refunds are processed within five business days.",
    "The Lima office ships for free.",
    "A backup completed successfully and the service restarted.",
)


def compare_embeddings(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """How far an exported package's embeddings are from its source checkpoint's."""
    if reference.shape != candidate.shape:
        return {"comparable": False, "shape_reference": list(reference.shape),
                "shape_candidate": list(candidate.shape)}
    difference = np.abs(np.asarray(reference, np.float64) - np.asarray(candidate, np.float64))
    return {
        "comparable": True,
        "max_abs_diff": float(difference.max()) if difference.size else 0.0,
        "mean_abs_diff": float(difference.mean()) if difference.size else 0.0,
        "dimension": int(candidate.shape[-1]),
    }


def package_modules(model: Any) -> list[str]:
    return [module.__class__.__name__ for module in model]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="distill_verify.py")
    ap.add_argument("--package", default=str(PACKAGE))
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--tolerance", type=float, default=1e-6,
                    help="an export is a copy, so this is a float-noise allowance, not a budget")
    ap.add_argument("--check", action="store_true", help="exit 1 if the export is not faithful")
    ap.add_argument("--out", default=str(ROOT / "eval" / "distill-verify"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    package, source = Path(args.package), Path(args.source)
    missing = [str(p) for p in (package, source) if not p.is_dir()]
    if missing:
        print("distill-verify: ABSENT — not a pass", file=sys.stderr)
        for path in missing:
            print(f"  missing: {path}", file=sys.stderr)
        return 2

    from sentence_transformers import SentenceTransformer

    loaded_package = SentenceTransformer(str(package))
    loaded_source = SentenceTransformer(str(source))
    candidate = loaded_package.encode(list(PROBES), normalize_embeddings=True)
    reference = loaded_source.encode(list(PROBES), normalize_embeddings=True)
    comparison = compare_embeddings(reference, candidate)

    report = {
        "package": str(package),
        "source": str(source),
        "modules": package_modules(loaded_package),
        "probes": len(PROBES),
        **comparison,
        "tolerance": args.tolerance,
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), report)

    if not comparison["comparable"]:
        print(f"distill-verify: FAILED — shapes differ {comparison}", file=sys.stderr)
        return 1
    faithful = comparison["max_abs_diff"] <= args.tolerance
    print(f"package modules: {', '.join(report['modules'])}")
    print(f"dimension {comparison['dimension']}, {len(PROBES)} probes, "
          f"max abs diff {comparison['max_abs_diff']:.3g} "
          f"({'faithful' if faithful else 'DIFFERS'})")
    print("wrote", str(out) + ".json")
    if args.check and not faithful:
        print(f"distill-verify: FAILED — max abs diff exceeds {args.tolerance}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
