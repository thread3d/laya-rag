"""Download MuSiQue (train + validation) into the workspace for the replication study.

MuSiQue matters here for two reasons beyond being a second multi-hop dataset:

* it ships a real **train/validation split**, which MultiHop-RAG does not — its 2,556
  records are all the data there is, so any supervised result on it is cross-validation,
  not a train/test claim;
* its gold evidence is *supporting paragraphs* with sentence-level annotations, so the
  coverage framework (gold = the set of facts that must all be retrieved) applies directly.

The corpus is embedded in the records (each question carries its context paragraphs), so
one download is the whole dataset. Written as JSON so the adapter does not depend on the
`datasets` library at measurement time.

    .venv/bin/python tools/fetch_musique.py --split validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eval" / "benchmarks" / "musique"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="fetch_musique.py")
    ap.add_argument("--dataset", default="dgslibisey/MuSiQue")
    ap.add_argument("--split", default="validation", choices=["validation", "train", "both"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Keep the HF cache inside the workspace: the sandbox does not write to ~/.cache.
    os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "beir" / "hf-cache"))
    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = ["validation", "train"] if args.split == "both" else [args.split]
    for split in splits:
        started = time.perf_counter()
        print(f"loading {args.dataset} [{split}] ...", flush=True)
        dataset = load_dataset(args.dataset, split=split)
        rows = [dict(row) for row in dataset]
        path = out_dir / f"{split}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(
            f"{split}: {len(rows)} records, {path.stat().st_size / 2**20:.1f} MB "
            f"in {time.perf_counter() - started:.0f}s",
            flush=True,
        )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
