"""Print every resumable job's manifest, so a resumed session knows what is in flight.

A long job writes `<stem>.manifest.json` beside its checkpoint (see `tools/jobkit.py`). This
turns those into one line each: status, progress, fingerprint, age. The first command of a
resumed session is `make job-status` — a run left at `running` with 640/2255 units is work to
continue, not a result to quote.

    make job-status
    .venv/bin/python tools/job_status.py --root eval --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import jobkit  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="job_status.py")
    ap.add_argument("--root", default=str(ROOT / "eval"), help="tree to scan for manifests")
    ap.add_argument("--json", action="store_true", help="emit the manifests as JSON")
    return ap.parse_args(argv)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root.parent))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    rows = list(jobkit.iter_manifests(root))
    if args.json:
        print(json.dumps([{"path": str(p), **m} for p, m in rows], indent=2))
        return 0
    if not rows:
        print(f"job-status: no jobkit manifests under {root}")
        return 0
    print(f"job-status: {len(rows)} job(s) under {root}")
    for path, manifest in rows:
        total = manifest.get("total")
        done = manifest.get("completed", 0)
        progress = f"{done}/{total}" if total else str(done)
        shards = manifest.get("shard_count")
        shard = f" shard {manifest.get('shard_index')}/{shards}" if shards and shards > 1 else ""
        print(
            f"  {str(manifest.get('status', '?')):9s} {progress:>12s}{shard:>12s}  "
            f"fp={str(manifest.get('fingerprint', ''))[:12]}  "
            f"{manifest.get('updated_at', '')}  {_relative(path, root)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
