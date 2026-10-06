"""Verify the hash-chained index audit log.

    make audit-verify AUDIT_LOG=eval/audit.jsonl
    .venv/bin/python tools/audit_verify.py --log eval/audit.jsonl --json

Exits 0 when the chain is intact (or the log does not exist — nothing to verify), 1 when a
record has been edited, reordered, removed or inserted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.audit import AuditLog  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="audit_verify.py")
    ap.add_argument("--log", default=None, help="path to the audit log")
    ap.add_argument("--json", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = (
        args.log
        or os.environ.get("LAYA_RAG_AUDIT_LOG")
        or str(ROOT / "eval" / "audit.jsonl")
    )
    if not Path(path).is_file():
        print(f"audit-verify: no log at {path} (nothing to verify)")
        return 0
    result = AuditLog(path).verify()
    if args.json:
        print(json.dumps({**result, "path": str(path)}, indent=2))
    elif result["ok"]:
        head = str(result["head"])[:16]
        print(f"audit-verify: {result['entries']} records, chain intact, head {head}…")
    else:
        print(f"audit-verify: BROKEN at record {result['first_bad']}: {result['reason']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
