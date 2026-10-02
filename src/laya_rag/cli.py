"""Command line interface for the Laya RAG engine."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .config import EngineConfig
from .engine import IndexNotBuiltError, RagEngine
from .model import ModelUnavailableError


def _emit(payload: Any, as_json: bool, human: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(human)


def _cmd_index(args: argparse.Namespace, config: EngineConfig) -> int:
    engine = RagEngine(config)
    extensions = None
    if args.ext:
        extensions = [e if e.startswith(".") else f".{e}" for e in args.ext.split(",")]
    try:
        report = engine.index_paths(args.paths, rebuild=args.rebuild, prune=args.prune, **(
            {"extensions": extensions} if extensions else {}
        ))
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    payload = report.as_dict()
    lines = [
        f"documents : {report.documents}",
        f"chunks    : {report.chunks} (removed {report.removed_chunks})",
        f"in index  : {report.total_chunks} chunks from {report.paths} files",
        f"embed     : {report.embed_seconds:.2f}s",
        f"total     : {report.total_seconds:.2f}s",
    ]
    if report.skipped:
        lines.append(f"skipped   : {len(report.skipped)}")
        for path, reason in report.skipped[:10]:
            lines.append(f"  - {path}: {reason}")
        if len(report.skipped) > 10:
            lines.append(f"  ... and {len(report.skipped) - 10} more")
    _emit(payload, args.json, "\n".join(lines))
    return 0 if report.documents else 1


def _cmd_query(args: argparse.Namespace, config: EngineConfig) -> int:
    engine = RagEngine(config)
    try:
        result = engine.query(
            args.query,
            top_k=args.top_k,
            retrieve_k=args.retrieve_k,
            relevance_threshold=args.relevance_threshold,
        )
    except IndexNotBuiltError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    payload = result.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0

    verdict = "answerable" if result.answerable else "NOT answerable from the indexed corpus"
    if result.answerability is not None:
        verdict = f"{verdict} (Laya answerability {result.answerability:.3f})"
    print(f"query       : {result.query}")
    print(f"verdict     : {verdict}")
    print(f"considered  : {result.considered} dense candidates -> {result.kept} kept "
          f"({result.dropped_below_threshold} below threshold)")
    print(f"timings     : retrieve {result.retrieval_seconds:.2f}s, rerank "
          f"{result.rerank_seconds:.2f}s, total {result.total_seconds:.2f}s")
    if result.note:
        print(f"note        : {result.note}")
    for i, passage in enumerate(result.passages, start=1):
        channels = "+".join(passage.get("channels") or []) or "unknown"
        print(f"\n[{i}] relevance {passage['relevance']:.3f}  "
              f"recall {passage['recall_score']:.4f} ({channels})")
        print(f"    {passage['path']}")
        snippet = passage["text"]
        if not args.full:
            snippet = snippet[:400].rstrip() + ("…" if len(snippet) > 400 else "")
        for line in snippet.splitlines():
            print(f"    {line}")
    return 0


def _cmd_stats(args: argparse.Namespace, config: EngineConfig) -> int:
    engine = RagEngine(config)
    payload = engine.stats()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0
    index = payload.get("index", {})
    model = payload.get("model", {})
    print(f"index dir   : {payload['index_dir']}")
    print(f"index       : {index.get('chunks', 0)} chunks, {index.get('documents', 0)} documents, "
          f"{index.get('paths', 0)} files, dim {index.get('dim', 0)}")
    print(f"checkpoint  : {config.checkpoint} on {config.device} (loaded: {model.get('loaded')})")
    embedder = payload.get("embedder", {})
    print(f"embedder    : {embedder.get('embedder') or config.checkpoint} "
          f"({embedder.get('kind', 'laya encoder')})")
    print(f"recall      : dense weight {config.resolved_dense_weight}, "
          f"lexical weight {config.lexical_weight}, order {config.resolved_rerank_order}")
    print(f"thresholds  : relevance {config.relevance_threshold}, "
          f"answerability {config.answerability_threshold}")
    print(f"funnel      : retrieve {config.retrieve_k} -> rerank {config.rerank_k}")
    return 0


def _cmd_reset(args: argparse.Namespace, config: EngineConfig) -> int:
    RagEngine(config).reset()
    _emit({"status": "ok", "index_dir": str(config.index_dir)}, args.json,
          f"removed index at {config.index_dir}")
    return 0


def _cmd_doctor(args: argparse.Namespace, config: EngineConfig) -> int:
    """Report whether the checkpoint can actually be loaded."""
    engine = RagEngine(config)
    problems: list[str] = []
    try:
        engine.model.ensure_loaded()
        dim = engine.model.dim
        model_state = f"loaded, embedding dim {dim}"
    except ModelUnavailableError as exc:
        model_state = "UNAVAILABLE"
        problems.append(str(exc))
    payload = {
        "checkpoint": config.checkpoint,
        "device": config.device,
        "models_dir": str(config.models_dir),
        "model": model_state,
        "index_dir": str(config.index_dir),
        "index_present": engine.has_index(),
        "problems": problems,
    }
    _emit(payload, args.json, "\n".join([
        f"models dir  : {config.models_dir}",
        f"checkpoint  : {config.checkpoint} on {config.device}",
        f"model       : {model_state}",
        f"index       : {'present' if engine.has_index() else 'absent'} at {config.index_dir}",
        *(f"problem     : {p}" for p in problems),
    ]))
    return 0 if not problems else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-rag",
        description="Retrieval-augmented generation over the Laya decision engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="index files or directories")
    index.add_argument("paths", nargs="+")
    index.add_argument("--rebuild", action="store_true", help="discard the existing index first")
    index.add_argument(
        "--no-prune",
        action="store_false",
        dest="prune",
        default=True,
        help="keep chunks of files deleted since the last run (default: prune them)",
    )
    index.add_argument("--ext", help="comma-separated extension allowlist, e.g. .md,.py")
    index.add_argument("--json", action="store_true")
    index.set_defaults(func=_cmd_index)

    query = sub.add_parser("query", help="find evidence for a question")
    query.add_argument("query")
    query.add_argument("--top-k", type=int, default=None, dest="top_k")
    query.add_argument("--retrieve-k", type=int, default=None, dest="retrieve_k")
    query.add_argument("--relevance-threshold", type=float, default=None, dest="relevance_threshold")
    query.add_argument("--full", action="store_true", help="print whole passages, not a 400-char preview")
    query.add_argument("--json", action="store_true")
    query.set_defaults(func=_cmd_query)

    stats = sub.add_parser("stats", help="show configuration and corpus statistics")
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(func=_cmd_stats)

    reset = sub.add_parser("reset", help="delete the index")
    reset.add_argument("--json", action="store_true")
    reset.set_defaults(func=_cmd_reset)

    doctor = sub.add_parser("doctor", help="check that the checkpoint loads")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = EngineConfig()
    return int(args.func(args, config))


if __name__ == "__main__":
    sys.exit(main())
