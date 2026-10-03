"""Command line interface for the Laya RAG engine."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import EngineConfig
from .engine import IndexNotBuiltError, RagEngine
from .model import ModelUnavailableError
from .verify import split_claims, verify_citations


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
            tenant=args.tenant,
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


def _read_answer(args: argparse.Namespace) -> str | None:
    """The answer text from the positional argument or ``--answer-file``."""
    if getattr(args, "answer", None):
        return args.answer
    if getattr(args, "answer_file", None):
        try:
            return Path(args.answer_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read {args.answer_file}: {exc}", file=sys.stderr)
            return None
    print("error: provide the answer as an argument or with --answer-file", file=sys.stderr)
    return None


def _cmd_verify(args: argparse.Namespace, config: EngineConfig) -> int:
    """Deterministic citation/quote checks, the same ones ``/v1/verify`` runs."""
    answer = _read_answer(args)
    if answer is None:
        return 2
    passages: list[str] = []
    for path in args.evidence or []:
        try:
            passages.append(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            return 2
    report = verify_citations(answer, passages, query=args.query or "")
    counts = report["counts"]
    human = "\n".join([
        report.get("summary", ""),
        f"clean: {'yes' if report['clean'] else 'no'}",
        f"claims {counts['claims']} (cited {counts['cited']}, uncited {counts['uncited']}, "
        f"invalid citations {counts['invalid_citations']})",
        str(report["content"].get("summary", "")),
        str(report["injection"].get("summary", "")),
    ])
    _emit(report, args.json, human)
    return 0


def _cmd_adjudicate(args: argparse.Namespace, config: EngineConfig) -> int:
    """Ask Laya whether the corpus supports each claim, as ``/v1/adjudicate`` does."""
    answer = _read_answer(args)
    if answer is None:
        return 2
    engine = RagEngine(config)
    try:
        report = engine.adjudicate_claims(split_claims(answer), tenant=args.tenant)
    except IndexNotBuiltError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    lines = [
        f"supported {report['supported']}  unsupported {report['unsupported']}  "
        f"not measured {report['not_measured']}  of {report['total']}",
        f"relevance threshold {report['relevance_threshold']}, "
        f"answerability threshold {report['answerability_threshold']}",
    ]
    for claim in report["claims"]:
        verdict = claim.get("verdict") or claim.get("supported")
        lines.append(f"  [{verdict}] {claim.get('text') or claim.get('claim')}")
    _emit(report, args.json, "\n".join(lines))
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


def _cmd_integrity(args: argparse.Namespace, config: EngineConfig) -> int:
    """Report index drift. Reads the index only; repairs nothing."""
    from .integrity import audit, summarize

    engine = RagEngine(config)
    if not engine.has_index():
        print("error: no index at {}".format(config.index_dir), file=sys.stderr)
        return 3
    chunks = engine.index.chunks
    report = audit(
        chunks,
        near_duplicates=bool(args.near_duplicates),
        threshold=float(args.threshold),
        max_chunks=int(args.max_chunks),
    )
    _emit(report, args.json, summarize(report))
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
    query.add_argument(
        "--tenant", default=None,
        help="caller identity for the trust manifest's per-tenant ACL, when one is configured",
    )
    query.add_argument("--json", action="store_true")
    query.set_defaults(func=_cmd_query)

    verify = sub.add_parser("verify", help="check an answer's citations against evidence")
    verify.add_argument("answer", nargs="?", default=None, help="the answer text")
    verify.add_argument("--answer-file", default=None, dest="answer_file")
    verify.add_argument("--query", default="", help="the question the answer responds to")
    verify.add_argument(
        "--evidence", action="append", default=[], metavar="FILE",
        help="a file whose text is one evidence passage; repeat for more (index [1] is the first)",
    )
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=_cmd_verify)

    adjudicate = sub.add_parser(
        "adjudicate", help="ask Laya whether the corpus supports each claim in an answer"
    )
    adjudicate.add_argument("answer", nargs="?", default=None, help="the answer text")
    adjudicate.add_argument("--answer-file", default=None, dest="answer_file")
    adjudicate.add_argument(
        "--tenant", default=None, help="caller identity for the trust manifest's per-tenant ACL"
    )
    adjudicate.add_argument("--json", action="store_true")
    adjudicate.set_defaults(func=_cmd_adjudicate)

    stats = sub.add_parser("stats", help="show configuration and corpus statistics")
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(func=_cmd_stats)

    integrity = sub.add_parser(
        "integrity", help="report index drift: deleted documents, duplicate and near-duplicate chunks"
    )
    integrity.add_argument(
        "--near-duplicates", action="store_true", dest="near_duplicates",
        help="also scan for near-duplicate chunks (MinHash LSH; bounded by --max-chunks)",
    )
    integrity.add_argument(
        "--threshold", type=float, default=0.85,
        help="jaccard at or above which two chunks are near-duplicates (default 0.85)",
    )
    integrity.add_argument(
        "--max-chunks", type=int, default=20000, dest="max_chunks",
        help="bound the near-duplicate scan; a truncated scan is reported as truncated",
    )
    integrity.add_argument("--json", action="store_true")
    integrity.set_defaults(func=_cmd_integrity)

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
