"""An MTEB v2 ``SearchProtocol`` adapter for the engine.

MTEB's standing objection to this project was that "MTEB scores a model, not an engine, so this
system cannot be entered as-is". MTEB v2 defines a `SearchProtocol` alongside the encoder and
cross-encoder protocols, and that objection expires: an engine that exposes ``index`` and ``search``
is first-class representable, and the thing being scored is the retrieval *system* — chunking, fusion,
the second stage and the gate — rather than one embedding model.

This module is that adapter. It is deliberately dependency-light: the engine and the corpus plumbing
import nothing from MTEB, and only :func:`build_model_meta` touches it, so the adapter can be tested
without MTEB installed.

    pip install -e '.[mteb]'
    MTEB_CACHE=/tmp/mteb-cache python -c "
        import mteb
        from tools.mteb_search import load_laya_rag_search
        model = load_laya_rag_search('laya-rag/engine-multilingual')
        task = mteb.get_task('SciFact')
        mteb.evaluate(model, task)"

The corpus is written once, by position, to a temporary directory — the same convention
``tools/engine_beir.py`` uses — and retrieved passages are mapped back to corpus ids through that
order.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

#: The MTEB model name. MTEB requires `organization/name`.
MODEL_NAME = "laya-rag/engine-multilingual"

#: Provenance for the model metadata, kept as literals so the adapter imports without MTEB.
MODEL_FACTS: dict[str, Any] = {
    "n_parameters": 322_000_000,      # Laya's decision head; the embedder is the same checkpoint
    "memory_usage_mb": 7168.0,        # 7.0 GiB measured peak RSS (docs/findings-cost-axis.md)
    "max_tokens": 8192,               # the multilingual checkpoint's reading window
    "license": "apache-2.0",          # vendor/laya/LICENSE
    "languages": ["eng-Latn"],        # honest: the lexical instrumentation is English-only
    "framework": ["PyTorch"],
    "model_type": ["hybrid"],         # lexical + dense fused, then a second stage and a gate
    "reference": "https://github.com/laya-ai/laya",
}


# ── corpus plumbing (no MTEB import) ─────────────────────────────────────

def _column_names(obj: Any) -> list[str]:
    names = getattr(obj, "column_names", None)
    if names is not None:
        return list(names)
    if isinstance(obj, Mapping):
        return list(obj.keys())
    raise TypeError(f"corpus/query object {type(obj).__name__!r} has no columns")


def corpus_rows(corpus: Any) -> Iterator[tuple[str, str]]:
    """Yield ``(id, text)`` for a corpus; a ``title`` column is prepended when present."""
    columns = _column_names(corpus)
    ids = list(corpus["id"])
    texts = list(corpus["text"])
    titles = list(corpus["title"]) if "title" in columns else [None] * len(ids)
    for doc_id, text, title in zip(ids, texts, titles):
        body = str(text)
        yield str(doc_id), f"{title}\n\n{body}" if title else body


def progress_line(done: int, total: int, elapsed: float) -> str:
    """One line saying how far a search pass has got, and how long it has left.

    MTEB shows a progress bar for its own *scoring* phase, not for this one, so the adapter's
    silence made a working run indistinguishable from a hung one: `findings-mteb-search-protocol.md`
    recorded "no result, no error, no progress output, for 30+ minutes" and read it as the harness
    failing to fork cleanly. A rate and an estimate are what make that decidable from the log alone.
    """
    total = max(int(total), 0)
    done = max(int(done), 0)
    rate = (elapsed / done) if done else 0.0
    remaining = rate * max(total - done, 0)
    return (
        f"  search: {done}/{total} queries in {elapsed:.0f}s "
        f"({rate:.2f}s/query, about {remaining / 60:.0f} min left)"
    )


def query_rows(queries: Any) -> Iterator[tuple[str, str]]:
    """Yield ``(id, text)`` for a query set."""
    ids = list(queries["id"])
    texts = list(queries["text"])
    for query_id, text in zip(ids, texts):
        yield str(query_id), str(text)


def write_corpus(rows: Iterable[tuple[str, str]], directory: Path) -> list[str]:
    """One ``.txt`` per document, named by position; returns the id order.

    The name is the position, not the corpus id: BEIR ids are not safe filenames, and the
    position is what maps a returned passage back to its document.
    """
    directory.mkdir(parents=True, exist_ok=True)
    doc_ids: list[str] = []
    for index, (doc_id, text) in enumerate(rows):
        (directory / f"{index:06d}.txt").write_text(text, encoding="utf-8")
        doc_ids.append(doc_id)
    return doc_ids


def _passage_score(passage: Mapping[str, Any]) -> float:
    """The engine's ranking score for a passage: calibrated relevance, else recall."""
    for key in ("relevance", "recall_score"):
        value = passage.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _default_engine_factory(config: Any = None) -> Any:
    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    return RagEngine(config or EngineConfig())


# ── the adapter ──────────────────────────────────────────────────────────

class LayaRagSearch:
    """The engine behind MTEB's ``SearchProtocol``.

    ``index`` writes the corpus and builds the engine's index; ``search`` runs the engine's own
    query path (recall, fusion, second stage, gate) and returns ``{query_id: {doc_id: score}}``.
    """

    def __init__(
        self,
        *,
        config: Any = None,
        engine: Any = None,
        engine_factory: Callable[[Any], Any] | None = None,
        workdir: Path | str | None = None,
        retrieve_k: int | None = None,
        model_meta: Any = None,
        device: str | None = None,
        progress_every: int = 25,
        limit_queries: int | None = None,
        first_stage_only: bool = False,
    ) -> None:
        if device is not None and config is not None:
            config.device = device
        elif device is not None:
            from laya_rag.config import EngineConfig

            config = EngineConfig(device=device)
        # MTEB scores the retrieval *system*, and the engine's second stage is a 322M-parameter
        # decision on every candidate — 90+ minutes for one SciFact task when it runs. With this
        # on, the ranking MTEB sees is the first stage's own, which is what the benchmark is for;
        # the decision model is measured separately by the answerability and trust reports. A
        # submission must say which configuration produced it, and the run prints it.
        #: Recorded rather than only applied: `load_laya_rag_search` builds the adapter with no
        #: config, so the flag has to survive until an `EngineConfig` exists.
        self.first_stage_only = bool(first_stage_only)
        if self.first_stage_only and config is not None:
            config.second_stage = False
        self.config = config
        self._engine = engine
        self._engine_factory = engine_factory or _default_engine_factory
        self._workdir = Path(workdir) if workdir is not None else None
        self._temp: tempfile.TemporaryDirectory | None = None
        self._doc_ids: list[str] = []
        #: Print a progress line every N queries; 0 silences it. See :func:`progress_line`.
        self.progress_every = max(int(progress_every or 0), 0)
        #: A smoke cap on the queries scored, for measuring the rate before committing to a full
        #: task. It is *not* a result: a capped pass is a subset and must never be quoted.
        self.limit_queries = limit_queries
        # A plain attribute, not a read-only property: `ModelMeta.load_model` assigns this onto
        # the object it just constructed.
        self.mteb_model_meta = model_meta
        #: Over-fetch floor for the pool. MTEB usually asks for top-100; the engine's default
        #: `RETRIEVE_K` is smaller, so the adapter widens it to at least `top_k`.
        self.retrieve_k = retrieve_k

    # -- lifecycle -------------------------------------------------------

    def _directory(self) -> Path:
        if self._workdir is not None:
            self._workdir.mkdir(parents=True, exist_ok=True)
            return self._workdir
        if self._temp is None:
            self._temp = tempfile.TemporaryDirectory(prefix="laya-mteb-")
        return Path(self._temp.name)

    def _engine_for(self) -> Any:
        if self._engine is None:
            if self.first_stage_only:
                from laya_rag.config import EngineConfig

                # The loader path reaches here with no config at all, so the flag is applied
                # where a config is guaranteed to exist.
                self.config = self.config or EngineConfig()
                self.config.second_stage = False
            self._engine = self._engine_factory(self.config)
        return self._engine

    def close(self) -> None:
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None

    def __enter__(self) -> "LayaRagSearch":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- SearchProtocol --------------------------------------------------

    def index(
        self,
        corpus: Any,
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        encode_kwargs: Mapping[str, Any] | None = None,
        num_proc: int | None = None,
    ) -> None:
        directory = self._directory()
        self._doc_ids = write_corpus(corpus_rows(corpus), directory)
        started = time.perf_counter()
        last = {"at": 0.0}

        def progress(event: Mapping[str, Any]) -> None:
            """Say what indexing is doing, at most once a second.

            The search phase got this and the index phase did not, so a run that was embedding 5,183
            documents looked exactly like one that had hung — the same defect, one stage earlier.
            """
            now = time.perf_counter()
            if now - last["at"] < 1.0:
                return
            last["at"] = now
            print(
                f"  index: {event.get('stage', '?')} documents="
                f"{event.get('documents', 0)}/{event.get('documents_total', 0)} "
                f"chunks={event.get('chunks', 0)} embedded={event.get('embedded', 0)} "
                f"({now - started:.0f}s)",
                flush=True,
            )

        self._engine_for().index_paths([directory], rebuild=True, progress=progress)
        print(f"  index: done in {time.perf_counter() - started:.0f}s", flush=True)

    def search(
        self,
        queries: Any,
        *,
        task_metadata: Any = None,
        hf_split: str = "test",
        hf_subset: str = "default",
        top_k: int = 100,
        encode_kwargs: Mapping[str, Any] | None = None,
        top_ranked: Mapping[str, Sequence[str]] | None = None,
        num_proc: int | None = None,
    ) -> dict[str, dict[str, float]]:
        engine = self._engine_for()
        retrieve_k = max(int(top_k), int(self.retrieve_k or 0))
        results: dict[str, dict[str, float]] = {}
        rows = list(query_rows(queries))
        if self.limit_queries:
            rows = rows[: self.limit_queries]
        started = time.perf_counter()
        reported = False
        for position, (query_id, text) in enumerate(rows, start=1):
            if self.progress_every and position % self.progress_every == 0:
                print(progress_line(position, len(rows), time.perf_counter() - started), flush=True)
                reported = True
            allowed = set(top_ranked[query_id]) if top_ranked and query_id in top_ranked else None
            # `relevance_threshold=0.0` returns the ranking rather than the gate's kept set:
            # a retrieval benchmark scores order, and the gate is measured separately.
            result = engine.query(
                text, top_k=top_k, retrieve_k=retrieve_k, relevance_threshold=0.0
            )
            scores: dict[str, float] = {}
            for passage in result.passages:
                doc_id = self._doc_id(passage)
                if doc_id is None or (allowed is not None and doc_id not in allowed):
                    continue
                score = _passage_score(passage)
                if doc_id not in scores or score > scores[doc_id]:
                    scores[doc_id] = score
            results[query_id] = scores
        elapsed = time.perf_counter() - started
        # A *short* call reports too, when it was slow. `MTEB`'s evaluator carries a `qid`, so it
        # may drive `search` once per query rather than once per query set — and a line every 25
        # queries then never prints, which is exactly how a working run looked hung for three
        # rounds of this investigation. A call that took real time says so whether it had 1 query
        # or 1,000, and a fast one stays quiet so a per-query driver does not print a flood.
        if self.progress_every and not reported and elapsed >= 2.0:
            print(progress_line(len(rows), len(rows), elapsed), flush=True)
        return results

    def _doc_id(self, passage: Mapping[str, Any]) -> str | None:
        stem = Path(str(passage.get("path") or "")).stem
        if not stem.isdigit():
            return None
        index = int(stem)
        if not (0 <= index < len(self._doc_ids)):
            return None
        return self._doc_ids[index]


# ── MTEB model metadata ──────────────────────────────────────────────────

def load_laya_rag_search(name: str = MODEL_NAME, revision: str | None = None, **kwargs: Any) -> LayaRagSearch:
    """The ``ModelMeta.loader``: build an adapter, ignoring arguments it does not use.

    MTEB forwards ``device`` and any ``experiment_kwargs``; only the ones the adapter understands
    are kept, so an unknown kwarg cannot make a submission fail on a typo.
    """
    accepted = {k: v for k, v in kwargs.items()
                if k in {"config", "engine", "engine_factory", "workdir", "retrieve_k",
                         "model_meta", "device"}}
    return LayaRagSearch(**accepted)


def build_model_meta(name: str = MODEL_NAME) -> Any:
    """Build the ``mteb.models.model_meta.ModelMeta`` for this engine.

    Requires MTEB (`pip install -e '.[mteb]'`). Kept separate from the adapter so the engine-side
    tests run without the dependency.
    """
    try:
        from mteb.models.model_meta import ModelMeta, ScoringFunction
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "MTEB is not installed; install the extra with `pip install -e '.[mteb]'`"
        ) from exc

    return ModelMeta(
        loader=load_laya_rag_search,
        loader_kwargs={},
        name=name,
        revision=None,
        release_date=None,
        languages=list(MODEL_FACTS["languages"]),
        n_parameters=MODEL_FACTS["n_parameters"],
        memory_usage_mb=MODEL_FACTS["memory_usage_mb"],
        max_tokens=MODEL_FACTS["max_tokens"],
        embed_dim=None,
        license=MODEL_FACTS["license"],
        open_weights=True,
        public_training_code=None,
        public_training_data=False,
        framework=list(MODEL_FACTS["framework"]),
        reference=MODEL_FACTS["reference"],
        similarity_fn_name=ScoringFunction.CUSTOM,
        use_instructions=False,
        training_datasets=None,
        model_type=list(MODEL_FACTS["model_type"]),
    )


def _takes_heavy_lock(args: argparse.Namespace) -> bool:
    """Whether this invocation must hold the one-heavy-job lock.

    A baseline run evaluates a stock MTEB model: it builds no engine index and holds nothing
    another job needs, so it is excluded. Every other invocation builds an index and is heavy.
    """
    return not getattr(args, "baseline_model", None)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="mteb_search.py")
    ap.add_argument("--model-name", default=MODEL_NAME,
                    help="the MTEB model name this run is filed under; a run that scores a "
                         "different encoder must change it (see --embed-model)")
    ap.add_argument("--task", default="SciFact", help="an MTEB task name to evaluate")
    ap.add_argument("--cache", default=None, help="MTEB_CACHE directory (must be writable)")
    ap.add_argument("--stack-dump-every", type=int, default=0, metavar="SECONDS",
                    help="dump every thread's stack every N seconds, so a slow run is "
                         "distinguishable from a hung one without a process inspector")
    ap.add_argument("--embed-model", default=None,
                    help="encoder for the engine's dense channel (sets LAYA_RAG_EMBED_MODEL); "
                         "--model-name must be given with it, so a row cannot be filed under a "
                         "name that does not describe the encoder that produced it")
    ap.add_argument("--baseline-model", default=None,
                    help="evaluate a stock MTEB model on the same task **instead of** this engine, "
                         "so a score can be read against something under the same protocol. "
                         "Comparing an MTEB row to BEIR-convention numbers is a cross-protocol "
                         "claim; this is what makes the comparison legitimate")
    ap.add_argument("--first-stage-only", action="store_true",
                    help="score the first stage's own order, with no decision model in the path "
                         "(the second stage costs 90+ minutes per SciFact task)")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print a progress line every N queries; a working run and a hung one "
                         "used to look identical in the log")
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="score only the first N queries — a SMOKE, never a result")
    ap.add_argument("--num-proc", type=int, default=1,
                    help="dataloading processes; 1 keeps the engine single-process")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    # A benchmark run is heavy — an index build plus hundreds of queries, and MTEB forks dozens of
    # helper processes of its own — but this tool never took the lock, so `make heavy-status` could
    # not see it and two runs would contend. That is exactly how this session accumulated 67 and
    # then 101 `mteb_search.py` processes on a shared host, which made runs slow enough to look
    # pathologically stalled. A baseline run is short and deliberately excluded: it is a stock model
    # with no engine index to build, so it holds nothing another job needs.
    if _takes_heavy_lock(args):
        from heavylock import heavy_job_guard

        heavy_job_guard("mteb_search.py")
    if args.stack_dump_every:
        import faulthandler

        # The instrument that produced this adapter's corrected diagnosis: every N seconds the
        # process prints the stack of every thread, so "slow" and "hung" stop looking identical.
        faulthandler.dump_traceback_later(args.stack_dump_every, repeat=True)
    if args.cache:
        os.environ.setdefault("MTEB_CACHE", args.cache)
    try:
        import mteb
    except ImportError as exc:  # pragma: no cover
        print(f"mteb_search: MTEB is not installed ({exc}); pip install -e '.[mteb]'")
        return 2
    if args.baseline_model:
        # Not this engine: a stock MTEB model scored by the same harness on the same task. That is
        # the only like-for-like baseline an MTEB row can have — the BEIR-convention numbers this
        # repository quotes (BM25 0.665, `bge-small` 0.72 on SciFact) come from a different corpus
        # text and a different evaluator, and reading across them is the cross-protocol claim
        # `BENCHMARKS.md` forbids.
        import mteb

        print(f"mteb_search: baseline {args.baseline_model} on {args.task} (not this engine)",
              flush=True)
        result = mteb.evaluate(mteb.get_model(args.baseline_model), mteb.get_task(args.task))
        print(result)
        return 0

    if args.embed_model and args.model_name == MODEL_NAME:
        print(
            "mteb_search: --embed-model needs --model-name — the default name says the encoder is "
            "the multilingual decision checkpoint, and a run that scores a different one under it "
            "would be filed under a description that is false.",
            file=sys.stderr,
        )
        return 2
    if args.embed_model:
        os.environ["LAYA_RAG_EMBED_MODEL"] = args.embed_model
        print(f"mteb_search: encoder {args.embed_model}, filed as {args.model_name}", flush=True)
    meta = build_model_meta(args.model_name)
    model = meta.load_model()
    if args.first_stage_only:
        # The flag has to reach the *adapter*, not just the banner. `load_model()` builds it with
        # no config, so `_engine_for` applies `first_stage_only` when it constructs the engine —
        # and the first version of this printed the banner and set nothing, which meant every
        # "first-stage-only" MTEB run still paid the 322M decision on every candidate. It was
        # found by dumping stacks during a five-query probe: `engine.query → rerank →
        # score_relevance → predict_many`, i.e. the second stage, exactly as the round-5 reading
        # had said before two rounds of hypotheses about the query embedding.
        model.first_stage_only = True
        print(
            "mteb_search: first-stage-only — the ranking is the engine's fusion, with no "
            "relevance decision in the path. Quote this configuration with the row.",
            flush=True,
        )
    if args.progress_every:
        model.progress_every = args.progress_every
    if args.limit_queries:
        model.limit_queries = args.limit_queries
        print(
            f"mteb_search: SMOKE — scoring at most {args.limit_queries} queries. "
            f"A capped pass is a subset: it is a rate measurement, not a score.",
            flush=True,
        )
    task = mteb.get_task(args.task)
    result = mteb.evaluate(model, task, num_proc=args.num_proc)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
