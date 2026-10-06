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
    ) -> None:
        if device is not None and config is not None:
            config.device = device
        elif device is not None:
            from laya_rag.config import EngineConfig

            config = EngineConfig(device=device)
        self.config = config
        self._engine = engine
        self._engine_factory = engine_factory or _default_engine_factory
        self._workdir = Path(workdir) if workdir is not None else None
        self._temp: tempfile.TemporaryDirectory | None = None
        self._doc_ids: list[str] = []
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
        self._engine_for().index_paths([directory], rebuild=True)

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
        for query_id, text in query_rows(queries):
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


def build_model_meta() -> Any:
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
        name=MODEL_NAME,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="mteb_search.py")
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--task", default="SciFact", help="an MTEB task name to evaluate")
    ap.add_argument("--cache", default=None, help="MTEB_CACHE directory (must be writable)")
    ap.add_argument("--num-proc", type=int, default=1,
                    help="dataloading processes; 1 keeps the engine single-process")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    if args.cache:
        os.environ.setdefault("MTEB_CACHE", args.cache)
    try:
        import mteb
    except ImportError as exc:  # pragma: no cover
        print(f"mteb_search: MTEB is not installed ({exc}); pip install -e '.[mteb]'")
        return 2
    meta = build_model_meta()
    model = meta.load_model()
    task = mteb.get_task(args.task)
    result = mteb.evaluate(model, task, num_proc=args.num_proc)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
