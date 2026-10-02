#!/usr/bin/env python
"""Distil a retrieval-trained projection onto Laya's frozen document embeddings.

Laya's own encoder is a decision encoder, not a retrieval encoder: mean-pooled cosine
similarity over its embeddings scores around nDCG@10 ~0.014 on SciFact, which is why the
engine ships with the dense channel off by default. Re-training the encoder is expensive
and the checkpoints are vendored, but a linear map on top of frozen embeddings is cheap
to fit and cheap to serve.

The map is distilled from a strong retrieval teacher (``BAAI/bge-small-en-v1.5`` by
default) whose cached *document* embeddings already exist beside the BEIR caches. Ridge
regression is fit from Laya's document vectors to the teacher's, and the same map is
applied to Laya's *query* vectors. Because both sides pass through one linear map, the
projected query and projected document live in a shared space even though the map was
only ever fit on documents.

The comparison this tool reports is deliberately three-way, because a projection result
conflates two things:

* ``raw``         — Laya documents vs Laya queries, the channel the engine disables;
* ``projected``   — the same Laya queries through the learned map;
* ``teacher_docs_only`` — teacher documents vs **Laya** queries.

That last control isolates query-encoder quality: if the teacher's documents cannot be
retrieved with Laya's queries, no document-side projection can fix it.

Everything here is pure NumPy until :func:`main` loads the Laya query encoder, so the
tool imports and unit-tests offline with no weights::

    .venv/bin/python tools/projection_head.py --dataset scifact
    .venv/bin/python -m pytest tests/test_projection_head.py -q
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# ``evalkit`` lives beside this file and is imported the way ``tools/beir.py`` imports
# it: the tools directory is put on the path rather than being made a package.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import evalkit  # noqa: E402

#: Cache stem of the Laya document embeddings, matching ``tools/beir.py``'s ``laya.name``.
LAYA_CACHE_STEM = "laya-multilingual"

#: On-disk format written by :meth:`LinearProjection.save`.
PROJECTION_VERSION = 1

DEFAULT_TEACHER = "BAAI/bge-small-en-v1.5"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "eval" / "beir"


# ── the projection ───────────────────────────────────────────────────────

class LinearProjection:
    """A frozen affine map, ``x -> x @ weights + bias``, fitted by ridge regression.

    Stored as two plain arrays so it round-trips through JSON with no pickle and no
    framework dependency. ``transform`` deliberately does **not** normalize: callers
    choose whether the downstream metric is cosine (normalize) or something else.
    """

    def __init__(self, weights: np.ndarray | None = None, bias: np.ndarray | None = None) -> None:
        if weights is None and bias is None:
            self.weights: np.ndarray | None = None
            self.bias: np.ndarray | None = None
            return
        if weights is None or bias is None:
            raise ValueError("weights and bias must be supplied together")
        matrix = np.asarray(weights, dtype=np.float64)
        vector = np.asarray(bias, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"weights must be 2-D, got shape {matrix.shape}")
        if vector.ndim != 1:
            raise ValueError(f"bias must be 1-D, got shape {vector.shape}")
        if matrix.shape[1] != vector.shape[0]:
            raise ValueError(
                f"bias length {vector.shape[0]} does not match weights output dim {matrix.shape[1]}"
            )
        self.weights = matrix.copy()
        self.bias = vector.copy()

    # -- construction ----------------------------------------------------

    @classmethod
    def fit(
        cls,
        source: np.ndarray,
        target: np.ndarray,
        *,
        l2: float = 1e-3,
    ) -> "LinearProjection":
        """Closed-form ridge map from ``source`` rows to ``target`` rows.

        Solves ``W = (X^T X + l2 I)^-1 X^T Y`` on an augmented design matrix ``X`` with a
        trailing bias column of ones. The regularization therefore covers the bias too;
        ``l2`` is a shrink toward zero, not a numerical afterthought, and a tiny value is
        what makes the in-sample recovery of a known linear map possible.
        """
        x = np.asarray(source, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"source and target must be 2-D, got {x.ndim}-D and {y.ndim}-D")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"row-count mismatch: source has {x.shape[0]} rows, target has {y.shape[0]}"
            )
        if x.shape[0] == 0 or x.shape[1] == 0 or y.shape[1] == 0:
            raise ValueError("cannot fit a projection on empty input")
        if l2 < 0:
            raise ValueError(f"l2 must be non-negative, got {l2!r}")

        rows = x.shape[0]
        design = np.hstack([x, np.ones((rows, 1), dtype=np.float64)])
        gram = design.T @ design + float(l2) * np.eye(design.shape[1])
        solution = _solve(gram, design.T @ y)
        return cls(weights=solution[:-1], bias=solution[-1])

    # -- use -------------------------------------------------------------

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Map one row (1-D) or many rows (2-D) to ``float64``; no normalization."""
        weights, bias = self._fitted()
        x = np.asarray(vectors, dtype=np.float64)
        if x.ndim == 1:
            if x.shape[0] != weights.shape[0]:
                raise ValueError(
                    f"expected input dimension {weights.shape[0]}, got {x.shape[0]}"
                )
            return x @ weights + bias
        if x.ndim == 2:
            if x.shape[1] != weights.shape[0]:
                raise ValueError(
                    f"expected input dimension {weights.shape[0]}, got {x.shape[1]}"
                )
            return x @ weights + bias
        raise ValueError(f"vectors must be 1-D or 2-D, got {x.ndim}-D")

    def transform_normalized(self, vectors: np.ndarray) -> np.ndarray:
        """Map then L2-normalize each row. Zero rows stay zero rather than becoming NaN."""
        projected = self.transform(vectors)
        if projected.ndim == 1:
            norm = float(np.linalg.norm(projected))
            return projected / norm if norm > 0 else projected
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return np.divide(projected, norms, out=np.zeros_like(projected), where=norms > 0)

    # -- persistence -----------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Write JSON ``{"version", "weights", "bias"}`` atomically."""
        weights, bias = self._fitted()
        payload = {
            "version": PROJECTION_VERSION,
            "weights": weights.tolist(),
            "bias": bias.tolist(),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False)
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path | str) -> "LinearProjection":
        """Read a projection written by :meth:`save`; unknown versions are rejected."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != PROJECTION_VERSION:
            raise ValueError(
                f"unsupported projection version {version!r}; expected {PROJECTION_VERSION}"
            )
        return cls(
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=np.asarray(payload["bias"], dtype=np.float64),
        )

    # -- introspection ---------------------------------------------------

    def _fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self.weights is None or self.bias is None:
            raise ValueError("projection is not fitted; call LinearProjection.fit or .load first")
        return self.weights, self.bias

    def __repr__(self) -> str:
        if self.weights is None:
            return "LinearProjection(unfitted)"
        return f"LinearProjection(weights={self.weights.shape}, bias={self.bias.shape})"


def _solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a ridge normal equation, falling back to least squares if singular."""
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def distillation_loss(
    source: np.ndarray,
    target: np.ndarray,
    projection: LinearProjection,
) -> float:
    """Mean cosine distance ``1 - cos(projected_source, target)``, for reporting only.

    It measures how well the map reproduces the teacher in the sample it was fit on, so
    it is an in-sample diagnostic rather than a held-out score.
    """
    projected = projection.transform(source)
    teacher = np.asarray(target, dtype=np.float64)
    if projected.ndim == 1:
        projected = projected[None, :]
    if teacher.ndim == 1:
        teacher = teacher[None, :]
    if projected.shape != teacher.shape:
        raise ValueError(
            f"projected source shape {projected.shape} does not match target shape {teacher.shape}"
        )
    projected_norms = np.linalg.norm(projected, axis=1)
    teacher_norms = np.linalg.norm(teacher, axis=1)
    denominators = projected_norms * teacher_norms
    cosine = np.divide(
        (projected * teacher).sum(axis=1),
        denominators,
        out=np.zeros_like(denominators),
        where=denominators > 0,
    )
    return float(np.mean(1.0 - cosine))


# ── ranking (no model needed) ────────────────────────────────────────────

def retrieval_rankings(
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray | Mapping[str, np.ndarray],
    ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    top_k: int = 100,
) -> dict[str, list[str]]:
    """Cosine-rank ``ids`` against each query, dropping the query's own document.

    ``query_vectors`` is either a 2-D array whose rows follow ``qrels``' key order, or a
    ``{query_id: vector}`` mapping. Ties are broken by ascending document id so a run is
    reproducible regardless of the order ``doc_vectors`` arrives in.
    """
    docs = np.asarray(doc_vectors, dtype=np.float64)
    doc_ids = [str(doc_id) for doc_id in ids]
    if docs.ndim != 2:
        raise ValueError(f"doc_vectors must be 2-D, got shape {docs.shape}")
    if docs.shape[0] != len(doc_ids):
        raise ValueError(
            f"doc_vectors has {docs.shape[0]} rows but {len(doc_ids)} ids were given"
        )
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k!r}")

    doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
    unit_docs = np.divide(docs, doc_norms, out=np.zeros_like(docs), where=doc_norms > 0)

    if isinstance(query_vectors, Mapping):
        queries = [(str(query_id), np.asarray(vector, dtype=np.float64))
                   for query_id, vector in query_vectors.items()]
    else:
        vectors = np.asarray(query_vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.ndim != 2:
            raise ValueError(f"query_vectors must be 2-D, got shape {vectors.shape}")
        query_ids = [str(query_id) for query_id in qrels]
        if vectors.shape[0] != len(query_ids):
            raise ValueError(
                f"query_vectors has {vectors.shape[0]} rows but qrels has {len(query_ids)} queries"
            )
        queries = list(zip(query_ids, vectors))

    rankings: dict[str, list[str]] = {}
    for query_id, query in queries:
        norm = float(np.linalg.norm(query))
        if norm > 0:
            unit_query = query / norm
            scores = unit_docs @ unit_query
        else:
            scores = np.zeros(len(doc_ids), dtype=np.float64)
        order = sorted(range(len(doc_ids)), key=lambda index: (-scores[index], doc_ids[index]))
        rankings[query_id] = [
            doc_ids[index] for index in order if doc_ids[index] != query_id
        ][:top_k]
    return rankings


# ── evaluation ───────────────────────────────────────────────────────────

def _query_matrix(
    laya_query: np.ndarray | Mapping[str, np.ndarray],
    qrels: Mapping[str, Mapping[str, float]],
) -> tuple[list[str], np.ndarray]:
    """Normalize the query side to ``(query_ids, matrix)`` in the order qrels is scored.

    A mapping is restricted to the labelled query ids; an array is assumed to follow
    ``sorted(qrels)`` so the alignment is stable and explicit.
    """
    if isinstance(laya_query, Mapping):
        normalized = {str(query_id): vector for query_id, vector in laya_query.items()}
        query_ids = [query_id for query_id in normalized if query_id in qrels]
        if not query_ids:
            raise ValueError("no query vectors correspond to a labelled query id")
        rows = [np.asarray(normalized[query_id], dtype=np.float64).reshape(-1) for query_id in query_ids]
        return query_ids, np.vstack(rows)

    vectors = np.asarray(laya_query, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    if vectors.ndim != 2:
        raise ValueError(f"laya_query must be 1-D or 2-D, got shape {vectors.shape}")
    query_ids = sorted(str(query_id) for query_id in qrels)
    if vectors.shape[0] != len(query_ids):
        raise ValueError(
            f"laya_query has {vectors.shape[0]} rows but qrels has {len(query_ids)} labelled queries"
        )
    return query_ids, vectors


def evaluate_projected(
    laya_doc: np.ndarray,
    teacher_doc: np.ndarray,
    laya_query: np.ndarray | Mapping[str, np.ndarray],
    qrels: Mapping[str, Mapping[str, float]],
    ids: Sequence[str],
    top_k: int = 100,
    *,
    l2: float = 1e-3,
) -> dict[str, Any]:
    """Fit the distillation projection and score three retrieval configurations.

    ``l2`` is accepted in addition to the documented positional signature so the CLI can
    expose it; the default matches :meth:`LinearProjection.fit`.
    """
    laya_documents = np.asarray(laya_doc, dtype=np.float64)
    teacher_documents = np.asarray(teacher_doc, dtype=np.float64)
    if laya_documents.ndim != 2 or teacher_documents.ndim != 2:
        raise ValueError("laya_doc and teacher_doc must be 2-D arrays")
    if laya_documents.shape[0] != teacher_documents.shape[0]:
        raise ValueError(
            f"laya_doc has {laya_documents.shape[0]} rows but teacher_doc has "
            f"{teacher_documents.shape[0]}"
        )
    doc_ids = [str(doc_id) for doc_id in ids]
    if laya_documents.shape[0] != len(doc_ids):
        raise ValueError(
            f"laya_doc has {laya_documents.shape[0]} rows but {len(doc_ids)} ids were given"
        )

    normalized_qrels = {
        str(query_id): {str(doc_id): float(score) for doc_id, score in relevant.items()}
        for query_id, relevant in qrels.items()
    }
    query_ids, query_matrix = _query_matrix(laya_query, normalized_qrels)
    ordered_qrels = {query_id: normalized_qrels[query_id] for query_id in query_ids}

    projection = LinearProjection.fit(laya_documents, teacher_documents, l2=l2)
    projected_documents = projection.transform_normalized(laya_documents)
    projected_queries = projection.transform_normalized(query_matrix)

    raw = retrieval_rankings(laya_documents, query_matrix, doc_ids, ordered_qrels, top_k)
    projected = retrieval_rankings(projected_documents, projected_queries, doc_ids, ordered_qrels, top_k)
    # Control: the teacher's documents against Laya's own queries. If this is weak, the
    # bottleneck is the query encoder and no document-side projection can rescue it.
    #
    # Raw Laya queries and teacher documents usually live in different dimensions, so the
    # control uses the raw query only when the spaces already match and otherwise uses the
    # projected query -- the same Laya query vectors, mapped into the teacher's space.
    if query_matrix.shape[1] == teacher_documents.shape[1]:
        control_queries, control_kind = query_matrix, "raw"
    else:
        control_queries, control_kind = projected_queries, "projected"
    teacher_only = retrieval_rankings(teacher_documents, control_queries, doc_ids, ordered_qrels, top_k)

    def summary(rankings: Mapping[str, Sequence[str]]) -> dict[str, float]:
        return evalkit.evaluate_rankings(rankings, ordered_qrels)["summary"]

    raw_metrics = summary(raw)
    projected_metrics = summary(projected)
    teacher_metrics = summary(teacher_only)

    return {
        "raw": raw_metrics,
        "projected": projected_metrics,
        "teacher_docs_only": teacher_metrics,
        "ndcg@10": {
            "raw": raw_metrics.get("nDCG@10", 0.0),
            "projected": projected_metrics.get("nDCG@10", 0.0),
            "teacher_docs_only": teacher_metrics.get("nDCG@10", 0.0),
            "delta": round(
                float(projected_metrics.get("nDCG@10", 0.0))
                - float(raw_metrics.get("nDCG@10", 0.0)),
                4,
            ),
        },
        "loss": round(distillation_loss(laya_documents, teacher_documents, projection), 6),
        "settings": {
            "top_k": top_k,
            "l2": l2,
            "documents": int(laya_documents.shape[0]),
            "queries": len(query_ids),
            "source_dim": int(laya_documents.shape[1]),
            "target_dim": int(teacher_documents.shape[1]),
            "teacher_control_query": control_kind,
        },
        "projection": repr(projection),
    }


# ── reporting ────────────────────────────────────────────────────────────

def render_markdown(results: Mapping[str, Any], *, dataset: str, teacher: str) -> str:
    """Human-readable companion to the JSON report."""
    lines = [
        f"# Projection head — {dataset}",
        "",
        "Ridge map fit from Laya document embeddings to the teacher's, then applied to "
        "Laya's query embeddings.",
        "",
        f"- teacher: `{teacher}`",
        f"- projection: `{results.get('projection', '?')}`",
        f"- in-sample distillation loss (mean cosine distance): {results.get('loss')}",
        f"- settings: `{json.dumps(results.get('settings', {}), sort_keys=True)}`",
        "",
    ]
    rows = []
    for name in ("raw", "projected", "teacher_docs_only"):
        metrics = results.get(name, {})
        row: dict[str, Any] = {"configuration": name}
        for metric in ("nDCG@10", "nDCG@100", "MRR@10", "MAP@100"):
            row[metric] = metrics.get(metric)
        rows.append(row)
    lines.append(evalkit.markdown_table(rows))
    lines.append("")
    comparison = results.get("ndcg@10", {})
    lines.append(
        f"nDCG@10: raw={comparison.get('raw', 0.0):.4f} "
        f"projected={comparison.get('projected', 0.0):.4f} "
        f"teacher_docs_only={comparison.get('teacher_docs_only', 0.0):.4f} "
        f"(delta={comparison.get('delta', 0.0):+.4f})"
    )
    lines.append("")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distil a linear projection onto Laya's frozen document embeddings.",
    )
    parser.add_argument("--dataset", default="scidocs", help="BEIR dataset name (default: scidocs)")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help=f"directory holding datasets and their embedding cache (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument("--out", default=None, help="output stem for .json/.md (default: data-root/projection-<dataset>)")
    parser.add_argument("--top-k", type=int, default=100, help="ranking depth (default: 100)")
    parser.add_argument("--l2", type=float, default=1e-3, help="ridge regularization (default: 1e-3)")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER, help=f"teacher model id (default: {DEFAULT_TEACHER})")
    return parser.parse_args(argv)


def load_qrels(path: Path | str) -> dict[str, dict[str, float]]:
    """Read a BEIR ``qrels/*.tsv`` file directly; header rows are ignored."""
    qrels: dict[str, dict[str, float]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        try:
            score = float(parts[2])
        except ValueError:
            continue
        qrels.setdefault(parts[0], {})[parts[1]] = score
    return qrels


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Loads the Laya query encoder lazily, only after cache checks."""
    args = parse_args(argv)
    data_root = Path(args.data_root)
    dataset = args.dataset
    cache_dir = data_root / "cache"

    laya_vectors = cache_dir / f"{dataset}.{LAYA_CACHE_STEM}.npy"
    laya_ids = cache_dir / f"{dataset}.{LAYA_CACHE_STEM}.ids.json"
    teacher_short = args.teacher.rsplit("/", 1)[-1]
    teacher_vectors = cache_dir / f"{dataset}.{teacher_short}.cls.npy"
    teacher_ids = cache_dir / f"{dataset}.{teacher_short}.cls.ids.json"

    if not (laya_vectors.is_file() and laya_ids.is_file()):
        print(
            f"[{dataset}] missing cached Laya document embeddings: {laya_vectors} "
            f"(and {laya_ids.name}); build them with "
            f"`tools/beir.py --datasets {dataset} --retrievers dense` first",
            flush=True,
        )
        return 1
    if not (teacher_vectors.is_file() and teacher_ids.is_file()):
        print(
            f"[{dataset}] missing cached teacher embeddings for {args.teacher}: "
            f"{teacher_vectors} (and {teacher_ids.name}); build them with "
            f"`tools/beir.py --datasets {dataset} --retrievers biencoder "
            f"--biencoder-model {args.teacher}` first",
            flush=True,
        )
        return 1

    qrels_path = data_root / dataset / "qrels" / "test.tsv"
    if not qrels_path.is_file():
        print(f"[{dataset}] missing qrels file: {qrels_path}", flush=True)
        return 1

    # Imported here so the module stays importable (and unit-testable) without the
    # engine package or any model dependency at import time.
    import beir  # noqa: E402

    documents = np.load(laya_vectors).astype(np.float64)
    document_ids = [str(doc_id) for doc_id in json.loads(laya_ids.read_text(encoding="utf-8"))]
    teacher = np.load(teacher_vectors).astype(np.float64)
    teacher_order = [str(doc_id) for doc_id in json.loads(teacher_ids.read_text(encoding="utf-8"))]

    if teacher_order != document_ids:
        position = {doc_id: index for index, doc_id in enumerate(teacher_order)}
        missing = [doc_id for doc_id in document_ids if doc_id not in position]
        if missing:
            print(
                f"[{dataset}] teacher cache is missing {len(missing)} Laya document ids "
                f"(first: {missing[0]})",
                flush=True,
            )
            return 1
        teacher = teacher[[position[doc_id] for doc_id in document_ids]]
    if documents.shape[0] != len(document_ids):
        print(
            f"[{dataset}] Laya cache has {documents.shape[0]} vectors but "
            f"{len(document_ids)} ids",
            flush=True,
        )
        return 1

    qrels = load_qrels(qrels_path)
    all_queries = beir.load_queries(data_root / dataset)
    query_ids = sorted(query_id for query_id in all_queries if query_id in qrels)
    if not query_ids:
        print(f"[{dataset}] no labelled query has text in queries.jsonl", flush=True)
        return 1

    # The only model load in the tool, and only after every cache check has passed.
    from laya_rag.config import EngineConfig

    laya = beir.LayaEmbedder(EngineConfig())
    if laya.name != LAYA_CACHE_STEM:
        print(
            f"[{dataset}] warning: query encoder {laya.name!r} does not match the cached "
            f"document embeddings {LAYA_CACHE_STEM!r}; their dimensions may disagree",
            flush=True,
        )
    query_vectors = np.asarray(laya.embed([all_queries[query_id] for query_id in query_ids]), dtype=np.float64)
    if query_vectors.shape[0] != len(query_ids):
        print(f"[{dataset}] embedder returned {query_vectors.shape[0]} query vectors", flush=True)
        return 1

    print(
        f"[{dataset}] {documents.shape[0]} documents x {documents.shape[1]} dims, "
        f"{len(query_ids)} labelled queries, query encoder={laya.name}, teacher={args.teacher}",
        flush=True,
    )
    results = evaluate_projected(
        documents,
        teacher,
        {query_id: query_vectors[index] for index, query_id in enumerate(query_ids)},
        qrels,
        document_ids,
        top_k=args.top_k,
        l2=args.l2,
    )

    out = Path(args.out) if args.out else data_root / f"projection-{dataset}"
    json_path = out.with_suffix(".json")
    markdown_path = out.with_suffix(".md")
    evalkit.write_json(json_path, results)
    evalkit.write_text(markdown_path, render_markdown(results, dataset=dataset, teacher=args.teacher))

    comparison = results["ndcg@10"]
    print(
        f"nDCG@10: raw={comparison['raw']:.4f} projected={comparison['projected']:.4f} "
        f"teacher_docs_only={comparison['teacher_docs_only']:.4f} "
        f"(delta={comparison['delta']:+.4f})",
        flush=True,
    )
    print(f"wrote {json_path} and {markdown_path}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
