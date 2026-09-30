#!/usr/bin/env python
"""Retrieval quality on a standard benchmark, so the numbers can be read beside other systems.

``tools/eval.py`` measures this project on a 13-chunk toy corpus. That answers "does it
work", not "how does it compare". This runs the same retrieval channels over a BEIR
dataset with its official qrels and reports the metrics the IR leaderboard reports —
nDCG@10, nDCG@100, MRR@10, Recall@100, MAP@100 — against baselines built the same way:

* ``bm25``      — lexical, at the published BEIR configuration (k1=0.9, b=0.4);
* ``dense``     — Laya's mean-pooled encoder, the channel this engine uses;
* ``hybrid``    — dense + BM25 fused by RRF, exactly as the engine fuses them;
* ``biencoder`` — a retrieval-trained bi-encoder, i.e. what Laya's README says a
                  dedicated shortlist model does better than its own encoder.

Nothing here is hardcoded from a leaderboard. The comparison is controlled: same corpus,
same qrels, same evaluator, same top-k, so the difference between rows is the retriever.

    .venv/bin/python tools/beir.py --datasets scifact
    .venv/bin/python tools/beir.py --datasets scifact,nfcorpus --retrievers bm25,dense

Datasets are downloaded from the BEIR mirror into ``eval/beir/data/`` and embeddings are
cached beside them, both gitignored. Needs network on the first run and the Laya
checkpoints for the dense rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# Hugging Face caches under ~/.cache/huggingface by default, which is outside this
# workspace and therefore unwritable when the file sandbox is active. Keep the
# bi-encoder download inside eval/ so the comparison runs without an escape hatch.
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "beir" / "hf-cache"))

import evalkit  # noqa: E402
from laya_rag.chunking import Chunk  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.index_store import VectorIndex  # noqa: E402
from laya_rag.lexical import BM25Index, reciprocal_rank_fusion  # noqa: E402

#: Public BEIR mirror used by the official scripts.
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

#: The BM25 configuration the published BEIR baselines are run at (Anserini/Pyserini).
BEIR_K1 = 0.9
BEIR_B = 0.4

DEFAULT_DATA_ROOT = ROOT / "eval" / "beir"

REFERENCES = {
    "BEIR paper": "https://arxiv.org/abs/2104.08663",
    "BEIR code": "https://github.com/UKPLab/beir",
    "note": (
        "Published BM25 baselines use Anserini/Pyserini: k1=0.9, b=0.4 and a Lucene "
        "analyzer with stemming. This harness uses the same k1/b but this project's "
        "regex tokenizer without stemming, so a small gap is expected and the rows here "
        "are a like-for-like comparison between retrievers, not a head-to-head with a "
        "leaderboard entry."
    ),
}


# ── dataset loading ──────────────────────────────────────────────────────

def ensure_dataset(name: str, data_root: Path, *, download: bool = True) -> Path:
    """Return the local directory for ``name``, downloading and unzipping if needed."""
    target = data_root / name
    if (target / "corpus.jsonl").is_file() and (target / "queries.jsonl").is_file():
        return target
    if not download:
        raise FileNotFoundError(f"{target} is not present and --no-download was given")
    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / f"{name}.zip"
    if not archive.is_file():
        url = BEIR_URL.format(name=name)
        print(f"  downloading {url}", flush=True)
        urllib.request.urlretrieve(url, archive)
    print(f"  extracting {archive.name}", flush=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(data_root)
    if not (target / "corpus.jsonl").is_file():
        raise FileNotFoundError(f"{archive} did not contain {name}/corpus.jsonl")
    return target


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_corpus(dataset_dir: Path) -> dict[str, str]:
    """``_id -> "title text"``, the concatenation BEIR's own baselines index."""
    corpus: dict[str, str] = {}
    for row in _read_jsonl(dataset_dir / "corpus.jsonl"):
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        corpus[str(row["_id"])] = f"{title} {text}".strip()
    return corpus


def load_queries(dataset_dir: Path) -> dict[str, str]:
    return {str(row["_id"]): str(row.get("text") or "") for row in _read_jsonl(dataset_dir / "queries.jsonl")}


def load_qrels(dataset_dir: Path) -> dict[str, dict[str, float]]:
    """Read ``qrels/*.tsv``, preferring the test split. Header rows are ignored."""
    qrels_dir = dataset_dir / "qrels"
    candidates = sorted(qrels_dir.glob("*.tsv"))
    if not candidates:
        raise FileNotFoundError(f"no qrels/*.tsv under {dataset_dir}")
    path = next((p for p in candidates if p.stem == "test"), candidates[0])

    qrels: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        try:
            score = float(parts[2])
        except ValueError:
            continue  # header row
        qrels.setdefault(parts[0], {})[parts[1]] = score
    return qrels


def make_chunks(corpus: Mapping[str, str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=doc_id,
            doc_id=doc_id,
            path=doc_id,
            text=text,
            index=0,
            start_char=0,
            end_char=len(text),
        )
        for doc_id, text in corpus.items()
    ]


# ── embedders ────────────────────────────────────────────────────────────

class LayaEmbedder:
    """Laya's mean-pooled checkpoint encoder — the dense channel this engine ships."""

    name = "laya"

    def __init__(self, config: EngineConfig) -> None:
        from laya_rag.model import LayaModel

        self._model = LayaModel(config)
        self._model.ensure_loaded()
        self.name = f"laya-{config.checkpoint}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.embed(list(texts)), dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts)


class BiEncoder:
    """A retrieval-trained bi-encoder, for the comparison Laya's README invites."""

    def __init__(
        self,
        model_id: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device in (None, "auto"):
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.name = model_id.rsplit("/", 1)[-1]
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = list(texts)
        out: list[np.ndarray] = []
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed([self.query_prefix + text for text in texts])


def cached_embed(
    embedder: Any,
    corpus: Mapping[str, str],
    cache: Path,
    *,
    use_cache: bool = True,
) -> np.ndarray:
    """Embed the corpus once, reusing a sidecar cache keyed by the doc id order."""
    ids = list(corpus)
    ids_file = cache.with_suffix(".ids.json")
    vectors_file = cache.with_suffix(".npy")
    if use_cache and vectors_file.is_file() and ids_file.is_file():
        if json.loads(ids_file.read_text(encoding="utf-8")) == ids:
            print(f"  reusing cached {embedder.name} embeddings ({vectors_file.name})", flush=True)
            return np.load(vectors_file)
    print(f"  embedding {len(ids)} documents with {embedder.name}", flush=True)
    started = time.perf_counter()
    vectors = np.asarray(embedder.embed([corpus[doc_id] for doc_id in ids]), dtype=np.float32)
    print(f"  embedded in {time.perf_counter() - started:.1f}s", flush=True)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(vectors_file, vectors)
        ids_file.write_text(json.dumps(ids), encoding="utf-8")
    return vectors


# ── retrievers ───────────────────────────────────────────────────────────

def bm25_rankings(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    top_k: int,
    k1: float = BEIR_K1,
    b: float = BEIR_B,
    drop_stopwords: bool = False,
) -> dict[str, list[str]]:
    """Lexical ranking at the published BEIR BM25 settings.

    Query stopword filtering is off here: the engine drops stopwords because its queries
    are short natural-language questions, but the Anserini baseline does not, and turning
    it off is what makes this row comparable.
    """
    index = BM25Index(k1=k1, b=b, drop_stopwords=drop_stopwords)
    index.build(make_chunks(corpus))
    return {
        query_id: [hit.chunk_id for hit in index.search(text, k=top_k)]
        for query_id, text in queries.items()
    }


def dense_rankings(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    embedder: Any,
    vectors: np.ndarray,
    top_k: int,
) -> dict[str, list[str]]:
    index = VectorIndex()
    index.add(make_chunks(corpus), vectors)
    query_vectors = np.asarray(embedder.embed_queries(list(queries.values())), dtype=np.float32)
    rankings: dict[str, list[str]] = {}
    for (query_id, _), vector in zip(queries.items(), query_vectors):
        rankings[query_id] = [hit.chunk.chunk_id for hit in index.search(vector, k=top_k)]
    return rankings


def hybrid_rankings(
    dense: Mapping[str, Sequence[str]],
    lexical: Mapping[str, Sequence[str]],
    top_k: int,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> dict[str, list[str]]:
    """RRF over the two channels, the same weighted fusion ``RagEngine._fuse`` applies."""
    fused: dict[str, list[str]] = {}
    for query_id, dense_ids in dense.items():
        lexical_ids = lexical.get(query_id, [])
        merged = reciprocal_rank_fusion(
            [list(dense_ids)[:top_k], list(lexical_ids)[:top_k]],
            limit=top_k,
            weights=[dense_weight, lexical_weight],
        )
        fused[query_id] = [doc_id for doc_id, _ in merged]
    return fused


# ── reporting ────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="beir.py", description="Standard retrieval evaluation on BEIR")
    parser.add_argument("--datasets", default="scifact,nfcorpus")
    parser.add_argument("--retrievers", default="bm25,dense,hybrid,biencoder,hybrid-biencoder")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=BEIR_K1)
    parser.add_argument("--b", type=float, default=BEIR_B)
    parser.add_argument("--biencoder-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out", default=str(ROOT / "eval" / "beir" / "report"))
    parser.add_argument("--no-download", action="store_true", help="fail instead of fetching a dataset")
    parser.add_argument("--no-cache", action="store_true", help="re-embed even if a cache exists")
    parser.add_argument("--limit-docs", type=int, default=None, help="truncate the corpus (smoke runs only)")
    parser.add_argument("--dense-weight", type=float, default=None,
                        help="dense RRF weight for the hybrid row (default: the engine's resolved value)")
    parser.add_argument("--biencoder-weight", type=float, default=1.0,
                        help="bi-encoder RRF weight for the hybrid-biencoder row")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    wanted = [name.strip() for name in args.retrievers.split(",") if name.strip()]
    config = EngineConfig()
    if args.device:
        config.device = args.device
    dense_weight = (
        config.resolved_dense_weight if args.dense_weight is None else args.dense_weight
    )

    laya: LayaEmbedder | None = None
    biencoder: BiEncoder | None = None
    report: dict[str, Any] = {
        "settings": {
            "top_k": args.top_k,
            "bm25_k1": args.k1,
            "bm25_b": args.b,
            "drop_stopwords": False,
            "dense_weight": dense_weight,
            "lexical_weight": config.lexical_weight,
            "checkpoint": config.checkpoint,
            "biencoder_model": args.biencoder_model if "biencoder" in wanted else None,
            "limit_docs": args.limit_docs,
        },
        "references": REFERENCES,
        "datasets": {},
    }

    for name in dataset_names:
        print(f"[{name}]", flush=True)
        dataset_dir = ensure_dataset(name, data_root, download=not args.no_download)
        corpus = load_corpus(dataset_dir)
        if args.limit_docs:
            corpus = dict(list(corpus.items())[: args.limit_docs])
        all_queries = load_queries(dataset_dir)
        qrels = load_qrels(dataset_dir)
        # Only rank the queries that have qrels, which is what BEIR scores.
        queries = {qid: text for qid, text in all_queries.items() if qid in qrels}
        print(f"  {len(corpus)} documents, {len(queries)} scored queries", flush=True)

        cache_dir = data_root / "cache"
        results: dict[str, Any] = {}
        rankings: dict[str, dict[str, list[str]]] = {}
        lexical: dict[str, list[str]] | None = None

        if "bm25" in wanted:
            started = time.perf_counter()
            rankings["bm25"] = bm25_rankings(corpus, queries, args.top_k, k1=args.k1, b=args.b)
            results["bm25"] = {"seconds": round(time.perf_counter() - started, 2)}

        # Any fused row needs the lexical channel, whether or not it is reported.
        if {"hybrid", "hybrid-biencoder"} & set(wanted):
            lexical = rankings.get("bm25") or bm25_rankings(
                corpus, queries, args.top_k, k1=args.k1, b=args.b
            )

        dense: dict[str, list[str]] | None = None
        if "dense" in wanted or "hybrid" in wanted:
            if laya is None:
                laya = LayaEmbedder(config)
            vectors = cached_embed(
                laya, corpus, cache_dir / f"{name}.{laya.name}", use_cache=not args.no_cache
            )
            started = time.perf_counter()
            dense = dense_rankings(corpus, queries, laya, vectors, args.top_k)
            seconds = round(time.perf_counter() - started, 2)
            if "dense" in wanted:
                rankings["dense"] = dense
                results["dense"] = {"seconds": seconds, "model": laya.name}
            if "hybrid" in wanted:
                rankings["hybrid"] = hybrid_rankings(
                    dense, lexical or {}, args.top_k,
                    dense_weight=dense_weight, lexical_weight=config.lexical_weight,
                )
                results["hybrid"] = {
                    "seconds": seconds,
                    "note": f"RRF of dense (w={dense_weight}) + bm25 (w={config.lexical_weight})",
                }

        if "biencoder" in wanted or "hybrid-biencoder" in wanted:
            try:
                if biencoder is None:
                    biencoder = BiEncoder(args.biencoder_model, device=args.device)
                vectors = cached_embed(
                    biencoder, corpus, cache_dir / f"{name}.{biencoder.name}", use_cache=not args.no_cache
                )
                started = time.perf_counter()
                bi = dense_rankings(corpus, queries, biencoder, vectors, args.top_k)
                seconds = round(time.perf_counter() - started, 2)
                if "biencoder" in wanted:
                    rankings["biencoder"] = bi
                    results["biencoder"] = {"seconds": seconds, "model": biencoder.name}
                if "hybrid-biencoder" in wanted:
                    rankings["hybrid-biencoder"] = hybrid_rankings(
                        bi, lexical or {}, args.top_k,
                        dense_weight=args.biencoder_weight, lexical_weight=config.lexical_weight,
                    )
                    results["hybrid-biencoder"] = {
                        "seconds": seconds,
                        "note": f"RRF of biencoder (w={args.biencoder_weight}) + bm25",
                    }
            except Exception as exc:  # noqa: BLE001 - a missing model must not lose the run
                for retriever in ("biencoder", "hybrid-biencoder"):
                    if retriever in wanted:
                        results[retriever] = {"error": f"{type(exc).__name__}: {exc}"}

        for retriever, ranking in rankings.items():
            evaluation = evalkit.evaluate_rankings(ranking, qrels, k_values=(10, 100))
            results[retriever].update(evaluation["summary"])
            results[retriever]["scored_queries"] = evaluation["queries"]

        report["datasets"][name] = {
            "documents": len(corpus),
            "queries": len(all_queries),
            "scored_queries": len(queries),
            "retrievers": results,
        }

    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    evalkit.write_text(out.with_suffix(".md"), render_markdown(report))
    print()
    print(render_console(report))
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


def _metric_columns(report: dict[str, Any]) -> list[str]:
    for dataset in report["datasets"].values():
        for result in dataset["retrievers"].values():
            return [key for key in result if key not in {"seconds", "model", "note", "error", "scored_queries"}]
    return ["nDCG@10", "nDCG@100", "MRR@10", "Recall@100", "MAP@100"]


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    columns = _metric_columns(report)
    rows: list[dict[str, Any]] = []
    for name, dataset in report["datasets"].items():
        for retriever, result in dataset["retrievers"].items():
            row: dict[str, Any] = {
                "dataset": name,
                "documents": dataset["documents"],
                "retriever": retriever,
            }
            if "error" in result:
                row["error"] = result["error"]
            else:
                row.update({column: result.get(column) for column in columns})
                row["seconds"] = result.get("seconds")
            rows.append(row)
    return rows


def render_console(report: dict[str, Any]) -> str:
    columns = _metric_columns(report)
    header = f"{'dataset':<10} {'retriever':<12} " + " ".join(f"{c:>9}" for c in columns) + f" {'secs':>7}"
    lines = [header, "-" * len(header)]
    for row in _rows(report):
        if "error" in row:
            lines.append(f"{row['dataset']:<10} {row['retriever']:<12} {row['error'][:60]}")
            continue
        cells = " ".join(f"{row[c]:>9.4f}" if isinstance(row[c], float) else f"{str(row[c]):>9}" for c in columns)
        lines.append(f"{row['dataset']:<10} {row['retriever']:<12} {cells} {row['seconds']:>7}")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    settings = report["settings"]
    parts = [
        "# BEIR retrieval evaluation",
        "",
        f"top_k={settings['top_k']}, BM25 k1={settings['bm25_k1']} b={settings['bm25_b']}, "
        f"query stopwords kept, checkpoint `{settings['checkpoint']}`.",
        "",
        "`nDCG@10` is the number the BEIR leaderboard reports; the rest are supporting metrics.",
        "All rows share corpus, qrels, evaluator and top-k, so the difference between rows is "
        "the retriever.",
        "",
        evalkit.markdown_table(_rows(report)),
        "",
        "## How to read this",
        "",
        REFERENCES["note"],
        "",
        f"- BEIR paper: {REFERENCES['BEIR paper']}",
        f"- BEIR code: {REFERENCES['BEIR code']}",
        "",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    sys.exit(main())
