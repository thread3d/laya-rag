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
from collections import Counter
import os
import random
import shutil
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
from laya_rag.lexical import STOPWORDS, BM25Index, reciprocal_rank_fusion, tokenize  # noqa: E402

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
        "analyzer with stemming. The `bm25` row keeps this project's regex tokenizer "
        "without stemming so it stays a clean control; `bm25-engine` runs the engine's own "
        "settings (stemming on, query stopwords dropped) and is what feeds the fused and "
        "reranked rows. The rows are a like-for-like comparison between retrievers, not a "
        "head-to-head with a leaderboard entry. Published scores, including these, come "
        "from BEIR's own evaluator with `ignore_identical_ids=True`, which drops a "
        "retrieved document whose id equals the query id; this harness does the same by "
        "default, which is the only way the ArguAna row is comparable."
    ),
}

#: Published BM25 nDCG@10, the anchor the `bm25` control is checked against.
#:
#: These are the Pyserini 2CR numbers (https://castorini.github.io/pyserini/2cr/beir.html),
#: which run a flat index with `--remove-query` -- Pyserini's name for the same
#: "drop the query's own document" rule this harness applies. That matters on ArguAna: 2CR
#: reports 0.397 there, the BEIR paper reports 0.315, and the difference is convention rather
#: than quality. The paper's Elasticsearch baseline indexed title and body as separate
#: *multifield* fields; a flat concatenation behaves differently, and a harness that keeps the
#: query's own document lands near 0.30. This project's flat title+text control with the rule
#: applied measures 0.394, so 0.397 is the baseline it is actually comparable to.
#: See https://github.com/beir-cellar/beir/issues/197.
#:
#: The other four datasets have no query/document id overlap, so no scoring convention moves
#: them and the values are stable across sources.
PUBLISHED_BM25 = {
    "scifact": 0.679,
    "nfcorpus": 0.325,
    "arguana": 0.397,
    "scidocs": 0.158,
    "fiqa": 0.236,
}

#: How far the `bm25` control may sit from :data:`PUBLISHED_BM25` before it is reported.
ANCHOR_TOLERANCE = 0.05


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


def load_corpus(dataset_dir: Path, include_titles: bool = True) -> dict[str, str]:
    """``_id -> document text``, as ``"title text"`` unless titles are excluded.

    BEIR's own baselines concatenate title and text, which is right for datasets whose
    titles are real titles. ArguAna's "titles" are shared topic keyword lists
    (``animals environment general health ...``) repeated across a whole cluster, so they
    add near-identical terms to every document in the cluster and flatten discrimination.
    ``include_titles=False`` measures that.
    """
    corpus: dict[str, str] = {}
    for row in _read_jsonl(dataset_dir / "corpus.jsonl"):
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        corpus[str(row["_id"])] = f"{title} {text}".strip() if include_titles else text
    return corpus


def load_titles(dataset_dir: Path) -> dict[str, str]:
    """``_id -> title``, for a title-weighted BM25F-style variant."""
    titles: dict[str, str] = {}
    for row in _read_jsonl(dataset_dir / "corpus.jsonl"):
        titles[str(row["_id"])] = (row.get("title") or "").strip()
    return titles


def restrict_qrels(qrels: Mapping[str, Mapping[str, float]],
                   queries: Mapping[str, str]) -> dict[str, Mapping[str, float]]:
    """Qrels limited to the queries actually ranked.

    `evaluate_rankings` macro-averages over every query in the qrels it is handed, so passing a full
    qrels alongside a subset's rankings scores every unsampled query as zero and divides the mean by
    the wrong denominator. On a 100-query SciFact subset that read **0.2047** where the full set scores
    **0.6762** — a 3.3x shortfall that looks like a retrieval failure and is an averaging one. It is
    the same dilution trap this project recorded for `arguana_stance` and `order_eval`.
    """
    return {qid: qrels[qid] for qid in queries if qid in qrels}


def sample_queries(queries: Mapping[str, str], limit: int | None, seed: int) -> dict[str, str]:
    """A seeded random subset of the scored queries, in a stable order.

    Random rather than the first N: query difficulty is not uniformly ordered in the file, so taking
    the head samples the easy ones and flatters the result. The seed is recorded with the report, so a
    subset can be reproduced exactly or widened later without re-deriving it.
    """
    if not limit or len(queries) <= limit:
        return dict(queries)
    rng = random.Random(seed)
    keep = sorted(rng.sample(sorted(queries), limit))
    return {qid: queries[qid] for qid in keep}


def load_queries(dataset_dir: Path) -> dict[str, str]:
    return {str(row["_id"]): str(row.get("text") or "") for row in _read_jsonl(dataset_dir / "queries.jsonl")}


def load_qrels(dataset_dir: Path, split: str | None = None) -> dict[str, dict[str, float]]:
    """Read ``qrels/*.tsv``. ``split`` selects one; ``None`` prefers the test split.

    Header rows are ignored.
    """
    qrels_dir = dataset_dir / "qrels"
    candidates = sorted(qrels_dir.glob("*.tsv"))
    if not candidates:
        raise FileNotFoundError(f"no qrels/*.tsv under {dataset_dir}")
    if split is not None:
        path = next((p for p in candidates if p.stem == split), None)
        if path is None:
            raise FileNotFoundError(f"{dataset_dir} has no qrels/{split}.tsv")
    else:
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
    """A retrieval-trained bi-encoder, for the comparison Laya's README invites.

    Pooling is not cosmetic. BGE models are trained and served with the **[CLS]** token's
    last hidden state (BAAI's model card uses ``model_output[0][:, 0]``), so mean pooling
    — the obvious default for a transformer — mismatches how the model was trained.
    ``pooling="auto"`` picks CLS for bge-family models and mean otherwise, and the name
    carries the pooling setting so the embedding cache cannot mix the two.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_id: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        pooling: str = "auto",
        query_instruction: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device in (None, "auto"):
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if pooling == "auto":
            # A local directory carries its own pooling; a name heuristic cannot tell that
            # eval/distill-nfcorpus-model is BGE-derived and would silently mean-pool it.
            from laya_rag.model import saved_pooling

            pooling = saved_pooling(model_id) or ("cls" if "bge" in model_id.lower() else "mean")
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"pooling must be cls, mean or auto, got {pooling!r}")

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.query_instruction = query_instruction
        self.model_id = model_id
        # Documents never take the query instruction, so the embedding cache is keyed by
        # model and pooling only — the instruction setting is a query-side choice.
        self.name = f"{model_id.rsplit('/', 1)[-1]}.{pooling}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    def embed(self, texts: Sequence[str], fixed_length: bool = False) -> np.ndarray:
        """Embed documents. ``fixed_length`` pads every batch to ``max_length``.

        Dynamic padding (the default) is faster but gives every batch a different tensor
        shape, and torch's MPS backend caches a compiled graph per shape — over tens of
        thousands of documents that cache grows without bound. Streaming construction
        therefore pads to a fixed width so memory stays flat.
        """
        rows = list(texts)
        out: list[np.ndarray] = []
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding="max_length" if fixed_length else True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)

    def embed_queries(self, texts: Sequence[str], fixed_length: bool = False) -> np.ndarray:
        if not self.query_instruction:
            return self.embed(texts, fixed_length=fixed_length)
        return self.embed([self.QUERY_PREFIX + text for text in texts], fixed_length=fixed_length)


class CrossEncoder:
    """A retrieval-trained cross-encoder reranker, the standard second stage.

    BAAI's own model card recommends exactly this shape: retrieve a wide candidate set
    with a cheap first stage, then re-rank it with a cross-encoder, which attends over
    the query and document together and is more accurate than a bi-encoder but too slow
    to run over a whole corpus.
    """

    def __init__(
        self,
        model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device in (None, "auto"):
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_id = model_id
        self.name = model_id.rsplit("/", 1)[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()

    def score(self, pairs: Sequence[tuple[str, str]], fixed_length: bool = False) -> list[float]:
        """Relevance score per ``(query, passage)`` pair; higher is more relevant.

        ``fixed_length`` pads every batch to ``max_length``. Dynamic padding is cheaper
        per batch but gives each batch a different tensor shape, and the MPS backend
        caches a compiled graph per shape — over thousands of batches that recompilation
        dominates. Fixed width trades some compute for a flat, predictable run.
        """
        rows = list(pairs)
        out: list[float] = []
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            encoded = self.tokenizer(
                [query for query, _ in batch],
                [passage for _, passage in batch],
                padding="max_length" if fixed_length else True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**encoded, return_dict=True).logits.view(-1).float()
            out.extend(logits.cpu().tolist())
        return out


def cached_embed(
    embedder: Any,
    corpus: Mapping[str, str],
    cache: Path,
    *,
    use_cache: bool = True,
    chunk: int = 0,
) -> np.ndarray:
    """Embed the corpus once, reusing a sidecar cache keyed by the doc id order.

    ``chunk`` above zero checkpoints the pass: each chunk is written beside the cache with a
    ``.done`` marker, so a run stopped part-way **resumes at the next chunk** instead of
    restarting. That matters because a corpus embedding is the long part of a dataset
    evaluation and this function used to write only at the end — an interrupted run lost
    everything.

    Chunk boundaries change batch composition, and therefore padding, by a few ULPs
    (measured at 4.17e-07 on 2,000 ``bge-small`` vectors, not bit-identical), so the
    **chunk size is part of the cache's identity**: a chunked run will not silently reuse an
    unchunked cache, or one built at a different chunk size. ``chunk=0`` reproduces the
    previous behaviour exactly, including reusing caches written before this option existed.
    """
    ids = list(corpus)
    # Append rather than `with_suffix`: a cache stem can contain dots (model.pooling),
    # and `with_suffix` would replace the pooling tag instead of adding an extension,
    # making every pooling variant share one cache file.
    ids_file = Path(str(cache) + ".ids.json")
    vectors_file = Path(str(cache) + ".npy")
    meta_file = Path(str(cache) + ".meta.json")
    if use_cache and vectors_file.is_file() and ids_file.is_file():
        if json.loads(ids_file.read_text(encoding="utf-8")) == ids:
            recorded = 0
            if meta_file.is_file():
                recorded = int(json.loads(meta_file.read_text(encoding="utf-8")).get("chunk", 0))
            if recorded == chunk:
                print(f"  reusing cached {embedder.name} embeddings ({vectors_file.name})", flush=True)
                return np.load(vectors_file)
            print(f"  cache was built at chunk={recorded}, want {chunk}; re-embedding", flush=True)

    print(f"  embedding {len(ids)} documents with {embedder.name}", flush=True)
    started = time.perf_counter()
    if not use_cache or chunk <= 0:
        vectors = np.asarray(embedder.embed([corpus[doc_id] for doc_id in ids]), dtype=np.float32)
    else:
        parts_dir = Path(str(cache) + ".parts")
        parts_dir.mkdir(parents=True, exist_ok=True)
        blocks: list[np.ndarray] = []
        for start in range(0, len(ids), chunk):
            part = parts_dir / f"part-{start:09d}.npy"
            marker = part.with_suffix(".done")
            block_vectors = None
            if part.is_file() and marker.is_file():
                try:
                    block_vectors = np.load(part)
                except Exception:  # a torn or corrupt part is re-embedded, not trusted
                    block_vectors = None
            if block_vectors is None:
                block = ids[start : start + chunk]
                block_vectors = np.asarray(
                    embedder.embed([corpus[doc_id] for doc_id in block]), dtype=np.float32
                )
                # Save first, then mark: an interruption before the marker leaves the part
                # untrusted and it is simply re-embedded.
                np.save(part, block_vectors)
                marker.write_text(str(len(block)), encoding="utf-8")
                print(f"    {min(start + chunk, len(ids))}/{len(ids)} "
                      f"({time.perf_counter() - started:.1f}s)", flush=True)
            blocks.append(block_vectors)
        vectors = (np.concatenate(blocks, axis=0) if blocks
                   else np.zeros((0, 0), dtype=np.float32))
    print(f"  embedded in {time.perf_counter() - started:.1f}s", flush=True)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(vectors_file, vectors)
        ids_file.write_text(json.dumps(ids), encoding="utf-8")
        meta_file.write_text(json.dumps({"chunk": chunk, "count": len(ids)}), encoding="utf-8")
        if chunk > 0:
            shutil.rmtree(Path(str(cache) + ".parts"), ignore_errors=True)
    return vectors


# ── retrievers ───────────────────────────────────────────────────────────

def bm25_rankings(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    top_k: int,
    k1: float = BEIR_K1,
    b: float = BEIR_B,
    drop_stopwords: bool = False,
    stem: bool = False,
    titles: Mapping[str, str] | None = None,
    title_boost: int = 1,
    checkpoint: Path | None = None,
) -> dict[str, list[str]]:
    """Lexical ranking at the published BEIR BM25 settings.

    Query stopword filtering is off here: the engine drops stopwords because its queries
    are short natural-language questions, but the Anserini baseline does not, and turning
    it off is what makes this row comparable.

    ``title_boost`` repeats each document's title before its body, a crude BM25F field
    boost; 1 leaves the document exactly as BEIR concatenates it.

    ``checkpoint`` makes the pass resumable. The corpus build is the *cheap* part of this
    function — measured at about 4 s on 25,657 documents against ~85% of the time in the
    per-query scoring — so each query's ranking is appended to a JSONL file as it is
    produced and a resumed run skips the queries already there, repeating the build rather
    than checkpointing it. The file opens with a **settings fingerprint**: rankings from a
    different ``top_k``, a different corpus size or a different query set are ignored, so a
    stale file cannot silently mix into a report.
    """
    if titles and title_boost > 1:
        corpus = {
            doc_id: (titles.get(doc_id, "") + " ") * title_boost + text
            for doc_id, text in corpus.items()
        }
    index = BM25Index(k1=k1, b=b, drop_stopwords=drop_stopwords, stem=stem)
    index.build(make_chunks(corpus))

    def rank(query_id: str, text: str) -> list[str]:
        return [hit.chunk_id for hit in index.search(text, k=top_k)]

    if checkpoint is None:
        return {query_id: rank(query_id, text) for query_id, text in queries.items()}

    fingerprint = {
        "top_k": top_k, "k1": k1, "b": b, "drop_stopwords": drop_stopwords, "stem": stem,
        "documents": len(corpus),
    }
    done: dict[str, list[str]] = {}
    checkpoint = Path(checkpoint)
    reset = False
    if checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "fingerprint" in row:
                if row["fingerprint"] != fingerprint:
                    print("  lexical checkpoint was built with different settings; rewriting it",
                          flush=True)
                    done.clear()
                    reset = True
                    break
                continue
            if row.get("q") in queries:
                done[row["q"]] = row["ids"]
    if done:
        print(f"  resuming lexical ranking: {len(done)}/{len(queries)} queries checkpointed",
              flush=True)

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # A mismatched file is *rewritten*, not appended to: appending would leave two
    # conflicting fingerprints, and every later reader trusts only the first one.
    with checkpoint.open("w" if reset else "a", encoding="utf-8") as handle:
        if reset or not done:
            handle.write(json.dumps({"fingerprint": fingerprint}) + "\n")
        for query_id, text in queries.items():
            if query_id in done:
                continue
            done[query_id] = rank(query_id, text)
            handle.write(json.dumps({"q": query_id, "ids": done[query_id]}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return {query_id: done[query_id] for query_id in queries if query_id in done}


def passage_rankings(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    top_k: int,
    passage_chars: int = 600,
    overlap_chars: int = 100,
    k1: float = BEIR_K1,
    b: float = BEIR_B,
    drop_stopwords: bool = True,
    stem: bool = True,
) -> dict[str, list[str]]:
    """BM25 over passages, aggregated to documents by their best passage.

    Long documents diluted BM25's length normalisation and mix several topics into one
    vector, which is why passage-level retrieval plus max-pooling is standard on
    long-document collections. Documents here are whole abstracts, so this measures
    whether splitting them helps even at this size.
    """
    from laya_rag.chunking import chunk_text

    class _Passage:
        __slots__ = ("chunk_id", "text")

        def __init__(self, chunk_id: str, text: str) -> None:
            self.chunk_id = chunk_id
            self.text = text

    passages: list[_Passage] = []
    document_of: dict[str, str] = {}
    for doc_id, text in corpus.items():
        pieces = chunk_text(
            text, chunk_chars=passage_chars, overlap_chars=overlap_chars, min_chunk_chars=1
        )
        for position, (piece, _start, _end) in enumerate(pieces):
            passage_id = f"{doc_id}#{position}"
            passages.append(_Passage(passage_id, piece))
            document_of[passage_id] = doc_id

    index = BM25Index(k1=k1, b=b, drop_stopwords=drop_stopwords, stem=stem)
    index.build(passages)
    depth = max(top_k * 5, 50)
    rankings: dict[str, list[str]] = {}
    for query_id, text in queries.items():
        best: list[str] = []
        seen: set[str] = set()
        # Hits are score-ordered, so a document's first appearance is its best passage.
        for hit in index.search(text, k=depth):
            doc_id = document_of[hit.chunk_id]
            if doc_id not in seen:
                seen.add(doc_id)
                best.append(doc_id)
        rankings[query_id] = best[:top_k]
    return rankings


def bm25_rankings_prf(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    top_k: int,
    feedback_docs: int = 10,
    expansion_terms: int = 15,
    expansion_weight: int = 2,
    k1: float = BEIR_K1,
    b: float = BEIR_B,
    drop_stopwords: bool = True,
    stem: bool = True,
    titles: Mapping[str, str] | None = None,
    title_boost: int = 1,
) -> dict[str, list[str]]:
    """BM25 with pseudo-relevance feedback (an RM3-style expansion).

    Retrieve once, collect the most frequent terms from the top documents, and search
    again with them appended. It assumes the first pass is mostly right; when it is not,
    the expansion drifts the query, which is why it is measured rather than assumed.
    """
    if titles and title_boost > 1:
        corpus = {
            doc_id: (titles.get(doc_id, "") + " ") * title_boost + text
            for doc_id, text in corpus.items()
        }
    index = BM25Index(k1=k1, b=b, drop_stopwords=drop_stopwords, stem=stem)
    index.build(make_chunks(corpus))

    rankings: dict[str, list[str]] = {}
    for query_id, query in queries.items():
        first = index.search(query, k=feedback_docs)
        counts: Counter[str] = Counter()
        for hit in first:
            counts.update(
                term
                for term in tokenize(corpus[hit.chunk_id], stem=stem)
                if (not drop_stopwords or term not in STOPWORDS) and len(term) > 2
            )
        for term in set(tokenize(query, stem=stem)):
            counts.pop(term, None)
        expanded = query
        if counts:
            extra = [term for term, _ in counts.most_common(expansion_terms)]
            expanded = query + " " + " ".join(extra * max(1, expansion_weight))
        rankings[query_id] = [hit.chunk_id for hit in index.search(expanded, k=top_k)]
    return rankings


def dense_rankings(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    embedder: Any,
    vectors: np.ndarray,
    top_k: int,
    *,
    fixed_length: bool = False,
) -> dict[str, list[str]]:
    index = VectorIndex()
    index.add(make_chunks(corpus), vectors)
    query_vectors = np.asarray(
        embedder.embed_queries(list(queries.values()), fixed_length=fixed_length),
        dtype=np.float32,
    )
    rankings: dict[str, list[str]] = {}
    for (query_id, _), vector in zip(queries.items(), query_vectors):
        rankings[query_id] = [hit.chunk.chunk_id for hit in index.search(vector, k=top_k)]
    return rankings


def crossencoder_rankings(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    base: Mapping[str, Sequence[str]],
    cross_encoder: "CrossEncoder",
    depth: int,
    top_k: int,
    blend: float = 1.0,
    rrf_k: int = 60,
) -> tuple[dict[str, list[str]], float]:
    """Re-rank each query's top ``depth`` candidates with a cross-encoder.

    Returns the new ranking and the seconds spent scoring, so the cost of the stage can
    be reported separately from the recall that fed it.

    ``blend`` controls how far the cross-encoder may move the first stage. ``1.0``
    (default) replaces the order with the cross-encoder's, the historical behaviour.
    Below that, the first-stage order and the cross-encoder order are fused by RRF with
    weights ``[1 - blend, blend]``, so a second stage nudges rather than replaces. That
    matters here: a fully trusted pointwise cross-encoder lost nDCG@10 on all five BEIR
    corpora and catastrophically on ArguAna, where relevance is an opposition relation
    it does not model.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend!r}")
    spans: list[tuple[str, list[str]]] = []
    pairs: list[tuple[str, str]] = []
    for query_id, text in queries.items():
        docs = list(base.get(query_id, []))[:depth]
        spans.append((query_id, docs))
        pairs.extend((text, corpus[doc_id]) for doc_id in docs)

    started = time.perf_counter()
    scores = cross_encoder.score(pairs)
    elapsed = time.perf_counter() - started

    reranked: dict[str, list[str]] = {}
    cursor = 0
    for query_id, docs in spans:
        block = scores[cursor : cursor + len(docs)]
        cursor += len(docs)
        if blend >= 1.0:
            order = sorted(range(len(docs)), key=lambda index: -block[index])
        elif blend <= 0.0:
            order = list(range(len(docs)))
        else:
            position = {doc_id: index for index, doc_id in enumerate(docs)}
            ce_order = sorted(range(len(docs)), key=lambda index: -block[index])
            ce_ranked = [docs[index] for index in ce_order]
            merged = reciprocal_rank_fusion(
                [list(docs), ce_ranked],
                k=rrf_k,
                limit=len(docs),
                weights=[1.0 - blend, blend],
            )
            order = [position[doc_id] for doc_id, _ in merged]
        reranked[query_id] = [docs[index] for index in order[:top_k]]
    return reranked, elapsed


def multi_channel_fusion(
    channels: Sequence[Mapping[str, Sequence[str]]],
    weights: Sequence[float],
    top_k: int,
    rrf_k: int = 60,
) -> dict[str, list[str]]:
    """RRF over any number of channels, for ensembling retrievers."""
    fused: dict[str, list[str]] = {}
    for query_id in channels[0]:
        lists = [list(channel.get(query_id, []))[:top_k] for channel in channels]
        merged = reciprocal_rank_fusion(
            lists, k=rrf_k, limit=top_k, weights=list(weights)
        )
        fused[query_id] = [doc_id for doc_id, _ in merged]
    return fused


def score_fusion_rankings(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    embedder: Any,
    vectors: np.ndarray,
    top_k: int,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    k1: float = BEIR_K1,
    b: float = BEIR_B,
    stem: bool = True,
    titles: Mapping[str, str] | None = None,
    title_boost: int = 1,
) -> dict[str, list[str]]:
    """Score fusion (CombSUM) instead of rank fusion.

    RRF throws away score magnitudes; CombSUM keeps them but has to reconcile BM25's
    unbounded scores with cosine, so each channel is max-normalised per query first.
    Same channels, same weights, different fusion -- the comparison isolates the fusion
    method rather than the retrievers.
    """
    lexical_corpus = corpus
    if titles and title_boost > 1:
        lexical_corpus = {
            doc_id: (titles.get(doc_id, "") + " ") * title_boost + text
            for doc_id, text in corpus.items()
        }
    lexical = BM25Index(k1=k1, b=b, drop_stopwords=True, stem=stem)
    lexical.build(make_chunks(lexical_corpus))
    dense = VectorIndex()
    dense.add(make_chunks(corpus), vectors)
    query_vectors = np.asarray(embedder.embed_queries(list(queries.values())), dtype=np.float32)

    fused: dict[str, list[str]] = {}
    for (query_id, text), query_vector in zip(queries.items(), query_vectors):
        dense_hits = dense.search(query_vector, k=top_k)
        lexical_hits = lexical.search(text, k=top_k)
        dense_max = max((hit.score for hit in dense_hits), default=0.0) or 1.0
        lexical_max = max((hit.score for hit in lexical_hits), default=0.0) or 1.0
        scores: dict[str, float] = {}
        for hit in dense_hits:
            scores[hit.chunk.chunk_id] = (
                scores.get(hit.chunk.chunk_id, 0.0) + dense_weight * hit.score / dense_max
            )
        for hit in lexical_hits:
            scores[hit.chunk_id] = (
                scores.get(hit.chunk_id, 0.0) + lexical_weight * hit.score / lexical_max
            )
        fused[query_id] = [
            doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: -item[1])[:top_k]
        ]
    return fused


def hybrid_rankings(
    dense: Mapping[str, Sequence[str]],
    lexical: Mapping[str, Sequence[str]],
    top_k: int,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    rrf_k: int = 60,
) -> dict[str, list[str]]:
    """RRF over the two channels, the same weighted fusion ``RagEngine._fuse`` applies."""
    fused: dict[str, list[str]] = {}
    for query_id, dense_ids in dense.items():
        lexical_ids = lexical.get(query_id, [])
        merged = reciprocal_rank_fusion(
            [list(dense_ids)[:top_k], list(lexical_ids)[:top_k]],
            k=rrf_k,
            limit=top_k,
            weights=[dense_weight, lexical_weight],
        )
        fused[query_id] = [doc_id for doc_id, _ in merged]
    return fused


# ── reporting ────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="beir.py", description="Standard retrieval evaluation on BEIR")
    parser.add_argument("--datasets", default="scifact,nfcorpus")
    parser.add_argument("--retrievers",
                        default="bm25,dense,hybrid,biencoder,hybrid-biencoder,crossencoder")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=BEIR_K1)
    parser.add_argument("--b", type=float, default=BEIR_B)
    parser.add_argument("--biencoder-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--biencoder-pooling", default="auto",
                        help="pooling for the bi-encoder: auto (CLS for bge-family, mean "
                             "otherwise), cls, or mean — SPECTER-family checkpoints are CLS")
    parser.add_argument("--dataset-models", default="",
                        help="per-dataset bi-encoder overrides, 'name=model,...' — e.g. "
                             "scidocs=allenai/specter2_base. Lets a domain encoder be routed "
                             "to the corpus whose relevance relation it was trained on")
    parser.add_argument("--dense-text-only-datasets", default="",
                        help="comma list of datasets whose dense channel embeds text without "
                             "titles (the global --dense-text-only applies to every dataset)")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out", default=str(ROOT / "eval" / "beir" / "report"))
    parser.add_argument("--no-download", action="store_true", help="fail instead of fetching a dataset")
    parser.add_argument("--no-cache", action="store_true", help="re-embed even if a cache exists")
    parser.add_argument("--lexical-checkpoint", action="store_true",
                        help="checkpoint the engine lexical ranking per query at <out>.lexical.jsonl "
                             "so an interrupted lexical pass resumes instead of restarting")
    parser.add_argument("--embed-chunk", type=int, default=0,
                        help="checkpoint the corpus embedding every N documents so an interrupted "
                             "run resumes instead of restarting; 0 keeps the single-pass behaviour")
    parser.add_argument("--limit-docs", type=int, default=None, help="truncate the corpus (smoke runs only)")
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="score a seeded random subset of the queries; for corpora whose full "
                             "query set costs hours. Recorded in the report so a subset can never be "
                             "mistaken for the full run")
    parser.add_argument("--query-seed", type=int, default=17,
                        help="seed for --limit-queries; the same seed scores the same queries")
    parser.add_argument("--dense-weight", type=float, default=None,
                        help="dense RRF weight for the hybrid row (default: the engine's resolved value)")
    parser.add_argument("--biencoder-weight", type=float, default=1.0,
                        help="bi-encoder RRF weight for the hybrid-biencoder row")
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="cross-encoder used by the crossencoder row")
    parser.add_argument("--rerank-depth", type=int, default=50,
                        help="how many candidates the cross-encoder re-ranks")
    parser.add_argument("--rerank-blend", type=float, default=1.0,
                        help="how far the cross-encoder may move the first stage, in [0, 1]; "
                             "1.0 replaces the order (historical), below it is RRF-fused with the base")
    parser.add_argument("--no-stem", dest="stem", action="store_false",
                        help="disable the engine's English stemming for its lexical path")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant for fusion rows")
    parser.add_argument("--title-boost", type=int, default=1,
                        help="repeat each title N times before the body in a bm25-title row")
    parser.add_argument("--passages", default="",
                        help="comma list of passage sizes in characters for bm25-passages rows")
    parser.add_argument("--prf", type=int, default=0,
                        help="feedback documents for a bm25-prf row (0 = off)")
    parser.add_argument("--prf-terms", type=int, default=15, help="expansion terms for bm25-prf")
    parser.add_argument("--prf-weight", type=int, default=2, help="times each expansion term is repeated")
    parser.add_argument("--combsum", action="store_true",
                        help="also report a score-fused (CombSUM) row alongside RRF")
    parser.add_argument("--no-titles", action="store_true",
                        help="index document text only, excluding the title field")
    parser.add_argument("--dense-text-only", action="store_true",
                        help="embed document text without the title, while BM25 keeps title+text")
    parser.add_argument("--query-instruction", action="store_true",
                        help="apply the bi-encoder's retrieval instruction to queries")
    parser.add_argument("--ensemble-model", default=None,
                        help="second bi-encoder to fuse as a hybrid-ensemble row")
    parser.add_argument("--ensemble-models", default=None,
                        help="comma list of extra bi-encoders; adds dense-ensemble (dense models "
                             "only) and hybrid-ensemble (dense models + lexical) rows")
    parser.add_argument("--splade-model", default=None,
                        help="learned-sparse encoder for a splade / hybrid-splade row")
    parser.add_argument("--keep-identical-ids", action="store_true",
                        help="score a retrieved document whose id equals the query id; BEIR "
                             "drops it (ArguAna only), so this makes the row non-comparable")
    parser.add_argument("--baseline", default="bm25-engine",
                        help="retriever every other row is significance-tested against")
    parser.add_argument("--bootstrap-samples", type=int, default=10000,
                        help="paired-bootstrap resamples for the significance table")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def _parse_pairs(spec: str) -> dict[str, str]:
    """Parse a ``name=value,name=value`` CLI spec into a dict."""
    out: dict[str, str] = {}
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected name=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    wanted = [name.strip() for name in args.retrievers.split(",") if name.strip()]
    dataset_models = _parse_pairs(args.dataset_models)
    dense_text_only_names = {
        name.strip() for name in args.dense_text_only_datasets.split(",") if name.strip()
    }
    config = EngineConfig()
    if args.device:
        config.device = args.device
    dense_weight = (
        config.resolved_dense_weight if args.dense_weight is None else args.dense_weight
    )

    laya: LayaEmbedder | None = None
    biencoders: dict[str, BiEncoder] = {}

    def active_biencoder(dataset: str) -> BiEncoder:
        """The bi-encoder for this dataset: a per-dataset override, else the global one."""
        model_id = dataset_models.get(dataset, args.biencoder_model)
        if model_id not in biencoders:
            biencoders[model_id] = BiEncoder(
                model_id, device=args.device, query_instruction=args.query_instruction,
                pooling=args.biencoder_pooling,
            )
        return biencoders[model_id]
    report: dict[str, Any] = {
        "settings": {
            "top_k": args.top_k,
            "bm25_k1": args.k1,
            "bm25_b": args.b,
            "drop_stopwords": True,
            "stem": args.stem,
            "rrf_k": args.rrf_k,
            "title_boost": args.title_boost,
            "passages": args.passages,
            "prf": args.prf,
            "prf_terms": args.prf_terms,
            "prf_weight": args.prf_weight,
            "combsum": args.combsum,
            "include_titles": not args.no_titles,
            "dense_text_only": args.dense_text_only,
            "query_instruction": args.query_instruction,
            "ensemble_model": args.ensemble_model,
            "ensemble_models": args.ensemble_models,
            "splade_model": args.splade_model,
            "dense_weight": dense_weight,
            "lexical_weight": config.lexical_weight,
            "checkpoint": config.checkpoint,
            "biencoder_model": args.biencoder_model if "biencoder" in wanted else None,
            "biencoder_pooling": args.biencoder_pooling,
            "dataset_models": dataset_models,
            "dense_text_only_datasets": sorted(dense_text_only_names),
            "rerank_model": args.rerank_model if "crossencoder" in wanted else None,
            "rerank_depth": args.rerank_depth,
            "limit_docs": args.limit_docs,
            "ignore_identical_ids": not args.keep_identical_ids,
            "baseline": args.baseline,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "references": REFERENCES,
        "anchors": {},
        "datasets": {},
    }

    for name in dataset_names:
        print(f"[{name}]", flush=True)
        dataset_dir = ensure_dataset(name, data_root, download=not args.no_download)
        corpus = load_corpus(dataset_dir, include_titles=not args.no_titles)
        # The two channels can see different document text: titles are strong lexical
        # signal but generic topic keywords (ArguAna) flatten an embedding. The choice can
        # be global or per dataset, because the effect is corpus-specific.
        per_dataset_text_only = args.dense_text_only or name in dense_text_only_names
        dense_corpus = (
            load_corpus(dataset_dir, include_titles=False)
            if per_dataset_text_only
            else corpus
        )
        dense_suffix = ".notitles" if (args.no_titles or per_dataset_text_only) else ""
        if args.limit_docs:
            corpus = dict(list(corpus.items())[: args.limit_docs])
        all_queries = load_queries(dataset_dir)
        titles = load_titles(dataset_dir)
        qrels = load_qrels(dataset_dir)
        # Only rank the queries that have qrels, which is what BEIR scores.
        queries = {qid: text for qid, text in all_queries.items() if qid in qrels}
        if args.limit_queries and len(queries) > args.limit_queries:
            total = len(queries)
            queries = sample_queries(queries, args.limit_queries, args.query_seed)
            # `evaluate_rankings` macro-averages over every query in the qrels it is handed, so
            # passing the full qrels alongside a partial ranking set scores every unsampled query as
            # zero and dilutes the mean -- SciFact read 0.2047 instead of 0.6762 before this line.
            # The restriction is applied only here, so a full run's qrels are untouched.
            qrels = restrict_qrels(qrels, queries)
            print(f"  SUBSET RUN: {len(queries)} of {total} qrels queries, seed {args.query_seed} "
                  f"-- not comparable to a full-set published figure", flush=True)
        print(f"  {len(corpus)} documents, {len(queries)} scored queries", flush=True)

        cache_dir = data_root / "cache"
        results: dict[str, Any] = {}
        rankings: dict[str, dict[str, list[str]]] = {}
        lexical: dict[str, list[str]] | None = None

        if "bm25" in wanted:
            started = time.perf_counter()
            # The control row is checkpointed too. It runs at *published* settings, so it does not
            # stem and its vocabulary is far larger: on NQ that made it a three-hour row with no
            # way to resume, which is exactly the waste the engine row's checkpoint exists to avoid.
            rankings["bm25"] = bm25_rankings(
                corpus, queries, args.top_k, k1=args.k1, b=args.b,
                checkpoint=(Path(str(args.out) + ".lexical-control.jsonl")
                            if args.lexical_checkpoint else None),
            )
            results["bm25"] = {"seconds": round(time.perf_counter() - started, 2)}

        # The engine's own lexical channel: query stopwords dropped, English stemming on.
        # It is reported as its own row and feeds every fused and reranked row, while the
        # `bm25` row above stays at the published settings so it remains a valid control.
        lexical = bm25_rankings(
            corpus, queries, args.top_k, k1=args.k1, b=args.b,
            drop_stopwords=True, stem=args.stem,
            checkpoint=(Path(str(args.out) + ".lexical.jsonl")
                        if args.lexical_checkpoint else None),
        )
        rankings["bm25-engine"] = lexical
        results["bm25-engine"] = {
            "note": f"engine settings: stopwords dropped, stem={args.stem}"
        }

        # Title-weighted BM25F-style row, and passage-level retrieval for long documents.
        if args.title_boost > 1:
            title_rankings = bm25_rankings(
                corpus, queries, args.top_k, k1=args.k1, b=args.b,
                drop_stopwords=True, stem=args.stem,
                titles=titles, title_boost=args.title_boost,
            )
            rankings["bm25-title"] = title_rankings
            results["bm25-title"] = {"note": f"title repeated {args.title_boost}x"}
            # Fuse the title-boosted channel: it measured better on both datasets, and
            # everything downstream inherits it.
            lexical = title_rankings

        if args.prf > 0:
            prf_ranking = bm25_rankings_prf(
                corpus, queries, args.top_k, feedback_docs=args.prf,
                expansion_terms=args.prf_terms, expansion_weight=args.prf_weight,
                k1=args.k1, b=args.b, drop_stopwords=True, stem=args.stem,
                titles=titles, title_boost=args.title_boost,
            )
            rankings["bm25-prf"] = prf_ranking
            results["bm25-prf"] = {
                "note": f"{args.prf} feedback docs, {args.prf_terms} terms x{args.prf_weight}"
            }

        for passage_size in [int(v) for v in str(args.passages).split(",") if v.strip()]:
            passage_ranking = passage_rankings(
                corpus, queries, args.top_k, passage_chars=passage_size,
                k1=args.k1, b=args.b, drop_stopwords=True, stem=args.stem,
            )
            label = f"bm25-passages@{passage_size}"
            rankings[label] = passage_ranking
            results[label] = {"note": f"{passage_size}-char passages, max-pooled"}

        dense: dict[str, list[str]] | None = None
        if "dense" in wanted or "hybrid" in wanted:
            if laya is None:
                laya = LayaEmbedder(config)
            vectors = cached_embed(
                laya, dense_corpus, cache_dir / f"{name}.{laya.name}{dense_suffix}",
                use_cache=not args.no_cache,
                    chunk=args.embed_chunk,
            )
            started = time.perf_counter()
            dense = dense_rankings(dense_corpus, queries, laya, vectors, args.top_k)
            seconds = round(time.perf_counter() - started, 2)
            if "dense" in wanted:
                rankings["dense"] = dense
                results["dense"] = {"seconds": seconds, "model": laya.name}
            if "hybrid" in wanted:
                rankings["hybrid"] = hybrid_rankings(
                    dense, lexical or {}, args.top_k,
                    dense_weight=dense_weight, lexical_weight=config.lexical_weight,
                    rrf_k=args.rrf_k,
                )
                results["hybrid"] = {
                    "seconds": seconds,
                    "note": f"RRF of dense (w={dense_weight}) + bm25 (w={config.lexical_weight})",
                }

        if "biencoder" in wanted or "hybrid-biencoder" in wanted:
            try:
                encoder = active_biencoder(name)
                vectors = cached_embed(
                    encoder,
                    dense_corpus,
                    cache_dir / f"{name}.{encoder.name}{dense_suffix}",
                    use_cache=not args.no_cache,
                    chunk=args.embed_chunk,
                )
                started = time.perf_counter()
                bi = dense_rankings(dense_corpus, queries, encoder, vectors, args.top_k)
                seconds = round(time.perf_counter() - started, 2)
                if "biencoder" in wanted:
                    rankings["biencoder"] = bi
                    results["biencoder"] = {"seconds": seconds, "model": encoder.name}
                if args.combsum:
                    rankings["hybrid-combsum"] = score_fusion_rankings(
                        corpus, queries, encoder, vectors, args.top_k,
                        dense_weight=args.biencoder_weight,
                        lexical_weight=config.lexical_weight,
                        k1=args.k1, b=args.b, stem=args.stem,
                        titles=titles, title_boost=args.title_boost,
                    )
                    results["hybrid-combsum"] = {"note": "CombSUM, max-normalised per query"}

                if "hybrid-biencoder" in wanted:
                    rankings["hybrid-biencoder"] = hybrid_rankings(
                        bi, lexical or {}, args.top_k,
                        dense_weight=args.biencoder_weight, lexical_weight=config.lexical_weight,
                        rrf_k=args.rrf_k,
                    )
                    results["hybrid-biencoder"] = {
                        "seconds": seconds,
                        "note": f"RRF of biencoder (w={args.biencoder_weight}) + bm25",
                    }
            except Exception as exc:  # noqa: BLE001 - a missing model must not lose the run
                for retriever in ("biencoder", "hybrid-biencoder"):
                    if retriever in wanted:
                        results[retriever] = {"error": f"{type(exc).__name__}: {exc}"}

        ensemble_spec = args.ensemble_models or args.ensemble_model
        if ensemble_spec:
            try:
                extra_models = [m.strip() for m in str(ensemble_spec).split(",") if m.strip()]
                base_encoder = active_biencoder(name)
                if "biencoder" not in rankings:
                    base_vectors = cached_embed(
                        base_encoder, dense_corpus,
                        cache_dir / f"{name}.{base_encoder.name}{dense_suffix}",
                        use_cache=not args.no_cache,
                    chunk=args.embed_chunk,
                    )
                    rankings["biencoder"] = dense_rankings(
                        dense_corpus, queries, base_encoder, base_vectors, args.top_k
                    )
                dense_channels = [rankings["biencoder"]]
                for model_id in extra_models:
                    extra = BiEncoder(
                        model_id, device=args.device, query_instruction=args.query_instruction,
                        pooling=args.biencoder_pooling,
                    )
                    extra_vectors = cached_embed(
                        extra, dense_corpus,
                        cache_dir / f"{name}.{extra.name}{dense_suffix}",
                        use_cache=not args.no_cache,
                    chunk=args.embed_chunk,
                    )
                    dense_channels.append(
                        dense_rankings(dense_corpus, queries, extra, extra_vectors, args.top_k)
                    )
                started = time.perf_counter()
                # Pure dense ensembling, and the same ensemble fused with the lexical channel:
                # the first isolates the multi-model dense gain, the second is the deployable row.
                rankings["dense-ensemble"] = multi_channel_fusion(
                    dense_channels, [1.0] * len(dense_channels), args.top_k, args.rrf_k
                )
                rankings["hybrid-ensemble"] = multi_channel_fusion(
                    dense_channels + [lexical or {}],
                    [1.0] * len(dense_channels) + [config.lexical_weight],
                    args.top_k,
                    args.rrf_k,
                )
                note = f"RRF of {base_encoder.name}, {', '.join(extra_models)} (dense only, and +bm25)"
                results["dense-ensemble"] = {"note": note}
                results["hybrid-ensemble"] = {
                    "seconds": round(time.perf_counter() - started, 2),
                    "note": note,
                }
            except Exception as exc:  # noqa: BLE001 - a missing model must not lose the run
                results["hybrid-ensemble"] = {"error": f"{type(exc).__name__}: {exc}"}

        if args.splade_model:
            try:
                import sparse as sparse_mod

                encoder = sparse_mod.SpladeEncoder(args.splade_model, device=args.device)
                short = args.splade_model.rsplit("/", 1)[-1]
                doc_vectors = sparse_mod.load_or_encode(
                    encoder,
                    [corpus[doc_id] for doc_id in corpus],
                    cache_dir / f"{name}.{short}.sparse",
                    use_cache=not args.no_cache,
                    chunk=args.embed_chunk,
                )
                sparse_ranking, seconds = sparse_mod.sparse_rankings(
                    corpus, queries, encoder, args.top_k, doc_vectors=doc_vectors
                )
                rankings["splade"] = sparse_ranking
                results["splade"] = {"seconds": round(seconds, 1), "model": args.splade_model}
                if "biencoder" in rankings and lexical:
                    rankings["hybrid-splade"] = multi_channel_fusion(
                        [rankings["biencoder"], lexical, sparse_ranking],
                        [1.0, config.lexical_weight, 1.0],
                        args.top_k,
                        args.rrf_k,
                    )
                    results["hybrid-splade"] = {
                        "note": "RRF of biencoder + bm25 + splade",
                        "seconds": round(seconds, 1),
                    }
            except Exception as exc:  # noqa: BLE001 - a missing model must not lose the run
                results["splade"] = {"error": f"{type(exc).__name__}: {exc}"}

        if "crossencoder" in wanted:
            try:
                cross_encoder = CrossEncoder(args.rerank_model, device=args.device)
                base = (
                    rankings.get("hybrid-biencoder")
                    or rankings.get("bm25")
                    or bm25_rankings(corpus, queries, args.top_k, k1=args.k1, b=args.b)
                )
                reranked, seconds = crossencoder_rankings(
                    corpus, queries, base, cross_encoder, args.rerank_depth, args.top_k,
                    blend=args.rerank_blend,
                )
                rankings["crossencoder"] = reranked
                results["crossencoder"] = {
                    "seconds": round(seconds, 2),
                    "model": cross_encoder.name,
                    "blend": args.rerank_blend,
                    "note": f"{args.rerank_model} over the best available first stage "
                            f"top-{args.rerank_depth}"
                            + ("" if args.rerank_blend >= 1.0
                               else f", blended with the first stage at {args.rerank_blend}"),
                }
            except Exception as exc:  # noqa: BLE001 - a missing model must not lose the run
                results["crossencoder"] = {"error": f"{type(exc).__name__}: {exc}"}

        ignore_identical = not args.keep_identical_ids
        per_query: dict[str, dict[str, float]] = {}
        for retriever, ranking in rankings.items():
            evaluation = evalkit.evaluate_rankings(
                ranking,
                qrels,
                k_values=(10, 100),
                ignore_identical_ids=ignore_identical,
            )
            results[retriever].update(evaluation["summary"])
            results[retriever]["scored_queries"] = evaluation["queries"]
            per_query[retriever] = evalkit.per_query_metric(evaluation["per_query"], "nDCG@10")

        # Where the control row lands against the published baseline. A large gap here
        # is the evaluator's problem rather than the retriever's, so it is reported
        # rather than left to whoever reads the table.
        control = results.get("bm25", {})
        anchor = (
            evalkit.anchor_deviations(
                {name: control["nDCG@10"]}, PUBLISHED_BM25, tolerance=ANCHOR_TOLERANCE
            )
            if "nDCG@10" in control
            else []
        )
        report["anchors"][name] = anchor[0] if anchor else None

        # Significance against the baseline row, scored on the same queries, so the
        # intervals reflect query difficulty rather than run-to-run variation.
        comparisons: dict[str, Any] = {}
        baseline_scores = per_query.get(args.baseline)
        if baseline_scores:
            for retriever, scores in per_query.items():
                if retriever == args.baseline:
                    continue
                try:
                    comparisons[retriever] = evalkit.paired_bootstrap(
                        baseline_scores, scores, samples=args.bootstrap_samples
                    )
                except ValueError:  # no query scored by both, e.g. a failed row
                    continue

        report["datasets"][name] = {
            "documents": len(corpus),
            "queries": len(all_queries),
            "scored_queries": len(queries),
            "retrievers": results,
            "per_query_nDCG@10": per_query,
            "comparisons_vs": args.baseline,
            "comparisons": comparisons,
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
        seconds = row["seconds"]
        secs = f"{seconds:>7}" if isinstance(seconds, (int, float)) else " " * 7
        lines.append(f"{row['dataset']:<10} {row['retriever']:<12} {cells} {secs}")
    return "\n".join(lines)


def _anchor_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, anchor in report.get("anchors", {}).items():
        if anchor:
            rows.append({
                "dataset": name,
                "bm25 control": anchor["measured"],
                "published BM25": anchor["published"],
                "delta": anchor["delta"],
                "within tolerance": "yes" if anchor["within_tolerance"] else "NO",
            })
    return rows


def _comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, dataset in report["datasets"].items():
        for retriever, comparison in dataset.get("comparisons", {}).items():
            rows.append({
                "dataset": name,
                "retriever": retriever,
                "vs": dataset.get("comparisons_vs", ""),
                "delta nDCG@10": comparison["delta"],
                "95% CI": f"[{comparison['ci_low']}, {comparison['ci_high']}]",
                "p": comparison["p_value"],
                "significant": "yes" if comparison["significant"] else "no",
            })
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    settings = report["settings"]
    identical = settings.get("ignore_identical_ids", True)
    parts = [
        "# BEIR retrieval evaluation",
        "",
        f"top_k={settings['top_k']}, BM25 k1={settings['bm25_k1']} b={settings['bm25_b']}, "
        f"query stopwords kept, checkpoint `{settings['checkpoint']}`.",
        f"`ignore_identical_ids={identical}` — a retrieved document whose id equals the "
        "query id is dropped, BEIR's own default.",
        "",
        "`nDCG@10` is the number the BEIR leaderboard reports; the rest are supporting metrics.",
        "All rows share corpus, qrels, evaluator and top-k, so the difference between rows is "
        "the retriever.",
        "",
        evalkit.markdown_table(_rows(report)),
        "",
    ]

    anchors = _anchor_rows(report)
    if anchors:
        parts += [
            "## Evaluator anchor",
            "",
            "The `bm25` control against the published baseline. A deviation beyond "
            f"{ANCHOR_TOLERANCE} means the harness, not the retriever, needs a look.",
            "",
            evalkit.markdown_table(anchors),
            "",
        ]

    comparisons = _comparison_rows(report)
    if comparisons:
        parts += [
            "## Significance",
            "",
            f"Paired bootstrap over queries, each row against `{settings.get('baseline')}`. "
            f"`delta nDCG@10` with a {settings.get('bootstrap_samples')}-resample 95% interval; "
            "the difference is only evidence when the interval excludes zero.",
            "",
            evalkit.markdown_table(comparisons),
            "",
        ]

    parts += [
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
