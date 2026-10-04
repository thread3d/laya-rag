"""Offline tests for resumable corpus embedding in the BEIR harness.

A corpus embedding is the long part of a dataset evaluation, and `cached_embed` used to write
only at the very end — so a run stopped part-way lost everything, which is how an hour of the
touche evaluation was lost. The checkpointing is therefore only worth having if it is *correct*:
a resumed run must produce the same vectors as an uninterrupted one, and a torn chunk must never
be trusted.

The embedder is a stand-in, so none of this needs weights.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_beir"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beir = _load()


class FakeEmbedder:
    """Embeds deterministically from the text, and can be made to fail on a given chunk."""

    name = "fake"

    def __init__(self, fail_on_call: int | None = None):
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def embed(self, texts):
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("interrupted")
        self.calls.append(list(texts))
        return np.asarray([[len(t), sum(map(ord, t)) % 97] for t in texts], dtype=np.float32)


def _corpus(n: int) -> dict[str, str]:
    return {f"d{i}": f"text number {i}" for i in range(n)}


def test_unchunked_embedding_is_unchanged_and_reuses_its_cache(tmp_path):
    corpus = _corpus(10)
    cache = tmp_path / "c"
    first = beir.cached_embed(FakeEmbedder(), corpus, cache, chunk=0)
    again = FakeEmbedder()
    second = beir.cached_embed(again, corpus, cache, chunk=0)
    assert np.array_equal(first, second)
    assert again.calls == [], "a matching cache must be reused, not recomputed"


def test_chunked_embedding_matches_an_unchunked_pass(tmp_path):
    """A resumed/chunked pass must produce the same vectors as one uninterrupted call."""
    corpus = _corpus(25)
    whole = beir.cached_embed(FakeEmbedder(), corpus, tmp_path / "whole", chunk=0)
    chunked = beir.cached_embed(FakeEmbedder(), corpus, tmp_path / "chunked", chunk=10)
    assert np.array_equal(whole, chunked)


def test_an_interrupted_pass_resumes_instead_of_restarting(tmp_path):
    """The point of the change: the work already done is not thrown away."""
    corpus = _corpus(25)
    cache = tmp_path / "c"
    with pytest.raises(RuntimeError):
        beir.cached_embed(FakeEmbedder(fail_on_call=2), corpus, cache, chunk=10)

    resumed = FakeEmbedder()
    vectors = beir.cached_embed(resumed, corpus, cache, chunk=10)
    assert len(resumed.calls) == 1, "the two completed chunks should be reused"
    assert vectors.shape[0] == 25
    assert np.array_equal(vectors, beir.cached_embed(FakeEmbedder(), corpus, tmp_path / "ref",
                                                     chunk=0))


def test_a_torn_chunk_is_not_trusted(tmp_path):
    """A part without its `.done` marker must be re-embedded, not concatenated."""
    corpus = _corpus(20)
    cache = tmp_path / "c"
    with pytest.raises(RuntimeError):
        beir.cached_embed(FakeEmbedder(fail_on_call=1), corpus, cache, chunk=10)
    torn = sorted((Path(str(cache) + ".parts")).glob("part-*.npy"))
    torn[0].write_bytes(b"not a numpy array")          # simulate a truncated write
    vectors = beir.cached_embed(FakeEmbedder(), corpus, cache, chunk=10)
    assert vectors.shape[0] == 20


def test_the_chunk_size_is_part_of_the_cache_identity(tmp_path):
    """Chunk boundaries move embeddings by ULPs, so caches must not be mixed silently."""
    corpus = _corpus(12)
    cache = tmp_path / "c"
    beir.cached_embed(FakeEmbedder(), corpus, cache, chunk=0)
    rebuilt = FakeEmbedder()
    beir.cached_embed(rebuilt, corpus, cache, chunk=6)
    assert rebuilt.calls, "an unchunked cache must not be reused for a chunked request"
    assert json.loads(Path(str(cache) + ".meta.json").read_text())["chunk"] == 6


def test_a_cache_from_before_this_option_is_still_reused(tmp_path):
    """chunk=0 must not invalidate the five datasets already embedded."""
    corpus = _corpus(8)
    cache = tmp_path / "c"
    vectors = beir.cached_embed(FakeEmbedder(), corpus, cache, chunk=0)
    Path(str(cache) + ".meta.json").unlink()            # as an older cache would be
    again = FakeEmbedder()
    assert np.array_equal(beir.cached_embed(again, corpus, cache, chunk=0), vectors)
    assert again.calls == []


def test_a_changed_document_order_re_embeds(tmp_path):
    corpus = _corpus(6)
    cache = tmp_path / "c"
    beir.cached_embed(FakeEmbedder(), corpus, cache, chunk=3)
    reordered = {k: corpus[k] for k in reversed(list(corpus))}
    again = FakeEmbedder()
    beir.cached_embed(again, corpus if False else reordered, cache, chunk=3)
    assert again.calls, "the cache is keyed by document id order"


def test_parts_are_cleaned_up_after_a_successful_pass(tmp_path):
    cache = tmp_path / "c"
    beir.cached_embed(FakeEmbedder(), _corpus(15), cache, chunk=5)
    assert not Path(str(cache) + ".parts").exists()
