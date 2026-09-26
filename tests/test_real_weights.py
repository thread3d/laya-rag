"""End-to-end tests against the real Laya checkpoints.

These are the tests that can actually fail on retrieval quality, so they assert on
behaviour that was measured on this machine rather than assumed:

* Laya's decision head separates a relevant passage from irrelevant ones far more
  sharply than the encoder's cosine similarity does (0.79 against 0.001 on the probe
  below, where cosine was 0.357 against 0.284);
* the answerability decision collapses when the evidence does not address the query;
* the whole index -> query path returns the right file.

They skip unless the checkpoints are present, so the offline suite still runs.
"""
from __future__ import annotations

import pytest

from laya_rag.config import EngineConfig
from laya_rag.engine import RagEngine
from laya_rag.model import LayaModel
from laya_rag.rerank import LayaReranker

from .support import make_config

pytestmark = pytest.mark.real_weights

PROBE_DOCS = [
    "Refunds are processed within five business days to the original payment method.",
    "Shipping is free on orders over fifty dollars in the continental United States.",
    "To reset your password, open Settings and choose Security, then Reset Password.",
    "Our API rate limit is 600 requests per minute per API key.",
]
PROBE_QUERY = "How long does a refund take?"


def checkpoints_available() -> bool:
    return (EngineConfig().models_dir / "laya").is_dir()


requires_checkpoints = pytest.mark.skipif(
    not checkpoints_available(),
    reason="Laya checkpoints are not downloaded; run vendor/laya/setup_laya.sh",
)


@pytest.fixture(scope="module")
def model() -> LayaModel:
    instance = LayaModel(EngineConfig())
    instance.ensure_loaded()
    return instance


@requires_checkpoints
def test_checkpoint_loads_and_embeds(model):
    vectors = model.embed(PROBE_DOCS)
    assert vectors.shape == (len(PROBE_DOCS), model.dim)
    assert model.dim > 0
    assert vectors.dtype.name == "float32"


@requires_checkpoints
def test_laya_relevance_beats_dense_cosine_on_a_probe(model):
    """The reason Laya is in this pipeline at all: calibrated, sharp separation."""
    import numpy as np

    vectors = model.embed(PROBE_DOCS)
    query = model.embed_one(PROBE_QUERY)
    normalised = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = normalised @ (query / np.linalg.norm(query))

    reranker = LayaReranker(model)
    relevance, usage = reranker.score_relevance(PROBE_QUERY, PROBE_DOCS)

    best_cosine, best_relevance = int(cosine.argmax()), int(np.argmax(relevance))
    assert best_cosine == 0, "the refund passage should lead on cosine too"
    assert best_relevance == 0, "Laya must rank the refund passage first"

    # Laya should separate the correct passage by a wide margin...
    others = [relevance[i] for i in range(len(PROBE_DOCS)) if i != 0]
    assert relevance[0] > max(others) + 0.3, f"weak separation: {relevance}"
    # ...where cosine similarity does not.
    assert cosine[0] - max(cosine[1:]) < 0.2, f"cosine unexpectedly sharp: {cosine}"
    assert usage.get("input_tokens", 0) > 0


@requires_checkpoints
def test_answerability_gates_unrelated_evidence(model):
    from laya_rag.rerank import ScoredPassage

    reranker = LayaReranker(model)

    def passage(text: str) -> ScoredPassage:
        return ScoredPassage(
            text=text,
            path="/probe.md",
            chunk_id="probe",
            recall_score=1.0,
            relevance=1.0,
            index=0,
            start_char=0,
            end_char=len(text),
        )

    supported, _ = reranker.check_answerability(PROBE_QUERY, [passage(PROBE_DOCS[0])])
    unsupported, _ = reranker.check_answerability(
        "What is the chief executive's home address?", [passage(PROBE_DOCS[0])]
    )

    assert supported > 0.5, f"relevant evidence judged unanswerable ({supported})"
    assert unsupported < 0.5, f"unrelated question judged answerable ({unsupported})"
    assert supported > unsupported


@requires_checkpoints
def test_full_pipeline_retrieves_the_right_file(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, body in (
        ("langchain.md", "Laya integrates with LangChain through the LayaRouter and LayaGuardrail wrappers."),
        ("docker.md", "Build the Laya image with docker build and run it with docker compose up."),
        ("structured.md", "Structured output uses Pydantic models to validate the decoded decisions."),
    ):
        (corpus / name).write_text(f"# {name[:-3]}\n\n{body}\n", encoding="utf-8")

    engine = RagEngine(make_config(tmp_path, models_dir=EngineConfig().models_dir))
    engine.index_paths([corpus])

    result = engine.query("How do I run Laya in Docker?")

    assert result.answerable is True
    assert result.passages
    assert "docker.md" in result.passages[0]["path"]
    assert result.passages[0]["relevance"] > 0.5
