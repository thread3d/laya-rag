"""Offline tests for the Laya encoder fine-tuner's data and checkpoint helpers.

The training loop itself needs a checkpoint and MPS, so these cover the parts that decide
*what* is trained — the pair builder, the streaming positive loader, the per-dataset caps and
the resume fingerprint — which is where a silent mistake is most expensive (a wrong pair set
looks like a training result). The trainer's `--init-from`, `LambdaLR` and export paths were
verified against real checkpoints in `docs/findings-encoder-adaptation.md`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import laya_encoder_finetune as lef  # noqa: E402


def test_parse_dataset_caps_reads_and_rejects():
    assert lef.parse_dataset_caps("") == {}
    assert lef.parse_dataset_caps("msmarco:150000,hotpotqa:40000") == {
        "msmarco": 150000,
        "hotpotqa": 40000,
    }
    assert lef.parse_dataset_caps(" nfcorpus:5 , ") == {"nfcorpus": 5}
    with pytest.raises(SystemExit):
        lef.parse_dataset_caps("msmarco")
    with pytest.raises(SystemExit):
        lef.parse_dataset_caps("msmarco:notanumber")


def test_gold_pairs_drops_nonpositive_and_missing_documents():
    queries = {"q1": "Q1", "q2": "Q2"}
    qrels = {"q1": {"d1": 1.0, "d2": 0.0}, "q2": {"d9": 1.0}}
    corpus = {"d1": "D1", "d2": "D2"}  # d9 is judged but absent from the corpus
    assert lef.gold_pairs(queries, qrels, corpus, 0) == [("Q1", "D1")]


def test_positive_texts_streams_only_needed_ids(tmp_path):
    rows = [
        {"_id": "a", "title": "T", "text": "body a"},
        {"_id": "b", "title": "", "text": "body b"},
        {"_id": "c", "title": "U", "text": "body c"},
    ]
    (tmp_path / "corpus.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    assert lef.positive_texts(tmp_path, {"a", "c"}) == {"a": "T body a", "c": "U body c"}
    # `include_titles=False` is the dense-text-only representation.
    assert lef.positive_texts(tmp_path, {"a"}, include_titles=False) == {"a": "body a"}
    assert lef.positive_texts(tmp_path, set()) == {}


def _args(**overrides):
    args = lef.parse_args(["--datasets", "nfcorpus", "--lr", "2e-5"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_fingerprint_id_is_stable_and_setting_sensitive():
    first = lef.fingerprint(_args())
    again = lef.fingerprint(_args())
    assert lef.fingerprint_id(first) == lef.fingerprint_id(again)
    changed = lef.fingerprint_id(lef.fingerprint(_args(lr=1e-5)))
    assert changed != lef.fingerprint_id(first)
    # A new dataset set must change the fingerprint so a stale state cannot resume.
    other = lef.fingerprint_id(lef.fingerprint(_args(datasets="scifact")))
    assert other != lef.fingerprint_id(first)


def test_fingerprint_covers_staged_training_inputs():
    keys = set(lef.fingerprint(_args()))
    assert {"init_from", "dataset_caps", "max_pairs", "teacher_models"} <= keys


def test_mean_pool_ignores_padding():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = lef.mean_pool(hidden, mask)
    assert pooled.tolist() == [[2.0, 2.0]]


def test_false_negative_blocks_marks_other_positives_of_the_same_query():
    torch = pytest.importorskip("torch")
    queries = ["q1", "q1", "q2"]
    documents = ["d1", "d2", "d3"]
    positives = {("q1", "d1"), ("q1", "d2"), ("q2", "d3")}
    block = lef.false_negative_blocks(torch, queries, documents, positives, "cpu")
    # The two q1 rows each see the other's document as a false negative; the diagonal is kept.
    assert block.tolist() == [[False, True, False], [True, False, False], [False, False, False]]


def test_masking_false_negatives_lowers_the_loss_when_the_batch_is_all_positives():
    torch = pytest.importorskip("torch")
    # Both rows share one query and both documents are its positives, so the off-diagonal is
    # not a real negative; blocking it must make the same batch easier, not harder.
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    document = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # row 0's own doc is the bad match
    plain = lef.info_nce(torch, query, document, 0.05)
    block = lef.false_negative_blocks(torch, ["q", "q"], ["dn", "dp"], {("q", "dp")}, "cpu")
    masked = lef.info_nce(torch, query, document, 0.05, block)
    assert float(masked) < float(plain)
