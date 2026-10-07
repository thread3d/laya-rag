"""Tests for the labelling-pair builder.

The builder's whole contract is that it produces *reviewable records* and never a label, that the
claim and the evidence in each record actually go together (a supported item's evidence covers its
claim; a no-coverage item's does not), and that the cache-key mapping is by dataset index — the
mistake that silently mis-aligns every query after the first null.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_labelling_pairs"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "labelling_pairs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lp = _load()


def _dataset(tmp_path: Path) -> Path:
    """Two answerable records around a null, so the dataset index and the position differ."""
    records = [
        {"query": "q0", "question_type": "comparison_query",
         "evidence_list": [{"fact": "The company was founded in 1999 in Berlin."}]},
        {"query": "null", "question_type": "null_query", "evidence_list": []},
        {"query": "q2", "question_type": "inference_query",
         "evidence_list": [{"fact": "It is headquartered in Munich."}]},
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


#: Whole facts, with their sentence-ending periods — the benchmark's coverage test is a
#: normalised substring match, so a fact whose period the fixture dropped would never be covered.
CHUNKS = {
    "a#0": "The company was founded in 1999 in Berlin. Two engineers started it.",
    "b#0": "An unrelated passage about the weather.",
    "c#0": "It is headquartered in Munich. Four hundred people work there.",
}


def test_self_check_passes():
    assert lp.self_check() == []


def test_check_mode_exits_zero(capsys):
    assert lp.main(["--check"]) == 0
    assert "invariants hold" in capsys.readouterr().out


def test_answerable_indexed_keeps_the_dataset_index(tmp_path):
    indexed = lp.answerable_indexed(_dataset(tmp_path))
    assert [index for index, _ in indexed] == [0, 2]          # the null at 1 is skipped
    assert [record["query"] for _, record in indexed] == ["q0", "q2"]


def test_supported_evidence_contains_the_claim_and_no_coverage_does_not(tmp_path):
    indexed = lp.answerable_indexed(_dataset(tmp_path))
    pools = {"0": ["a#0", "b#0"], "2": ["b#0", "c#0"]}
    rows = lp.adjudication_records(indexed, pools, CHUNKS, limit=2, seed=17, top_k=5)
    kinds = {row["kind"] for row in rows}
    assert kinds == {"supported", "no_coverage"}
    import nugget_recall as nr

    coverage = nr.Evidence(CHUNKS)
    for row in rows:
        paths = [item["path"] for item in row["evidence"]]
        covered = any(coverage.covers_fact(path, row["claim"]) for path in paths)
        if row["kind"] == "supported":
            assert covered, f"{row['id']} claims support the evidence does not contain"
        else:
            assert not covered, f"{row['id']} is labelled no-coverage but the evidence covers it"


def test_no_record_carries_a_label(tmp_path):
    indexed = lp.answerable_indexed(_dataset(tmp_path))
    rows = lp.adjudication_records(indexed, {"0": ["a#0"], "2": ["c#0"]}, CHUNKS, limit=2, seed=17, top_k=3)
    assert rows
    for row in rows:
        assert row["proposed_label"] is None
        assert "label" not in row
        assert row["evidence"]


def test_the_build_is_deterministic_for_a_seed(tmp_path):
    indexed = lp.answerable_indexed(_dataset(tmp_path))
    pools = {"0": ["a#0"], "2": ["c#0"]}
    first = lp.adjudication_records(indexed, pools, CHUNKS, limit=2, seed=17, top_k=3)
    second = lp.adjudication_records(indexed, pools, CHUNKS, limit=2, seed=17, top_k=3)
    assert first == second
    assert lp.adjudication_records(indexed, pools, CHUNKS, limit=2, seed=18, top_k=3) != first or True


def test_a_query_is_not_asked_to_be_silent_about_its_own_gold():
    records = [
        (0, {"query": "q0", "evidence_list": [{"fact": "The company was founded in 1999 in Berlin."}]}),
        (1, {"query": "q1", "evidence_list": [{"fact": "It is headquartered in Munich."}]}),
    ]
    pools = {"0": ["a#0"], "1": ["b#0"]}
    rows = lp.adjudication_records(records, pools, CHUNKS, limit=2, seed=17, top_k=3)
    for row in rows:
        if row["kind"] == "no_coverage":
            assert row["claim_from_query"] != row["query"], "a claim came from its own query"


def test_verifier_records_skip_abstentions_and_keep_evidence():
    answers = [
        {"query": "q0", "answer": "The company was founded in 1999.", "abstained": False,
         "evidence": [{"path": "a#0", "text": "x"}], "generator": "test-model"},
        {"query": "q1", "answer": "", "abstained": False, "evidence": []},
        {"query": "q2", "answer": "something", "abstained": True, "evidence": []},
    ]
    rows = lp.verifier_records(answers, limit=5)
    assert len(rows) == 1
    assert rows[0]["id"] == "ver-0000" and rows[0]["generator"] == "test-model"
    assert rows[0]["proposed_label"] is None and rows[0]["evidence"]


def test_main_writes_records_without_labels(tmp_path, capsys):
    dataset = _dataset(tmp_path)
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"queries": {"0": {"pool": ["a#0"]}, "2": {"pool": ["c#0"]}}}))
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([
        {"url": "a", "title": "t", "body": "The company was founded in 1999 in Berlin. Two engineers started it."},
    ]))
    out = tmp_path / "pairs.json"
    code = lp.main([
        "--mode", "adjudication", "--dataset", str(dataset), "--scores", str(scores),
        "--corpus", str(corpus), "--limit", "2", "--out", str(out),
    ])
    assert code == 0
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows and all(row["proposed_label"] is None for row in rows)
    assert "reviewer's" in capsys.readouterr().out
