"""Tests for the generator-free-iteration Stage-1 mechanism test.

The experiment's logic is probe mining and set-union coverage, so those are pinned directly, plus a
synthetic corpus where the gold fact is reachable **only** through a probe entity — the exact
mechanism the bet claims.
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
    name = "laya_rag_generator_free_iteration"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools" / "generator_free_iteration.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gi = _load()


# -- probe mining ----------------------------------------------------------

def test_probes_are_entities_absent_from_the_query_most_frequent_first():
    texts = [
        "Acme Corp met Globex Industries.",
        "Globex Industries filed a report.",
    ]
    probes = gi.probes_from(texts, "What did Acme Corp say?", max_probes=5)
    assert probes == ["Globex Industries"]      # most frequent; Acme Corp is in the query


def test_probes_respect_the_cap_and_minimum_length():
    text = "Globex Industries and Beta Holdings appeared."
    probes = gi.probes_from([text], "q", max_probes=1)
    assert len(probes) == 1
    assert probes[0] in {"Beta Holdings", "Globex Industries"}
    # A capitalised run shorter than the minimum is not a probe.
    assert gi.probes_from(["Ab appeared."], "q", max_probes=5) == []


# -- coverage --------------------------------------------------------------

def test_coverage_is_the_fraction_of_gold_facts_present():
    import nugget_recall as nugget

    evidence = nugget.Evidence({"c0": "the cat sat", "c1": "a dog ran"})
    assert gi.coverage(["c0"], ["the cat sat"], evidence) == 1.0
    assert gi.coverage(["c0"], ["the cat sat", "a dog ran"], evidence) == 0.5
    assert gi.coverage([], ["the cat sat"], evidence) == 0.0


# -- the mechanism, end to end on a synthetic corpus -----------------------

CHUNKS = {
    "d0a": "Alpha Corp reported earnings. Beta Holdings was mentioned.",
    "d1a": "Beta Holdings was founded in 1991.",
    "d0b": "Gamma Inc reported earnings. Delta LLC was mentioned.",
    "d1b": "Delta LLC was founded in 1992.",
    "d0c": "Epsilon Ltd reported earnings. Zeta Group was mentioned.",
    "d1c": "Zeta Group was founded in 1993.",
}
QUERIES = {"qa": "Alpha Corp", "qb": "Gamma Inc", "qc": "Epsilon Ltd"}
GOLDS = {
    "qa": ["Beta Holdings was founded in 1991."],
    "qb": ["Delta LLC was founded in 1992."],
    "qc": ["Zeta Group was founded in 1993."],
}


def test_expansion_reaches_a_gold_fact_the_first_pool_misses():
    import nugget_recall as nugget

    index = gi.build_bm25(CHUNKS)
    evidence = nugget.Evidence(CHUNKS)
    result = gi.run(
        QUERIES, GOLDS, index, evidence, CHUNKS,
        depth=1, iterations=1, max_probes=1, top_texts=1,
    )
    baseline = result["iterations"][0]
    expanded = result["iterations"][1]
    # Depth 1 retrieves only the earnings document, which does not contain the gold fact.
    assert baseline["real_coverage"] == 0.0
    # The probe entity (Beta Holdings / Delta LLC / Zeta Group) retrieves the founding document.
    assert expanded["real_coverage"] == 1.0
    # The shuffled-probe control uses another query's entity, which cannot cover this one's gold.
    assert expanded["shuffled_coverage"] == 0.0
    # And the added coverage is headroom for an oracle ordering.
    assert baseline["real_ceiling_map@10"] == 0.0
    assert expanded["real_ceiling_map@10"] == 1.0


def test_ceiling_map_at_10_is_the_oracle_ordering():
    assert gi.ceiling_map_at_10(1, 1) == 1.0
    assert gi.ceiling_map_at_10(2, 2) == 0.75          # (1 + 1/2) / 2
    assert gi.ceiling_map_at_10(0, 2) == 0.0
    assert gi.ceiling_map_at_10(5, 1) == 1.0           # capped by the gold count


def test_fact_counts_returns_found_and_total():
    import nugget_recall as nugget

    evidence = nugget.Evidence({"c0": "the cat sat", "c1": "a dog ran"})
    assert gi.fact_counts(["c0"], ["the cat sat", "a dog ran"], evidence) == (1, 2)
    assert gi.fact_counts([], ["the cat sat"], evidence) == (0, 1)


def test_load_queries_skips_null_queries(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps([
        {"query": "Q1", "question_type": "inference_query",
         "evidence_list": [{"fact": "f1"}, {"fact": "f2"}]},
        {"query": "Qnull", "question_type": "null_query", "evidence_list": []},
    ]))
    queries, golds = gi.load_queries(path)
    assert queries == {"0": "Q1"}
    assert golds == {"0": ["f1", "f2"]}


def test_load_queries_limit(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps([
        {"query": "Q1", "question_type": "inference_query", "evidence_list": []},
        {"query": "Q2", "question_type": "inference_query", "evidence_list": []},
    ]))
    queries, _golds = gi.load_queries(path, limit=1)
    assert list(queries) == ["0"]


def test_load_question_types_skips_null_queries(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps([
        {"query": "Q1", "question_type": "inference_query", "evidence_list": []},
        {"query": "Qnull", "question_type": "null_query", "evidence_list": []},
        {"query": "Q2", "question_type": "comparison_query", "evidence_list": []},
    ]))
    # Keys are the *original* record indices (the null record keeps index 1), matching load_queries.
    assert gi.load_question_types(path) == {"0": "inference_query", "2": "comparison_query"}


def test_by_question_type_averages_each_group():
    coverage = {"a": {"0": 0.0, "1": 1.0}, "b": {"0": 1.0, "1": 1.0}}
    ceiling = {"a": {"0": 0.0, "1": 1.0}, "b": {"0": 0.5, "1": 1.0}}
    types = {"a": "inference_query", "b": "comparison_query"}
    out = gi._by_question_type(["a", "b"], types, coverage, ceiling, 1)
    assert out["inference_query"]["queries"] == 1
    assert out["inference_query"]["coverage_baseline"] == 0.0
    assert out["inference_query"]["coverage_final"] == 1.0
    assert out["comparison_query"]["ceiling_baseline"] == 0.5


def test_by_question_type_is_empty_without_types():
    assert gi._by_question_type(["a"], None, {"a": {"0": 0.0}}, {"a": {"0": 0.0}}, 0) == {}


def test_the_synthetic_run_breaks_the_gain_down_by_type():
    import nugget_recall as nugget

    index = gi.build_bm25(CHUNKS)
    evidence = nugget.Evidence(CHUNKS)
    types = {"qa": "inference_query", "qb": "inference_query", "qc": "comparison_query"}
    result = gi.run(
        QUERIES, GOLDS, index, evidence, CHUNKS,
        depth=1, iterations=1, max_probes=1, top_texts=1, types=types,
    )
    by_type = result["by_question_type"]
    assert by_type["inference_query"]["queries"] == 2
    assert by_type["inference_query"]["coverage_final"] == 1.0
    assert by_type["comparison_query"]["coverage_final"] == 1.0
