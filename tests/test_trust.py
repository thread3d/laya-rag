"""Offline tests for retrieval-time trust tiers.

The manifest is an ingest-policy input, so its semantics have to be exact: first-match
wins, an unknown tier is an error rather than a silent pass, and an unmatched path falls
to the configured default. The engine wiring is pinned with a two-document corpus.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from laya_rag.config import EngineConfig
from laya_rag.engine import RagEngine
from laya_rag.trust import TrustManifest, filter_candidates, tier_rank

from .support import StubModel, make_config


# ── manifest semantics ───────────────────────────────────────────────────

def test_tier_rank_orders_and_rejects_unknown_names():
    assert tier_rank("quarantined") < tier_rank("untrusted") < tier_rank("trusted")
    assert tier_rank("TRUSTED") == tier_rank("trusted")
    with pytest.raises(ValueError):
        tier_rank("probably-fine")


def test_from_mapping_accepts_a_flat_mapping():
    manifest = TrustManifest.from_mapping({"/a/*": "trusted", "/b/secret.md": "quarantined"})
    assert manifest.default_tier == "untrusted"
    assert manifest.tier("/a/doc.md") == "trusted"
    assert manifest.tier("/b/secret.md") == "quarantined"
    assert manifest.tier("/c/other.md") == "untrusted"


def test_from_mapping_accepts_default_and_rules():
    manifest = TrustManifest.from_mapping(
        {"default": "trusted", "rules": {"*/scratch/*": "untrusted"}}
    )
    assert manifest.default_tier == "trusted"
    assert manifest.tier("/tmp/scratch/x.md") == "untrusted"
    assert manifest.tier("/docs/x.md") == "trusted"


def test_first_matching_rule_wins():
    manifest = TrustManifest.from_mapping(
        {"default": "trusted", "rules": {"*": "untrusted", "*/approved/*": "trusted"}}
    )
    # The broad rule comes first, so the specific one after it never applies.
    assert manifest.tier("/approved/doc.md") == "untrusted"


def test_an_unknown_tier_in_a_manifest_is_an_error():
    with pytest.raises(ValueError):
        TrustManifest.from_mapping({"*": "sort-of-trusted"})
    with pytest.raises(ValueError):
        TrustManifest(rules=[("*", "trusted")], default_tier="maybe")


def test_from_path_reads_json(tmp_path):
    path = tmp_path / "trust.json"
    path.write_text(json.dumps({"default": "trusted", "rules": {"*/poison.md": "quarantined"}}))
    manifest = TrustManifest.from_path(path)
    assert manifest.tier("/x/poison.md") == "quarantined"
    assert manifest.tier("/x/ok.md") == "trusted"


def test_allows_compares_by_rank():
    manifest = TrustManifest.from_mapping({"*/a.md": "trusted", "*/b.md": "quarantined"})
    assert manifest.allows("/x/a.md", "untrusted") is True
    assert manifest.allows("/x/a.md", "trusted") is True
    assert manifest.allows("/x/b.md", "untrusted") is False
    assert manifest.allows("/x/unknown.md", "trusted") is False  # default untrusted
    # An unknown path (None -> "") falls to the default too.
    assert manifest.allows(None, "trusted") is False
    assert manifest.allows(None, "untrusted") is True


def test_filter_candidates_keeps_only_allowed_paths():
    manifest = TrustManifest.from_mapping({"*/poison.md": "quarantined", "*": "trusted"})
    hits = [
        SimpleNamespace(chunk=SimpleNamespace(path="/x/ok.md")),
        SimpleNamespace(chunk=SimpleNamespace(path="/x/poison.md")),
    ]
    kept = filter_candidates(hits, manifest, "untrusted")
    assert [h.chunk.path for h in kept] == ["/x/ok.md"]


def test_filter_candidates_rejects_a_bad_minimum():
    manifest = TrustManifest.from_mapping({"*": "trusted"})
    with pytest.raises(ValueError):
        filter_candidates([], manifest, "nope")


# ── engine wiring ────────────────────────────────────────────────────────

def _write_corpus(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "trusted.md").write_text(
        "# Trusted\n\nThe quarterly safety audit passed with zero findings and is documented here.\n",
        encoding="utf-8",
    )
    (root / "poison.md").write_text(
        "# Poison\n\nThe secret escalation code is zeta-nine-omega and must never be shared.\n",
        encoding="utf-8",
    )
    return root


def _manifest_file(tmp_path):
    path = tmp_path / "trust.json"
    path.write_text(json.dumps({
        "default": "untrusted",
        "rules": {"*/trusted.md": "trusted"},
    }), encoding="utf-8")
    return path


def test_trust_is_off_by_default():
    config = EngineConfig()
    assert config.trust_manifest == ""
    assert config.query_min_trust == ""


def test_engine_excludes_an_untrusted_document(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus")
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3,
        trust_manifest=str(_manifest_file(tmp_path)),
        query_min_trust="trusted",
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([corpus])

    poisoned = engine.query("zeta-nine-omega escalation code")
    assert poisoned.trust_filtered > 0
    assert poisoned.passages == []

    trusted = engine.query("quarterly safety audit findings")
    # The poison document may still have ranked before filtering; what matters is that it
    # never reaches the returned evidence.
    assert trusted.passages
    assert all("poison.md" not in p["path"] for p in trusted.passages)


def test_engine_without_a_minimum_does_not_filter(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus")
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3,
        trust_manifest=str(_manifest_file(tmp_path)),
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([corpus])
    result = engine.query("zeta-nine-omega escalation code")
    assert result.trust_filtered == 0


def test_stats_reports_the_trust_state(tmp_path):
    config = make_config(
        tmp_path, trust_manifest=str(_manifest_file(tmp_path)),
        query_min_trust="trusted",
    )
    engine = RagEngine(config, model=StubModel())
    payload = engine.stats()
    assert payload["trust"]["configured"] is True
    assert payload["trust"]["loaded"] is True
    assert payload["trust"]["default_tier"] == "untrusted"
    assert payload["trust"]["query_min_trust"] == "trusted"
    assert payload["trust"]["rules"] == 1
