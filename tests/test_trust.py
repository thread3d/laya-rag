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
    assert payload["trust"]["tenants"] == []
    assert payload["trust"]["acl_enabled"] is False


# ── per-tenant ACL ───────────────────────────────────────────────────────

def test_tenant_block_parses_and_enables_the_acl():
    manifest = TrustManifest.from_mapping(
        {"default": "trusted", "tenants": {"acme": ["*/acme/*"], "beta": "*/shared/*"}}
    )
    assert manifest.acl_enabled is True
    assert manifest.tenants["acme"] == ("*/acme/*",)
    assert manifest.tenants["beta"] == ("*/shared/*",)
    assert TrustManifest.from_mapping({"*": "trusted"}).acl_enabled is False


def test_no_tenant_block_allows_every_caller():
    manifest = TrustManifest.from_mapping({"*": "trusted"})
    assert manifest.tenant_allowed("/any/doc.md", None) is True
    assert manifest.tenant_allowed("/any/doc.md", "acme") is True


def test_tenant_acl_fails_closed():
    manifest = TrustManifest.from_mapping({"tenants": {"acme": ["*/acme/*"]}})
    assert manifest.tenant_allowed("/x/acme/doc.md", "acme") is True
    # Another tenant's document, an unknown tenant, and no identity are all denied.
    assert manifest.tenant_allowed("/x/beta/doc.md", "acme") is False
    assert manifest.tenant_allowed("/x/acme/doc.md", "beta") is False
    assert manifest.tenant_allowed("/x/acme/doc.md", None) is False
    assert manifest.allows("/x/acme/doc.md", "untrusted", "acme") is True
    assert manifest.allows("/x/acme/doc.md", "untrusted", "beta") is False


def test_tenant_block_rejects_bad_shapes():
    with pytest.raises(ValueError):
        TrustManifest.from_mapping({"tenants": {"a": 5}})
    with pytest.raises(ValueError):
        TrustManifest.from_mapping({"tenants": ["a"]})


def _write_tenant_corpus(root):
    (root / "acme").mkdir(parents=True, exist_ok=True)
    (root / "beta").mkdir(parents=True, exist_ok=True)
    (root / "acme" / "secret.md").write_text(
        "# Acme\n\nThe acme secret codename is falcon-nine and is confidential.\n",
        encoding="utf-8",
    )
    (root / "beta" / "secret.md").write_text(
        "# Beta\n\nThe beta secret codename is otter-four and is confidential.\n",
        encoding="utf-8",
    )
    return root


def _tenant_manifest(tmp_path):
    path = tmp_path / "tenants.json"
    path.write_text(json.dumps({
        "default": "trusted",
        "tenants": {"acme": ["*/acme/*"], "beta": ["*/beta/*"]},
    }), encoding="utf-8")
    return path


def test_engine_isolates_tenants(tmp_path):
    corpus = _write_tenant_corpus(tmp_path / "corpus")
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3, relevance_threshold=0.0,
        trust_manifest=str(_tenant_manifest(tmp_path)),
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([corpus])

    acme = engine.query("acme secret codename", tenant="acme")
    assert acme.passages
    assert all("beta" not in passage["path"] for passage in acme.passages)

    beta = engine.query("beta secret codename", tenant="beta")
    assert beta.passages
    assert all("acme" not in passage["path"] for passage in beta.passages)

    # Asking for the other tenant's content never returns it, and the exclusion is counted.
    # (The acme document legitimately matches "secret codename", so it may still appear.)
    stolen = engine.query("beta secret codename", tenant="acme")
    assert all("beta" not in passage["path"] for passage in stolen.passages)
    assert stolen.acl_filtered > 0

    # With an ACL configured, a request with no identity retrieves nothing.
    anonymous = engine.query("acme secret codename")
    assert anonymous.passages == []
    assert anonymous.acl_filtered > 0


def test_stats_reports_tenants(tmp_path):
    config = make_config(tmp_path, trust_manifest=str(_tenant_manifest(tmp_path)))
    engine = RagEngine(config, model=StubModel())
    payload = engine.stats()
    assert payload["trust"]["tenants"] == ["acme", "beta"]
    assert payload["trust"]["acl_enabled"] is True


def test_adjudicate_claims_respects_the_tenant_acl(tmp_path):
    corpus = _write_tenant_corpus(tmp_path / "corpus")
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3, relevance_threshold=0.0,
        trust_manifest=str(_tenant_manifest(tmp_path)),
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([corpus])

    # The second verification layer must not resurrect what the first one filters.
    report = engine.adjudicate_claims(["beta secret codename is otter-four"], tenant="acme")
    sources = [
        source["path"]
        for claim in report["claims"]
        for source in claim.get("sources", [])
    ]
    assert all("beta" not in path for path in sources)
    assert report["acl_filtered"] > 0


def test_adjudicate_claims_denies_an_anonymous_caller_when_an_acl_exists(tmp_path):
    corpus = _write_tenant_corpus(tmp_path / "corpus")
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3, relevance_threshold=0.0,
        trust_manifest=str(_tenant_manifest(tmp_path)),
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([corpus])
    report = engine.adjudicate_claims(["acme secret codename is falcon-nine"])
    assert report["acl_filtered"] > 0
    assert all(not claim.get("sources") for claim in report["claims"])
