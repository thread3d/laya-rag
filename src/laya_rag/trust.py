"""Retrieval-time trust for indexed documents.

The index has no notion of authority: every chunk is equally retrievable, so a document
added by a self-serve ``/v1/index`` call, pulled in from a dependency, or edited on disk
can rank for a query and be spliced into the composer prompt. A trust *manifest* maps
paths (exact or glob) to a tier, and a query can set a minimum tier, so a document whose
provenance is not trusted is excluded before it reaches any decision.

This is deliberately a **filter, not a classifier**: the manifest is an ingest-policy
input, not a model, and it is deterministic. With no manifest configured nothing is
filtered, so the shipped behaviour is unchanged. The honest default *when* a manifest is
configured is that anything it does not name is ``untrusted`` — an allow-list is the only
safe reading of "these are the documents I vouch for" — but the default is configurable
for deployments that want a deny-list.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: Ordered from least to most trusted; a query's minimum is compared by rank, not name.
TIERS: tuple[str, ...] = ("quarantined", "untrusted", "trusted")

_RANK = {tier: index for index, tier in enumerate(TIERS)}


def tier_rank(tier: str) -> int:
    """Rank of a tier name; raises on an unknown tier rather than guessing."""
    try:
        return _RANK[str(tier).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown trust tier {tier!r}; expected one of {TIERS}") from exc


@dataclass
class TrustManifest:
    """Path patterns mapped to tiers, an optional per-tenant allow-list, and a default."""

    rules: list[tuple[str, str]] = field(default_factory=list)
    default_tier: str = "untrusted"
    #: tenant name -> path globs that tenant may retrieve. Empty means no ACL at all.
    tenants: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate eagerly so a typo in a manifest fails at load, not at query time.
        tier_rank(self.default_tier)
        for _pattern, tier in self.rules:
            tier_rank(tier)

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], *, default_tier: str | None = None
    ) -> "TrustManifest":
        """Build from ``{pattern: tier}`` or ``{"default": ..., "rules": {...}, "tenants": {...}}``."""
        default = default_tier or str(mapping.get("default", "untrusted"))
        rules_source = mapping.get("rules", None)
        if rules_source is None:
            rules_source = {
                k: v for k, v in mapping.items() if k not in {"default", "tenants"}
            }
        if not isinstance(rules_source, Mapping):
            raise ValueError("a trust manifest's 'rules' must be an object of pattern -> tier")
        rules = [(str(pattern), str(tier)) for pattern, tier in rules_source.items()]

        raw_tenants = mapping.get("tenants") or {}
        if not isinstance(raw_tenants, Mapping):
            raise ValueError("a trust manifest's 'tenants' must be an object of tenant -> globs")
        tenants: dict[str, tuple[str, ...]] = {}
        for name, patterns in raw_tenants.items():
            if isinstance(patterns, str):
                patterns = [patterns]
            if not isinstance(patterns, (list, tuple)):
                raise ValueError(f"tenant {name!r} must map to a glob or a list of globs")
            tenants[str(name)] = tuple(str(pattern) for pattern in patterns)
        return cls(rules=rules, default_tier=default, tenants=tenants)

    @classmethod
    def from_path(cls, path: str | Path, *, default_tier: str | None = None) -> "TrustManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} must contain a JSON object")
        return cls.from_mapping(payload, default_tier=default_tier)

    @property
    def acl_enabled(self) -> bool:
        """Whether any tenant allow-list is configured (and therefore enforced)."""
        return bool(self.tenants)

    def tenant_allowed(self, path: str | None, tenant: str | None) -> bool:
        """Whether ``tenant`` may retrieve ``path``.

        With no tenant block configured every caller is allowed, so the shipped behaviour
        is unchanged. Once a tenant block exists the check **fails closed**: a request with
        no tenant, or an unknown tenant, retrieves nothing. That is the only safe reading
        of "these tenants may see these documents".
        """
        if not self.tenants:
            return True
        if not tenant:
            return False
        patterns = self.tenants.get(str(tenant))
        if patterns is None:
            return False
        candidate = str(path or "")
        return any(fnmatch.fnmatch(candidate, pattern) for pattern in patterns)

    def tier(self, path: str | None) -> str:
        """The tier for one path: the first matching rule, else the default.

        First match wins, so a manifest can put a broad ``*`` rule first and then
        exceptions after it in the file's own order.
        """
        candidate = str(path or "")
        for pattern, tier in self.rules:
            if fnmatch.fnmatch(candidate, pattern):
                return tier
        return self.default_tier

    def allows(self, path: str | None, min_tier: str, tenant: str | None = None) -> bool:
        return self.tenant_allowed(path, tenant) and tier_rank(self.tier(path)) >= tier_rank(min_tier)

    def describe(self) -> dict[str, Any]:
        return {
            "default_tier": self.default_tier,
            "tiers": list(TIERS),
            "rules": [{"pattern": pattern, "tier": tier} for pattern, tier in self.rules],
            "tenants": sorted(self.tenants),
            "acl_enabled": self.acl_enabled,
        }


def filter_candidates(
    candidates: Sequence[Any],
    manifest: TrustManifest,
    min_tier: str,
    tenant: str | None = None,
) -> list[Any]:
    """Keep only candidates whose chunk path passes the tenant ACL and the tier minimum.

    Candidates are engine ``Hit`` objects; anything without a chunk path is treated as
    the empty path and therefore falls to the manifest's default, which is the safe
    behaviour for an unknown provenance.
    """
    tier_rank(min_tier)  # fail loudly on a bad minimum rather than filtering nothing
    kept: list[Any] = []
    for hit in candidates:
        chunk = getattr(hit, "chunk", None)
        path = getattr(chunk, "path", None)
        if manifest.allows(path, min_tier, tenant):
            kept.append(hit)
    return kept
