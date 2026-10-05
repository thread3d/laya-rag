"""Principals, scopes, tenant binding and index path scope.

Two of the API's live hazards are fixed here.

**Tenant spoofing.** ``tenant`` used to be a body field, so the trust manifest's
per-tenant ACL enforced against an identity the caller chose. A token may now carry a
tenant (``LAYA_RAG_API_KEYS``); that tenant is the only one it can reach, and a body
``tenant`` that disagrees is refused with 403.

**Unrestricted index paths.** ``POST /v1/index`` handed client paths straight to the
loader. A token can now be limited to read/write scopes, and paths must resolve inside
``LAYA_RAG_INDEX_ROOTS``. With no allowlist the old behaviour survives *only* on a
loopback bind; a service reachable from elsewhere refuses to read arbitrary host paths.

The single legacy ``LAYA_RAG_API_KEY`` is a principal with no bound tenant — an admin
that may still select one — so existing deployments keep working.
"""
from __future__ import annotations

import hmac
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import EngineConfig
from .problems import (
    CODE_PATH_NOT_ALLOWED,
    CODE_SCOPE_REQUIRED,
    CODE_TENANT_MISMATCH,
    APIProblem,
)

log = logging.getLogger("laya_rag.auth")

SCOPE_READ = "read"
SCOPE_WRITE = "write"
ALL_SCOPES = frozenset({SCOPE_READ, SCOPE_WRITE})
KNOWN_SCOPES = ALL_SCOPES

#: Values that mean "no tenant binding" when used as the shorthand string form.
_ANY_TENANT = {"", "*", "admin", "all", "any", "none"}


@dataclass(frozen=True)
class Principal:
    """The authenticated caller behind a request."""

    token: str
    tenant: str | None = None
    scopes: frozenset[str] = ALL_SCOPES
    name: str | None = None
    anonymous: bool = False

    def has(self, scope: str) -> bool:
        return scope in self.scopes


#: Used when no authentication is configured at all (the loopback default).
ANONYMOUS = Principal(token="", anonymous=True)


def _parse_scopes(raw: Any) -> frozenset[str]:
    if raw is None:
        return ALL_SCOPES
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.replace(",", " ").split()]
    elif isinstance(raw, (list, tuple, set)):
        parts = [str(part).strip() for part in raw]
    else:
        raise ValueError(f"scopes must be a list or string, got {type(raw).__name__}")
    scopes = frozenset(part for part in parts if part)
    unknown = scopes - KNOWN_SCOPES
    if unknown:
        raise ValueError(f"unknown scope(s): {', '.join(sorted(unknown))}")
    return scopes or ALL_SCOPES


def _entry_principal(token: str, entry: Any) -> Principal:
    if isinstance(entry, str):
        tenant = None if entry.strip().lower() in _ANY_TENANT else entry.strip()
        return Principal(token=token, tenant=tenant, scopes=ALL_SCOPES)
    if isinstance(entry, dict):
        tenant = entry.get("tenant")
        if isinstance(tenant, str):
            tenant = None if tenant.strip().lower() in _ANY_TENANT else tenant.strip()
        elif tenant is not None:
            raise ValueError("a principal's tenant must be a string")
        name = entry.get("name")
        return Principal(
            token=token,
            tenant=tenant,
            scopes=_parse_scopes(entry.get("scopes")),
            name=str(name) if name else None,
        )
    raise ValueError("each api_keys entry must be a string tenant or an object")


def parse_api_keys(raw: str) -> dict[str, Principal]:
    """Parse ``LAYA_RAG_API_KEYS``: inline JSON, or a path to a JSON file."""
    text = (raw or "").strip()
    if not text:
        return {}
    candidate = Path(text).expanduser()
    if not text.startswith("{") and candidate.is_file():
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot read api_keys file {candidate}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LAYA_RAG_API_KEYS is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("LAYA_RAG_API_KEYS must be a JSON object mapping tokens to principals")
    principals: dict[str, Principal] = {}
    for token, entry in data.items():
        token = str(token)
        if not token:
            raise ValueError("LAYA_RAG_API_KEYS contains an empty token")
        principals[token] = _entry_principal(token, entry)
    return principals


class Authenticator:
    """Resolves a bearer token to a :class:`Principal`."""

    def __init__(self, config: EngineConfig) -> None:
        self._principals = parse_api_keys(config.api_keys)
        self._legacy = config.api_key
        if not self._principals and self._legacy:
            self._principals = {
                self._legacy: Principal(token=self._legacy, tenant=None, scopes=ALL_SCOPES)
            }

    @property
    def required(self) -> bool:
        return bool(self._principals)

    def authenticate(self, authorization: str | None) -> Principal:
        """Return the caller, or raise 401 with a ``WWW-Authenticate`` challenge."""
        if not self._principals:
            return ANONYMOUS
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise APIProblem(
                401,
                "unauthorized",
                "a bearer token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        matched: Principal | None = None
        # Constant-time over every candidate so a wrong token cannot be found byte by
        # byte; no early exit on the first mismatch.
        for candidate_token, principal in self._principals.items():
            if hmac.compare_digest(token, candidate_token):
                matched = principal
        if matched is None:
            raise APIProblem(
                401,
                "unauthorized",
                "invalid bearer token",
                headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
            )
        return matched

    def check_scope(self, principal: Principal, scope: str) -> None:
        """Refuse a principal without ``scope`` (RFC 6750 ``insufficient_scope``)."""
        if principal.has(scope):
            return
        raise APIProblem(
            403,
            CODE_SCOPE_REQUIRED,
            f"this token lacks the {scope!r} scope",
            headers={"WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{scope}"'},
        )

    def resolve_tenant(self, principal: Principal, requested: str | None) -> str | None:
        """The tenant a request may use, refusing a body tenant that conflicts."""
        if principal.tenant is not None:
            if requested and requested != principal.tenant:
                raise APIProblem(
                    403,
                    CODE_TENANT_MISMATCH,
                    "this token is bound to a different tenant",
                )
            return principal.tenant
        return requested


def check_index_paths(config: EngineConfig, paths: Iterable[str]) -> list[str]:
    """Return ``paths`` if every one is allowed, else raise 403.

    Configured roots are enforced absolutely. With no roots, indexing is allowed only
    when the service is loopback-only — a reachable service must opt in to the paths it
    is willing to read.
    """
    requested = list(paths)
    roots = config.resolved_index_roots
    if not roots:
        if config.loopback_only:
            return requested
        raise APIProblem(
            403,
            CODE_PATH_NOT_ALLOWED,
            "this service is reachable beyond loopback and has no LAYA_RAG_INDEX_ROOTS "
            "allowlist; set it to the directories that may be indexed",
        )
    resolved_roots = [root.resolve() for root in roots]
    for path in requested:
        resolved = Path(path).expanduser().resolve()
        if not any(
            resolved == root or resolved.is_relative_to(root) for root in resolved_roots
        ):
            raise APIProblem(
                403,
                CODE_PATH_NOT_ALLOWED,
                f"path {path!r} is outside the configured LAYA_RAG_INDEX_ROOTS",
            )
    return requested
