"""Optional answer composition by a generative model.

Everything else in this engine is Laya's: retrieval, calibrated relevance, and the
answerability gate. This module is the single exception. Laya cannot generate text,
so if a caller wants prose rather than an evidence bundle, something else has to
write it — and that something can hallucinate.

Two consequences shape the design:

* It is **opt-in per request**. Nothing here runs unless the caller asks, and the
  interface that exposes it is expected to say plainly that the prose is unverified.
* The model is **not configured here**. The engine reads the harness's own
  ``settings.yaml`` for its endpoint and model, so there is one source of truth for
  which model is in play rather than a second copy that drifts.

The transport is the OpenAI-compatible ``/chat/completions`` route that Ollama
serves, using only the standard library: this project has no HTTP client dependency
and does not need one.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .config import EngineConfig

#: Where Ollama serves its OpenAI-compatible API when nothing overrides it.
DEFAULT_BASE_URL = "http://localhost:11434/v1"

#: The instruction that keeps this step as close to the evidence as a generative
#: model can be kept. It is a mitigation, never a guarantee.
#:
#: The citation requirement is stated as a hard rule rather than a preference,
#: because the model otherwise omits citations often enough to make them useless as
#: a check: a first real run produced twelve uncited claims out of twelve while every
#: quoted span was verbatim.
SYSTEM_PROMPT = (
    "You answer strictly from the numbered evidence the user provides. "
    "Rules, in order of importance:\n"
    "1. End every sentence that states a fact with the number of the evidence passage "
    "that supports it, in square brackets — like this: [1]. A sentence supported by two "
    "passages ends [1][2].\n"
    "2. Quote code, paths and identifiers exactly as they appear in the evidence.\n"
    "3. If the evidence does not answer the question, say so plainly instead of using "
    "outside knowledge.\n"
    "Do not speculate, and do not add facts that are not present in the evidence. "
    "Sentences without a supporting passage number are treated as unsupported."
)


def _scan_settings(text: str) -> dict[str, str]:
    """Read the two keys we need without a YAML parser.

    A deliberately tiny, indentation-aware scan for ``baseURL`` and the
    ``agent-default-model.model`` value, used when PyYAML is not installed. It is
    not a general YAML reader and does not pretend to be one.
    """
    found: dict[str, str] = {}
    section = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1].strip()
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key, value = key.strip(), value.strip().strip("'\"")
        if not value:
            continue
        if key == "baseURL" and "baseURL" not in found:
            found["baseURL"] = value
        elif key == "model" and section == "agent-default-model" and "model" not in found:
            found["model"] = value
    return found


def read_dsh_settings(path: Path | None) -> dict[str, str]:
    """Best-effort read of the harness's settings for its LLM endpoint and model.

    Any problem — no file, no parser, an unexpected shape — yields an empty mapping
    so the caller falls back rather than failing. A missing model simply means
    composition stays unavailable, which is reported rather than guessed at.
    """
    if path is None:
        return {}
    try:
        if not Path(path).is_file():
            return {}
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}

    parsed: dict[str, Any] = {}
    try:
        import yaml  # noqa: PLC0415 - optional, and imported only when present

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            parsed = loaded
    except Exception:  # noqa: BLE001 - any parser problem falls back to the scan
        parsed = {}

    settings = _scan_settings(text)
    if isinstance(parsed, dict):
        # Only the agent default names the model this engine should use; a model key
        # in some other section is a different concern and must not be picked up.
        agent = parsed.get("agent-default-model")
        if isinstance(agent, dict):
            model = agent.get("model")
            if isinstance(model, str) and model.strip():
                settings["model"] = model.strip()
        if not settings.get("baseURL"):
            for name in ("llm-deepseek", "llm", "llm-openai"):
                section = parsed.get(name)
                if isinstance(section, dict):
                    base_url = section.get("baseURL")
                    if isinstance(base_url, str) and base_url.strip():
                        settings["baseURL"] = base_url.strip()
                        break
    return settings


@dataclass(frozen=True)
class ComposeTarget:
    """A resolved endpoint and model to compose with."""

    base_url: str
    model: str
    timeout_s: int
    max_tokens: int
    source: str
    api: str = "ollama"

    @property
    def request_url(self) -> str:
        """The route to call, which differs by API flavour.

        ``openai`` speaks ``<base>/chat/completions``. ``ollama`` speaks the native
        ``<root>/api/chat``, where the base URL's ``/v1`` suffix is dropped.
        """
        base = self.base_url.rstrip("/")
        if self.api == "openai":
            return base + "/chat/completions"
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        return root + "/api/chat"

    def payload(self, query: str, passages: Sequence[tuple[str, str]], max_tokens: int) -> dict[str, Any]:
        """The request body for this flavour.

        Both set ``think: False``; only the native route actually honours it.
        """
        messages = build_messages(query, passages)
        if self.api == "openai":
            return {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "stream": True,
                "stream_options": {"include_usage": True},
                "think": False,
            }
        return {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }


def resolve_target(config: EngineConfig) -> tuple[ComposeTarget | None, str]:
    """Resolve where and with what to compose. Returns ``(target, reason)``.

    Precedence for the model is environment, then the harness's own settings; a
    model name is never invented, because guessing one would fail confusingly later.
    """
    settings = read_dsh_settings(config.dsh_settings)
    base_url = (config.compose_base_url or settings.get("baseURL") or "").strip()
    model = (config.compose_model or settings.get("model") or "").strip()

    if config.compose_model:
        source = "environment"
    elif settings.get("model"):
        source = f"dsh settings ({config.dsh_settings})"
    else:
        source = "unset"

    if not model:
        return None, (
            "no composition model is configured: set LAYA_DSH_RAG_COMPOSE_MODEL, or give "
            f"`agent-default-model.model` in {config.dsh_settings}"
        )

    return (
        ComposeTarget(
            base_url=base_url or DEFAULT_BASE_URL,
            model=model,
            timeout_s=int(config.compose_timeout_s),
            max_tokens=int(config.compose_max_tokens),
            source=source,
            api="openai" if (config.compose_api or "").strip().lower() == "openai" else "ollama",
        ),
        "",
    )


def build_messages(
    query: str,
    passages: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    """Build the chat messages: the evidence, numbered, then the question."""
    blocks: list[str] = []
    for index, (path, text) in enumerate(passages, start=1):
        header = f"[{index}] {path}" if path else f"[{index}]"
        blocks.append(f"{header}\n{text}")
    evidence = "\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Evidence:\n\n{evidence}\n\nQuestion: {query}"},
    ]


class Composer:
    """Streams composed prose from an OpenAI-compatible chat endpoint."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def describe(self) -> dict[str, Any]:
        """Report the resolved composition target, without contacting it.

        Cheap enough to sit in ``/v1/stats``, which the interface polls. Whether the
        endpoint is actually up is discovered on first use, and reported then.
        """
        target, reason = resolve_target(self.config)
        if target is None:
            return {"available": False, "reason": reason}
        return {
            "available": True,
            "model": target.model,
            "base_url": target.base_url,
            "api": target.api,
            "request_url": target.request_url,
            "source": target.source,
            "timeout_s": target.timeout_s,
            "max_tokens": target.max_tokens,
        }

    def stream(
        self,
        query: str,
        passages: Iterable[tuple[str, str]],
        max_tokens: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield ``delta`` events as the model writes, then a ``done`` or ``error``.

        An ``error`` can follow some ``delta`` events: the connection may drop after
        the model has already written part of an answer, and the caller needs to know
        the text it holds is incomplete.
        """
        target, reason = resolve_target(self.config)
        if target is None:
            yield {"type": "error", "message": reason}
            return

        payload = target.payload(query, list(passages), int(max_tokens or target.max_tokens))
        request = urllib.request.Request(
            target.request_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        written = 0
        usage: dict[str, Any] = {}
        finish_reason = ""
        try:
            with urllib.request.urlopen(request, timeout=target.timeout_s) as response:
                for line in response:
                    parsed = self._parse_line(line)
                    if parsed["usage"]:
                        usage = parsed["usage"]
                    if parsed["finish_reason"]:
                        finish_reason = parsed["finish_reason"]
                    content = parsed["content"]
                    if content:
                        written += len(content)
                        yield {"type": "delta", "text": content}
                    if parsed["done"]:
                        break
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001 - the body is best-effort context
                detail = ""
            yield {
                "type": "error",
                "message": f"the model endpoint returned HTTP {exc.code}"
                + (f": {detail.strip()}" if detail.strip() else ""),
            }
            return
        except Exception as exc:  # noqa: BLE001 - reported to the caller, never raised
            yield {
                "type": "error",
                "message": f"could not reach the model at {target.base_url} ({target.model}): {exc}",
            }
            return

        yield {
            "type": "done",
            "model": target.model,
            "chars": written,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            # "length" means the token budget ran out mid-answer. Without this a
            # truncated answer is indistinguishable from a complete one.
            "finish_reason": finish_reason,
            "usage": usage,
        }

    @staticmethod
    def _parse_line(line: bytes) -> dict[str, Any]:
        """Pull one chunk out of a streaming line from either API flavour.

        Ollama's native route streams NDJSON objects shaped ``{"message": {"content":
        ...}, "done": bool}``; the OpenAI-compatible route streams SSE lines wrapping
        ``choices[0].delta.content`` and ending with ``[DONE]``. Both are handled here
        so the caller sees one event shape.
        """
        empty: dict[str, Any] = {
            "content": None, "usage": {}, "finish_reason": "", "done": False,
        }
        text = line.decode("utf-8", "replace").strip()
        if not text:
            return empty
        if text.startswith("data:"):
            text = text[len("data:") :].strip()
        if text == "[DONE]":
            return {**empty, "done": True}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return empty
        if not isinstance(payload, dict):
            return empty

        usage = payload.get("usage")
        usage = dict(usage) if isinstance(usage, dict) else {}
        if not usage and isinstance(payload.get("eval_count"), int):
            # The native route reports counts at the top level instead.
            usage = {
                "completion_tokens": payload.get("eval_count"),
                "prompt_tokens": payload.get("prompt_eval_count"),
            }

        finish = payload.get("done_reason")
        finish = finish if isinstance(finish, str) else ""
        done = payload.get("done") is True

        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            return {
                "content": content if isinstance(content, str) and content else None,
                "usage": usage,
                "finish_reason": finish,
                "done": done,
            }

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return {**empty, "usage": usage, "finish_reason": finish}
        choice = choices[0] if isinstance(choices[0], dict) else {}
        reason = choice.get("finish_reason")
        finish = reason if isinstance(reason, str) else finish

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            return {
                "content": content if isinstance(content, str) and content else None,
                "usage": usage,
                "finish_reason": finish,
                "done": done,
            }
        # Some endpoints return a whole message instead of a delta.
        whole = choice.get("message")
        if isinstance(whole, dict) and isinstance(whole.get("content"), str):
            return {
                "content": whole["content"] or None,
                "usage": usage,
                "finish_reason": finish,
                "done": done,
            }
        return {**empty, "usage": usage, "finish_reason": finish}
