"""The model client: one request shape, cached on disk, with usage accounting.

Every call goes through `CachedClient.create`, which hashes the full request --
model, system, messages, tools, tool_choice, output config, max tokens,
thinking -- and keys a response file by that digest. Two consequences the
evaluation depends on:

Re-scoring is free. The scorer can be rewritten and re-run over the same
answers without re-paying for a single generation, which is what makes "every
reported number reproduces from the cache" a claim rather than an aspiration.

Offline is enforceable. With `offline=True` a cache miss raises instead of
sending, so the whole test suite and any re-run of a completed evaluation
provably makes no network call and costs nothing. That is the difference
between a run that should be free and one that is.

The key covers the request and nothing else. It deliberately does not include
a timestamp, a run id or an attempt counter, because any of those would make
every call a miss and silently turn the cache off.

The API key is read from the environment by the SDK. It is never read into a
variable here, never written to the cache, and never included in a digest.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ticker.cited.config import Config, price_usd

CACHE_FORMAT_VERSION = 2


class OfflineCacheMiss(RuntimeError):
    """A request was not cached and sending was disabled.

    Carries the digest so the missing entry can be located or recorded.
    """

    def __init__(self, digest: str, model: str) -> None:
        super().__init__(
            f"offline mode: no cached response for {model} request {digest[:16]}. "
            "Run without --offline to generate it, or add a recorded fixture."
        )
        self.digest = digest
        self.model = model


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    def cost_usd(self, model: str) -> float | None:
        return price_usd(
            model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
        )

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


@dataclass(slots=True)
class Response:
    """What a call returns, independent of the SDK's object graph.

    `content` is the raw list of content blocks as plain dicts, so a cached
    response round-trips through JSON without needing the SDK to rehydrate it.
    """

    model: str
    content: list[dict]
    stop_reason: str | None
    usage: Usage
    latency_s: float
    from_cache: bool

    def text(self) -> str:
        return "\n".join(
            block.get("text", "")
            for block in self.content
            if block.get("type") == "text"
        ).strip()

    def tool_uses(self) -> list[dict]:
        return [block for block in self.content if block.get("type") == "tool_use"]


def request_digest(payload: dict) -> str:
    """Stable digest of a request.

    `sort_keys` is what makes this stable: Python dict ordering would otherwise
    make the same logical request hash differently depending on the order the
    caller happened to build it in, and every call would miss.
    """
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(
        f"v{CACHE_FORMAT_VERSION}\n{encoded}".encode()
    ).hexdigest()


@dataclass(slots=True)
class CachedClient:
    config: Config
    _client: Any = field(default=None, repr=False)
    calls: int = 0
    cache_hits: int = 0
    total_usage: Usage = field(default_factory=Usage)

    def _cache_path(self, digest: str) -> Path:
        # Two-level fan-out: a flat directory of thousands of files is slow to
        # list and awkward to inspect by hand.
        return self.config.llm_cache_dir / digest[:2] / f"{digest}.json"

    def _sdk(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _payload(self, request: dict) -> dict:
        """Build the exact request body, in one place.

        `create` and `record` both go through this. Two code paths building the
        payload separately would hash differently on any divergence, which
        would make every recorded fixture a cache miss while looking correct.
        """
        model = request.get("model") or self.config.answer_model
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": request.get("max_tokens") or self.config.max_tokens,
            "messages": request["messages"],
        }
        for key in ("system", "tools", "tool_choice"):
            if request.get(key) is not None:
                payload[key] = request[key]

        output_config = dict(request.get("output_config") or {})
        if self.config.effort:
            output_config.setdefault("effort", self.config.effort)
        if output_config:
            payload["output_config"] = output_config
        return payload

    def create(
        self,
        *,
        model: str | None = None,
        system: str | list | None = None,
        messages: list,
        tools: list | None = None,
        tool_choice: dict | None = None,
        output_config: dict | None = None,
        max_tokens: int | None = None,
    ) -> Response:
        """One model call, from cache when possible."""
        payload = self._payload(
            {
                "model": model,
                "system": system,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "output_config": output_config,
                "max_tokens": max_tokens,
            }
        )
        model = payload["model"]
        digest = request_digest(payload)
        path = self._cache_path(digest)
        self.calls += 1

        if path.exists():
            cached = json.loads(path.read_text())
            self.cache_hits += 1
            usage = Usage(**cached["usage"])
            self.total_usage += usage
            return Response(
                model=cached["model"],
                content=cached["content"],
                stop_reason=cached.get("stop_reason"),
                usage=usage,
                latency_s=cached.get("latency_s", 0.0),
                from_cache=True,
            )

        if self.config.offline:
            raise OfflineCacheMiss(digest, model)

        started = time.perf_counter()
        raw = self._sdk().messages.create(**payload)
        latency_s = time.perf_counter() - started

        usage = Usage(
            input_tokens=getattr(raw.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw.usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(raw.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(raw.usage, "cache_creation_input_tokens", 0) or 0,
        )
        content = [block.model_dump(mode="json") for block in raw.content]
        record = {
            "model": raw.model,
            "content": content,
            "stop_reason": raw.stop_reason,
            "usage": usage.to_dict(),
            "latency_s": latency_s,
            "request_digest": digest,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, sort_keys=True, ensure_ascii=False))

        self.total_usage += usage
        return Response(
            model=raw.model,
            content=content,
            stop_reason=raw.stop_reason,
            usage=usage,
            latency_s=latency_s,
            from_cache=False,
        )

    def spend_usd(self, model: str | None = None) -> float | None:
        return self.total_usage.cost_usd(model or self.config.answer_model)

    def record(
        self,
        *,
        content: list[dict],
        stop_reason: str = "tool_use",
        usage: Usage | None = None,
        **request,
    ) -> str:
        """Write a cache entry for a request that has not been made.

        This is how the test suite gets recorded fixtures: a test seeds the
        response it wants, then runs the real code path with `offline=True`.
        Building the payload through `_payload` rather than by hand is the
        whole point -- a fixture keyed by a hand-written digest would drift
        silently the moment the request shape changed, and the test would pass
        against a cache entry nothing reads.

        Returns the digest, so a test can assert on cache identity.
        """
        payload = self._payload(request)
        digest = request_digest(payload)
        path = self._cache_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model": payload["model"],
                    "content": content,
                    "stop_reason": stop_reason,
                    "usage": (usage or Usage()).to_dict(),
                    "latency_s": 0.0,
                    "request_digest": digest,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return digest
