"""The response cache: key stability, offline enforcement, usage and cost.

The cache key is what makes re-scoring free and offline runs provable, so the
properties tested here are the ones the evaluation's reproducibility claim
rests on. No test in this file may construct an SDK client; a cache miss with
`offline=True` must raise, and that is asserted rather than assumed.
"""

from __future__ import annotations

import json

import pytest

from ticker.cited.config import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    Config,
    price_usd,
)
from ticker.cited.llm import (
    CachedClient,
    OfflineCacheMiss,
    Usage,
    request_digest,
)


@pytest.fixture
def client(tmp_path):
    return CachedClient(Config(offline=True, llm_cache_dir=tmp_path / "llm"))


MESSAGES = [{"role": "user", "content": "What was revenue?"}]


class TestDigest:
    def test_key_order_does_not_matter(self):
        # Python dict ordering would otherwise make the same logical request
        # hash differently depending on how the caller built it, and every
        # call would miss.
        assert request_digest({"a": 1, "b": 2}) == request_digest({"b": 2, "a": 1})

    def test_nested_key_order_does_not_matter(self):
        left = {"messages": [{"role": "user", "content": "x"}], "model": "m"}
        right = {"model": "m", "messages": [{"content": "x", "role": "user"}]}
        assert request_digest(left) == request_digest(right)

    def test_different_requests_differ(self):
        assert request_digest({"model": "a"}) != request_digest({"model": "b"})

    def test_list_order_does_matter(self):
        # Message order is semantic; two different conversations must not
        # collide on one cache entry.
        assert request_digest({"m": [1, 2]}) != request_digest({"m": [2, 1]})


class TestOffline:
    def test_miss_raises_rather_than_sending(self, client):
        with pytest.raises(OfflineCacheMiss) as excinfo:
            client.create(messages=MESSAGES)
        assert excinfo.value.model == client.config.answer_model
        assert len(excinfo.value.digest) == 64

    def test_recorded_response_is_served(self, client):
        client.record(
            messages=MESSAGES,
            content=[{"type": "text", "text": "Revenue was $53.1 billion."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=100, output_tokens=20),
        )
        response = client.create(messages=MESSAGES)
        assert response.from_cache
        assert response.text() == "Revenue was $53.1 billion."
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 100

    def test_record_and_create_agree_on_the_key(self, client):
        # The whole fixture mechanism depends on these two building the
        # payload identically; a divergence would make every fixture a miss
        # while looking correct.
        digest = client.record(
            model="claude-sonnet-5",
            system="rubric",
            messages=MESSAGES,
            tools=[{"name": "t", "input_schema": {"type": "object"}}],
            tool_choice={"type": "tool", "name": "t"},
            content=[{"type": "text", "text": "ok"}],
        )
        response = client.create(
            model="claude-sonnet-5",
            system="rubric",
            messages=MESSAGES,
            tools=[{"name": "t", "input_schema": {"type": "object"}}],
            tool_choice={"type": "tool", "name": "t"},
        )
        assert response.from_cache
        path = next((client.config.llm_cache_dir / digest[:2]).glob("*.json"))
        assert json.loads(path.read_text())["request_digest"] == digest

    def test_a_changed_request_misses(self, client):
        client.record(messages=MESSAGES, content=[{"type": "text", "text": "ok"}])
        with pytest.raises(OfflineCacheMiss):
            client.create(messages=[{"role": "user", "content": "different"}])

    def test_effort_is_part_of_the_key(self, tmp_path):
        high = CachedClient(
            Config(offline=True, llm_cache_dir=tmp_path / "llm", effort="high")
        )
        low = CachedClient(
            Config(offline=True, llm_cache_dir=tmp_path / "llm", effort="low")
        )
        high.record(messages=MESSAGES, content=[{"type": "text", "text": "ok"}])
        assert high.create(messages=MESSAGES).from_cache
        # Same prompt, different settings, so it must not be served the other
        # one's response.
        with pytest.raises(OfflineCacheMiss):
            low.create(messages=MESSAGES)


class TestUsageAccounting:
    def test_usage_adds(self):
        total = Usage(input_tokens=10, output_tokens=2) + Usage(
            input_tokens=5, output_tokens=1, cache_read_tokens=100
        )
        assert total.input_tokens == 15
        assert total.output_tokens == 3
        assert total.cache_read_tokens == 100

    def test_client_accumulates_across_calls(self, client):
        for content in ("a", "b"):
            client.record(
                messages=[{"role": "user", "content": content}],
                content=[{"type": "text", "text": content}],
                usage=Usage(input_tokens=100, output_tokens=10),
            )
            client.create(messages=[{"role": "user", "content": content}])
        assert client.calls == 2
        assert client.cache_hits == 2
        assert client.total_usage.input_tokens == 200

    def test_price_uses_published_rates(self):
        # Opus 5 at $5 in and $25 out per million.
        assert price_usd(
            "claude-opus-5", input_tokens=1_000_000, output_tokens=0
        ) == pytest.approx(5.0)
        assert price_usd(
            "claude-opus-5", input_tokens=0, output_tokens=1_000_000
        ) == pytest.approx(25.0)

    def test_cache_tokens_are_discounted_and_surcharged(self):
        read = price_usd("claude-opus-5", input_tokens=0, output_tokens=0,
                         cache_read_tokens=1_000_000)
        write = price_usd("claude-opus-5", input_tokens=0, output_tokens=0,
                          cache_write_tokens=1_000_000)
        assert read == pytest.approx(5.0 * CACHE_READ_MULTIPLIER)
        assert write == pytest.approx(5.0 * CACHE_WRITE_MULTIPLIER)

    def test_unknown_model_has_no_price(self):
        # None rather than zero: a run whose cost could not be computed must
        # not be reported as free.
        assert price_usd("not-a-model", input_tokens=10, output_tokens=10) is None
        assert Usage(input_tokens=10).cost_usd("not-a-model") is None


class TestConfig:
    def test_redacted_has_no_credentials_and_stringifies_paths(self):
        payload = Config().redacted()
        assert isinstance(payload["db_path"], str)
        joined = json.dumps(payload).lower()
        # Credential-shaped names only. "token" on its own would match
        # `max_tokens`, which is a request parameter and belongs in a results
        # file.
        for secret in ("api_key", "apikey", "sk-ant", "secret", "password",
                       "auth_token", "credential"):
            assert secret not in joined

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("TICKER_ANSWER_MODEL", "claude-haiku-4-5")
        monkeypatch.setenv("TICKER_MAX_TOOL_CALLS", "3")
        monkeypatch.setenv("TICKER_OFFLINE", "1")
        config = Config.from_env()
        assert config.answer_model == "claude-haiku-4-5"
        assert config.max_tool_calls == 3
        assert config.offline is True

    def test_no_env_leaves_defaults(self, monkeypatch):
        for key in ("TICKER_ANSWER_MODEL", "TICKER_JUDGE_MODEL",
                    "TICKER_MAX_TOOL_CALLS", "TICKER_EFFORT", "TICKER_OFFLINE"):
            monkeypatch.delenv(key, raising=False)
        assert Config.from_env() == Config()
