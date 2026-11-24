"""Configuration for the cited-answer feature.

Model names are config, never literals in call sites: swapping the answering
model must not require editing the agent, and a results file has to be able to
say which model produced it. Every field can be overridden from the
environment so a run is reproducible from a recorded config rather than from
remembered flags.

The API key is read from the environment by the SDK and is never stored here,
never logged, and never written to a results file. `redacted()` exists so a
config can be serialised into a results file without carrying anything secret.

Default models are the higher-quality options rather than the cheapest. The
cheap preset is one flag away (`--preset cheap`) and the cost of each is in the
README; picking the spend is the operator's call, and quietly defaulting to a
weaker model would misrepresent what the numbers mean.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# Published per-million-token prices, input then output. Used only to report
# what a run cost; nothing here affects what is sent. A model missing from this
# table still runs, and its cost is reported as None rather than guessed.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Cache reads bill at a tenth of the input rate; cache writes at 1.25x for the
# default five-minute TTL.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

DEFAULT_ANSWER_MODEL = "claude-opus-5"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"

DEFAULT_MAX_TOOL_CALLS = 8
DEFAULT_SEARCH_K = 8
DEFAULT_SINGLE_SHOT_K = 10

DEFAULT_RESULTS_DIR = Path("results/cited_answers")
DEFAULT_LLM_CACHE_DIR = Path(".cache/llm")


@dataclass(frozen=True, slots=True)
class Config:
    answer_model: str = DEFAULT_ANSWER_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL

    # Cap on tool calls per question. The agent is told the cap and how many
    # calls remain, so hitting it is a decision it can see coming rather than
    # a truncation it cannot explain.
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    search_k: int = DEFAULT_SEARCH_K
    single_shot_k: int = DEFAULT_SINGLE_SHOT_K

    max_tokens: int = 8192
    effort: str = "high"

    db_path: Path = Path("data/ticker.duckdb")
    index_root: Path = Path("data/index")
    questions_path: Path = Path("eval/questions.jsonl")
    results_dir: Path = DEFAULT_RESULTS_DIR
    llm_cache_dir: Path = DEFAULT_LLM_CACHE_DIR

    # Refuse to send a request when the cache misses. The whole feature is
    # built to run offline against recorded responses; this makes an
    # accidental paid run impossible rather than merely unlikely.
    offline: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        config = cls()
        overrides: dict[str, object] = {}
        if value := os.environ.get("TICKER_ANSWER_MODEL"):
            overrides["answer_model"] = value
        if value := os.environ.get("TICKER_JUDGE_MODEL"):
            overrides["judge_model"] = value
        if value := os.environ.get("TICKER_MAX_TOOL_CALLS"):
            overrides["max_tool_calls"] = int(value)
        if value := os.environ.get("TICKER_EFFORT"):
            overrides["effort"] = value
        if os.environ.get("TICKER_OFFLINE"):
            overrides["offline"] = True
        return replace(config, **overrides) if overrides else config

    def redacted(self) -> dict:
        """Serialisable form, with paths as strings and no credentials.

        There is nothing to strip today because no secret is ever stored on
        this object. The method exists so that stays true: a field added here
        has to be considered for this output before it can reach a results
        file.
        """
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in payload.items()
        }


CHEAP_PRESET = Config(
    answer_model="claude-haiku-4-5",
    judge_model="claude-sonnet-5",
)

PRESETS: dict[str, Config] = {
    "default": Config(),
    "cheap": CHEAP_PRESET,
}


def price_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Cost of one call, or None for a model with no published price here.

    None rather than zero: a run whose cost could not be computed must not be
    reported as free.
    """
    prices = PRICES_USD_PER_MTOK.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    per_token_in = input_price / 1_000_000
    per_token_out = output_price / 1_000_000
    return (
        input_tokens * per_token_in
        + output_tokens * per_token_out
        + cache_read_tokens * per_token_in * CACHE_READ_MULTIPLIER
        + cache_write_tokens * per_token_in * CACHE_WRITE_MULTIPLIER
    )
