"""The agent loop and the baselines, driven entirely by recorded responses.

Every model turn here is a cache fixture, so the loop runs offline. What is
under test is the control flow: that tool results feed back, that retrieved
chunk ids accumulate into the allow-list citations are checked against, that
the tool-call cap forces a submission rather than truncating, and that a model
answering in prose instead of calling the answer tool is recorded as a
protocol failure rather than parsed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from conftest_cited import StubRetriever, build_corpus  # noqa: E402

from ticker.cited.agent import (
    SYSTEM_PROMPT,
    TOOL_READ_NEIGHBORS,
    TOOL_SEARCH,
    TOOL_SUBMIT,
    answer_question,
    tool_definitions,
)
from ticker.cited.answers import SYSTEM_AGENT, SYSTEM_CLOSED, SYSTEM_SINGLE
from ticker.cited.baselines import answer_closed_book, answer_single_shot
from ticker.cited.config import Config
from ticker.cited.llm import CachedClient, Usage
from ticker.cited.questions import KIND_QUALITATIVE, SOURCE_FILING_TEXT, Question
from ticker.cited.retrieval import HybridSearcher


@pytest.fixture
def corpus():
    return build_corpus()


@pytest.fixture
def config(tmp_path):
    return Config(offline=True, llm_cache_dir=tmp_path / "llm", max_tool_calls=2)


QUESTION = Question(
    question_id="q01",
    kind=KIND_QUALITATIVE,
    text="What share of revenue came from outside the United States?",
    gold_answer="53%.",
    gold_source=SOURCE_FILING_TEXT,
    answerable=True,
    rationale="test",
    companies=("NVDA",),
    form_type="10-K",
    gold_chunk_ids=("0000000001-25-000001#1A#chunk0",),
)

CACHED_SYSTEM = [
    {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
]


def searcher(con, ranking):
    return HybridSearcher(con, {"stub": StubRetriever("stub", ranking)})


def tool_use(name, payload, block_id="toolu_1"):
    return {"type": "tool_use", "id": block_id, "name": name, "input": payload}


def submit(claims, declined=False, summary="A summary."):
    return tool_use(
        TOOL_SUBMIT,
        {"declined": declined, "summary": summary, "claims": claims},
        block_id="toolu_submit",
    )


def record_turn(client, config, messages, content, tool_choice=None):
    client.record(
        model=config.answer_model,
        system=CACHED_SYSTEM,
        messages=messages,
        tools=tool_definitions(config),
        tool_choice=tool_choice,
        content=content,
        usage=Usage(input_tokens=1000, output_tokens=100),
    )


def user_prompt(question=QUESTION):
    lines = [f"Question: {question.text}"]
    if question.companies:
        lines.append(f"Companies in scope: {', '.join(question.companies)}")
    if question.form_type:
        lines.append(f"Form: {question.form_type}")
    return [{"role": "user", "content": "\n".join(lines)}]


class TestToolDefinitions:
    def test_four_tools_in_a_fixed_order(self, config):
        # Tools render at the front of the prompt, so a reordering would
        # invalidate the cached prefix for every question.
        names = [tool["name"] for tool in tool_definitions(config)]
        assert names == ["search", "read_chunk", "read_neighbors", "submit_answer"]

    def test_submit_is_strict_and_closed(self, config):
        submit_tool = tool_definitions(config)[-1]
        assert submit_tool["strict"] is True
        schema = submit_tool["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"declined", "summary", "claims"}
        claim_schema = schema["properties"]["claims"]["items"]
        assert set(claim_schema["required"]) == {"text", "chunk_ids"}


class TestAgentLoop:
    def test_searches_then_submits(self, corpus, config):
        con, ids = corpus
        chunk = ids["nvda_2024"][0]
        client = CachedClient(config)

        first = [tool_use(TOOL_SEARCH, {"query": "revenue outside the United States"})]
        record_turn(client, config, user_prompt(), first)

        search = searcher(con, ids["nvda_2024"])
        result = search.search(
            "revenue outside the United States", company="NVDA",
            form_type="10-K", k=config.search_k,
        )
        from ticker.cited.agent import _format_search

        followup = user_prompt() + [
            {"role": "assistant", "content": first},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": _format_search(result, config.max_tool_calls - 1),
                    }
                ],
            },
        ]
        record_turn(
            client, config, followup,
            [submit([{"text": "53% came from outside the US.", "chunk_ids": [chunk]}])],
        )

        answer = answer_question(
            QUESTION, con=con, searcher=search, client=client, config=config
        )
        assert answer.system == SYSTEM_AGENT
        assert not answer.declined
        assert answer.claims[0].chunk_ids == (chunk,)
        assert answer.rejected_citations == ()
        assert [call.name for call in answer.tool_calls] == [TOOL_SEARCH]
        assert chunk in answer.retrieved_chunk_ids
        assert not answer.hit_tool_cap
        assert answer.error is None

    def test_a_fabricated_citation_is_rejected_even_after_searching(
        self, corpus, config
    ):
        con, ids = corpus
        client = CachedClient(config)
        record_turn(
            client, config, user_prompt(),
            [submit([{"text": "A claim.", "chunk_ids": ["0000000001-25-000001#1A#chunk99"]}])],
        )
        answer = answer_question(
            QUESTION, con=con, searcher=searcher(con, ids["nvda_2024"]),
            client=client, config=config,
        )
        assert answer.rejected_citations == ("0000000001-25-000001#1A#chunk99",)
        assert answer.claims[0].chunk_ids == ()

    def test_declining_discards_claims(self, corpus, config):
        # The schema says claims must be empty when declining, but a model can
        # still set both. Keeping the claims would let a system be scored as
        # correct on an unanswerable question it said it could not answer.
        con, ids = corpus
        client = CachedClient(config)
        record_turn(
            client, config, user_prompt(),
            [submit([{"text": "A claim.", "chunk_ids": []}], declined=True,
                    summary="Not in the filings.")],
        )
        answer = answer_question(
            QUESTION, con=con, searcher=searcher(con, ids["nvda_2024"]),
            client=client, config=config,
        )
        assert answer.declined and answer.claims == ()

    def test_prose_instead_of_a_tool_call_is_a_protocol_failure(self, corpus, config):
        # Parsing prose here would quietly exempt a system from the citation
        # requirement.
        con, ids = corpus
        client = CachedClient(config)
        record_turn(
            client, config, user_prompt(),
            [{"type": "text", "text": "Revenue was mostly international."}],
        )
        answer = answer_question(
            QUESTION, con=con, searcher=searcher(con, ids["nvda_2024"]),
            client=client, config=config,
        )
        assert answer.error == "model returned no tool call"
        assert answer.claims == ()

    def test_a_tool_error_is_returned_to_the_model_not_raised(self, corpus, config):
        # A malformed argument is recoverable on the next call; raising would
        # score a recoverable mistake as a system failure.
        con, ids = corpus
        client = CachedClient(config)
        first = [tool_use(TOOL_READ_NEIGHBORS, {})]  # missing chunk_id
        record_turn(client, config, user_prompt(), first)

        followup = user_prompt() + [
            {"role": "assistant", "content": first},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "Tool error: 'chunk_id'",
                        "is_error": True,
                    }
                ],
            },
        ]
        record_turn(
            client, config, followup,
            [submit([], declined=True, summary="Could not retrieve.")],
        )
        answer = answer_question(
            QUESTION, con=con, searcher=searcher(con, ids["nvda_2024"]),
            client=client, config=config,
        )
        assert answer.tool_calls[0].ok is False
        assert answer.tool_calls[0].note == "KeyError"
        assert answer.error is None

    def test_the_cap_forces_a_submission(self, corpus, config):
        con, ids = corpus
        client = CachedClient(config)
        search = searcher(con, ids["nvda_2024"])
        from ticker.cited.agent import _format_search

        messages = user_prompt()
        # Two searches, which is the configured cap.
        for index in range(config.max_tool_calls):
            call = [tool_use(TOOL_SEARCH, {"query": f"q{index}"}, f"toolu_{index}")]
            record_turn(client, config, messages, call)
            result = search.search(
                f"q{index}", company="NVDA", form_type="10-K", k=config.search_k
            )
            messages = messages + [
                {"role": "assistant", "content": call},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"toolu_{index}",
                            "content": _format_search(
                                result, config.max_tool_calls - index - 1
                            ),
                        }
                    ],
                },
            ]

        # The loop appends a budget-spent message and forces the answer tool.
        messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your tool call budget is spent. Submit your answer now from "
                    "what you have retrieved. If the passages you have do not "
                    "answer the question, decline."
                ),
            }
        ]
        record_turn(
            client, config, messages,
            [submit([{"text": "53% was international.",
                      "chunk_ids": [ids["nvda_2024"][0]]}])],
            tool_choice={"type": "tool", "name": TOOL_SUBMIT},
        )

        answer = answer_question(
            QUESTION, con=con, searcher=search, client=client, config=config
        )
        assert answer.hit_tool_cap
        assert len(answer.tool_calls) == config.max_tool_calls
        assert not answer.declined
        assert answer.claims[0].chunk_ids == (ids["nvda_2024"][0],)


class TestBaselines:
    def test_closed_book_citations_are_always_rejected(self, corpus, config):
        con, _ = corpus
        client = CachedClient(config)
        from ticker.cited.baselines import CLOSED_BOOK_SYSTEM, _submit_only_tools

        client.record(
            model=config.answer_model,
            system=[{"type": "text", "text": CLOSED_BOOK_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Question: {QUESTION.text}"}],
            tools=_submit_only_tools(config),
            tool_choice={"type": "tool", "name": TOOL_SUBMIT},
            content=[submit([{"text": "53% was international.",
                              "chunk_ids": ["0000000001-25-000001#1A#chunk0"]}])],
            usage=Usage(input_tokens=400, output_tokens=120),
        )
        answer = answer_closed_book(QUESTION, client=client, config=config)
        assert answer.system == SYSTEM_CLOSED
        assert answer.retrieved_chunk_ids == ()
        # It retrieved nothing, so any citation it offers is a guess.
        assert answer.rejected_citations == ("0000000001-25-000001#1A#chunk0",)
        assert answer.claims[0].chunk_ids == ()

    def test_single_shot_records_its_one_search(self, corpus, config):
        con, ids = corpus
        client = CachedClient(config)
        search = searcher(con, ids["nvda_2024"])
        from ticker.cited.baselines import SINGLE_SHOT_SYSTEM, _submit_only_tools

        result = search.search(
            QUESTION.text, company="NVDA", form_type="10-K", k=config.single_shot_k
        )
        passages = "\n\n".join(
            f"[{hit.chunk_id}] {hit.chunk.citation()}\n{hit.chunk.text}"
            for hit in result.hits
        )
        client.record(
            model=config.answer_model,
            system=[{"type": "text", "text": SINGLE_SHOT_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": f"Question: {QUESTION.text}\n\nRetrieved passages:\n\n{passages}",
            }],
            tools=_submit_only_tools(config),
            tool_choice={"type": "tool", "name": TOOL_SUBMIT},
            content=[submit([{"text": "53% was international.",
                              "chunk_ids": [ids["nvda_2024"][0]]}])],
            usage=Usage(input_tokens=2500, output_tokens=150),
        )
        answer = answer_single_shot(
            QUESTION, searcher=search, client=client, config=config
        )
        assert answer.system == SYSTEM_SINGLE
        assert len(answer.tool_calls) == 1
        assert answer.tool_calls[0].name == "search"
        # Retrieved, so the citation stands.
        assert answer.claims[0].chunk_ids == (ids["nvda_2024"][0],)
        assert answer.rejected_citations == ()

    def test_all_arms_share_one_answer_schema(self, config):
        # A schema change that applied to one arm and not another would make
        # the arms incomparable.
        from ticker.cited.baselines import _submit_only_tools

        agent_submit = tool_definitions(config)[-1]
        assert _submit_only_tools(config) == [agent_submit]
