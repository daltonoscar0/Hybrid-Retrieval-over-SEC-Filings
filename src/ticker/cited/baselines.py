"""The two baselines the agent is measured against.

Both answer the same frozen questions, in the same claim-and-citation schema,
through the same cached client, so the only thing that differs between the
three arms is how much retrieval each gets.

**Closed book.** The model, the question, and no tools but `submit_answer`.
Its retrieved set is empty by construction, so `validate_citations` rejects
every citation it offers. That is the measurement, not a defect: the count of
rejected citations is a direct count of fabricated chunk ids, and the figures
in its answers are a direct count of recalled-or-invented numbers. It is the
arm that says how much of the agent's score is retrieval and how much was
already in the weights.

**Single shot.** One retrieval against the question text, top-k passages
pasted into the prompt, one generation, no loop. It separates "retrieval
helps" from "the agent loop helps". Its query is the question verbatim,
deliberately: letting the model rewrite the query first would make it a
one-step agent and blur the comparison the arm exists to draw.

The filters come from the question record rather than from the model, which is
the most favourable setup for this baseline -- it gets the company and form
handed to it. The agent has to work them out. Where the agent beats single
shot, it beats a baseline that was given the scoping for free.
"""

from __future__ import annotations

import time

from ticker.cited.agent import (
    SYSTEM_PROMPT,
    TOOL_SUBMIT,
    answer_from_tool_input,
    tool_definitions,
)
from ticker.cited.answers import (
    SYSTEM_CLOSED,
    SYSTEM_SINGLE,
    Answer,
    ToolCall,
    validate_citations,
)
from ticker.cited.config import Config
from ticker.cited.llm import CachedClient
from ticker.cited.questions import Question
from ticker.cited.retrieval import HybridSearcher

CLOSED_BOOK_SYSTEM = """\
You answer questions about US public company filings from your own knowledge. \
You have no retrieval tools in this session.

Call submit_answer exactly once. Break the answer into claims. Give figures \
with their units, scale and period.

The cite field expects chunk ids of the form accession#item#chunkN from an \
indexed filing corpus. You have not retrieved anything in this session, so \
leave chunk_ids empty for every claim.

If you do not know the answer, set declined to true rather than estimating. An \
approximate figure presented as a reported one is worse than declining.
"""

SINGLE_SHOT_SYSTEM = (
    SYSTEM_PROMPT
    + """
In this session you have no search tools. The passages you may use are given \
in the user message below, each with its chunk id. Cite only those chunk ids. \
If they do not answer the question, decline.
"""
)


def _submit_only_tools(config: Config) -> list[dict]:
    """Just `submit_answer`, so the schema is identical across arms.

    Reusing the agent's definition rather than restating it means a change to
    the answer schema cannot apply to one arm and not another, which would
    make the arms incomparable.
    """
    return [tool for tool in tool_definitions(config) if tool["name"] == TOOL_SUBMIT]


def _finish(
    question: Question,
    system_name: str,
    config: Config,
    response,
    *,
    retrieved: tuple[str, ...],
    tool_calls: tuple[ToolCall, ...],
    started: float,
) -> Answer:
    declined, summary, claims = False, "", ()
    error = None
    submissions = [
        block for block in response.tool_uses() if block.get("name") == TOOL_SUBMIT
    ]
    if submissions:
        declined, summary, claims = answer_from_tool_input(submissions[0].get("input") or {})
    else:
        error = "model returned no submit_answer call"
        summary = response.text()

    answer = Answer(
        question_id=question.question_id,
        system=system_name,
        model=config.answer_model,
        declined=declined,
        summary=summary,
        claims=claims,
        tool_calls=tool_calls,
        retrieved_chunk_ids=retrieved,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=response.usage.cache_read_tokens,
        cache_write_tokens=response.usage.cache_write_tokens,
        cost_usd=response.usage.cost_usd(config.answer_model),
        latency_s=time.perf_counter() - started,
        from_cache=response.from_cache,
        stop_reason=response.stop_reason,
        error=error,
    )
    return validate_citations(answer)


def answer_closed_book(
    question: Question, *, client: CachedClient, config: Config
) -> Answer:
    started = time.perf_counter()
    system = [
        {
            "type": "text",
            "text": CLOSED_BOOK_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    response = client.create(
        model=config.answer_model,
        system=system,
        messages=[{"role": "user", "content": f"Question: {question.text}"}],
        tools=_submit_only_tools(config),
        tool_choice={"type": "tool", "name": TOOL_SUBMIT},
    )
    return _finish(
        question,
        SYSTEM_CLOSED,
        config,
        response,
        retrieved=(),
        tool_calls=(),
        started=started,
    )


def answer_single_shot(
    question: Question,
    *,
    searcher: HybridSearcher,
    client: CachedClient,
    config: Config,
) -> Answer:
    started = time.perf_counter()
    search_started = time.perf_counter()
    result = searcher.search(
        question.text,
        company=question.company,
        form_type=question.form_type,
        fiscal_year=question.fiscal_year,
        k=config.single_shot_k,
    )
    search_ms = (time.perf_counter() - search_started) * 1000

    passages = "\n\n".join(
        f"[{hit.chunk_id}] {hit.chunk.citation()}\n{hit.chunk.text}"
        for hit in result.hits
    )
    body = (
        f"Question: {question.text}\n\n"
        f"Retrieved passages:\n\n{passages}"
        if result.hits
        else f"Question: {question.text}\n\nNo passages were retrieved."
    )

    system = [
        {
            "type": "text",
            "text": SINGLE_SHOT_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    response = client.create(
        model=config.answer_model,
        system=system,
        messages=[{"role": "user", "content": body}],
        tools=_submit_only_tools(config),
        tool_choice={"type": "tool", "name": TOOL_SUBMIT},
    )
    return _finish(
        question,
        SYSTEM_SINGLE,
        config,
        response,
        retrieved=tuple(result.chunk_ids),
        tool_calls=(
            ToolCall(
                index=1,
                name="search",
                arguments={
                    "query": question.text,
                    "company": question.company,
                    "form_type": question.form_type,
                    "fiscal_year": question.fiscal_year,
                    "k": config.single_shot_k,
                },
                ok=True,
                chunk_ids_returned=tuple(result.chunk_ids),
                latency_ms=search_ms,
            ),
        ),
        started=started,
    )
