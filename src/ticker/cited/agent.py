"""The cited-answer agent: three read tools, one terminal answer tool, a loop.

The answer is produced by a tool call, not by parsing prose. `submit_answer`
is declared `strict`, so the claim-and-citation structure is guaranteed by the
API rather than by a regex over free text that works until a model phrases
something differently. Declining is a field on that same schema, which makes
"the filings do not answer this" a first-class outcome the model selects
rather than a string the scorer has to recognise.

The tool-call cap is told to the model and counted down in each tool result.
A model that does not know its budget cannot spend it sensibly: it will either
stop early or get cut off mid-investigation with no chance to commit to an
answer. When the last call is spent the loop makes one final request with
`tool_choice` forcing `submit_answer`, so the cap produces a considered answer
under budget pressure rather than a truncation. `Answer.hit_tool_cap` records
when that happened, because an answer given under a forced stop is a different
object from one the model chose to give.

Every tool call is recorded with its arguments and the chunk ids it returned.
The returned ids accumulate into the session's retrieved set, which is the
allow-list `ticker.cited.answers.validate_citations` checks against -- so the
grounding guarantee comes from what the tools actually handed over, not from
what the model says it read.

The system prompt is a frozen constant and sits first in the request so the
prompt prefix is cacheable across all sixty questions. Interpolating the
question into it would make every request a distinct prefix and turn caching
off; the question goes in the user turn.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

from ticker.cited.answers import (
    SYSTEM_AGENT,
    Answer,
    Claim,
    ToolCall,
    validate_citations,
)
from ticker.cited.config import Config
from ticker.cited.corpus import read_chunk, read_neighbors
from ticker.cited.llm import CachedClient, Usage
from ticker.cited.questions import Question
from ticker.cited.retrieval import HybridSearcher

TOOL_SEARCH = "search"
TOOL_READ_CHUNK = "read_chunk"
TOOL_READ_NEIGHBORS = "read_neighbors"
TOOL_SUBMIT = "submit_answer"

READ_TOOLS = (TOOL_SEARCH, TOOL_READ_CHUNK, TOOL_READ_NEIGHBORS)

MAX_NEIGHBORS = 3

SYSTEM_PROMPT = """\
You answer questions about US public company filings using only the passages \
you retrieve with your tools.

The corpus holds 10-K, 10-Q and 8-K filings for twenty companies -- ten \
semiconductor companies and ten regional banks -- covering 2020 through 2025. \
From each 10-K it indexes items 1, 1A, 3, 7 and 7A only. It does not index \
item 5, item 8, the financial statement notes, the cover page, or proxy \
statements. From each 10-Q it indexes Part I item 2 and Part II items 1 and \
1A. From each 8-K it indexes the exhibit 99.1 earnings release.

Every passage has a chunk id of the form accession#item#chunkN. Cite by chunk \
id.

How to work:

- Search first. Use the company, form_type and fiscal_year filters when the \
question names them; fiscal_year matches the period the filing covers, not the \
year it was filed, so a 10-K for a year ending December 2024 has fiscal_year \
2024 even though it was filed in 2025.
- Read neighbouring passages when a figure or sentence appears to continue \
past the edge of a chunk. Filing tables often put the scale ("in millions") in \
a different passage from the number.
- You have a limited number of tool calls. Each tool result tells you how many \
remain. Spend them on the question asked.

How to answer:

- Call submit_answer exactly once, at the end.
- Break the answer into claims. Each claim is one assertion a reader could \
check against a single passage, and each must cite the chunk ids that support \
it.
- You may only cite chunk ids that your tool calls returned to you in this \
session. Do not construct a chunk id, and do not cite a passage you have not \
seen. A citation to an id you did not retrieve is rejected and counted against \
the answer.
- Give figures with their units and scale, and name the period and the filing. \
"Revenue was 53,101" is not an answer; "revenue was $53.1 billion for the year \
ended December 28, 2024" is.
- If the indexed filings do not answer the question, set declined to true and \
say what is missing. Declining is the correct answer when the corpus does not \
carry the information, and is better than an unsupported guess. Do not decline \
merely because the answer took several searches to find.
"""


def tool_definitions(config: Config) -> list[dict]:
    """The four tools, in a fixed order.

    Order is fixed because tools render at the front of the prompt and a
    reordering would invalidate the cached prefix for every question.
    """
    return [
        {
            "name": TOOL_SEARCH,
            "description": (
                "Hybrid BM25 and dense retrieval over the indexed filing passages, "
                "fused by reciprocal rank. Returns chunk ids with their text and "
                "filing metadata. Use the filters to narrow to a company, form or "
                "fiscal year; fiscal_year is the year the filing's period ends."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language or keyword query.",
                    },
                    "company": {
                        "type": "string",
                        "description": "Ticker to restrict to, e.g. NVDA. Omit for all.",
                    },
                    "form_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q", "8-K"],
                        "description": "Filing form to restrict to. Omit for all.",
                    },
                    "fiscal_year": {
                        "type": "integer",
                        "description": (
                            "Year the filing's reporting period ends, 2019 to 2025. "
                            "Omit for all."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": f"Passages to return, 1 to 20. Default {config.search_k}.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": TOOL_READ_CHUNK,
            "description": "Fetch one passage by chunk id, with its filing metadata.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "accession#item#chunkN"}
                },
                "required": ["chunk_id"],
            },
        },
        {
            "name": TOOL_READ_NEIGHBORS,
            "description": (
                "Fetch a passage together with the n passages either side of it "
                "within the same filing item, in document order. Use this when a "
                "sentence or table appears to be cut off."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "accession#item#chunkN"},
                    "n": {
                        "type": "integer",
                        "description": f"Passages either side, 1 to {MAX_NEIGHBORS}. Default 1.",
                    },
                },
                "required": ["chunk_id"],
            },
        },
        {
            "name": TOOL_SUBMIT,
            "description": (
                "Submit the final answer. Call exactly once. Either set declined "
                "to true, or give claims each citing the chunk ids that support it."
            ),
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "declined": {
                        "type": "boolean",
                        "description": (
                            "True if the indexed filings do not answer the question. "
                            "When true, claims must be empty."
                        ),
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "One or two sentences answering the question, or saying "
                            "what is missing if declining."
                        ),
                    },
                    "claims": {
                        "type": "array",
                        "description": "Empty when declining.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "One checkable assertion.",
                                },
                                "chunk_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "Chunk ids supporting this claim. Must have "
                                        "been returned by a tool call in this session."
                                    ),
                                },
                            },
                            "required": ["text", "chunk_ids"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["declined", "summary", "claims"],
                "additionalProperties": False,
            },
        },
    ]


def _render_hit(chunk_id: str, text: str, citation: str, score: float | None) -> str:
    head = f"[{chunk_id}] {citation}"
    if score is not None:
        head += f" (score {score:.4f})"
    return f"{head}\n{text}"


def _format_search(result, remaining: int) -> str:
    if not result.hits:
        return (
            f"No passages matched those filters. Pool searched: {result.pool_size} "
            f"candidates. Try a broader query or drop a filter. "
            f"Tool calls remaining: {remaining}."
        )
    body = "\n\n".join(
        _render_hit(hit.chunk_id, hit.chunk.text, hit.chunk.citation(), hit.score)
        for hit in result.hits
    )
    note = ""
    if result.exhausted_pool:
        note = (
            "\n\nFewer passages than requested matched these filters; the corpus "
            "may hold no more that match."
        )
    return f"{body}{note}\n\nTool calls remaining: {remaining}."


def _format_chunks(records, remaining: int) -> str:
    if not records:
        return f"No such passage. Tool calls remaining: {remaining}."
    body = "\n\n".join(
        _render_hit(record.chunk_id, record.text, record.citation(), None)
        for record in records
    )
    return f"{body}\n\nTool calls remaining: {remaining}."


class ToolRunner:
    """Executes the read tools and accumulates what they returned."""

    def __init__(self, con, searcher: HybridSearcher, config: Config) -> None:
        self._con = con
        self._searcher = searcher
        self._config = config
        self.retrieved: dict[str, None] = {}
        self.calls: list[ToolCall] = []

    def run(self, index: int, name: str, arguments: dict, remaining: int) -> tuple[str, bool]:
        started = time.perf_counter()
        try:
            text, returned = self._dispatch(name, arguments, remaining)
            ok = True
            note = ""
        except Exception as exc:  # surfaced to the model, not raised
            # A malformed argument is something the model can correct on its
            # next call, so the error goes back as a tool result rather than
            # ending the run. Raising here would score a recoverable mistake
            # as a system failure.
            text, returned, ok = f"Tool error: {exc}", (), False
            note = type(exc).__name__

        for chunk_id in returned:
            self.retrieved.setdefault(chunk_id, None)
        self.calls.append(
            ToolCall(
                index=index,
                name=name,
                arguments=arguments,
                ok=ok,
                chunk_ids_returned=tuple(returned),
                latency_ms=(time.perf_counter() - started) * 1000,
                note=note,
            )
        )
        return text, ok

    def _dispatch(self, name: str, arguments: dict, remaining: int) -> tuple[str, tuple[str, ...]]:
        if name == TOOL_SEARCH:
            k = int(arguments.get("k") or self._config.search_k)
            result = self._searcher.search(
                arguments["query"],
                company=arguments.get("company"),
                form_type=arguments.get("form_type"),
                fiscal_year=arguments.get("fiscal_year"),
                k=max(1, min(k, 20)),
            )
            return _format_search(result, remaining), tuple(result.chunk_ids)

        if name == TOOL_READ_CHUNK:
            record = read_chunk(self._con, arguments["chunk_id"])
            records = [record] if record else []
            return _format_chunks(records, remaining), tuple(r.chunk_id for r in records)

        if name == TOOL_READ_NEIGHBORS:
            n = int(arguments.get("n") or 1)
            records = read_neighbors(
                self._con, arguments["chunk_id"], max(0, min(n, MAX_NEIGHBORS))
            )
            return _format_chunks(records, remaining), tuple(r.chunk_id for r in records)

        raise ValueError(f"unknown tool {name!r}")


def _question_prompt(question: Question) -> str:
    lines = [f"Question: {question.text}"]
    if question.companies:
        lines.append(f"Companies in scope: {', '.join(question.companies)}")
    if question.form_type:
        lines.append(f"Form: {question.form_type}")
    return "\n".join(lines)


def answer_from_tool_input(payload: dict) -> tuple[bool, str, tuple[Claim, ...]]:
    declined = bool(payload.get("declined"))
    summary = (payload.get("summary") or "").strip()
    claims = tuple(
        Claim(
            text=(item.get("text") or "").strip(),
            chunk_ids=tuple(item.get("chunk_ids") or []),
        )
        for item in (payload.get("claims") or [])
        if (item.get("text") or "").strip()
    )
    # The schema says claims must be empty when declining, but a model can
    # still set both. Declining wins: a system that says it cannot answer and
    # then answers anyway has declined, and keeping the claims would let it be
    # scored as a correct answer on an unanswerable question.
    if declined:
        claims = ()
    return declined, summary, claims


def answer_question(
    question: Question,
    *,
    con,
    searcher: HybridSearcher,
    client: CachedClient,
    config: Config,
) -> Answer:
    """Run the agent loop for one question and return a validated answer."""
    tools = tool_definitions(config)
    runner = ToolRunner(con, searcher, config)
    messages: list[dict] = [{"role": "user", "content": _question_prompt(question)}]

    usage = Usage()
    started = time.perf_counter()
    all_from_cache = True
    stop_reason: str | None = None
    hit_cap = False
    error: str | None = None
    declined, summary, claims = False, "", ()

    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    tool_calls_used = 0
    submitted = False

    while not submitted:
        remaining = config.max_tool_calls - tool_calls_used
        force_submit = remaining <= 0
        if force_submit:
            hit_cap = True
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your tool call budget is spent. Submit your answer now from "
                        "what you have retrieved. If the passages you have do not "
                        "answer the question, decline."
                    ),
                }
            )

        try:
            response = client.create(
                model=config.answer_model,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice={"type": "tool", "name": TOOL_SUBMIT} if force_submit else None,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break

        usage += response.usage
        all_from_cache = all_from_cache and response.from_cache
        stop_reason = response.stop_reason

        tool_uses = response.tool_uses()
        if not tool_uses:
            # No tool call and no submission: the model answered in prose. That
            # is a protocol failure, recorded as one rather than parsed out of
            # the text, because accepting prose here would quietly exempt a
            # system from the citation requirement.
            error = error or "model returned no tool call"
            summary = response.text()
            break

        messages.append({"role": "assistant", "content": response.content})

        results: list[dict] = []
        for block in tool_uses:
            name = block.get("name", "")
            arguments = block.get("input") or {}
            if name == TOOL_SUBMIT:
                declined, summary, claims = answer_from_tool_input(arguments)
                submitted = True
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": "Answer recorded.",
                    }
                )
                continue

            tool_calls_used += 1
            text, ok = runner.run(
                tool_calls_used,
                name,
                arguments,
                max(0, config.max_tool_calls - tool_calls_used),
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": text,
                    **({"is_error": True} if not ok else {}),
                }
            )

        messages.append({"role": "user", "content": results})

    answer = Answer(
        question_id=question.question_id,
        system=SYSTEM_AGENT,
        model=config.answer_model,
        declined=declined,
        summary=summary,
        claims=claims,
        tool_calls=tuple(runner.calls),
        retrieved_chunk_ids=tuple(runner.retrieved),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cost_usd=usage.cost_usd(config.answer_model),
        latency_s=time.perf_counter() - started,
        from_cache=all_from_cache,
        stop_reason=stop_reason,
        hit_tool_cap=hit_cap,
        error=error,
    )
    return validate_citations(answer)
