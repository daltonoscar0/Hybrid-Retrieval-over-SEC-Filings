"""Answer records, the on-disk format, and citation validation.

An answer is a list of claims, each attached to the chunk ids that support it,
plus an explicit declining option. The shape is enforced at generation time by
a strict tool schema (see `ticker.cited.agent`); this module is where it is
checked after the fact and where it is persisted.

Citation validation is the part worth reading. A system may only cite a chunk
it actually retrieved in that session. A chunk id that was never returned by
any tool call is not a citation, it is a well-formed guess: chunk ids are
structural (`{accession}#{item}#chunk{n}`) and so are trivially fabricable
from a known accession, which means an unchecked citation field would let a
system look fully grounded while having read nothing.

Rather than discard an answer with an invalid citation, validation moves the
offending ids into `rejected_citations` and keeps the claim with whatever
valid ids remain. Two reasons. A claim can cite three chunks of which one is
fabricated, and throwing away the whole answer would lose the information that
the other two were real. And "cited a chunk it never retrieved" is a failure
category the analysis needs to count per claim, which it cannot do if such
answers never reach the scorer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

SYSTEM_CLOSED = "closed"
SYSTEM_SINGLE = "single"
SYSTEM_AGENT = "agent"

SYSTEMS = (SYSTEM_CLOSED, SYSTEM_SINGLE, SYSTEM_AGENT)

DECLINED_MARKER = "not answerable from the indexed filings"


@dataclass(frozen=True, slots=True)
class Claim:
    """One assertion and the passages offered as support for it."""

    text: str
    chunk_ids: tuple[str, ...] = ()

    @property
    def is_uncited(self) -> bool:
        return not self.chunk_ids


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation, recorded whether or not it helped.

    The whole sequence is kept, not a summary: the failure analysis needs to
    distinguish a wrong answer reached in one search from the same answer
    reached after six, and to see when a filter was set to the wrong year.
    """

    index: int
    name: str
    arguments: dict
    ok: bool
    chunk_ids_returned: tuple[str, ...] = ()
    latency_ms: float = 0.0
    note: str = ""


@dataclass(frozen=True, slots=True)
class Answer:
    question_id: str
    system: str
    model: str

    declined: bool = False
    summary: str = ""
    claims: tuple[Claim, ...] = ()

    tool_calls: tuple[ToolCall, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()
    rejected_citations: tuple[str, ...] = ()

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    from_cache: bool = False

    stop_reason: str | None = None
    hit_tool_cap: bool = False
    error: str | None = None

    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.system not in SYSTEMS:
            raise ValueError(f"unknown system {self.system!r}; expected one of {SYSTEMS}")
        if self.declined and self.claims:
            raise ValueError(
                f"{self.question_id}: declined answer also carries claims; "
                "declining and asserting are mutually exclusive"
            )

    @property
    def cited_chunk_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for claim in self.claims:
            for chunk_id in claim.chunk_ids:
                seen.setdefault(chunk_id, None)
        return tuple(seen)

    @property
    def answer_text(self) -> str:
        """The answer as prose, for a judge that compares against a gold."""
        if self.declined:
            return DECLINED_MARKER
        parts = [self.summary] if self.summary else []
        parts.extend(claim.text for claim in self.claims)
        return "\n".join(part for part in parts if part).strip()


def validate_citations(answer: Answer) -> Answer:
    """Strip citations to chunks the session never retrieved.

    Comparison is exact string equality against the ids the tools returned.
    Nothing is normalised and nothing is fuzzily matched: a near-miss id is a
    fabrication, and silently repairing it would hide exactly the behaviour
    this check exists to surface.

    A closed-book answer retrieves nothing, so every citation it offers is
    rejected. That is the correct result and is the point of running that arm.
    """
    allowed = set(answer.retrieved_chunk_ids)
    rejected: dict[str, None] = {}
    cleaned: list[Claim] = []

    for claim in answer.claims:
        kept = tuple(chunk_id for chunk_id in claim.chunk_ids if chunk_id in allowed)
        for chunk_id in claim.chunk_ids:
            if chunk_id not in allowed:
                rejected.setdefault(chunk_id, None)
        cleaned.append(replace(claim, chunk_ids=kept))

    return replace(
        answer,
        claims=tuple(cleaned),
        rejected_citations=tuple(rejected),
    )


def _claim_to_dict(claim: Claim) -> dict:
    return {"text": claim.text, "chunk_ids": list(claim.chunk_ids)}


def to_json(answer: Answer) -> str:
    payload = asdict(answer)
    payload["claims"] = [_claim_to_dict(claim) for claim in answer.claims]
    payload["tool_calls"] = [
        {**asdict(call), "chunk_ids_returned": list(call.chunk_ids_returned)}
        for call in answer.tool_calls
    ]
    for key in ("retrieved_chunk_ids", "rejected_citations", "tags"):
        payload[key] = list(payload[key])
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def from_json(line: str) -> Answer:
    payload = json.loads(line)
    payload["claims"] = tuple(
        Claim(text=item["text"], chunk_ids=tuple(item.get("chunk_ids") or []))
        for item in payload.get("claims") or []
    )
    payload["tool_calls"] = tuple(
        ToolCall(
            index=item["index"],
            name=item["name"],
            arguments=item.get("arguments") or {},
            ok=item.get("ok", True),
            chunk_ids_returned=tuple(item.get("chunk_ids_returned") or []),
            latency_ms=item.get("latency_ms", 0.0),
            note=item.get("note", ""),
        )
        for item in payload.get("tool_calls") or []
    )
    for key in ("retrieved_chunk_ids", "rejected_citations", "tags"):
        payload[key] = tuple(payload.get(key) or [])
    return Answer(**payload)


def write_answers(path: Path, answers: Iterable[Answer]) -> int:
    ordered = sorted(answers, key=lambda answer: answer.question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for answer in ordered:
            handle.write(to_json(answer) + "\n")
    return len(ordered)


def iter_answers(path: Path) -> Iterator[Answer]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield from_json(line)


def load_answers(path: Path) -> list[Answer]:
    return list(iter_answers(path))
