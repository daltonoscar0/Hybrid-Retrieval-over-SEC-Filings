"""The two LLM judges, and the rubrics they are held to.

Rubrics are module constants rather than inline strings so they are
reviewable, diffable, and versioned with `RUBRIC_VERSION`. The judge-agreement
check is only meaningful against a named rubric version: if the rubric is
edited after agreement was measured, the measured agreement no longer applies
to the judge that produced the reported numbers, and the version string is
what makes that visible instead of silent.

**The citation judge sees the claim and the cited text and nothing else.** Not
the question, not the gold answer, not the other claims, not which system
produced it. This is the single most important design choice here. A judge
that can see the gold answer will mark a citation supported because the claim
is true, and the question being asked is not "is this claim true" but "does
this passage support it" -- a claim can be perfectly correct and cited to a
passage that says nothing about it, which is exactly the failure a citation
metric exists to catch. Blinding the judge is also what makes the human
agreement check cheap to run: Oscar labels the same two fields, so he and the
judge are answering the same question from the same evidence.

**The answer judge** sees the question, the gold answer and the system's
answer. It does not see the citations or the retrieved passages, because
whether the answer is right is a separate measurement from whether it is
grounded, and a judge shown good citations will mark a wrong answer right.

Both judges return a structured verdict through a strict tool schema and run
through the cached client, so re-scoring never re-pays and a verdict is
reproducible from the cache.
"""

from __future__ import annotations

from dataclasses import dataclass

from ticker.cited.config import Config
from ticker.cited.llm import CachedClient, Usage

RUBRIC_VERSION = "1"

SUPPORT_SUPPORTED = "supported"
SUPPORT_PARTIAL = "partially_supported"
SUPPORT_NOT = "not_supported"

SUPPORT_LABELS = (SUPPORT_SUPPORTED, SUPPORT_PARTIAL, SUPPORT_NOT)

CITATION_RUBRIC = """\
You are checking whether a passage from a company's SEC filing supports a \
claim someone made while citing that passage.

You will be shown only the claim and the passage. You will not be shown the \
question that prompted the claim, and you must not try to infer whether the \
claim is true in general. Judge only one thing: can a reader who has this \
passage in front of them, and nothing else, verify what the claim asserts?

Rate the claim as exactly one of:

supported
  Everything the claim asserts is stated in the passage, or follows from the \
passage by reading it plainly. Figures match, including their scale and \
period. Paraphrase is fine. A claim that restates the passage in different \
words is supported. An arithmetic step the passage supplies both operands for \
is supported.

partially_supported
  The passage supports part of what the claim asserts and is silent on the \
rest. Typical cases: the claim names two facts and the passage carries one; \
the claim gives a figure the passage carries but attributes it to a period the \
passage does not name; the claim adds a causal or comparative statement the \
passage does not make.

not_supported
  The passage does not support the claim. It may be about a different topic, a \
different company, or a different period, or it may state something the claim \
contradicts. A passage that is merely topically related, without carrying what \
the claim asserts, is not_supported.

Rules that decide the hard cases:

- Silence is not support. If the passage does not say it, the claim is not \
supported on that point, however plausible it is.
- A figure must match in scale. A passage reading "53,101" in a table \
supports a claim of "$53.1 billion" when the table's units are evident or the \
magnitude is unambiguous, but a claim of "$53,101" is not supported by it.
- Contradiction is not_supported, not partially_supported, even if the topic \
matches closely.
- Judge the passage as given. Do not assume a neighbouring passage would \
supply the missing part.
"""

ANSWER_RUBRIC = """\
You are marking an answer to a question about SEC filings against a gold \
answer written from the filings in advance.

Mark the answer correct if it conveys what the gold answer conveys on every \
point the question asked about. Mark it incorrect otherwise.

Apply these rules:

- Wording does not matter. A correct answer in different words is correct.
- Extra correct detail does not make an answer incorrect.
- Extra incorrect detail does make it incorrect, if it bears on what the \
question asked.
- A missing part makes it incorrect. If the question asks two things and the \
answer gives one, that is incorrect, not partially correct.
- Figures must agree with the gold, including scale and period. A figure in \
the wrong units or attributed to the wrong year is incorrect.
- An answer that declines to answer is incorrect here. These questions are \
answerable from the filings; declining is scored separately and is never \
correct on this set.
- Hedging is acceptable if the substance is right. "Approximately 45%" against \
a gold of 45% is correct.
"""


@dataclass(frozen=True, slots=True)
class CitationVerdict:
    label: str
    reason: str
    usage: Usage
    from_cache: bool

    @property
    def is_supported(self) -> bool:
        return self.label == SUPPORT_SUPPORTED

    @property
    def is_supported_or_partial(self) -> bool:
        return self.label in (SUPPORT_SUPPORTED, SUPPORT_PARTIAL)


@dataclass(frozen=True, slots=True)
class AnswerVerdict:
    correct: bool
    reason: str
    usage: Usage
    from_cache: bool


_CITATION_TOOL = {
    "name": "rate_support",
    "description": "Record whether the passage supports the claim.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": list(SUPPORT_LABELS),
                "description": "The support rating.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "One sentence naming what in the passage does or does not "
                    "carry the claim."
                ),
            },
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    },
}

_ANSWER_TOOL = {
    "name": "mark_answer",
    "description": "Record whether the answer matches the gold answer.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "correct": {"type": "boolean", "description": "True if correct."},
            "reason": {
                "type": "string",
                "description": "One sentence giving the deciding difference or match.",
            },
        },
        "required": ["correct", "reason"],
        "additionalProperties": False,
    },
}


def _first_tool_input(response, name: str) -> dict | None:
    for block in response.tool_uses():
        if block.get("name") == name:
            return block.get("input") or {}
    return None


def judge_citation(
    claim_text: str,
    passage: str,
    *,
    client: CachedClient,
    config: Config,
) -> CitationVerdict:
    """Rate whether `passage` supports `claim_text`, seeing nothing else."""
    response = client.create(
        model=config.judge_model,
        system=[
            {
                "type": "text",
                "text": CITATION_RUBRIC,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Claim:\n{claim_text}\n\nPassage:\n{passage}",
            }
        ],
        tools=[_CITATION_TOOL],
        tool_choice={"type": "tool", "name": "rate_support"},
    )
    payload = _first_tool_input(response, "rate_support") or {}
    label = payload.get("label")
    if label not in SUPPORT_LABELS:
        # An unusable verdict is recorded as not_supported with the reason
        # saying so, rather than dropped. Dropping would shrink the
        # denominator and quietly inflate the support rate.
        return CitationVerdict(
            label=SUPPORT_NOT,
            reason=f"judge returned no usable label ({label!r})",
            usage=response.usage,
            from_cache=response.from_cache,
        )
    return CitationVerdict(
        label=label,
        reason=(payload.get("reason") or "").strip(),
        usage=response.usage,
        from_cache=response.from_cache,
    )


def judge_answer(
    question_text: str,
    gold_answer: str,
    candidate: str,
    *,
    client: CachedClient,
    config: Config,
) -> AnswerVerdict:
    """Mark `candidate` against `gold_answer` for `question_text`."""
    response = client.create(
        model=config.judge_model,
        system=[
            {
                "type": "text",
                "text": ANSWER_RUBRIC,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question:\n{question_text}\n\n"
                    f"Gold answer:\n{gold_answer}\n\n"
                    f"Answer to mark:\n{candidate}"
                ),
            }
        ],
        tools=[_ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "mark_answer"},
    )
    payload = _first_tool_input(response, "mark_answer") or {}
    if "correct" not in payload:
        return AnswerVerdict(
            correct=False,
            reason="judge returned no usable verdict",
            usage=response.usage,
            from_cache=response.from_cache,
        )
    return AnswerVerdict(
        correct=bool(payload["correct"]),
        reason=(payload.get("reason") or "").strip(),
        usage=response.usage,
        from_cache=response.from_cache,
    )
