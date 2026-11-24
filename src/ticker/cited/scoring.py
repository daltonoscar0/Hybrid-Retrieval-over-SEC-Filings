"""Scoring one answer, and the rules that decide each metric.

Five measurements per answer, each with a stated method so a reported number
can be traced to the rule that produced it.

**Answer correctness** branches on question kind, because the kinds are not
comparable. A numeric answer is checked arithmetically against an XBRL value
with no model in the loop. An unanswerable question is correct only if the
system declined. Everything else goes to the answer judge. The branch matters:
scoring all three the same way would let a judge grade arithmetic, and would
score declining on an unanswerable question by comparing prose to prose.

**Citation support** is per claim, and prefers the deterministic path. A claim
asserting figures is checked by looking for those figures in the cited text --
no judge, no rubric, no error rate. Only a claim with no reported figure in it
goes to the citation judge. That keeps the judge off the cases where it adds
risk without adding anything, and it means the numeric arm's support rate is
not a judge-dependent number at all.

**Citation recall** asks whether any cited chunk is one the gold answer was
drafted from. It applies only where a gold chunk set exists, so numeric
questions (whose gold is an XBRL value, not a passage) and unanswerable
questions (which have no gold passage by construction) are excluded rather
than scored as zero.

**Hallucinated figures** are reported quantities in the answer that appear in
no cited chunk. A figure in a declined answer counts: a system that declines
and then states a number has still produced an unsupported figure.

**Citations to chunks never retrieved** are counted from the answer's
`rejected_citations`, which `validate_citations` filled in before the answer
was ever written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from ticker.cited.answers import Answer
from ticker.cited.config import Config
from ticker.cited.corpus import load_chunks
from ticker.cited.judge import (
    SUPPORT_NOT,
    SUPPORT_PARTIAL,
    SUPPORT_SUPPORTED,
    RUBRIC_VERSION,
    judge_answer,
    judge_citation,
)
from ticker.cited.llm import CachedClient, Usage
from ticker.cited.questions import (
    KIND_MULTI_FILING,
    KIND_NUMERIC,
    KIND_QUALITATIVE,
    KIND_UNANSWERABLE,
    Question,
)
from ticker.cited.units import (
    DEFAULT_RELATIVE_TOLERANCE,
    DIRECTION_DOWN,
    DIRECTION_UNSTATED,
    DIRECTION_UP,
    PERCENTAGE_POINT_TOLERANCE,
    number_appears_in,
    reportable_quantities,
    stated_direction,
    within_tolerance,
)

SUPPORT_UNCITED = "uncited"

METHOD_NUMERIC = "numeric_tolerance"
METHOD_DECLINED = "declined_check"
METHOD_JUDGE = "llm_judge"
METHOD_DETERMINISTIC = "number_in_passage"
METHOD_UNCITED = "no_citation"


@dataclass(frozen=True, slots=True)
class ClaimScore:
    index: int
    text: str
    chunk_ids: tuple[str, ...]
    support: str
    method: str
    reason: str
    figures_checked: int = 0
    figures_found: int = 0


@dataclass(frozen=True, slots=True)
class AnswerScore:
    question_id: str
    kind: str
    system: str

    answer_correct: bool
    correctness_method: str
    correctness_detail: str
    declined: bool

    claim_scores: tuple[ClaimScore, ...] = ()
    citation_recall: bool | None = None
    hallucinated_figures: tuple[str, ...] = ()
    figures_in_answer: int = 0
    rejected_citations: tuple[str, ...] = ()

    cost_usd: float | None = None
    judge_cost_usd: float | None = None
    latency_s: float = 0.0
    tool_calls: int = 0
    hit_tool_cap: bool = False
    error: str | None = None

    rubric_version: str = RUBRIC_VERSION
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def cited_claims(self) -> tuple[ClaimScore, ...]:
        return tuple(score for score in self.claim_scores if score.chunk_ids)

    @property
    def citation_support_rate(self) -> float | None:
        """Share of cited claims rated fully supported.

        None when the answer makes no cited claim. Returning zero instead
        would average a declined answer in as a support failure, when it made
        no citation to fail.
        """
        cited = self.cited_claims
        if not cited:
            return None
        return sum(1 for score in cited if score.support == SUPPORT_SUPPORTED) / len(cited)

    @property
    def uncited_claim_count(self) -> int:
        return sum(1 for score in self.claim_scores if not score.chunk_ids)


def _cited_text(
    con: duckdb.DuckDBPyConnection, chunk_ids: tuple[str, ...]
) -> str:
    records = load_chunks(con, list(chunk_ids))
    return "\n\n".join(
        records[chunk_id].text for chunk_id in chunk_ids if chunk_id in records
    )


def score_numeric_answer(
    question: Question,
    answer: Answer,
    *,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    percentage_point_tolerance: float = PERCENTAGE_POINT_TOLERANCE,
) -> tuple[bool, str]:
    """Check a numeric answer against its XBRL gold, arithmetically.

    Every reported quantity in the answer is tried, and the answer is correct
    if any of them matches the gold. That is deliberate: a correct answer to a
    year-over-year question usually states the change and both operands, and
    requiring the gold to be the only figure present would fail it for showing
    its work.

    No scale is guessed. A quantity is read with the scale the answer gave it,
    so "revenue was 53,101" is fifty-three thousand and fails against a gold
    of $53.101 billion. That is a unit error, and it is one of the failure
    categories worth counting rather than forgiving.

    A percentage gold is matched only by a percent-marked quantity, so an
    answer that recites the two operands without computing the change does not
    pass by having the right digits somewhere in it. For such a gold the
    magnitude is compared unsigned and the direction is read from the answer's
    wording, because "revenue declined 2.1%" is how a change of -2.08% is
    actually written; see `stated_direction`. An answer that states a magnitude
    with no direction at all fails, since it has not said which way the figure
    went.
    """
    if question.gold_value is None:
        return False, "question has no gold value"
    if answer.declined:
        return False, "declined an answerable numeric question"

    text = answer.answer_text
    candidates = reportable_quantities(text)
    if not candidates:
        return False, "no figure found in the answer"

    if question.gold_is_percentage:
        percent_candidates = [q for q in candidates if q.is_percent]
        if not percent_candidates:
            return False, "answer states no percentage for a percentage-valued question"

        gold_direction = DIRECTION_DOWN if question.gold_value < 0 else DIRECTION_UP
        best = ""
        for quantity in percent_candidates:
            check = within_tolerance(
                abs(quantity.value),
                abs(question.gold_value),
                relative_tolerance=relative_tolerance,
                is_percentage=True,
                percentage_point_tolerance=percentage_point_tolerance,
            )
            if not check.passed:
                if not best:
                    best = (
                        f"closest was {quantity.raw!r} against gold magnitude "
                        f"{abs(question.gold_value):.4f}%, relative error "
                        f"{check.relative_error:.4f}"
                        if check.relative_error is not None
                        else f"closest was {quantity.raw!r}"
                    )
                continue
            direction = stated_direction(text, quantity)
            if direction == DIRECTION_UNSTATED:
                return False, (
                    f"{quantity.raw!r} matches the gold magnitude but the answer does "
                    "not state whether the figure rose or fell"
                )
            if direction != gold_direction:
                return False, (
                    f"{quantity.raw!r} matches the gold magnitude but states the "
                    f"direction as {'down' if direction == DIRECTION_DOWN else 'up'} "
                    f"where the gold is "
                    f"{'down' if gold_direction == DIRECTION_DOWN else 'up'}"
                )
            return True, (
                f"{quantity.raw!r} matches gold {question.gold_value:+.4f}% by "
                f"{check.rule}, direction agrees"
            )
        return False, best or "no percentage matched the gold"

    best = ""
    for quantity in candidates:
        check = within_tolerance(
            quantity.value,
            question.gold_value,
            relative_tolerance=relative_tolerance,
            is_percentage=False,
        )
        if check.passed:
            return True, (
                f"{quantity.raw!r} matches gold {question.gold_value:.6g} by "
                f"{check.rule}"
            )
        if not best:
            best = (
                f"closest was {quantity.raw!r} = {quantity.value:.6g} against gold "
                f"{question.gold_value:.6g}, relative error {check.relative_error:.4f}"
                if check.relative_error is not None
                else f"closest was {quantity.raw!r}"
            )
    return False, best or "no figure matched the gold"


def score_claim_support(
    claim_index: int,
    claim_text: str,
    chunk_ids: tuple[str, ...],
    *,
    con: duckdb.DuckDBPyConnection,
    client: CachedClient,
    config: Config,
) -> tuple[ClaimScore, Usage]:
    """Rate one claim's citation, deterministically where possible."""
    if not chunk_ids:
        return (
            ClaimScore(
                index=claim_index,
                text=claim_text,
                chunk_ids=(),
                support=SUPPORT_UNCITED,
                method=METHOD_UNCITED,
                reason="claim carries no citation",
            ),
            Usage(),
        )

    passage = _cited_text(con, chunk_ids)
    if not passage:
        return (
            ClaimScore(
                index=claim_index,
                text=claim_text,
                chunk_ids=chunk_ids,
                support=SUPPORT_NOT,
                method=METHOD_DETERMINISTIC,
                reason="cited chunk ids resolve to no text in the corpus",
            ),
            Usage(),
        )

    figures = reportable_quantities(claim_text)
    if figures:
        found = [
            quantity
            for quantity in figures
            if number_appears_in(
                quantity.value, passage, is_percentage=quantity.is_percent
            ).found
        ]
        if len(found) == len(figures):
            support, reason = SUPPORT_SUPPORTED, (
                f"all {len(figures)} figure(s) in the claim appear in the cited text"
            )
        elif found:
            missing = [q.raw for q in figures if q not in found]
            support, reason = SUPPORT_PARTIAL, (
                f"{len(found)} of {len(figures)} figures appear; missing {missing}"
            )
        else:
            support, reason = SUPPORT_NOT, (
                f"none of {[q.raw for q in figures]} appear in the cited text"
            )
        return (
            ClaimScore(
                index=claim_index,
                text=claim_text,
                chunk_ids=chunk_ids,
                support=support,
                method=METHOD_DETERMINISTIC,
                reason=reason,
                figures_checked=len(figures),
                figures_found=len(found),
            ),
            Usage(),
        )

    verdict = judge_citation(claim_text, passage, client=client, config=config)
    return (
        ClaimScore(
            index=claim_index,
            text=claim_text,
            chunk_ids=chunk_ids,
            support=verdict.label,
            method=METHOD_JUDGE,
            reason=verdict.reason,
        ),
        verdict.usage,
    )


def _distinct_figures(text: str) -> list:
    """Reported figures in `text`, one entry per distinct value.

    An answer states its headline figure in both the summary and a claim, so
    counting occurrences would report one unsupported number as two
    hallucinations and make the ratio depend on how repetitive the prose is.
    Distinctness is on the normalised value and percent-ness, not the raw
    string, so "$53.1 billion" and "53.1 billion" collapse to one figure.
    """
    seen: dict[tuple[float, bool], object] = {}
    for quantity in reportable_quantities(text):
        seen.setdefault((quantity.value, quantity.is_percent), quantity)
    return list(seen.values())


def _hallucinated_figures(
    answer: Answer, con: duckdb.DuckDBPyConnection
) -> tuple[tuple[str, ...], int]:
    """Distinct reported figures in the answer that appear in no cited chunk."""
    figures = _distinct_figures(answer.answer_text)
    if not figures:
        return (), 0
    passage = _cited_text(con, answer.cited_chunk_ids)
    unsupported = tuple(
        quantity.raw
        for quantity in figures
        if not number_appears_in(
            quantity.value, passage, is_percentage=quantity.is_percent
        ).found
    )
    return unsupported, len(figures)


def score_answer(
    question: Question,
    answer: Answer,
    *,
    con: duckdb.DuckDBPyConnection,
    client: CachedClient,
    config: Config,
) -> AnswerScore:
    """Score one answer against one question."""
    judge_usage = Usage()

    if question.kind == KIND_UNANSWERABLE:
        correct = answer.declined
        method = METHOD_DECLINED
        detail = "declined" if correct else "answered an unanswerable question"
    elif question.kind == KIND_NUMERIC:
        correct, detail = score_numeric_answer(question, answer)
        method = METHOD_NUMERIC
    else:
        if answer.declined:
            correct, detail = False, "declined an answerable question"
            method = METHOD_DECLINED
        else:
            verdict = judge_answer(
                question.text,
                question.gold_answer,
                answer.answer_text,
                client=client,
                config=config,
            )
            judge_usage += verdict.usage
            correct, detail, method = verdict.correct, verdict.reason, METHOD_JUDGE

    claim_scores: list[ClaimScore] = []
    for index, claim in enumerate(answer.claims):
        score, usage = score_claim_support(
            index,
            claim.text,
            claim.chunk_ids,
            con=con,
            client=client,
            config=config,
        )
        claim_scores.append(score)
        judge_usage += usage

    recall: bool | None = None
    if question.gold_chunk_ids and question.kind in (KIND_QUALITATIVE, KIND_MULTI_FILING):
        recall = bool(set(answer.cited_chunk_ids) & set(question.gold_chunk_ids))

    hallucinated, figure_count = _hallucinated_figures(answer, con)

    return AnswerScore(
        question_id=question.question_id,
        kind=question.kind,
        system=answer.system,
        answer_correct=correct,
        correctness_method=method,
        correctness_detail=detail,
        declined=answer.declined,
        claim_scores=tuple(claim_scores),
        citation_recall=recall,
        hallucinated_figures=hallucinated,
        figures_in_answer=figure_count,
        rejected_citations=answer.rejected_citations,
        cost_usd=answer.cost_usd,
        judge_cost_usd=judge_usage.cost_usd(config.judge_model),
        latency_s=answer.latency_s,
        tool_calls=len(answer.tool_calls),
        hit_tool_cap=answer.hit_tool_cap,
        error=answer.error,
        tags=question.tags,
    )
