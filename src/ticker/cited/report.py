"""Aggregating scores into the results table, with bootstrap intervals.

The unit of analysis is the question, everywhere. Every interval here comes
from `ticker.evaluation.bootstrap_ci`, the same percentile bootstrap the
retrieval harness uses, resampling questions with replacement. Reusing it
rather than writing a second one means the two halves of this repo cannot
disagree about what a confidence interval is.

Rates that are only defined on some answers are averaged over those answers
and report their own `n`. Citation support is undefined for an answer that
made no cited claim, and citation recall is undefined for a question with no
gold passage -- a numeric question's gold is an XBRL value and an unanswerable
question has no supporting passage by construction. Folding those in as zeros
would be the easiest way to make these numbers wrong: a system that declines
everything would post a perfect support rate over an empty denominator, or a
terrible one, depending on which way the zero fell.

Metrics are macro over questions rather than pooled over claims. An answer
with eleven claims should not count eleven times as much as one with a single
claim when the thing being estimated is per-question behaviour, and the
bootstrap needs the per-question value to resample. Pooled claim counts are
reported alongside so the macro figure can be checked against them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ticker.cited.judge import SUPPORT_SUPPORTED
from ticker.cited.questions import KINDS
from ticker.cited.scoring import AnswerScore, ClaimScore
from ticker.evaluation import DEFAULT_N_BOOT, bootstrap_ci


@dataclass(frozen=True, slots=True)
class Metric:
    """A mean with its bootstrap interval and the n it was computed over."""

    name: str
    value: float | None
    low: float | None
    high: float | None
    n: int

    def render(self, *, percent: bool = True, places: int = 1) -> str:
        if self.value is None or self.n == 0:
            return "n/a"
        if percent:
            return (
                f"{self.value * 100:.{places}f}% "
                f"[{self.low * 100:.{places}f}, {self.high * 100:.{places}f}]"
            )
        return f"{self.value:.{places}f} [{self.low:.{places}f}, {self.high:.{places}f}]"


def _metric(name: str, values: list[float], *, n_boot: int = DEFAULT_N_BOOT) -> Metric:
    if not values:
        return Metric(name=name, value=None, low=None, high=None, n=0)
    array = np.asarray(values, dtype=float)
    low, high = bootstrap_ci(array, n_boot=n_boot)
    return Metric(
        name=name, value=float(array.mean()), low=low, high=high, n=len(values)
    )


@dataclass(frozen=True, slots=True)
class SystemSummary:
    system: str
    kind: str  # "all" or one of KINDS
    n_questions: int

    answer_accuracy: Metric
    citation_support: Metric
    citation_recall: Metric
    hallucinated_figure_rate: Metric
    answers_with_rejected_citation: Metric
    declined_rate: Metric
    uncited_claim_rate: Metric

    cost_per_question_usd: Metric
    latency_per_question_s: Metric
    tool_calls_per_question: Metric

    claims_total: int = 0
    cited_claims_total: int = 0
    supported_claims_total: int = 0
    figures_total: int = 0
    hallucinated_figures_total: int = 0
    errors: int = 0
    hit_tool_cap: int = 0


def summarise(
    scores: Iterable[AnswerScore], *, system: str, kind: str = "all"
) -> SystemSummary:
    selected = [
        score
        for score in scores
        if score.system == system and (kind == "all" or score.kind == kind)
    ]

    accuracy = [float(score.answer_correct) for score in selected]
    support = [
        score.citation_support_rate
        for score in selected
        if score.citation_support_rate is not None
    ]
    recall = [
        float(score.citation_recall)
        for score in selected
        if score.citation_recall is not None
    ]
    # Share of answers containing at least one unsupported figure, not the
    # count of such figures. One answer with six invented numbers and six
    # answers with one each are different failures, and the per-answer rate is
    # the one that bounds how often a reader would be misled.
    hallucinated = [
        float(bool(score.hallucinated_figures))
        for score in selected
        if score.figures_in_answer
    ]
    rejected = [float(bool(score.rejected_citations)) for score in selected]
    declined = [float(score.declined) for score in selected]
    uncited = [
        score.uncited_claim_count / len(score.claim_scores)
        for score in selected
        if score.claim_scores
    ]
    cost = [
        (score.cost_usd or 0.0) + (score.judge_cost_usd or 0.0)
        for score in selected
        if score.cost_usd is not None
    ]
    latency = [score.latency_s for score in selected]
    tool_calls = [float(score.tool_calls) for score in selected]

    return SystemSummary(
        system=system,
        kind=kind,
        n_questions=len(selected),
        answer_accuracy=_metric("answer_accuracy", accuracy),
        citation_support=_metric("citation_support", support),
        citation_recall=_metric("citation_recall", recall),
        hallucinated_figure_rate=_metric("hallucinated_figure_rate", hallucinated),
        answers_with_rejected_citation=_metric("rejected_citation_rate", rejected),
        declined_rate=_metric("declined_rate", declined),
        uncited_claim_rate=_metric("uncited_claim_rate", uncited),
        cost_per_question_usd=_metric("cost_per_question_usd", cost),
        latency_per_question_s=_metric("latency_per_question_s", latency),
        tool_calls_per_question=_metric("tool_calls_per_question", tool_calls),
        claims_total=sum(len(score.claim_scores) for score in selected),
        cited_claims_total=sum(len(score.cited_claims) for score in selected),
        supported_claims_total=sum(
            1
            for score in selected
            for claim in score.cited_claims
            if claim.support == SUPPORT_SUPPORTED
        ),
        figures_total=sum(score.figures_in_answer for score in selected),
        hallucinated_figures_total=sum(
            len(score.hallucinated_figures) for score in selected
        ),
        errors=sum(1 for score in selected if score.error),
        hit_tool_cap=sum(1 for score in selected if score.hit_tool_cap),
    )


@dataclass(frozen=True, slots=True)
class Report:
    questions_sha256: str
    questions_path: str
    answer_model: str
    judge_model: str
    rubric_version: str
    config: dict
    overall: tuple[SystemSummary, ...] = ()
    by_kind: tuple[SystemSummary, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


def build_report(
    scores: list[AnswerScore],
    *,
    questions_sha256: str,
    questions_path: str,
    answer_model: str,
    judge_model: str,
    rubric_version: str,
    config: dict,
    notes: tuple[str, ...] = (),
) -> Report:
    systems = sorted({score.system for score in scores})
    return Report(
        questions_sha256=questions_sha256,
        questions_path=questions_path,
        answer_model=answer_model,
        judge_model=judge_model,
        rubric_version=rubric_version,
        config=config,
        overall=tuple(summarise(scores, system=system) for system in systems),
        by_kind=tuple(
            summarise(scores, system=system, kind=kind)
            for system in systems
            for kind in KINDS
        ),
        notes=notes,
    )


HEADLINE_COLUMNS = (
    ("system", lambda s: s.system),
    ("n", lambda s: str(s.n_questions)),
    ("answer accuracy", lambda s: s.answer_accuracy.render()),
    ("citation support", lambda s: s.citation_support.render()),
    ("hallucinated figures", lambda s: s.hallucinated_figure_rate.render()),
    ("cost / question", lambda s: f"${s.cost_per_question_usd.value:.4f}"
     if s.cost_per_question_usd.value is not None else "n/a"),
)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(report: Report) -> str:
    """The results tables, in the form the README carries them."""
    out: list[str] = []

    out.append("Question set "
               f"`{report.questions_path}`, SHA-256 `{report.questions_sha256[:16]}`. "
               f"Answering model `{report.answer_model}`, judge `{report.judge_model}`, "
               f"rubric version {report.rubric_version}.")
    out.append("")

    if not report.overall:
        out.append("No scored answers. The table below is empty because no "
                   "evaluation run has been made.")
        return "\n".join(out)

    out.append("### Headline")
    out.append("")
    out.append(
        _table(
            [name for name, _ in HEADLINE_COLUMNS],
            [[render(summary) for _, render in HEADLINE_COLUMNS]
             for summary in report.overall],
        )
    )
    out.append("")
    out.append("Intervals are 95% percentile bootstrap over questions.")
    out.append("")

    out.append("### Grounding")
    out.append("")
    out.append(
        _table(
            ["system", "citation recall", "cited a chunk never retrieved",
             "uncited claims", "declined", "claims", "cited claims"],
            [
                [
                    summary.system,
                    summary.citation_recall.render(),
                    summary.answers_with_rejected_citation.render(),
                    summary.uncited_claim_rate.render(),
                    summary.declined_rate.render(),
                    str(summary.claims_total),
                    str(summary.cited_claims_total),
                ]
                for summary in report.overall
            ],
        )
    )
    out.append("")

    out.append("### By question type")
    out.append("")
    out.append(
        _table(
            ["system", "type", "n", "answer accuracy", "citation support",
             "citation recall", "declined"],
            [
                [
                    summary.system,
                    summary.kind,
                    str(summary.n_questions),
                    summary.answer_accuracy.render(),
                    summary.citation_support.render(),
                    summary.citation_recall.render(),
                    summary.declined_rate.render(),
                ]
                for summary in report.by_kind
                if summary.n_questions
            ],
        )
    )
    out.append("")

    out.append("### Cost and latency")
    out.append("")
    out.append(
        _table(
            ["system", "cost / question", "latency / question", "tool calls / question",
             "hit tool cap", "errors"],
            [
                [
                    summary.system,
                    summary.cost_per_question_usd.render(percent=False, places=4),
                    summary.latency_per_question_s.render(percent=False, places=2),
                    summary.tool_calls_per_question.render(percent=False, places=2),
                    str(summary.hit_tool_cap),
                    str(summary.errors),
                ]
                for summary in report.overall
            ],
        )
    )

    if report.notes:
        out.append("")
        out.append("### Notes")
        out.append("")
        for note in report.notes:
            out.append(f"- {note}")

    return "\n".join(out)


def _claim_to_dict(claim: ClaimScore) -> dict:
    payload = asdict(claim)
    payload["chunk_ids"] = list(claim.chunk_ids)
    return payload


def score_to_json(score: AnswerScore) -> str:
    payload = asdict(score)
    payload["claim_scores"] = [_claim_to_dict(claim) for claim in score.claim_scores]
    for key in ("hallucinated_figures", "rejected_citations", "tags"):
        payload[key] = list(payload[key])
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def score_from_json(line: str) -> AnswerScore:
    payload = json.loads(line)
    payload["claim_scores"] = tuple(
        ClaimScore(
            index=item["index"],
            text=item["text"],
            chunk_ids=tuple(item.get("chunk_ids") or []),
            support=item["support"],
            method=item["method"],
            reason=item.get("reason", ""),
            figures_checked=item.get("figures_checked", 0),
            figures_found=item.get("figures_found", 0),
        )
        for item in payload.get("claim_scores") or []
    )
    for key in ("hallucinated_figures", "rejected_citations", "tags"):
        payload[key] = tuple(payload.get(key) or [])
    return AnswerScore(**payload)


def write_scores(path: Path, scores: Iterable[AnswerScore]) -> int:
    ordered = sorted(scores, key=lambda score: (score.system, score.question_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for score in ordered:
            handle.write(score_to_json(score) + "\n")
    return len(ordered)


def iter_scores(path: Path) -> Iterator[AnswerScore]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield score_from_json(line)


def load_scores(path: Path) -> list[AnswerScore]:
    return list(iter_scores(path))
