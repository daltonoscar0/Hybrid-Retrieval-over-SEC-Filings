"""Running a system over the frozen question set, and scoring the result.

Thin orchestration. The only judgement in here is about failure handling: a
question that raises is recorded as an answer carrying the error, not dropped.
A dropped question silently shrinks the denominator, which makes a run that
crashed on a third of the set look like a run that scored well on all of it.

Every output file records the question set's SHA-256 alongside the numbers, so
a results file is self-describing: the numbers and the exact inputs that
produced them travel together.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ticker.cited.agent import answer_question
from ticker.cited.answers import (
    SYSTEM_AGENT,
    SYSTEM_CLOSED,
    SYSTEM_SINGLE,
    Answer,
    load_answers,
    write_answers,
)
from ticker.cited.baselines import answer_closed_book, answer_single_shot
from ticker.cited.config import Config
from ticker.cited.corpus import connect
from ticker.cited.judge import RUBRIC_VERSION
from ticker.cited.llm import CachedClient
from ticker.cited.questions import Question, freeze_hash, load_questions
from ticker.cited.report import build_report, render_markdown, write_scores
from ticker.cited.retrieval import load_searcher
from ticker.cited.scoring import score_answer


def answers_path(config: Config, system: str) -> Path:
    return config.results_dir / f"answers_{system}.jsonl"


def scores_path(config: Config) -> Path:
    return config.results_dir / "scores.jsonl"


def report_path(config: Config) -> Path:
    return config.results_dir / "report.md"


def report_json_path(config: Config) -> Path:
    return config.results_dir / "report.json"


@dataclass(frozen=True, slots=True)
class RunOutcome:
    system: str
    answers: list[Answer]
    path: Path
    spend_usd: float | None
    cache_hits: int
    calls: int


def run_system(
    system: str,
    *,
    config: Config,
    questions: Iterable[Question] | None = None,
    progress: Callable[[str], None] = lambda _message: None,
) -> RunOutcome:
    """Answer every question with one system and write the answers out."""
    questions = list(questions or load_questions(config.questions_path))
    con = connect(config.db_path)
    client = CachedClient(config)

    # The dense index and its query encoder cost several seconds and most of a
    # gigabyte to load, so the closed-book arm, which retrieves nothing, does
    # not load them at all.
    searcher = None
    if system in (SYSTEM_SINGLE, SYSTEM_AGENT):
        searcher = load_searcher(con, index_root=config.index_root)

    answers: list[Answer] = []
    for index, question in enumerate(questions, start=1):
        progress(f"[{index}/{len(questions)}] {system} {question.question_id}")
        try:
            if system == SYSTEM_CLOSED:
                answer = answer_closed_book(question, client=client, config=config)
            elif system == SYSTEM_SINGLE:
                answer = answer_single_shot(
                    question, searcher=searcher, client=client, config=config
                )
            elif system == SYSTEM_AGENT:
                answer = answer_question(
                    question, con=con, searcher=searcher, client=client, config=config
                )
            else:
                raise ValueError(f"unknown system {system!r}")
        except Exception as exc:
            answer = Answer(
                question_id=question.question_id,
                system=system,
                model=config.answer_model,
                error=f"{type(exc).__name__}: {exc}",
            )
        answers.append(answer)

    path = answers_path(config, system)
    write_answers(path, answers)
    return RunOutcome(
        system=system,
        answers=answers,
        path=path,
        spend_usd=client.spend_usd(config.answer_model),
        cache_hits=client.cache_hits,
        calls=client.calls,
    )


def score_all(
    *,
    config: Config,
    systems: Iterable[str] = (SYSTEM_CLOSED, SYSTEM_SINGLE, SYSTEM_AGENT),
    progress: Callable[[str], None] = lambda _message: None,
) -> tuple[list, dict]:
    """Score whatever answer files exist, and write scores plus the report.

    A missing answer file is skipped rather than fatal: scoring the arms that
    have been run is useful, and demanding all three would make the scorer
    unusable until the last one finished.
    """
    questions = {q.question_id: q for q in load_questions(config.questions_path)}
    con = connect(config.db_path)
    client = CachedClient(config)

    scores = []
    present: list[str] = []
    for system in systems:
        path = answers_path(config, system)
        if not path.exists():
            progress(f"skipping {system}: no answers at {path}")
            continue
        present.append(system)
        for answer in load_answers(path):
            question = questions.get(answer.question_id)
            if question is None:
                # An answer for a question id not in the frozen set means the
                # answers were produced against a different question file.
                # Scoring it anyway would mix two question sets in one table.
                raise ValueError(
                    f"{path}: answer for unknown question {answer.question_id!r}; "
                    "answers were produced against a different question set"
                )
            progress(f"scoring {system} {answer.question_id}")
            scores.append(
                score_answer(question, answer, con=con, client=client, config=config)
            )

    write_scores(scores_path(config), scores)

    report = build_report(
        scores,
        questions_sha256=freeze_hash(config.questions_path),
        questions_path=str(config.questions_path),
        answer_model=config.answer_model,
        judge_model=config.judge_model,
        rubric_version=RUBRIC_VERSION,
        config=config.redacted(),
        notes=(
            f"Systems scored: {', '.join(present) or 'none'}.",
            f"Judge spend for this scoring pass: "
            f"{_format_spend(client.spend_usd(config.judge_model))}.",
            f"Judge calls: {client.calls}, of which {client.cache_hits} served "
            "from cache.",
        ),
    )

    report_path(config).parent.mkdir(parents=True, exist_ok=True)
    report_path(config).write_text(render_markdown(report) + "\n")
    report_json_path(config).write_text(
        json.dumps(_report_payload(report), indent=2, sort_keys=True) + "\n"
    )
    return scores, _report_payload(report)


def _format_spend(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def _report_payload(report) -> dict:
    """The report as plain data.

    Deliberately carries no timestamp. A generated-at field would make
    `report.json` differ on every scoring pass, which would break the one
    property the results are meant to have: re-running the scorer over the
    cached answers reproduces the file byte for byte. The question-set hash
    and the model names already identify what produced it.
    """
    from dataclasses import asdict

    payload = asdict(report)
    payload["generated_by"] = "ticker eval score"
    return payload
