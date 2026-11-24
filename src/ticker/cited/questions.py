"""The question set: record shape, on-disk format, and the freeze hash.

One JSONL line per question, written once and then treated as immutable. The
file's SHA-256 is recorded in every results file, so a reported number can be
tied to the exact question set it was produced against; a question edited after
a run changes the hash and the mismatch is visible rather than silent.

Gold provenance differs by kind and is recorded as such in `gold_source`:

`xbrl`
    The value as reported in a named accession, pulled from EDGAR's
    company-facts API. Machine-checkable end to end -- nothing drafted the
    number, and `gold_accessions` names the filing that reported it.

`filing_text`
    A one- or two-sentence answer drafted from passages located by substring
    scan (`ticker.cited.locate`), never by the rankers under evaluation. See
    that module for why that distinction is load-bearing.

`not_in_corpus`
    An unanswerable question. `gold_chunk_ids` is empty by construction and
    `rationale` records what makes it unanswerable: a figure the indexed
    sections do not carry, a year outside the corpus, or an event that did not
    happen.

`answerable` is the field the scorer branches on. A system is correct on an
unanswerable question only by declining, and correct on an answerable one only
by answering; the two are not symmetric and conflating them would let a system
that declines everything score well on a third of the set.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_QUESTIONS_PATH = Path("eval/questions.jsonl")

KIND_NUMERIC = "numeric"
KIND_QUALITATIVE = "qualitative"
KIND_UNANSWERABLE = "unanswerable"
KIND_MULTI_FILING = "multi_filing"

KINDS = (KIND_NUMERIC, KIND_QUALITATIVE, KIND_UNANSWERABLE, KIND_MULTI_FILING)

SOURCE_XBRL = "xbrl"
SOURCE_FILING_TEXT = "filing_text"
SOURCE_NOT_IN_CORPUS = "not_in_corpus"

# The intended composition. `validate_composition` checks a built set against
# it, so a question dropped during drafting fails the build rather than
# quietly shipping a 58-question set described in the README as 60.
TARGET_COMPOSITION: dict[str, int] = {
    KIND_NUMERIC: 20,
    KIND_QUALITATIVE: 25,
    KIND_UNANSWERABLE: 10,
    KIND_MULTI_FILING: 5,
}

TOTAL_QUESTIONS = sum(TARGET_COMPOSITION.values())


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    kind: str
    text: str
    gold_answer: str
    gold_source: str
    answerable: bool
    rationale: str

    companies: tuple[str, ...] = ()
    form_type: str | None = None
    fiscal_year: int | None = None

    gold_accessions: tuple[str, ...] = ()
    gold_chunk_ids: tuple[str, ...] = ()

    # The substring pattern the gold chunks were located with, recorded so the
    # gold set is re-derivable rather than pasted in. Re-running the scan with
    # this pattern must reproduce `gold_chunk_ids`; `ticker eval build-questions
    # --verify` checks exactly that, which turns a hand-drafted gold set into
    # one with a reproducible provenance chain back to the filing text.
    gold_locator: str | None = None

    # Numeric questions only.
    gold_value: float | None = None
    gold_unit: str | None = None
    gold_is_percentage: bool = False
    gold_concepts: tuple[str, ...] = ()
    gold_period_end: str | None = None

    # The operands a derived gold was computed from. A year-over-year question's
    # gold is a percentage the filing need never print, so answerability has to
    # be judged on whether both operands are in the text, not on whether the
    # quotient is.
    gold_components: tuple[float, ...] = ()

    # Set by the answerability check, not by drafting: whether the gold figure
    # is actually present in the indexed text of the filings it came from.
    figure_in_indexed_text: bool | None = None

    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {KINDS}")
        if self.gold_source not in (SOURCE_XBRL, SOURCE_FILING_TEXT, SOURCE_NOT_IN_CORPUS):
            raise ValueError(f"unknown gold_source {self.gold_source!r}")
        if self.answerable and self.kind == KIND_UNANSWERABLE:
            raise ValueError(f"{self.question_id}: unanswerable question marked answerable")
        if not self.answerable and self.gold_chunk_ids:
            raise ValueError(
                f"{self.question_id}: unanswerable question carries gold chunk ids"
            )
        if self.kind == KIND_NUMERIC and self.gold_value is None:
            raise ValueError(f"{self.question_id}: numeric question has no gold value")
        if self.kind == KIND_NUMERIC and not self.gold_accessions:
            raise ValueError(
                f"{self.question_id}: numeric question must name the filing it came from"
            )
        if self.kind == KIND_MULTI_FILING and len(self.gold_accessions) < 2:
            raise ValueError(
                f"{self.question_id}: multi-filing question needs at least two accessions"
            )

    @property
    def company(self) -> str | None:
        """The single company a question is about, when there is one."""
        return self.companies[0] if len(self.companies) == 1 else None


def to_json(question: Question) -> str:
    """One canonical line per question.

    Keys are sorted and tuples become lists, so the same question set always
    serialises to the same bytes and therefore the same freeze hash regardless
    of the order fields were assigned in.
    """
    payload = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(question).items()
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def from_json(line: str) -> Question:
    payload = json.loads(line)
    tuple_fields = (
        "companies",
        "gold_accessions",
        "gold_chunk_ids",
        "gold_concepts",
        "gold_components",
        "tags",
    )
    for key in tuple_fields:
        if key in payload and payload[key] is not None:
            payload[key] = tuple(payload[key])
    return Question(**payload)


def write_questions(path: Path, questions: Iterable[Question]) -> int:
    """Write the set, ordered by question id. Returns the count written."""
    ordered = sorted(questions, key=lambda question: question.question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for question in ordered:
            handle.write(to_json(question) + "\n")
    return len(ordered)


def iter_questions(path: Path) -> Iterator[Question]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield from_json(line)


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[Question]:
    questions = list(iter_questions(path))
    ids = [question.question_id for question in questions]
    duplicates = {qid for qid in ids if ids.count(qid) > 1}
    if duplicates:
        raise ValueError(f"duplicate question ids in {path}: {sorted(duplicates)}")
    return questions


def freeze_hash(path: Path = DEFAULT_QUESTIONS_PATH) -> str:
    """SHA-256 of the question file as it sits on disk.

    Hashes the bytes rather than a re-serialisation of the parsed records, so
    any edit at all -- including one that round-trips to the same objects --
    changes the hash. The point is to detect that the file moved, not to decide
    whether the move mattered.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def composition(questions: Iterable[Question]) -> dict[str, int]:
    counts = dict.fromkeys(KINDS, 0)
    for question in questions:
        counts[question.kind] += 1
    return counts


def validate_composition(
    questions: Iterable[Question], *, target: dict[str, int] | None = None
) -> None:
    """Raise unless the set matches the intended composition exactly."""
    target = target or TARGET_COMPOSITION
    actual = composition(questions)
    if actual != target:
        raise ValueError(
            f"question set composition {actual} does not match target {target}"
        )
