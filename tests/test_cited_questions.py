"""The question record, the freeze hash, and the report aggregation.

The frozen question file is the input every reported number is keyed to, so
its invariants are worth pinning: an unanswerable question cannot carry gold
passages, a numeric question cannot ship without a gold value, and the hash
must change when the file does.
"""

from __future__ import annotations

import pytest

from ticker.cited.questions import (
    KIND_MULTI_FILING,
    KIND_NUMERIC,
    KIND_QUALITATIVE,
    KIND_UNANSWERABLE,
    SOURCE_FILING_TEXT,
    SOURCE_NOT_IN_CORPUS,
    SOURCE_XBRL,
    TARGET_COMPOSITION,
    TOTAL_QUESTIONS,
    Question,
    composition,
    freeze_hash,
    from_json,
    load_questions,
    to_json,
    validate_composition,
    write_questions,
)
from ticker.cited.report import Metric, build_report, render_markdown, summarise
from ticker.cited.scoring import AnswerScore, ClaimScore
from ticker.cited.judge import SUPPORT_NOT, SUPPORT_SUPPORTED
from ticker.cited.units import UNIT_USD


def qualitative(**overrides) -> Question:
    base = dict(
        question_id="q01",
        kind=KIND_QUALITATIVE,
        text="What did it say?",
        gold_answer="It said a thing.",
        gold_source=SOURCE_FILING_TEXT,
        answerable=True,
        rationale="stated in item 1",
        companies=("NVDA",),
        form_type="10-K",
        fiscal_year=2025,
        gold_accessions=("0000000001-25-000001",),
        gold_chunk_ids=("0000000001-25-000001#1#chunk0",),
        gold_locator="a thing",
    )
    base.update(overrides)
    return Question(**base)


class TestInvariants:
    def test_unanswerable_cannot_carry_gold_chunks(self):
        # A gold passage on an unanswerable question would make the
        # declining-is-correct metric incoherent.
        with pytest.raises(ValueError, match="gold chunk ids"):
            qualitative(
                kind=KIND_UNANSWERABLE,
                answerable=False,
                gold_source=SOURCE_NOT_IN_CORPUS,
            )

    def test_unanswerable_cannot_be_marked_answerable(self):
        with pytest.raises(ValueError, match="marked answerable"):
            qualitative(
                kind=KIND_UNANSWERABLE,
                answerable=True,
                gold_chunk_ids=(),
                gold_source=SOURCE_NOT_IN_CORPUS,
            )

    def test_numeric_needs_a_gold_value(self):
        with pytest.raises(ValueError, match="no gold value"):
            qualitative(kind=KIND_NUMERIC, gold_source=SOURCE_XBRL, gold_value=None)

    def test_numeric_needs_an_accession(self):
        # Without it, a gold figure cannot be traced to the filing that
        # reported it.
        with pytest.raises(ValueError, match="name the filing"):
            qualitative(
                kind=KIND_NUMERIC,
                gold_source=SOURCE_XBRL,
                gold_value=1.0,
                gold_accessions=(),
            )

    def test_multi_filing_needs_two_accessions(self):
        with pytest.raises(ValueError, match="two accessions"):
            qualitative(kind=KIND_MULTI_FILING)

    def test_multi_filing_accepts_two(self):
        question = qualitative(
            kind=KIND_MULTI_FILING,
            gold_accessions=("0000000001-25-000001", "0000000001-23-000001"),
            fiscal_year=None,
        )
        assert len(question.gold_accessions) == 2

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown kind"):
            qualitative(kind="trick")

    def test_company_is_none_when_several(self):
        # A filter built from a two-company question would exclude half its
        # own gold set.
        assert qualitative(companies=("NVDA", "INTC")).company is None
        assert qualitative(companies=("NVDA",)).company == "NVDA"


class TestSerialisation:
    def test_round_trips(self):
        question = qualitative()
        assert from_json(to_json(question)) == question

    def test_numeric_round_trips_with_components(self):
        question = qualitative(
            kind=KIND_NUMERIC,
            gold_source=SOURCE_XBRL,
            gold_value=-2.0783,
            gold_is_percentage=True,
            gold_unit="percent",
            gold_components=(53_101e6, 54_228e6),
            gold_chunk_ids=(),
        )
        assert from_json(to_json(question)) == question

    def test_serialisation_is_key_order_stable(self):
        # The freeze hash is over the file bytes, so a field-order change
        # would change the hash without changing the questions.
        question = qualitative()
        assert to_json(question) == to_json(from_json(to_json(question)))


class TestFreezeHash:
    def test_hash_changes_when_a_question_changes(self, tmp_path):
        path = tmp_path / "questions.jsonl"
        write_questions(path, [qualitative()])
        before = freeze_hash(path)
        write_questions(path, [qualitative(text="Something else?")])
        assert freeze_hash(path) != before

    def test_hash_is_stable_across_writes_of_the_same_set(self, tmp_path):
        first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        questions = [qualitative(question_id="q02"), qualitative(question_id="q01")]
        write_questions(first, questions)
        write_questions(second, list(reversed(questions)))
        # Written sorted by id, so input order must not affect the hash.
        assert freeze_hash(first) == freeze_hash(second)

    def test_duplicate_ids_are_rejected_on_load(self, tmp_path):
        path = tmp_path / "questions.jsonl"
        path.write_text(to_json(qualitative()) + "\n" + to_json(qualitative()) + "\n")
        with pytest.raises(ValueError, match="duplicate"):
            load_questions(path)


class TestComposition:
    def test_target_sums_to_sixty(self):
        assert TOTAL_QUESTIONS == 60
        assert TARGET_COMPOSITION == {
            KIND_NUMERIC: 20,
            KIND_QUALITATIVE: 25,
            KIND_UNANSWERABLE: 10,
            KIND_MULTI_FILING: 5,
        }

    def test_short_set_is_rejected(self):
        # A question dropped during drafting must fail the build rather than
        # ship a 59-question set described as 60.
        with pytest.raises(ValueError, match="composition"):
            validate_composition([qualitative()])

    def test_counts_every_kind(self):
        counts = composition([qualitative(), qualitative(question_id="q02")])
        assert counts[KIND_QUALITATIVE] == 2
        assert counts[KIND_NUMERIC] == 0


def score(system, **overrides) -> AnswerScore:
    base = dict(
        question_id="q01",
        kind=KIND_QUALITATIVE,
        system=system,
        answer_correct=True,
        correctness_method="llm_judge",
        correctness_detail="",
        declined=False,
    )
    base.update(overrides)
    return AnswerScore(**base)


class TestReportAggregation:
    def test_undefined_support_is_excluded_not_zeroed(self):
        # An answer with no cited claim has no support rate. Folding it in as
        # zero would let a declining system post a support failure it never
        # had the chance to commit.
        scores = [
            score("agent", question_id="q01", claim_scores=(
                ClaimScore(0, "c", ("a",), SUPPORT_SUPPORTED, "m", ""),
            )),
            score("agent", question_id="q02", declined=True, claim_scores=()),
        ]
        summary = summarise(scores, system="agent")
        assert summary.citation_support.n == 1
        assert summary.citation_support.value == pytest.approx(1.0)
        assert summary.n_questions == 2

    def test_support_rate_is_share_of_fully_supported(self):
        scores = [
            score("agent", question_id="q01", claim_scores=(
                ClaimScore(0, "c", ("a",), SUPPORT_SUPPORTED, "m", ""),
                ClaimScore(1, "d", ("b",), SUPPORT_NOT, "m", ""),
            )),
        ]
        summary = summarise(scores, system="agent")
        assert summary.citation_support.value == pytest.approx(0.5)
        assert summary.cited_claims_total == 2
        assert summary.supported_claims_total == 1

    def test_recall_excludes_questions_with_no_gold_passage(self):
        scores = [
            score("agent", question_id="q01", citation_recall=True),
            score("agent", question_id="n01", kind=KIND_NUMERIC, citation_recall=None),
        ]
        summary = summarise(scores, system="agent")
        assert summary.citation_recall.n == 1

    def test_hallucination_rate_is_over_answers_with_figures(self):
        scores = [
            score("agent", question_id="q01", figures_in_answer=2,
                  hallucinated_figures=("$9 billion",)),
            score("agent", question_id="q02", figures_in_answer=1,
                  hallucinated_figures=()),
            # No figures at all, so it cannot have hallucinated one.
            score("agent", question_id="q03", figures_in_answer=0),
        ]
        summary = summarise(scores, system="agent")
        assert summary.hallucinated_figure_rate.n == 2
        assert summary.hallucinated_figure_rate.value == pytest.approx(0.5)

    def test_by_kind_filters(self):
        scores = [
            score("agent", question_id="q01", kind=KIND_QUALITATIVE),
            score("agent", question_id="n01", kind=KIND_NUMERIC,
                  answer_correct=False),
        ]
        assert summarise(scores, system="agent", kind=KIND_NUMERIC).n_questions == 1
        assert summarise(
            scores, system="agent", kind=KIND_NUMERIC
        ).answer_accuracy.value == pytest.approx(0.0)

    def test_empty_metric_renders_as_not_available(self):
        metric = Metric(name="x", value=None, low=None, high=None, n=0)
        assert metric.render() == "n/a"

    def test_empty_report_says_so_rather_than_printing_zeros(self):
        report = build_report(
            [], questions_sha256="abc", questions_path="p", answer_model="m",
            judge_model="j", rubric_version="1", config={},
        )
        rendered = render_markdown(report)
        assert "no evaluation run has been made" in rendered

    def test_report_carries_the_question_hash_and_models(self):
        report = build_report(
            [score("agent")], questions_sha256="deadbeef" * 8,
            questions_path="eval/questions.jsonl", answer_model="claude-opus-5",
            judge_model="claude-sonnet-5", rubric_version="1", config={},
        )
        rendered = render_markdown(report)
        assert "deadbeef" in rendered
        assert "claude-opus-5" in rendered and "claude-sonnet-5" in rendered
