"""The scorer, on hand-made answers whose scores are known in advance.

Every case here is an answer written to have one specific property -- correct,
wrong by a unit error, wrong by direction, declining when it should not,
citing a passage that does not support it -- and the test asserts the score
the rules should give it. The judge is supplied through recorded cache entries,
so nothing here makes a network call and the deterministic paths are shown to
make no model call at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from conftest_cited import build_corpus  # noqa: E402

from ticker.cited.answers import (
    SYSTEM_AGENT,
    SYSTEM_CLOSED,
    Answer,
    Claim,
    validate_citations,
)
from ticker.cited.config import Config
from ticker.cited.judge import (
    SUPPORT_NOT,
    SUPPORT_PARTIAL,
    SUPPORT_SUPPORTED,
    ANSWER_RUBRIC,
    CITATION_RUBRIC,
)
from ticker.cited.llm import CachedClient, OfflineCacheMiss, Usage
from ticker.cited.questions import (
    KIND_NUMERIC,
    KIND_QUALITATIVE,
    KIND_UNANSWERABLE,
    SOURCE_FILING_TEXT,
    SOURCE_NOT_IN_CORPUS,
    SOURCE_XBRL,
    Question,
)
from ticker.cited.scoring import (
    METHOD_DECLINED,
    METHOD_DETERMINISTIC,
    METHOD_JUDGE,
    METHOD_NUMERIC,
    score_answer,
    score_numeric_answer,
)
from ticker.cited.units import UNIT_PERCENT, UNIT_USD


@pytest.fixture
def corpus():
    con, ids = build_corpus()
    return con, ids


@pytest.fixture
def config(tmp_path):
    return Config(offline=True, llm_cache_dir=tmp_path / "llm")


@pytest.fixture
def client(config):
    return CachedClient(config)


def record_citation_verdict(client, claim, passage, label, reason="recorded"):
    """Seed a citation-judge response, exactly as the real call would key it."""
    client.record(
        model=client.config.judge_model,
        system=[
            {
                "type": "text",
                "text": CITATION_RUBRIC,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": f"Claim:\n{claim}\n\nPassage:\n{passage}"}],
        tools=[
            tool
            for tool in [_citation_tool()]
        ],
        tool_choice={"type": "tool", "name": "rate_support"},
        content=[
            {
                "type": "tool_use",
                "id": "toolu_test",
                "name": "rate_support",
                "input": {"label": label, "reason": reason},
            }
        ],
        usage=Usage(input_tokens=500, output_tokens=40),
    )


def record_answer_verdict(client, question, gold, candidate, correct, reason="recorded"):
    client.record(
        model=client.config.judge_model,
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
                    f"Question:\n{question}\n\n"
                    f"Gold answer:\n{gold}\n\n"
                    f"Answer to mark:\n{candidate}"
                ),
            }
        ],
        tools=[_answer_tool()],
        tool_choice={"type": "tool", "name": "mark_answer"},
        content=[
            {
                "type": "tool_use",
                "id": "toolu_test",
                "name": "mark_answer",
                "input": {"correct": correct, "reason": reason},
            }
        ],
        usage=Usage(input_tokens=600, output_tokens=50),
    )


def _citation_tool():
    from ticker.cited.judge import _CITATION_TOOL

    return _CITATION_TOOL


def _answer_tool():
    from ticker.cited.judge import _ANSWER_TOOL

    return _ANSWER_TOOL


NUMERIC_LEVEL = Question(
    question_id="n01",
    kind=KIND_NUMERIC,
    text="What was ZION's total deposits?",
    gold_answer="Total deposits were $75.241 billion.",
    gold_source=SOURCE_XBRL,
    answerable=True,
    rationale="test",
    companies=("ZION",),
    gold_accessions=("0000000002-25-000001",),
    gold_value=75_241_000_000.0,
    gold_unit=UNIT_USD,
)

NUMERIC_YOY = Question(
    question_id="n02",
    kind=KIND_NUMERIC,
    text="By what percent did revenue change?",
    gold_answer="Revenue changed by -2.08%.",
    gold_source=SOURCE_XBRL,
    answerable=True,
    rationale="test",
    companies=("NVDA",),
    gold_accessions=("0000000001-25-000001",),
    gold_value=-2.0783,
    gold_unit=UNIT_PERCENT,
    gold_is_percentage=True,
)

UNANSWERABLE = Question(
    question_id="u01",
    kind=KIND_UNANSWERABLE,
    text="How many shares were outstanding on the cover page?",
    gold_answer="The indexed filings do not answer this.",
    gold_source=SOURCE_NOT_IN_CORPUS,
    answerable=False,
    rationale="cover page is not indexed",
    companies=("NVDA",),
)


class TestNumericCorrectness:
    def make(self, text):
        return validate_citations(
            Answer(
                question_id="n01",
                system=SYSTEM_AGENT,
                model="m",
                summary=text,
                retrieved_chunk_ids=(),
            )
        )

    def test_correct_with_scale_word(self):
        ok, _detail = score_numeric_answer(NUMERIC_LEVEL, self.make("Deposits were $75.241 billion."))
        assert ok

    def test_correct_when_rounded(self):
        ok, _ = score_numeric_answer(NUMERIC_LEVEL, self.make("Deposits were $75.2 billion."))
        assert ok

    def test_unit_error_is_wrong(self):
        # 75,241 in a table means millions; stating it bare is a unit error.
        ok, detail = score_numeric_answer(NUMERIC_LEVEL, self.make("Deposits were 75,241."))
        assert not ok and "relative error" in detail

    def test_declining_an_answerable_numeric_is_wrong(self):
        answer = Answer(
            question_id="n01", system=SYSTEM_AGENT, model="m", declined=True,
            summary="not found",
        )
        ok, detail = score_numeric_answer(NUMERIC_LEVEL, answer)
        assert not ok and "declined" in detail

    def test_no_figure_is_wrong(self):
        ok, detail = score_numeric_answer(NUMERIC_LEVEL, self.make("Deposits grew."))
        assert not ok and "no figure" in detail

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Revenue declined 2.1%.", True),
            ("Revenue decreased by 2.08%.", True),
            ("Revenue changed by -2.1%.", True),
            ("Revenue grew 2.1%.", False),
            ("Revenue changed 2.1%.", False),
            ("Revenue fell 5.4%.", False),
        ],
    )
    def test_percentage_direction(self, text, expected):
        # A filing writes a fall as "declined 2.1%", never "-2.1%", so the
        # direction has to come from the wording. Getting the magnitude right
        # and the direction wrong is still wrong.
        ok, _ = score_numeric_answer(NUMERIC_YOY, self.make(text))
        assert ok is expected

    def test_operands_without_the_change_is_wrong(self):
        ok, detail = score_numeric_answer(
            NUMERIC_YOY, self.make("Revenue fell from $54.2 billion to $53.1 billion.")
        )
        assert not ok and "no percentage" in detail


class TestUnanswerable:
    def test_declining_is_correct(self, corpus, client, config):
        con, _ = corpus
        answer = Answer(
            question_id="u01", system=SYSTEM_AGENT, model="m", declined=True,
            summary="The cover page is not indexed.",
        )
        score = score_answer(UNANSWERABLE, answer, con=con, client=client, config=config)
        assert score.answer_correct
        assert score.correctness_method == METHOD_DECLINED
        assert client.calls == 0  # no judge needed

    def test_answering_is_wrong(self, corpus, client, config):
        con, _ = corpus
        answer = validate_citations(
            Answer(
                question_id="u01", system=SYSTEM_AGENT, model="m",
                summary="There were 2,470,000 shares outstanding.",
                claims=(Claim("There were 2,470,000 shares outstanding.", ()),),
                retrieved_chunk_ids=(),
            )
        )
        score = score_answer(UNANSWERABLE, answer, con=con, client=client, config=config)
        assert not score.answer_correct
        assert score.hallucinated_figures == ("2,470,000",)


class TestCitationSupport:
    def test_figures_in_the_cited_passage_are_supported_deterministically(
        self, corpus, client, config
    ):
        con, ids = corpus
        # The ZION filing's first chunk carries "$374 million".
        chunk_id = ids["zion_2024"][0]
        answer = validate_citations(
            Answer(
                question_id="q01", system=SYSTEM_AGENT, model="m",
                summary="Preferred stock fell.",
                claims=(Claim("Preferred stock decreased $374 million.", (chunk_id,)),),
                retrieved_chunk_ids=(chunk_id,),
            )
        )
        question = Question(
            question_id="q01", kind=KIND_QUALITATIVE,
            text="Why did preferred stock fall?",
            gold_answer="It fell $374 million on redemption of Series G, I and J.",
            gold_source=SOURCE_FILING_TEXT, answerable=True, rationale="test",
            companies=("ZION",), gold_chunk_ids=(chunk_id,),
        )
        record_answer_verdict(
            client, question.text, question.gold_answer, answer.answer_text, True
        )
        score = score_answer(question, answer, con=con, client=client, config=config)
        assert score.claim_scores[0].support == SUPPORT_SUPPORTED
        assert score.claim_scores[0].method == METHOD_DETERMINISTIC
        assert score.citation_recall is True

    def test_a_figure_absent_from_the_passage_is_unsupported(self, corpus, client, config):
        con, ids = corpus
        chunk_id = ids["zion_2024"][0]
        answer = validate_citations(
            Answer(
                question_id="q01", system=SYSTEM_AGENT, model="m",
                summary="Preferred stock fell.",
                claims=(Claim("Preferred stock decreased $999 million.", (chunk_id,)),),
                retrieved_chunk_ids=(chunk_id,),
            )
        )
        question = Question(
            question_id="q01", kind=KIND_QUALITATIVE, text="Why?",
            gold_answer="It fell $374 million.", gold_source=SOURCE_FILING_TEXT,
            answerable=True, rationale="test", companies=("ZION",),
            gold_chunk_ids=(chunk_id,),
        )
        record_answer_verdict(
            client, question.text, question.gold_answer, answer.answer_text, False
        )
        score = score_answer(question, answer, con=con, client=client, config=config)
        assert score.claim_scores[0].support == SUPPORT_NOT
        assert score.hallucinated_figures == ("$999 million",)

    def test_some_figures_present_is_partial(self, corpus, client, config):
        con, ids = corpus
        chunk_id = ids["zion_2024"][0]
        answer = validate_citations(
            Answer(
                question_id="q01", system=SYSTEM_AGENT, model="m", summary="",
                claims=(
                    Claim(
                        "Preferred stock decreased $374 million and deposits were $999 million.",
                        (chunk_id,),
                    ),
                ),
                retrieved_chunk_ids=(chunk_id,),
            )
        )
        question = Question(
            question_id="q01", kind=KIND_QUALITATIVE, text="Why?",
            gold_answer="x", gold_source=SOURCE_FILING_TEXT, answerable=True,
            rationale="test", companies=("ZION",), gold_chunk_ids=(chunk_id,),
        )
        record_answer_verdict(client, question.text, "x", answer.answer_text, False)
        score = score_answer(question, answer, con=con, client=client, config=config)
        assert score.claim_scores[0].support == SUPPORT_PARTIAL
        assert score.claim_scores[0].figures_found == 1
        assert score.claim_scores[0].figures_checked == 2

    def test_a_claim_with_no_figure_goes_to_the_judge(self, corpus, client, config):
        con, ids = corpus
        chunk_id = ids["zion_2024"][0]
        claim = "The redemption covered the Series G, I and J preferred shares."
        answer = validate_citations(
            Answer(
                question_id="q01", system=SYSTEM_AGENT, model="m", summary="",
                claims=(Claim(claim, (chunk_id,)),),
                retrieved_chunk_ids=(chunk_id,),
            )
        )
        question = Question(
            question_id="q01", kind=KIND_QUALITATIVE, text="Which series?",
            gold_answer="Series G, I and J.", gold_source=SOURCE_FILING_TEXT,
            answerable=True, rationale="test", companies=("ZION",),
            gold_chunk_ids=(chunk_id,),
        )
        passage = con.execute(
            "SELECT text FROM chunks WHERE chunk_id = ?", [chunk_id]
        ).fetchone()[0]
        record_citation_verdict(client, claim, passage, SUPPORT_SUPPORTED)
        record_answer_verdict(
            client, question.text, question.gold_answer, answer.answer_text, True
        )
        score = score_answer(question, answer, con=con, client=client, config=config)
        assert score.claim_scores[0].method == METHOD_JUDGE
        assert score.claim_scores[0].support == SUPPORT_SUPPORTED

    def test_missing_judge_fixture_raises_offline(self, corpus, client, config):
        con, ids = corpus
        chunk_id = ids["zion_2024"][0]
        question = Question(
            question_id="q01", kind=KIND_QUALITATIVE, text="Which series?",
            gold_answer="Series G, I and J.", gold_source=SOURCE_FILING_TEXT,
            answerable=True, rationale="test", companies=("ZION",),
            gold_chunk_ids=(chunk_id,),
        )
        answer = validate_citations(
            Answer(
                question_id="q01", system=SYSTEM_AGENT, model="m", summary="",
                claims=(Claim("Some unfigured prose.", (chunk_id,)),),
                retrieved_chunk_ids=(chunk_id,),
            )
        )
        # Nothing recorded, so offline mode must refuse rather than send.
        with pytest.raises(OfflineCacheMiss):
            score_answer(question, answer, con=con, client=client, config=config)


class TestUncitedAndRejected:
    def test_uncited_claim_is_labelled_not_judged(self, corpus, client, config):
        con, _ = corpus
        question = Question(
            question_id="n01", kind=KIND_NUMERIC, text="What were deposits?",
            gold_answer="x", gold_source=SOURCE_XBRL, answerable=True,
            rationale="test", companies=("ZION",),
            gold_accessions=("0000000002-25-000001",),
            gold_value=75_241_000_000.0, gold_unit=UNIT_USD,
        )
        answer = validate_citations(
            Answer(
                question_id="n01", system=SYSTEM_CLOSED, model="m",
                summary="Deposits were $75.241 billion.",
                claims=(Claim("Deposits were $75.241 billion.", ("FAKE#1#chunk0",)),),
                retrieved_chunk_ids=(),
            )
        )
        score = score_answer(question, answer, con=con, client=client, config=config)
        assert score.answer_correct  # right number, no grounding
        assert score.correctness_method == METHOD_NUMERIC
        assert score.rejected_citations == ("FAKE#1#chunk0",)
        assert score.uncited_claim_count == 1
        assert score.citation_support_rate is None
        assert client.calls == 0

    def test_support_rate_is_none_with_no_cited_claims(self, corpus, client, config):
        con, _ = corpus
        answer = Answer(
            question_id="u01", system=SYSTEM_AGENT, model="m", declined=True,
            summary="Not in the filings.",
        )
        score = score_answer(UNANSWERABLE, answer, con=con, client=client, config=config)
        # Returning zero here would average a declined answer in as a support
        # failure when it made no citation to fail.
        assert score.citation_support_rate is None
