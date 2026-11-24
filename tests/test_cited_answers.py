"""Citation validation, the answer schema's invariants, and round-tripping."""

from __future__ import annotations

import pytest

from ticker.cited.answers import (
    DECLINED_MARKER,
    SYSTEM_AGENT,
    SYSTEM_CLOSED,
    Answer,
    Claim,
    ToolCall,
    from_json,
    load_answers,
    to_json,
    validate_citations,
    write_answers,
)

REAL_A = "0000000001-25-000001#1A#chunk0"
REAL_B = "0000000001-25-000001#1A#chunk2"
FABRICATED = "0000000001-25-000001#1A#chunk9999"


def answer(**overrides) -> Answer:
    base = dict(
        question_id="q01",
        system=SYSTEM_AGENT,
        model="test-model",
        summary="A summary.",
        claims=(Claim("A claim.", (REAL_A,)),),
        retrieved_chunk_ids=(REAL_A, REAL_B),
    )
    base.update(overrides)
    return Answer(**base)


class TestCitationValidation:
    def test_keeps_retrieved_citations(self):
        validated = validate_citations(answer())
        assert validated.claims[0].chunk_ids == (REAL_A,)
        assert validated.rejected_citations == ()

    def test_rejects_a_chunk_never_retrieved(self):
        # Chunk ids are structural, so a model can build a plausible one from a
        # known accession without having read anything. That is the case this
        # check exists for.
        validated = validate_citations(
            answer(claims=(Claim("A claim.", (FABRICATED,)),))
        )
        assert validated.rejected_citations == (FABRICATED,)
        assert validated.claims[0].chunk_ids == ()
        assert validated.claims[0].is_uncited

    def test_keeps_the_valid_part_of_a_mixed_citation(self):
        # Discarding the whole answer would lose the information that one of
        # the two citations was real.
        validated = validate_citations(
            answer(claims=(Claim("A claim.", (REAL_A, FABRICATED)),))
        )
        assert validated.claims[0].chunk_ids == (REAL_A,)
        assert validated.rejected_citations == (FABRICATED,)

    def test_closed_book_citations_are_all_rejected(self):
        # A system that retrieved nothing cannot have a valid citation. This is
        # the measurement, not a defect.
        validated = validate_citations(
            answer(
                system=SYSTEM_CLOSED,
                retrieved_chunk_ids=(),
                claims=(Claim("A claim.", (REAL_A,)),),
            )
        )
        assert validated.rejected_citations == (REAL_A,)
        assert validated.claims[0].chunk_ids == ()

    def test_rejections_are_deduplicated(self):
        validated = validate_citations(
            answer(
                claims=(
                    Claim("One.", (FABRICATED,)),
                    Claim("Two.", (FABRICATED,)),
                )
            )
        )
        assert validated.rejected_citations == (FABRICATED,)

    def test_no_fuzzy_repair(self):
        # A near-miss id is a fabrication. Normalising case or whitespace into
        # a match would hide exactly what this check looks for.
        validated = validate_citations(
            answer(claims=(Claim("A claim.", (REAL_A.upper(),)),))
        )
        assert validated.claims[0].chunk_ids == ()


class TestSchemaInvariants:
    def test_declining_and_claiming_is_rejected(self):
        with pytest.raises(ValueError, match="declined"):
            answer(declined=True, claims=(Claim("A claim.", ()),))

    def test_unknown_system_is_rejected(self):
        with pytest.raises(ValueError, match="unknown system"):
            answer(system="magic")

    def test_declined_answer_text_is_the_marker(self):
        assert answer(declined=True, claims=()).answer_text == DECLINED_MARKER

    def test_answer_text_joins_summary_and_claims(self):
        text = answer(
            summary="Summary.", claims=(Claim("First.", ()), Claim("Second.", ()))
        ).answer_text
        assert "Summary." in text and "First." in text and "Second." in text

    def test_cited_chunk_ids_are_unique_and_ordered(self):
        subject = answer(
            claims=(Claim("One.", (REAL_B, REAL_A)), Claim("Two.", (REAL_A,)))
        )
        assert subject.cited_chunk_ids == (REAL_B, REAL_A)


class TestRoundTrip:
    def test_round_trips_through_json(self):
        subject = validate_citations(
            answer(
                tool_calls=(
                    ToolCall(
                        index=1,
                        name="search",
                        arguments={"query": "revenue", "k": 8},
                        ok=True,
                        chunk_ids_returned=(REAL_A, REAL_B),
                        latency_ms=12.5,
                    ),
                ),
                cost_usd=0.0123,
                latency_s=4.5,
                hit_tool_cap=True,
            )
        )
        assert from_json(to_json(subject)) == subject

    def test_file_round_trip(self, tmp_path):
        path = tmp_path / "answers.jsonl"
        subjects = [answer(question_id="q02"), answer(question_id="q01")]
        assert write_answers(path, subjects) == 2
        loaded = load_answers(path)
        # Written sorted by question id, so a results file is stable.
        assert [a.question_id for a in loaded] == ["q01", "q02"]
