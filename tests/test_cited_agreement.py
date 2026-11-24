"""The judge-agreement check: blind sampling, determinism, and kappa.

The property that matters most is that the blind sample carries no judge
verdict. An agreement number measured against a sample that showed the
labeller the judge's answer measures compliance, not agreement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from conftest_cited import build_corpus  # noqa: E402

from ticker.cited.agreement import (
    DEFAULT_SEED,
    compare,
    eligible_pairs,
    load_labels,
    render_markdown,
    sample_pairs,
    write_judge_verdicts,
    write_sample,
)
from ticker.cited.judge import SUPPORT_NOT, SUPPORT_PARTIAL, SUPPORT_SUPPORTED
from ticker.cited.scoring import (
    METHOD_DETERMINISTIC,
    METHOD_JUDGE,
    AnswerScore,
    ClaimScore,
)


@pytest.fixture
def corpus():
    return build_corpus()


def make_score(system, question_id, claims):
    return AnswerScore(
        question_id=question_id,
        kind="qualitative",
        system=system,
        answer_correct=True,
        correctness_method="llm_judge",
        correctness_detail="",
        declined=False,
        claim_scores=tuple(claims),
    )


def claim(index, chunk_ids, method=METHOD_JUDGE, support=SUPPORT_SUPPORTED):
    return ClaimScore(
        index=index,
        text=f"Claim {index}.",
        chunk_ids=tuple(chunk_ids),
        support=support,
        method=method,
        reason="because",
    )


class TestEligibility:
    def test_only_judge_rated_pairs_are_eligible(self, corpus):
        # A claim checked by the number-in-passage rule has no judge verdict to
        # agree or disagree with, so including it would inflate agreement with
        # cases the judge never saw.
        con, ids = corpus
        chunk = ids["zion_2024"][0]
        scores = [
            make_score("agent", "q01", [
                claim(0, [chunk], method=METHOD_JUDGE),
                claim(1, [chunk], method=METHOD_DETERMINISTIC),
            ])
        ]
        pairs = eligible_pairs(scores, con=con)
        assert [pair.claim_index for pair in pairs] == [0]

    def test_uncited_claims_are_not_eligible(self, corpus):
        con, _ = corpus
        scores = [make_score("closed", "q01", [claim(0, [])])]
        assert eligible_pairs(scores, con=con) == []

    def test_unresolvable_chunk_ids_are_not_eligible(self, corpus):
        con, _ = corpus
        scores = [make_score("agent", "q01", [claim(0, ["ghost#1A#chunk0"])])]
        assert eligible_pairs(scores, con=con) == []

    def test_pair_carries_the_cited_text(self, corpus):
        con, ids = corpus
        pairs = eligible_pairs(
            [make_score("agent", "q01", [claim(0, [ids["zion_2024"][0]])])], con=con
        )
        assert "$374 million" in pairs[0].cited_text
        assert pairs[0].pair_id == "agent:q01:0"


class TestSampling:
    def build(self, con, ids, per_system=10):
        scores = []
        for system in ("agent", "closed", "single"):
            for index in range(per_system):
                scores.append(
                    make_score(system, f"q{index:02d}", [claim(0, [ids["zion_2024"][0]])])
                )
        return eligible_pairs(scores, con=con)

    def test_is_deterministic_for_a_seed(self, corpus):
        con, ids = corpus
        pairs = self.build(con, ids)
        first = [p.pair_id for p in sample_pairs(pairs, size=9, seed=7)]
        second = [p.pair_id for p in sample_pairs(pairs, size=9, seed=7)]
        assert first == second

    def test_a_different_seed_gives_a_different_sample(self, corpus):
        con, ids = corpus
        pairs = self.build(con, ids)
        assert [p.pair_id for p in sample_pairs(pairs, size=9, seed=1)] != [
            p.pair_id for p in sample_pairs(pairs, size=9, seed=2)
        ]

    def test_is_stratified_across_systems(self, corpus):
        # An unstratified draw under-samples whichever arm made fewer cited
        # claims, which is usually the one whose citations are most suspect.
        con, ids = corpus
        drawn = sample_pairs(self.build(con, ids), size=9, seed=DEFAULT_SEED)
        counts = {}
        for pair in drawn:
            counts[pair.system] = counts.get(pair.system, 0) + 1
        assert set(counts) == {"agent", "closed", "single"}
        assert all(count == 3 for count in counts.values())

    def test_redistributes_when_a_system_is_short(self, corpus):
        con, ids = corpus
        scores = [make_score("agent", f"q{i:02d}", [claim(0, [ids["zion_2024"][0]])])
                  for i in range(8)]
        scores.append(make_score("closed", "q99", [claim(0, [ids["zion_2024"][0]])]))
        drawn = sample_pairs(eligible_pairs(scores, con=con), size=6, seed=DEFAULT_SEED)
        assert len(drawn) == 6

    def test_asks_for_more_than_exists(self, corpus):
        con, ids = corpus
        pairs = self.build(con, ids, per_system=1)
        assert len(sample_pairs(pairs, size=40, seed=DEFAULT_SEED)) == 3

    def test_empty_input(self, corpus):
        assert sample_pairs([], size=10) == []

    def test_zero_size_is_rejected(self, corpus):
        con, ids = corpus
        with pytest.raises(ValueError):
            sample_pairs(self.build(con, ids), size=0)


class TestBlindSample:
    def test_the_blind_file_contains_no_judge_verdict(self, corpus, tmp_path):
        con, ids = corpus
        scores = [
            make_score("agent", "q01", [
                claim(0, [ids["zion_2024"][0]], support=SUPPORT_NOT)
            ])
        ]
        pairs = sample_pairs(eligible_pairs(scores, con=con), size=1)
        path = tmp_path / "judge_sample.jsonl"
        write_sample(path, pairs)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]["label"] is None
        # The judge said not_supported; the blind file must not reveal it.
        assert SUPPORT_NOT not in json.dumps(rows[0]).replace(
            json.dumps(rows[0]["label_options"]), ""
        )
        assert set(rows[0]["label_options"]) == {
            SUPPORT_SUPPORTED, SUPPORT_PARTIAL, SUPPORT_NOT
        }

    def test_verdicts_go_to_a_separate_file(self, corpus, tmp_path):
        con, ids = corpus
        scores = [
            make_score("agent", "q01", [
                claim(0, [ids["zion_2024"][0]], support=SUPPORT_PARTIAL)
            ])
        ]
        pairs = sample_pairs(eligible_pairs(scores, con=con), size=1)
        path = tmp_path / "verdicts.jsonl"
        assert write_judge_verdicts(path, scores, pairs) == 1
        assert load_labels(path) == {"agent:q01:0": SUPPORT_PARTIAL}

    def test_labels_file_last_line_wins(self, tmp_path):
        # A correction is a new line, matching how `ticker.qrels` treats its
        # append-only judgment files.
        path = tmp_path / "labels.jsonl"
        path.write_text(
            json.dumps({"pair_id": "a:q:0", "label": SUPPORT_NOT}) + "\n"
            + json.dumps({"pair_id": "a:q:0", "label": SUPPORT_SUPPORTED}) + "\n"
        )
        assert load_labels(path) == {"a:q:0": SUPPORT_SUPPORTED}

    def test_missing_labels_file_is_empty_not_an_error(self, tmp_path):
        assert load_labels(tmp_path / "nope.jsonl") == {}


class TestAgreement:
    def test_perfect_agreement(self):
        labels = {f"p{i}": SUPPORT_SUPPORTED for i in range(10)}
        report = compare(judge_labels=labels, human_labels=labels)
        assert report.raw_agreement == 1.0
        assert report.n_labelled == 10

    def test_total_disagreement_gives_negative_kappa(self):
        judge = {"a": SUPPORT_SUPPORTED, "b": SUPPORT_NOT}
        human = {"a": SUPPORT_NOT, "b": SUPPORT_SUPPORTED}
        report = compare(judge_labels=judge, human_labels=human)
        assert report.raw_agreement == 0.0
        assert report.kappa < 0

    def test_partial_agreement(self):
        judge = {"a": SUPPORT_SUPPORTED, "b": SUPPORT_SUPPORTED,
                 "c": SUPPORT_NOT, "d": SUPPORT_NOT}
        human = {"a": SUPPORT_SUPPORTED, "b": SUPPORT_NOT,
                 "c": SUPPORT_NOT, "d": SUPPORT_NOT}
        report = compare(judge_labels=judge, human_labels=human)
        assert report.raw_agreement == 0.75

    def test_only_shared_pairs_are_compared(self):
        judge = {"a": SUPPORT_SUPPORTED, "b": SUPPORT_NOT}
        human = {"a": SUPPORT_SUPPORTED, "z": SUPPORT_NOT}
        report = compare(judge_labels=judge, human_labels=human)
        assert report.n_labelled == 1
        assert report.missing_pair_ids == ("b",)
        assert report.unknown_pair_ids == ("z",)

    def test_no_human_labels_reports_nothing(self):
        report = compare(judge_labels={"a": SUPPORT_SUPPORTED}, human_labels={})
        assert report.n_labelled == 0
        assert report.raw_agreement is None and report.kappa is None
        # Must not claim acceptability on no evidence.
        assert not report.is_acceptable
        assert "No human labels" in render_markdown(report)

    def test_acceptability_threshold(self):
        # 0.6 is the conventional "substantial agreement" boundary, and the
        # gate that decides whether judge-based numbers get reported.
        labels = {f"p{i}": SUPPORT_SUPPORTED for i in range(5)}
        labels.update({f"n{i}": SUPPORT_NOT for i in range(5)})
        report = compare(judge_labels=labels, human_labels=labels)
        assert report.kappa == pytest.approx(1.0)
        assert report.is_acceptable

    def test_confusion_matrix_is_human_rows_judge_columns(self):
        judge = {"a": SUPPORT_NOT}
        human = {"a": SUPPORT_SUPPORTED}
        report = compare(judge_labels=judge, human_labels=human)
        assert report.confusion[SUPPORT_SUPPORTED][SUPPORT_NOT] == 1
        assert report.confusion[SUPPORT_NOT][SUPPORT_SUPPORTED] == 0
