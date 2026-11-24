"""Unit normalisation, the numeric tolerance, and the passage match.

Hand-made cases only. Every expectation here is a judgement about what a
figure in a filing means, so each case is written out rather than generated;
a generated case would need the same normalisation logic to decide its own
expected answer.
"""

from __future__ import annotations

import pytest

from ticker.cited.units import (
    BARE_FIGURE_MIN_DIGITS,
    DEFAULT_RELATIVE_TOLERANCE,
    DIRECTION_DOWN,
    DIRECTION_UNSTATED,
    DIRECTION_UP,
    MIN_SIGNIFICANT_DIGITS,
    PERCENTAGE_POINT_TOLERANCE,
    UNIT_PERCENT,
    UNIT_USD,
    extract_quantities,
    format_usd,
    number_appears_in,
    parse_quantity,
    percent_change,
    reportable_quantities,
    significant_digits,
    stated_direction,
    within_tolerance,
)


class TestParsing:
    @pytest.mark.parametrize(
        "text,value,unit",
        [
            ("$53.1 billion", 53_100_000_000.0, UNIT_USD),
            ("$2.1 million", 2_100_000.0, UNIT_USD),
            ("53,101 million", 53_101_000_000.0, "unknown"),
            ("16,546", 16_546.0, "unknown"),
            ("2.1%", 2.1, UNIT_PERCENT),
            ("2.1 percent", 2.1, UNIT_PERCENT),
            ("9.73", 9.73, "unknown"),
            ("$101 billion", 101_000_000_000.0, UNIT_USD),
            ("1,035 thousand", 1_035_000.0, "unknown"),
        ],
    )
    def test_reads_scale_and_unit(self, text, value, unit):
        quantity = parse_quantity(text)
        assert quantity is not None
        assert quantity.value == pytest.approx(value)
        assert quantity.unit == unit

    @pytest.mark.parametrize(
        "text,value",
        [
            ("(18,756)", -18_756.0),
            ("-18,756", -18_756.0),
            ("−18,756", -18_756.0),
        ],
    )
    def test_negatives(self, text, value):
        quantity = parse_quantity(text)
        assert quantity is not None
        assert quantity.value == pytest.approx(value)

    def test_two_figures_is_not_a_single_quantity(self):
        # Taking the first would silently score an answer against the wrong
        # operand of a from-to statement.
        assert parse_quantity("revenue fell from 54,228 to 53,101") is None

    def test_bare_years_are_not_quantities(self):
        raws = [q.raw for q in extract_quantities("In 2024 revenue was 53,101")]
        assert raws == ["53,101"]

    def test_years_kept_when_asked(self):
        raws = [q.raw for q in extract_quantities("In 2024", exclude_years=False)]
        assert raws == ["2024"]

    def test_dollar_year_is_a_quantity_not_a_year(self):
        # "$2024" is money, whatever it looks like.
        quantity = parse_quantity("$2024")
        assert quantity is not None and quantity.unit == UNIT_USD

    def test_table_row(self):
        row = "Net revenue 53,101 100.0 $ 54,228 100.0 63,054 100.0"
        values = [q.value for q in extract_quantities(row)]
        assert 53_101.0 in values and 54_228.0 in values and 63_054.0 in values


class TestSignificantDigits:
    @pytest.mark.parametrize(
        "digits,expected",
        [
            ("83", 2),
            ("53,101", 5),
            ("14,089.8", 6),
            ("0.027", 2),
            ("111,601", 6),
            ("9.73", 3),
            ("100.0", 4),
        ],
    )
    def test_counts(self, digits, expected):
        assert significant_digits(digits) == expected


class TestReportableQuantities:
    def test_drops_dates_and_small_ordinals(self):
        text = "Revenue was $53.1 billion for the year ended December 28, 2024"
        assert [q.raw for q in reportable_quantities(text)] == ["$53.1 billion"]

    def test_drops_small_counts(self):
        text = "It has two active banking subsidiaries and is among the 25 largest"
        assert reportable_quantities(text) == []

    def test_keeps_percentages(self):
        assert [q.raw for q in reportable_quantities("45% of net sales")] == ["45%"]

    def test_keeps_long_bare_figures(self):
        assert [q.raw for q in reportable_quantities("Net revenue 53,101")] == ["53,101"]

    def test_threshold_boundary(self):
        # Three digits is below the bare-figure floor of four; four is at it.
        assert reportable_quantities("the figure 123 appears") == []
        assert [q.raw for q in reportable_quantities("the figure 1234 appears")] == ["1234"]
        assert BARE_FIGURE_MIN_DIGITS == 4


class TestTolerance:
    GOLD = 53_101_000_000.0

    def test_exact(self):
        assert within_tolerance(self.GOLD, self.GOLD).passed

    def test_rounded_to_three_digits_passes(self):
        check = within_tolerance(53_100_000_000.0, self.GOLD)
        assert check.passed and check.rule == "relative"

    def test_half_a_percent_is_the_boundary(self):
        just_inside = self.GOLD * (1 + DEFAULT_RELATIVE_TOLERANCE * 0.99)
        just_outside = self.GOLD * (1 + DEFAULT_RELATIVE_TOLERANCE * 1.01)
        assert within_tolerance(just_inside, self.GOLD).passed
        assert not within_tolerance(just_outside, self.GOLD).passed

    def test_scale_error_fails(self):
        # 53,101 against $53.101 billion is a unit error, not a rounding one.
        assert not within_tolerance(53_101.0, self.GOLD).passed

    def test_sign_is_never_ignored(self):
        # A loss reported as a profit is the error this check must catch.
        assert not within_tolerance(18_756e6, -18_756e6).passed

    def test_zero_gold_requires_exact(self):
        assert within_tolerance(0.0, 0.0).passed
        assert not within_tolerance(0.01, 0.0).passed
        assert within_tolerance(0.0, 0.0).rule == "exact_zero"

    def test_percentage_point_allowance_only_for_percentages(self):
        gold = percent_change(53_101e6, 54_228e6)  # -2.0783%
        assert within_tolerance(-2.1, gold, is_percentage=True).passed
        assert within_tolerance(-2.1, gold, is_percentage=True).rule == "percentage_point"
        # Without the percentage flag the same pair fails on relative error.
        assert not within_tolerance(-2.1, gold, is_percentage=False).passed

    def test_percentage_point_allowance_is_bounded(self):
        gold = -2.0
        assert within_tolerance(
            gold - PERCENTAGE_POINT_TOLERANCE * 0.9, gold, is_percentage=True
        ).passed
        assert not within_tolerance(-2.5, gold, is_percentage=True).passed


class TestPercentChange:
    def test_decline(self):
        assert percent_change(53_101e6, 54_228e6) == pytest.approx(-2.0783, abs=1e-4)

    def test_growth(self):
        assert percent_change(63_887e6, 51_574e6) == pytest.approx(23.8744, abs=1e-4)

    def test_zero_base_raises(self):
        # Returning infinity would put an unmatchable value in a gold answer.
        with pytest.raises(ValueError):
            percent_change(1.0, 0.0)


class TestNumberInPassage:
    GOLD = 53_101_000_000.0

    def test_finds_table_figure_without_a_scale_word(self):
        match = number_appears_in(self.GOLD, "Net revenue 53,101 100.0")
        assert match.found and match.matched_scale == 1e6

    def test_finds_prose_figure_with_a_scale_word(self):
        match = number_appears_in(self.GOLD, "revenue of $53.1 billion")
        assert match.found and match.matched_scale == 1.0

    def test_rejects_a_different_figure(self):
        assert not number_appears_in(self.GOLD, "revenue of $54,228").found

    def test_rejects_short_coincidences(self):
        # At scale 1e9 the bare token "83" is within 0.5% of $82.798 billion.
        # The significant-digit floor is what stops that counting as a match.
        assert not number_appears_in(82_798_000_000.0, "across 83 locations").found
        assert MIN_SIGNIFICANT_DIGITS == 3

    def test_keeps_genuine_three_digit_matches(self):
        assert number_appears_in(53_100_000_000.0, "revenue of $53.1 billion").found

    def test_sign_is_ignored_in_a_passage(self):
        # Filings write a loss as "(18,756)" in a table and the word "loss" is
        # often in a different chunk, so magnitude is the right comparison here.
        assert number_appears_in(-18_756_000_000.0, "Net loss (18,756)").found

    def test_percentage_needs_a_percent_marker(self):
        gold = percent_change(53_101e6, 54_228e6)
        assert number_appears_in(gold, "decreased 2.1% in 2024", is_percentage=True).found
        assert not number_appears_in(
            gold, "cost of sales was 2.1 and other items", is_percentage=True
        ).found

    def test_percentage_rounding_allowance(self):
        gold = percent_change(53_101e6, 54_228e6)
        assert number_appears_in(gold, "a decline of 2.1%", is_percentage=True).found
        assert not number_appears_in(gold, "a decline of 5.4%", is_percentage=True).found

    def test_explicit_scale_is_not_rescaled(self):
        # "$53.1 billion" means what it says; it must not also be tried at
        # 1e6 and so match a gold of $53.1 quadrillion.
        assert not number_appears_in(53.1e15, "revenue of $53.1 billion").found


class TestStatedDirection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Revenue declined 2.1%", DIRECTION_DOWN),
            ("Revenue decreased by 2.1%", DIRECTION_DOWN),
            ("Revenue fell 2.1%", DIRECTION_DOWN),
            ("Revenue was down 2.1%", DIRECTION_DOWN),
            ("Revenue grew 2.1%", DIRECTION_UP),
            ("Revenue increased 2.1%", DIRECTION_UP),
            ("Revenue rose 2.1%", DIRECTION_UP),
            ("Revenue changed 2.1%", DIRECTION_UNSTATED),
        ],
    )
    def test_reads_direction_from_wording(self, text, expected):
        assert stated_direction(text) == expected

    def test_explicit_sign_wins(self):
        quantity = parse_quantity("-2.1%")
        assert stated_direction("Revenue grew -2.1%", quantity) == DIRECTION_DOWN

    def test_both_directions_is_unstated(self):
        # Picking whichever word came first would be arbitrary.
        assert stated_direction("revenue fell as costs increased") == DIRECTION_UNSTATED


class TestFormatUsd:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (53_101_000_000.0, "$53.101 billion"),
            (335_000_000.0, "$335.0 million"),
            (-18_756_000_000.0, "-$18.756 billion"),
            (9.73, "$10"),
        ],
    )
    def test_renders(self, value, expected):
        assert format_usd(value) == expected
