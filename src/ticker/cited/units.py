"""Parsing, normalising and comparing the figures that appear in answers.

Three jobs, deliberately kept apart because they need different strictness.

**Answer correctness.** The system states a figure and states its scale, so
`parse_quantity` reads the scale the answer gave and compares at the 0.5%
relative tolerance the specification sets. Nothing is guessed: an answer of
"53,101" with no scale is parsed as fifty-three thousand, which is what it
says, and is wrong against a gold of $53.101 billion. Scale-guessing here
would turn a unit error -- one of the failure categories worth counting -- into
a pass.

**Number in a cited passage.** The opposite problem. Filing tables carry their
scale in a column header ("(In Millions, Except Per Share Amounts)") that sits
in a different chunk from the row holding the figure, so the cited passage very
often reads `53,101` for $53.101 billion with nothing nearby to say so.
`number_appears_in` therefore tries every plausible scale and reports which one
matched, so a human can audit the match rather than trust it. Being strict here
would fail citations that are in fact correct, for a reason that is an artefact
of chunking.

**Percentages.** A relative 0.5% tolerance on a gold of -2.08% admits only
±0.0104 percentage points, so an answer of "declined 2.1%" -- correct by any
reading, and the way a filing itself would write it -- scores wrong. That
measures rounding, not correctness. Percentage-valued golds therefore also
accept an absolute `PERCENTAGE_POINT_TOLERANCE`, which is a stated widening of
the specification rather than a silent one: it is a named constant, it is
configurable, and `ToleranceCheck.rule` records which of the two rules let a
given answer through so the split is visible in the results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

UNIT_USD = "USD"
UNIT_SHARES = "shares"
UNIT_PERCENT = "percent"
UNIT_UNKNOWN = "unknown"

# The specification's figure: 0.5% relative.
DEFAULT_RELATIVE_TOLERANCE = 0.005

# Rounding allowance for percentage-valued golds, in percentage points. One
# decimal place is how filings quote a growth rate, so this is the width of
# the rounding, not a margin of error on top of it.
PERCENTAGE_POINT_TOLERANCE = 0.05

SCALES: dict[str, float] = {
    "hundred": 1e2,
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "trillions": 1e12,
}

# Scales a bare table figure might be expressed in, for the passage check.
# Ordered so the smallest multiplier wins a tie, which makes the reported
# match the least surprising of the options.
CANDIDATE_SCALES: tuple[float, ...] = (1.0, 1e3, 1e6, 1e9)

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"

_SCALE_ALTERNATION = "|".join(
    sorted(SCALES, key=len, reverse=True)
)

# One quantity: optional sign or opening paren, optional currency symbol, the
# digits, optional closing paren, then either a percent sign or a scale word.
# `\b` around the scale word stops "m" matching the start of "margin".
_QUANTITY_RE = re.compile(
    rf"""
    (?P<open>[-−(]\s*)?
    (?P<currency>[$])?\s*
    (?P<digits>{_NUMBER})
    \s*
    (?P<close>\))?
    \s*
    (?:
        (?P<percent>%|\bpercent(?:age)?\b)
      | (?P<scale>\b(?:{_SCALE_ALTERNATION})\b)
    )?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


@dataclass(frozen=True, slots=True)
class Quantity:
    """A figure read out of text, carried with everything needed to audit it."""

    value: float
    unit: str
    scale: float
    raw: str
    start: int
    end: int

    @property
    def is_percent(self) -> bool:
        return self.unit == UNIT_PERCENT


def _digits_to_float(digits: str) -> float:
    return float(digits.replace(",", ""))


def _looks_like_year(match: re.Match) -> bool:
    """True for a bare four-digit integer in calendar range.

    Filings are dense with years, and counting them as figures makes the
    hallucinated-figure metric mostly a count of years. A token only escapes
    this test by carrying something a year never does: a currency symbol, a
    decimal point, a thousands separator, a sign, or a scale word.
    """
    if match.group("currency") or match.group("scale") or match.group("percent"):
        return False
    if match.group("open") or match.group("close"):
        return False
    return bool(_YEAR_RE.match(match.group("digits")))


def extract_quantities(
    text: str, *, exclude_years: bool = True, default_unit: str = UNIT_UNKNOWN
) -> list[Quantity]:
    """Every figure in `text`, left to right.

    `exclude_years` drops bare calendar years; see `_looks_like_year`. Callers
    that genuinely want years -- a wrong-year diagnosis, say -- pass False.
    """
    found: list[Quantity] = []
    for match in _QUANTITY_RE.finditer(text):
        if exclude_years and _looks_like_year(match):
            continue
        scale_word = match.group("scale")
        scale = SCALES[scale_word.lower()] if scale_word else 1.0
        magnitude = _digits_to_float(match.group("digits")) * scale

        negative = bool(match.group("open")) or (
            bool(match.group("close")) and bool(match.group("open"))
        )
        # Accounting parentheses: a closing paren with no opener still signals
        # a negative when the opener was consumed as part of the preceding
        # token, which happens for "(1,234)" inside a table row.
        if match.group("close") and not match.group("open"):
            negative = True

        if match.group("percent"):
            unit = UNIT_PERCENT
        elif match.group("currency"):
            unit = UNIT_USD
        else:
            unit = default_unit

        found.append(
            Quantity(
                value=-magnitude if negative else magnitude,
                unit=unit,
                scale=scale,
                raw=match.group(0).strip(),
                start=match.start(),
                end=match.end(),
            )
        )
    return found


def parse_quantity(text: str, *, default_unit: str = UNIT_UNKNOWN) -> Quantity | None:
    """The single figure in `text`, or None if there is not exactly one.

    Returns None on two figures rather than picking the first: "revenue fell
    from 54,228 to 53,101" has no single answer, and silently taking the
    leading number would score a correct answer against the wrong operand.
    """
    found = extract_quantities(text, exclude_years=True, default_unit=default_unit)
    return found[0] if len(found) == 1 else None


DIRECTION_UP = 1
DIRECTION_DOWN = -1
DIRECTION_UNSTATED = 0

_DOWN_WORDS = re.compile(
    r"\b(declin\w*|decreas\w*|fell|fall\w*|dropp?\w*|down|lower\w*|reduc\w*|"
    r"contract\w*|shrank|shrunk|loss|negative)\b",
    re.IGNORECASE,
)
_UP_WORDS = re.compile(
    r"\b(increas\w*|grew|grow\w*|rose|rising|risen|up|higher|gain\w*|improv\w*|"
    r"expand\w*|climb\w*|jump\w*)\b",
    re.IGNORECASE,
)


def stated_direction(text: str, quantity: "Quantity | None" = None) -> int:
    """Which way a change is stated to go: up, down, or unstated.

    A percentage change is almost never written with a sign. Filings and
    analysts write "revenue declined 2.1%", not "revenue changed by -2.1%", so
    comparing a signed gold against the signed value of the figure alone marks
    the ordinary phrasing wrong. The direction therefore comes from an explicit
    sign on the figure when there is one, and from the surrounding verb
    otherwise.

    `DIRECTION_UNSTATED` is returned when neither is present, and the caller
    treats that as a failure rather than a pass. "2.1%" on its own does not
    say whether revenue rose or fell, and guessing from the gold's sign would
    mark an ambiguous answer correct half the time for no reason.

    Both directions present returns `DIRECTION_UNSTATED` too: "revenue fell as
    costs increased" does not determine the direction of the figure, and
    picking whichever word came first would be arbitrary.
    """
    if quantity is not None and quantity.value < 0:
        return DIRECTION_DOWN
    down = bool(_DOWN_WORDS.search(text))
    up = bool(_UP_WORDS.search(text))
    if down and not up:
        return DIRECTION_DOWN
    if up and not down:
        return DIRECTION_UP
    return DIRECTION_UNSTATED


@dataclass(frozen=True, slots=True)
class ToleranceCheck:
    """The outcome of comparing an answer figure to a gold figure."""

    passed: bool
    gold: float
    candidate: float
    relative_error: float | None
    absolute_error: float
    rule: str  # "relative" | "percentage_point" | "exact_zero" | "failed"


def within_tolerance(
    candidate: float,
    gold: float,
    *,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    is_percentage: bool = False,
    percentage_point_tolerance: float = PERCENTAGE_POINT_TOLERANCE,
) -> ToleranceCheck:
    """Is `candidate` within tolerance of `gold`?

    Sign is never ignored. A net loss of $18.8 billion reported as a profit of
    $18.8 billion is the single most consequential error this check can catch,
    and an absolute-value comparison would pass it.
    """
    absolute_error = abs(candidate - gold)

    if gold == 0.0:
        passed = absolute_error == 0.0
        return ToleranceCheck(
            passed=passed,
            gold=gold,
            candidate=candidate,
            relative_error=None,
            absolute_error=absolute_error,
            rule="exact_zero" if passed else "failed",
        )

    relative_error = absolute_error / abs(gold)
    if relative_error <= relative_tolerance:
        return ToleranceCheck(
            passed=True,
            gold=gold,
            candidate=candidate,
            relative_error=relative_error,
            absolute_error=absolute_error,
            rule="relative",
        )

    if is_percentage and absolute_error <= percentage_point_tolerance:
        return ToleranceCheck(
            passed=True,
            gold=gold,
            candidate=candidate,
            relative_error=relative_error,
            absolute_error=absolute_error,
            rule="percentage_point",
        )

    return ToleranceCheck(
        passed=False,
        gold=gold,
        candidate=candidate,
        relative_error=relative_error,
        absolute_error=absolute_error,
        rule="failed",
    )


@dataclass(frozen=True, slots=True)
class PassageMatch:
    """Where and how a figure was found in a passage."""

    found: bool
    matched_scale: float | None
    matched_raw: str | None
    matched_value: float | None


def significant_digits(digits: str) -> int:
    """Significant digits in a number as written, ignoring separators.

    Leading zeros do not count; trailing zeros after a decimal point do, since
    a filing writing "14,089.80" is asserting that precision.
    """
    cleaned = digits.replace(",", "").lstrip("0")
    if "." in cleaned:
        cleaned = cleaned.replace(".", "").lstrip("0")
    return len(cleaned)


# Minimum significant digits a passage figure needs before a scale-blind match
# counts. Trying every scale makes short numbers promiscuous: at scale 1e9 the
# bare token "83" matches a gold of $82.798 billion to within 0.5%, and "27"
# matches $26.9 billion. Neither is a disclosure of the figure; both are
# coincidences produced by the scale search itself. Three digits is the point
# at which a match carries more information than the scale guess does.
MIN_SIGNIFICANT_DIGITS = 3

DEFAULT_RE = re.compile(r"(?P<digits>[\d,]+(?:\.\d+)?)")


def number_appears_in(
    gold: float,
    passage: str,
    *,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    candidate_scales: tuple[float, ...] = CANDIDATE_SCALES,
    is_percentage: bool = False,
    percentage_point_tolerance: float = PERCENTAGE_POINT_TOLERANCE,
    min_significant_digits: int = MIN_SIGNIFICANT_DIGITS,
) -> PassageMatch:
    """Does `gold` appear in `passage`, allowing for a scale the chunk omits?

    Each figure in the passage is tried at each candidate scale. A passage
    figure that already carries an explicit scale word is tried only as
    written -- "$53.1 billion" means what it says and does not also get
    multiplied by a million.

    Sign is compared on the magnitude only. Filings write a loss as `(18,756)`
    in a table and as "a net loss of $18.8 billion" in prose, and the chunk
    boundary frequently splits the word "loss" from the figure, so requiring
    the sign to be recoverable from the passage alone would reject correct
    citations. The sign is checked where it belongs, against the answer, in
    `within_tolerance`.

    A percentage gold is matched only against something the passage also wrote
    as a percentage, and only at scale 1: no filing states a growth rate in
    millions, and admitting those scales would let any `2.1` in a table match
    a 2.1% change by coincidence. It gets the same percentage-point rounding
    allowance as `within_tolerance`, for the same reason -- a filing quoting
    "decreased 2.1%" is citing a 2.0783% change, and that is a correct
    citation.
    """
    target = abs(gold)
    for quantity in extract_quantities(passage, exclude_years=False):
        if is_percentage and not quantity.is_percent:
            continue
        if is_percentage:
            # Percentages are exempt from the digit floor. They are quoted to
            # one or two digits by convention, and the explicit `%` marker
            # already rules out the coincidence the floor exists to stop.
            scales: tuple[float, ...] = (1.0,)
        else:
            scales = (1.0,) if quantity.scale != 1.0 else candidate_scales
            digit_match = DEFAULT_RE.search(quantity.raw)
            if (
                digit_match is not None
                and significant_digits(digit_match.group("digits"))
                < min_significant_digits
            ):
                continue
        for scale in scales:
            scaled = abs(quantity.value) * scale
            if scaled == 0.0 and target == 0.0:
                return PassageMatch(True, scale, quantity.raw, scaled)
            if target == 0.0:
                continue
            if abs(scaled - target) / target <= relative_tolerance:
                return PassageMatch(True, scale, quantity.raw, scaled)
            if is_percentage and abs(scaled - target) <= percentage_point_tolerance:
                return PassageMatch(True, scale, quantity.raw, scaled)
    return PassageMatch(False, None, None, None)


# Significant digits at which a bare integer counts as a reported quantity
# even with no currency symbol, scale word or percent sign. Four admits a
# table figure like "53,101" and excludes a day of the month, a count of
# segments, or a note number.
BARE_FIGURE_MIN_DIGITS = 4


def reportable_quantities(text: str) -> list[Quantity]:
    """Figures in `text` that are plausibly reported quantities.

    `extract_quantities` is deliberately greedy, which is right for finding a
    known value in a passage but wrong for deciding what an answer is
    asserting. "Revenue was $53.1 billion for the year ended December 28,
    2024" contains one reported quantity; the greedy reading also finds 28.
    Requiring a claim's citation to carry "28" would fail correct citations,
    and counting "28" as an unsupported figure would inflate the hallucinated
    figure count with calendar fragments.

    A figure qualifies by carrying something a date or a small ordinal does
    not: a currency symbol, a scale word, a percent sign, or at least
    `BARE_FIGURE_MIN_DIGITS` significant digits.
    """
    kept: list[Quantity] = []
    for quantity in extract_quantities(text, exclude_years=True):
        if quantity.unit in (UNIT_USD, UNIT_PERCENT) or quantity.scale != 1.0:
            kept.append(quantity)
            continue
        digit_match = DEFAULT_RE.search(quantity.raw)
        if (
            digit_match is not None
            and significant_digits(digit_match.group("digits")) >= BARE_FIGURE_MIN_DIGITS
        ):
            kept.append(quantity)
    return kept


def format_usd(value: float) -> str:
    """Render a dollar figure the way a filing would, for gold answer text."""
    magnitude = abs(value)
    sign = "-" if value < 0 else ""
    if magnitude >= 1e9:
        return f"{sign}${magnitude / 1e9:,.3f} billion".replace(".000 ", " ")
    if magnitude >= 1e6:
        return f"{sign}${magnitude / 1e6:,.1f} million"
    return f"{sign}${magnitude:,.0f}"


def percent_change(current: float, prior: float) -> float:
    """Year-over-year change in percent.

    Undefined for a zero base, and raising is better than returning infinity
    into a gold answer that would then be unmatchable.
    """
    if prior == 0.0:
        raise ValueError("percent change is undefined against a zero base")
    return (current - prior) / abs(prior) * 100.0
