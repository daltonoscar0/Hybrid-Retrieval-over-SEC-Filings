"""Building the question set, before any answering system exists.

Two halves, with different provenance and different trust.

The numeric half is generated. `NUMERIC_PLAN` names a company, a concept and
whether the question asks for a level or a year-over-year change; everything
else -- the value, its unit, the period, the accession -- is read out of
EDGAR's company-facts API. No figure here is drafted, and `build_numeric`
raises rather than skipping when a planned concept turns out not to be tagged,
so a question can never ship with a gold value that was quietly substituted
for the one the plan asked for.

Year-over-year golds take both operands from the *latest* filing rather than
one from each year's own filing. The comparative a company prints in this
year's MD&A is the figure it computed its own growth rate against, and it is
the figure sitting in the indexed text; taking the prior year from the prior
filing would occasionally pick up a number that was later restated, making the
gold disagree with every passage a system could cite for it.

The text half is drafted, and `TEXT_SPECS` records the substring pattern each
answer was found with instead of the chunk ids it was found at. `resolve_gold`
re-runs that scan to produce `gold_chunk_ids`, so the gold set is derived from
the filing text by a reproducible rule rather than asserted. The patterns are
phrases copied out of the filings; the scan is a sequential `LIKE` over
`chunks.text` and shares nothing with the rankers these questions are used to
measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb

from ticker.cited.corpus import FilingRecord, load_filings
from ticker.cited.locate import count_matching_chunks, grep_chunks
from ticker.cited.questions import (
    KIND_MULTI_FILING,
    KIND_NUMERIC,
    KIND_QUALITATIVE,
    KIND_UNANSWERABLE,
    SOURCE_FILING_TEXT,
    SOURCE_NOT_IN_CORPUS,
    SOURCE_XBRL,
    Question,
)
from ticker.cited.units import UNIT_PERCENT, format_usd, percent_change
from ticker.cited.xbrl import Fact, annual_value, fetch_company_facts, instant_value, iter_facts

MODE_LEVEL = "level"
MODE_YOY = "yoy"
MODE_INSTANT = "instant"

# Human phrasing for each concept, so a question reads like an analyst's
# question rather than a tag name. The second element is the sentence-leading
# form used in a gold answer, written out rather than derived with
# `str.capitalize`, which lowercases the rest of the string and turns
# "diluted EPS" into "Diluted eps".
CONCEPT_PHRASING: dict[str, tuple[str, str]] = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": ("revenue", "Revenue"),
    "Revenues": ("revenue", "Revenue"),
    "NetIncomeLoss": ("net income", "Net income"),
    "ResearchAndDevelopmentExpense": (
        "research and development expense",
        "Research and development expense",
    ),
    "GrossProfit": ("gross profit", "Gross profit"),
    "OperatingIncomeLoss": ("operating income", "Operating income"),
    "EarningsPerShareDiluted": ("diluted earnings per share", "Diluted EPS"),
    "InterestIncomeExpenseNet": ("net interest income", "Net interest income"),
    "NoninterestIncome": ("noninterest income", "Noninterest income"),
    "ProvisionForLoanLeaseAndOtherLosses": (
        "provision for credit losses",
        "Provision for credit losses",
    ),
    "InterestAndDividendIncomeOperating": (
        "total interest income",
        "Total interest income",
    ),
    "Assets": ("total assets", "Total assets"),
    "Deposits": ("total deposits", "Total deposits"),
    "StockholdersEquity": ("total shareholders' equity", "Shareholders' equity"),
}


@dataclass(frozen=True, slots=True)
class NumericSpec:
    ticker: str
    concept: str
    mode: str


# One question per company: twelve levels and eight year-over-year changes.
# Concepts vary by sector because the statements do -- a bank has no gross
# profit and a semiconductor company has no provision for credit losses -- and
# every pairing here was checked against the latest 10-K for that company
# before being written down.
NUMERIC_PLAN: tuple[NumericSpec, ...] = (
    NumericSpec("INTC", "RevenueFromContractWithCustomerExcludingAssessedTax", MODE_YOY),
    NumericSpec("NVDA", "Revenues", MODE_YOY),
    NumericSpec("TXN", "GrossProfit", MODE_LEVEL),
    NumericSpec("QCOM", "ResearchAndDevelopmentExpense", MODE_LEVEL),
    NumericSpec("AVGO", "RevenueFromContractWithCustomerExcludingAssessedTax", MODE_YOY),
    NumericSpec("MU", "NetIncomeLoss", MODE_YOY),
    NumericSpec("ADI", "OperatingIncomeLoss", MODE_LEVEL),
    NumericSpec("NXPI", "EarningsPerShareDiluted", MODE_LEVEL),
    NumericSpec("MCHP", "RevenueFromContractWithCustomerExcludingAssessedTax", MODE_YOY),
    NumericSpec("ON", "GrossProfit", MODE_LEVEL),
    NumericSpec("ZION", "InterestIncomeExpenseNet", MODE_LEVEL),
    NumericSpec("RF", "NoninterestIncome", MODE_LEVEL),
    NumericSpec("HBAN", "Deposits", MODE_INSTANT),
    NumericSpec("KEY", "ProvisionForLoanLeaseAndOtherLosses", MODE_LEVEL),
    NumericSpec("FITB", "EarningsPerShareDiluted", MODE_LEVEL),
    NumericSpec("CFG", "InterestIncomeExpenseNet", MODE_YOY),
    NumericSpec("WAL", "NetIncomeLoss", MODE_YOY),
    NumericSpec("EWBC", "NoninterestIncome", MODE_YOY),
    NumericSpec("CFR", "InterestAndDividendIncomeOperating", MODE_LEVEL),
    NumericSpec("CMA", "Assets", MODE_INSTANT),
)


def latest_two_tenks(con: duckdb.DuckDBPyConnection) -> dict[str, tuple[FilingRecord, FilingRecord]]:
    """The two most recent 10-Ks per company, newest first.

    Both of every company's two latest 10-Ks carry all five indexed items. The
    nine filings with no item 7 are all earlier years, where those registrants
    incorporated MD&A by reference, so anchoring every numeric question to the
    latest pair sidesteps that gap rather than working around it per question.
    """
    by_ticker: dict[str, list[FilingRecord]] = {}
    for filing in load_filings(con, form="10-K"):
        by_ticker.setdefault(filing.ticker, []).append(filing)
    result = {}
    for ticker, filings in by_ticker.items():
        ordered = sorted(filings, key=lambda f: f.period_end or date.min, reverse=True)
        if len(ordered) < 2:
            raise ValueError(f"{ticker} has fewer than two 10-Ks indexed")
        result[ticker] = (ordered[0], ordered[1])
    return result


def _fiscal_label(period_end: date) -> str:
    """How a filing refers to its own year.

    A fiscal year ending in January belongs to the prior calendar year by the
    company's own labelling -- NVIDIA's year ending 2025-01-26 is fiscal 2025,
    but Micron's ending 2025-08-28 is also fiscal 2025. The XBRL period end is
    the unambiguous anchor, so questions name the period end explicitly and
    this label is only used for readable prose.
    """
    return f"the fiscal year ended {period_end:%B %-d, %Y}"


def _value_for(
    facts: list[Fact], accession: str, concept: str, period_end: date, mode: str
) -> Fact | None:
    if mode == MODE_INSTANT:
        return instant_value(facts, accession, concept, period_end)
    return annual_value(facts, accession, concept, period_end)


def build_numeric(
    con: duckdb.DuckDBPyConnection,
    *,
    plan: tuple[NumericSpec, ...] = NUMERIC_PLAN,
    cik_by_ticker: dict[str, int] | None = None,
) -> list[Question]:
    """Generate the numeric questions from company facts."""
    from ticker.universe import UNIVERSE

    cik_by_ticker = cik_by_ticker or {c.ticker: c.cik for c in UNIVERSE}
    tenks = latest_two_tenks(con)
    questions: list[Question] = []

    for index, spec in enumerate(plan, start=1):
        latest, prior = tenks[spec.ticker]
        if latest.period_end is None or prior.period_end is None:
            raise ValueError(f"{spec.ticker}: 10-K with no period end")

        facts = list(iter_facts(fetch_company_facts(cik_by_ticker[spec.ticker])))
        current = _value_for(facts, latest.accession, spec.concept, latest.period_end, spec.mode)
        if current is None:
            raise ValueError(
                f"{spec.ticker} {latest.accession} does not tag {spec.concept}; "
                "fix NUMERIC_PLAN rather than letting the build substitute a concept"
            )

        phrase, short = CONCEPT_PHRASING[spec.concept]
        qid = f"n{index:02d}"
        unit = current.unit

        if spec.mode == MODE_YOY:
            previous = _value_for(
                facts, latest.accession, spec.concept, prior.period_end, MODE_LEVEL
            )
            if previous is None:
                raise ValueError(
                    f"{spec.ticker} {latest.accession} does not tag a prior-year "
                    f"comparative for {spec.concept}"
                )
            change = percent_change(current.value, previous.value)
            text = (
                f"By what percent did {spec.ticker}'s {phrase} change in "
                f"{_fiscal_label(latest.period_end)} compared with the prior year?"
            )
            gold_answer = (
                f"{short} changed by {change:+.2f}%, from "
                f"{format_usd(previous.value)} for the year ended {prior.period_end} to "
                f"{format_usd(current.value)} for the year ended {latest.period_end}, "
                f"both as reported in accession {latest.accession}."
            )
            questions.append(
                Question(
                    question_id=qid,
                    kind=KIND_NUMERIC,
                    text=text,
                    gold_answer=gold_answer,
                    gold_source=SOURCE_XBRL,
                    answerable=True,
                    rationale=(
                        f"Computed from two values of us-gaap:{spec.concept} reported in "
                        f"{latest.accession}: {current.value:.0f} for the period ending "
                        f"{current.end} and {previous.value:.0f} for the period ending "
                        f"{previous.end}. Both operands come from the same filing, which "
                        "is the comparative the company itself computes against."
                    ),
                    companies=(spec.ticker,),
                    form_type="10-K",
                    fiscal_year=latest.period_end.year,
                    gold_accessions=(latest.accession,),
                    gold_value=change,
                    gold_unit=UNIT_PERCENT,
                    gold_is_percentage=True,
                    gold_concepts=(f"us-gaap:{spec.concept}",),
                    gold_period_end=latest.period_end.isoformat(),
                    gold_components=(current.value, previous.value),
                    tags=("yoy", spec.concept),
                )
            )
            continue

        # A balance is as at a date; a flow is for a period. Asking for total
        # deposits "as at the fiscal year ended December 31" reads as though
        # the balance spans the year, which is the confusion a unit-and-period
        # question set should not be introducing itself.
        if spec.mode == MODE_INSTANT:
            when = f"as at {latest.period_end:%B %-d, %Y}"
        else:
            when = f"for {_fiscal_label(latest.period_end)}"
        text = f"What was {spec.ticker}'s {phrase} {when}?"

        rendered = (
            format_usd(current.value)
            if unit == "USD"
            else f"{current.value:,.2f} {unit}"
        )
        gold_answer = (
            f"{short} was {rendered}, as reported in accession "
            f"{latest.accession} for the period ending {current.end}."
        )
        questions.append(
            Question(
                question_id=qid,
                kind=KIND_NUMERIC,
                text=text,
                gold_answer=gold_answer,
                gold_source=SOURCE_XBRL,
                answerable=True,
                rationale=(
                    f"us-gaap:{spec.concept} = {current.value:.2f} {unit}, reported in "
                    f"{latest.accession} for the period ending {current.end}."
                ),
                companies=(spec.ticker,),
                form_type="10-K",
                fiscal_year=current.end.year,
                gold_accessions=(latest.accession,),
                gold_value=current.value,
                gold_unit=unit,
                gold_is_percentage=False,
                gold_concepts=(f"us-gaap:{spec.concept}",),
                gold_period_end=current.end.isoformat(),
                tags=(spec.mode, spec.concept),
            )
        )
    return questions


class SpecResolutionError(Exception):
    """A drafted spec did not resolve against the corpus.

    Always fatal to a build. A locator that stops matching means the question
    no longer has the gold it was written against, and shipping it with an
    empty or substituted gold set would put an unverifiable row in the results
    table.
    """


def _filing_for(
    con: duckdb.DuckDBPyConnection, company: str, form_type: str, fiscal_year: int
) -> FilingRecord:
    candidates = [
        filing
        for filing in load_filings(con, form=form_type, ticker=company)
        if filing.period_end and filing.period_end.year == fiscal_year
    ]
    if len(candidates) != 1:
        raise SpecResolutionError(
            f"{company} {form_type} with period ending in {fiscal_year}: "
            f"expected exactly one filing, found {len(candidates)}"
        )
    return candidates[0]


def resolve_source(con: duckdb.DuckDBPyConnection, source) -> tuple[str, tuple[str, ...]]:
    """Re-run a spec's locator scan, returning its accession and gold chunks.

    The chunk count is bounded, not pinned. Stride-2 windows put a phrase near
    a boundary into two or three adjacent chunks, and some filings repeat a
    risk sentence verbatim, so a gold set of every chunk containing the claim
    is correct. Zero matches means the locator no longer resolves; more than
    `max_chunks` means it has broadened into something generic and the gold set
    it would produce is no longer the passage the answer was drafted from.
    """
    filing = _filing_for(con, source.company, source.form_type, source.fiscal_year)
    matches = grep_chunks(
        con,
        source.locator,
        company=source.company,
        form_type=source.form_type,
        fiscal_year=source.fiscal_year,
        item=source.item,
        limit=200,
    )
    chunk_ids = tuple(match.chunk_id for match in matches)
    if not 1 <= len(chunk_ids) <= source.max_chunks:
        raise SpecResolutionError(
            f"{source.company} FY{source.fiscal_year} item {source.item}: locator "
            f"{source.locator!r} matched {len(chunk_ids)} chunks, spec allows 1 to "
            f"{source.max_chunks}. Matches: {list(chunk_ids)[:6]}"
        )
    return filing.accession, chunk_ids


def verify_absence(con: duckdb.DuckDBPyConnection, spec) -> None:
    """Confirm an unanswerable question really is unanswerable.

    Every probe phrase must match zero chunks across the company's indexed
    filings. A probe that matches means the corpus does carry the answer and
    the question is mislabelled -- the one failure mode that would corrupt the
    declining-is-correct metric, since a system that answered such a question
    would be marked wrong for being right.
    """
    for probe in spec.absence_probes:
        count = count_matching_chunks(con, probe, company=spec.absence_company)
        if count:
            raise SpecResolutionError(
                f"{spec.question_id}: probe {probe!r} matches {count} chunks for "
                f"{spec.absence_company}, so the question is not unanswerable"
            )


def build_text(con: duckdb.DuckDBPyConnection, specs) -> list[Question]:
    """Resolve drafted specs into questions with derived gold chunk ids."""
    questions: list[Question] = []
    for spec in specs:
        if spec.kind == KIND_UNANSWERABLE:
            verify_absence(con, spec)
            questions.append(
                Question(
                    question_id=spec.question_id,
                    kind=KIND_UNANSWERABLE,
                    text=spec.text,
                    gold_answer=spec.gold_answer,
                    gold_source=SOURCE_NOT_IN_CORPUS,
                    answerable=False,
                    rationale=spec.rationale,
                    companies=(spec.absence_company,) if spec.absence_company else (),
                    gold_locator=" | ".join(spec.absence_probes) or None,
                    tags=spec.tags,
                )
            )
            continue

        accessions: list[str] = []
        chunk_ids: list[str] = []
        companies: list[str] = []
        years: list[int] = []
        for source in spec.sources:
            accession, resolved = resolve_source(con, source)
            accessions.append(accession)
            chunk_ids.extend(resolved)
            if source.company not in companies:
                companies.append(source.company)
            years.append(source.fiscal_year)

        questions.append(
            Question(
                question_id=spec.question_id,
                kind=spec.kind,
                text=spec.text,
                gold_answer=spec.gold_answer,
                gold_source=SOURCE_FILING_TEXT,
                answerable=True,
                rationale=spec.rationale,
                companies=tuple(companies),
                form_type=spec.sources[0].form_type,
                # A multi-filing question spans years, so pinning one would
                # license a filter that excludes half its own gold set.
                fiscal_year=years[0] if len(set(years)) == 1 else None,
                gold_accessions=tuple(accessions),
                gold_chunk_ids=tuple(chunk_ids),
                gold_locator=" | ".join(source.locator for source in spec.sources),
                tags=spec.tags,
            )
        )
    return questions


def check_numeric_answerability(
    con: duckdb.DuckDBPyConnection, questions: list[Question]
) -> list[Question]:
    """Record, per numeric question, whether the gold figure is in the text.

    Measured, not assumed, and it is the honest caveat on the numeric arm: a
    question whose gold figure appears nowhere in the indexed sections cannot
    be answered from retrieval however good the ranker is, and the results
    table has to separate those from genuine retrieval failures.

    A year-over-year question is judged on its operands rather than its
    answer. The gold is a percentage the filing need never print -- a company
    that reports revenue for both years has fully disclosed the change without
    ever stating the rate -- so looking for the quotient would mark a question
    unanswerable that an agent can answer by dividing two figures it retrieved.
    Both operands must be present for such a question to count as answerable.

    Scale-tolerant, matching `number_appears_in`, and searching the whole
    filing rather than any single chunk.
    """
    from dataclasses import replace

    from ticker.cited.units import number_appears_in

    updated: list[Question] = []
    for question in questions:
        if question.kind != KIND_NUMERIC or question.gold_value is None:
            updated.append(question)
            continue
        blob = "\n".join(
            row[0]
            for row in con.execute(
                """
                SELECT c.text FROM chunks c
                JOIN sections sec ON c.section_id = sec.section_id
                JOIN filings f ON sec.accession = f.accession
                WHERE f.accession = ?
                """,
                [question.gold_accessions[0]],
            ).fetchall()
        )
        if question.gold_components:
            found = all(
                number_appears_in(component, blob).found
                for component in question.gold_components
            )
        else:
            found = number_appears_in(
                question.gold_value, blob, is_percentage=question.gold_is_percentage
            ).found
        updated.append(replace(question, figure_in_indexed_text=found))
    return updated


def build_all(con: duckdb.DuckDBPyConnection) -> list[Question]:
    """The whole question set: generated numerics plus resolved text specs."""
    from ticker.cited.specs import ALL_TEXT_SPECS

    questions = build_numeric(con) + build_text(con, ALL_TEXT_SPECS)
    return check_numeric_answerability(con, questions)
