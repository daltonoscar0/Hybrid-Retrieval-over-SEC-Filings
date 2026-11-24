"""EDGAR company-facts client: identity, rate limit, on-disk cache.

`ticker.download` goes through edgartools, which owns its own token-bucket
limiter. The company-facts endpoint lives on `data.sec.gov` and edgartools does
not wrap it, so this module talks to it directly and therefore has to supply
both halves of EDGAR's fair-access contract itself: a real contact string in
the `User-Agent` header, and a request rate well under the published ceiling.
`MIN_INTERVAL_S` fixes the latter at 5 requests a second against a limit of 10.
The identity comes from `EDGAR_IDENTITY`, the same variable
`ticker.download.ensure_identity` reads, so there is one place to set it.

CIKs are zero-padded to 10 digits via `ticker.download.zero_pad_cik`. That is
not cosmetic here: `/api/xbrl/companyfacts/CIK50863.json` is a 404 where
`CIK0000050863.json` is the record for Intel. This is the raw-endpoint case
that module's docstring anticipated.

Gold figures are selected by accession, not by fiscal year. Company facts
report the same figure many times over -- once in the filing that first
disclosed it and again as a prior-year comparative in later filings -- and the
`fy`/`fp` fields describe the report a fact was filed in rather than the period
the fact covers. Matching on `accn` against an accession that is already in
`filings` means a gold value is, by construction, the number as reported in a
filing this corpus actually indexes. A fiscal-year filter would also silently
mis-handle the six fiscal-year companies here, where FY2025 ends in January or
August rather than December.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from ticker.download import ensure_identity, zero_pad_cik

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

DEFAULT_CACHE_DIR = Path("data/xbrl")

# 5 requests a second against EDGAR's published ceiling of 10.
MIN_INTERVAL_S = 0.2

REQUEST_TIMEOUT_S = 60.0

# Taxonomies worth reading. `dei` carries the cover-page facts (shares
# outstanding, entity name); `us-gaap` carries everything else used here.
TAXONOMIES = ("us-gaap", "dei")


@dataclass(frozen=True, slots=True)
class Fact:
    """One reported value, with everything needed to cite it.

    `start` is None for instants (balance-sheet items, share counts) and set
    for durations (income- and cash-flow-statement items). `accession` is the
    filing that reported this value, which is the field gold answers join on.
    """

    cik: int
    taxonomy: str
    concept: str
    unit: str
    value: float
    start: date | None
    end: date
    form: str
    accession: str
    filed: date
    fiscal_year: int | None
    fiscal_period: str | None

    @property
    def qname(self) -> str:
        return f"{self.taxonomy}:{self.concept}"

    @property
    def is_duration(self) -> bool:
        return self.start is not None

    @property
    def duration_days(self) -> int | None:
        if self.start is None:
            return None
        return (self.end - self.start).days


class RateLimiter:
    """Minimum wall-clock gap between requests.

    A token bucket would allow a burst up to its capacity; a hard floor on the
    gap cannot burst at all. Against a shared public endpoint with a published
    limit, the un-burstable version is the one that cannot accidentally exceed
    it, and the cost is a few seconds across a twenty-company sweep.
    """

    def __init__(self, min_interval_s: float = MIN_INTERVAL_S) -> None:
        self.min_interval_s = min_interval_s
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            elapsed = time.monotonic() - self._last
            remaining = self.min_interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last = time.monotonic()


def user_agent() -> str:
    """The `User-Agent` EDGAR requires, from `EDGAR_IDENTITY`.

    Delegates the missing-variable error to `ensure_identity` so an unset
    identity fails with one message no matter which EDGAR path hit it first.
    """
    ensure_identity()
    import os

    return os.environ["EDGAR_IDENTITY"]


def _cache_path(cik: int, cache_dir: Path) -> Path:
    return cache_dir / f"CIK{zero_pad_cik(cik)}.json"


def fetch_company_facts(
    cik: int,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    limiter: RateLimiter | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """The company-facts payload for `cik`, from cache when present.

    Cache hits make no request and take no rate-limit slot, so re-running the
    question build after an interruption costs nothing and re-running it in a
    test costs no network at all.
    """
    path = _cache_path(cik, cache_dir)
    if path.exists():
        return json.loads(path.read_text())

    url = COMPANY_FACTS_URL.format(cik=zero_pad_cik(cik))
    headers = {"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"}

    (limiter or RateLimiter()).wait()
    if client is None:
        with httpx.Client(timeout=REQUEST_TIMEOUT_S) as owned:
            response = owned.get(url, headers=headers)
    else:
        response = client.get(url, headers=headers, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
    return payload


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def iter_facts(payload: dict, *, taxonomies: tuple[str, ...] = TAXONOMIES) -> Iterator[Fact]:
    """Flatten a company-facts payload into one `Fact` per reported value.

    The payload nests taxonomy -> concept -> units -> list of values, and the
    same concept can appear under more than one unit (`USD` and `USD/shares`,
    say). Unit is carried on the fact rather than normalized away, because a
    gold answer that drops its unit is not checkable.
    """
    cik = int(payload["cik"])
    facts = payload.get("facts") or {}
    for taxonomy in taxonomies:
        concepts = facts.get(taxonomy) or {}
        for concept, body in concepts.items():
            for unit, entries in (body.get("units") or {}).items():
                for entry in entries:
                    end = _parse_date(entry.get("end"))
                    accession = entry.get("accn")
                    value = entry.get("val")
                    if end is None or accession is None or value is None:
                        continue
                    yield Fact(
                        cik=cik,
                        taxonomy=taxonomy,
                        concept=concept,
                        unit=unit,
                        value=float(value),
                        start=_parse_date(entry.get("start")),
                        end=end,
                        form=entry.get("form") or "",
                        accession=accession,
                        filed=_parse_date(entry.get("filed")) or end,
                        fiscal_year=entry.get("fy"),
                        fiscal_period=entry.get("fp"),
                    )


# An annual duration is nominally 365 days but 52/53-week filers land anywhere
# in this window, and a 53-week year runs to 371. Quarters sit near 91 and are
# excluded by the lower bound.
ANNUAL_DAYS_MIN = 340
ANNUAL_DAYS_MAX = 380


def is_annual(fact: Fact) -> bool:
    """True for a full-year duration. Instants are not annual and not periodic;
    callers that want a year-end balance want `is_instant_at` instead."""
    days = fact.duration_days
    return days is not None and ANNUAL_DAYS_MIN <= days <= ANNUAL_DAYS_MAX


def facts_in_filing(
    facts: list[Fact],
    accession: str,
    *,
    concept: str | None = None,
) -> list[Fact]:
    """Facts reported in one filing, optionally for one concept."""
    return [
        fact
        for fact in facts
        if fact.accession == accession and (concept is None or fact.concept == concept)
    ]


def annual_value(
    facts: list[Fact],
    accession: str,
    concept: str,
    period_end: date,
    *,
    tolerance_days: int = 7,
) -> Fact | None:
    """The full-year value of `concept` reported in `accession` for the year
    ending at `period_end`.

    `tolerance_days` absorbs the gap between the period end recorded in
    `filings` and the one tagged in XBRL, which differ by a day or two for
    52/53-week filers. Returns None rather than guessing when the filing does
    not tag the concept, which is the common case and not an error: a question
    can only be built where a gold value exists.
    """
    candidates = [
        fact
        for fact in facts_in_filing(facts, accession, concept=concept)
        if is_annual(fact) and abs((fact.end - period_end).days) <= tolerance_days
    ]
    if not candidates:
        return None
    # Latest end, then largest absolute value: a filing occasionally tags the
    # same concept for both the consolidated total and a narrower member, and
    # the total is the one a question about the company means.
    return max(candidates, key=lambda fact: (fact.end, abs(fact.value)))


def instant_value(
    facts: list[Fact],
    accession: str,
    concept: str,
    period_end: date,
    *,
    tolerance_days: int = 7,
) -> Fact | None:
    """The point-in-time value of `concept` reported in `accession` as at
    `period_end`. Balance-sheet items and share counts take this path."""
    candidates = [
        fact
        for fact in facts_in_filing(facts, accession, concept=concept)
        if not fact.is_duration and abs((fact.end - period_end).days) <= tolerance_days
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda fact: (fact.end, abs(fact.value)))
