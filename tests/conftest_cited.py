"""Shared fixtures for the cited-answer tests: a tiny in-memory corpus.

The real corpus is a 900 MB DuckDB file that is not in version control, so
nothing here may depend on it. These helpers build a handful of filings,
sections and chunks in memory through the same `ticker.db` insert functions
the ingest pipeline uses, which means the tests exercise the real schema and
the real joins rather than a mock that agrees with whatever the code does.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import duckdb

from ticker.chunker import chunk_sentences
from ticker.db import (
    create_schema,
    insert_chunk,
    insert_filing,
    insert_section,
    insert_sentence,
)
from ticker.records import Filing, Section, Sentence

EASTERN = timezone.utc


def make_filing(
    con: duckdb.DuckDBPyConnection,
    *,
    accession: str,
    ticker: str,
    item: str,
    sentences: list[str],
    form: str = "10-K",
    period_end: date = date(2024, 12, 31),
    filed_at: datetime | None = None,
    cik: int = 1,
    sector: str = "semiconductors",
) -> list[str]:
    """Insert one filing with one section, and return its chunk ids."""
    filed_at = filed_at or datetime(2025, 2, 1, tzinfo=EASTERN)
    existing = con.execute(
        "SELECT count(*) FROM filings WHERE accession = ?", [accession]
    ).fetchone()
    if not existing[0]:
        insert_filing(
            con,
            Filing(
                accession=accession,
                cik=cik,
                ticker=ticker,
                sector=sector,
                form=form,
                filed_at=filed_at,
                period_end=period_end,
                url=f"https://example.invalid/{accession}",
            ),
        )

    section_id = f"{accession}#{item}"
    insert_section(
        con,
        Section(
            section_id=section_id,
            accession=accession,
            item=item,
            text=" ".join(sentences),
            char_start=0,
            char_end=len(" ".join(sentences)),
        ),
    )

    records = []
    for ordinal, text in enumerate(sentences):
        sentence = Sentence(
            sentence_id=f"{section_id}#{ordinal}",
            section_id=section_id,
            ordinal=ordinal,
            text=text,
            filed_at=filed_at,
        )
        insert_sentence(con, sentence)
        records.append(sentence)

    chunk_ids = []
    for chunk in chunk_sentences(records):
        insert_chunk(con, chunk)
        chunk_ids.append(chunk.chunk_id)
    return chunk_ids


def build_corpus(con: duckdb.DuckDBPyConnection | None = None):
    """A small corpus with the properties the tests need.

    One filing carries a figure in a table-like line with no scale word, which
    is the case the scale-tolerant passage matcher exists for. Another carries
    the same company in a different year, so year filtering has something to
    get wrong.
    """
    con = con or duckdb.connect(":memory:")
    create_schema(con)

    nvda_2024 = make_filing(
        con,
        accession="0000000001-25-000001",
        ticker="NVDA",
        item="1A",
        cik=1045810,
        period_end=date(2025, 1, 26),
        # Ten sentences, so 4-sentence windows at stride 2 produce four
        # chunks. Tests that walk a neighbour window need more than two.
        sentences=[
            "We generated 53% of our revenue in fiscal year 2025 from sales outside the United States.",
            "The market in China is limited by export controls and is highly competitive.",
            "Our wafers are manufactured by third parties located outside the United States.",
            "We expect competition to remain intense.",
            "Net revenue 130,497 and cost of revenue 32,639 for the year.",
            "Operating expenses rose in line with headcount.",
            "We depend on a limited number of partners and distributors.",
            "Changes in trade policy could affect our results.",
            "We may not realise the benefits of our investments.",
            "Our gross margin may fluctuate from period to period.",
        ],
    )

    nvda_2023 = make_filing(
        con,
        accession="0000000001-23-000001",
        ticker="NVDA",
        item="1A",
        cik=1045810,
        period_end=date(2023, 1, 29),
        sentences=[
            "We generated 69% of our revenue during fiscal year 2023 from sales outside the United States.",
            "International operations expose us to a number of risks.",
            "We sell our products internationally.",
            "Currency fluctuations may affect results.",
        ],
    )

    zion_2024 = make_filing(
        con,
        accession="0000000002-25-000001",
        ticker="ZION",
        item="7",
        cik=109380,
        sector="regional_banks",
        period_end=date(2024, 12, 31),
        sentences=[
            "Preferred stock decreased $374 million due to the redemption of Series G, I and J.",
            "The redemption reduced net earnings applicable to common shareholders by $6 million.",
            "Total deposits were 75,241 at year end.",
            "AOCI improved $312 million compared with December 31, 2023.",
            "We use interest rate swaps designated as hedges of our AFS securities.",
            "Net interest income benefited from higher yields on earning assets.",
            "Credit quality remained stable through the period.",
            "We repurchased 0.9 million common shares for $35 million.",
        ],
    )
    return con, {
        "nvda_2024": nvda_2024,
        "nvda_2023": nvda_2023,
        "zion_2024": zion_2024,
    }


class StubRetriever:
    """Returns a fixed ranking, so filtering and fusion can be tested alone.

    Deliberately not a real index: these tests are about the search tool's
    filtering, ordering and text-joining behaviour, and a real BM25 index
    would make them depend on a 95 MB artefact and on tokenisation choices
    that are already tested in `tests/test_bm25.py`.
    """

    def __init__(self, name: str, ranking: list[str]) -> None:
        self.name = name
        self._ranking = ranking

    def search(self, query: str, k: int):
        from ticker.retrieval_types import RankedChunk

        return [
            RankedChunk(chunk_id=chunk_id, score=float(len(self._ranking) - index))
            for index, chunk_id in enumerate(self._ranking[:k])
        ]
