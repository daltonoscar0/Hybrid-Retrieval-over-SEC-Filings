"""Read-side access to the indexed corpus: chunks, their neighbours, filings.

Every citation in this feature is a `chunks.chunk_id`, so the ability to turn
an id back into displayable text with its filing provenance is load-bearing
rather than a convenience. That lookup lives here so the agent's tools, the
baselines, the scorer and the CLI all resolve a citation the same way.

Chunk ids are structural, not opaque: `{accession}#{item}#chunk{n}`, where
`{accession}#{item}` is the `section_id` and `n` is the index of the chunk's
first sentence within its section (see `ticker.chunker`, 4-sentence windows at
stride 2). `read_neighbors` exploits that, ordering a section's chunks by `n`
and walking outward, so "the passage either side of this one" needs no extra
table and no recomputation.

Connections are opened read-only. A read-write DuckDB handle takes an exclusive
file lock, which would mean an `ask` in one terminal locks an evaluation run out
of the corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb

from ticker.db import connect_readonly

DEFAULT_DB = Path("data/ticker.duckdb")

_CHUNK_SUFFIX = re.compile(r"#chunk(\d+)$")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A passage plus the provenance a reader needs to check it."""

    chunk_id: str
    section_id: str
    accession: str
    item: str
    ticker: str
    sector: str
    form: str
    filed_at: datetime
    period_end: date | None
    url: str
    text: str
    sentence_ids: tuple[str, ...]

    @property
    def fiscal_year(self) -> int | None:
        """The year of the period the filing covers, not the year it was filed.

        A 10-K for a fiscal year ending December 2024 is filed in early 2025,
        and `filed_at` would answer a question about "fiscal 2024" wrongly.
        """
        return self.period_end.year if self.period_end else None

    def citation(self) -> str:
        period = self.period_end.isoformat() if self.period_end else "n/a"
        return (
            f"{self.ticker} {self.form} item {self.item} "
            f"(period ending {period}, filed {self.filed_at.date()}, "
            f"accession {self.accession})"
        )


def chunk_ordinal(chunk_id: str) -> int:
    """The sentence offset encoded in a chunk id.

    Raises on a malformed id rather than returning a sentinel: every caller
    uses the result for ordering, and a silent 0 would sort an unparseable id
    to the front of its section.
    """
    match = _CHUNK_SUFFIX.search(chunk_id)
    if match is None:
        raise ValueError(f"not a chunk id: {chunk_id!r}")
    return int(match.group(1))


def section_id_of(chunk_id: str) -> str:
    match = _CHUNK_SUFFIX.search(chunk_id)
    if match is None:
        raise ValueError(f"not a chunk id: {chunk_id!r}")
    return chunk_id[: match.start()]


def connect(db_path: Path | str = DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    return connect_readonly(db_path)


_SELECT = """
SELECT c.chunk_id, c.section_id, f.accession, sec.item, f.ticker, f.sector,
       f.form, f.filed_at, f.period_end, f.url, c.text, c.sentence_ids
FROM chunks c
JOIN sections sec ON c.section_id = sec.section_id
JOIN filings f ON sec.accession = f.accession
"""


def _to_record(row: tuple) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=row[0],
        section_id=row[1],
        accession=row[2],
        item=row[3],
        ticker=row[4],
        sector=row[5],
        form=row[6],
        filed_at=row[7],
        period_end=row[8],
        url=row[9],
        text=row[10],
        sentence_ids=tuple(row[11]),
    )


def load_chunks(
    con: duckdb.DuckDBPyConnection, chunk_ids: list[str]
) -> dict[str, ChunkRecord]:
    """Fetch many chunks at once, keyed by id.

    Returns only the ids that exist. A caller holding an id that is not in the
    corpus is the case citation validation exists to catch, so a missing key
    here is information rather than an error to raise.
    """
    if not chunk_ids:
        return {}
    unique = list(dict.fromkeys(chunk_ids))
    placeholders = ",".join("?" for _ in unique)
    rows = con.execute(
        f"{_SELECT} WHERE c.chunk_id IN ({placeholders})", unique
    ).fetchall()
    return {row[0]: _to_record(row) for row in rows}


def read_chunk(con: duckdb.DuckDBPyConnection, chunk_id: str) -> ChunkRecord | None:
    return load_chunks(con, [chunk_id]).get(chunk_id)


def read_neighbors(
    con: duckdb.DuckDBPyConnection, chunk_id: str, n: int = 1
) -> list[ChunkRecord]:
    """The `n` chunks either side of `chunk_id`, in document order, inclusive
    of the chunk itself.

    Scoped to the chunk's own section. A sentence that runs past a section
    boundary is a different item of the filing -- often a different topic
    entirely -- and silently returning it as adjacent context would invite a
    claim cited to item 1A that is really supported by item 7.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    section_id = section_id_of(chunk_id)
    rows = con.execute(
        f"{_SELECT} WHERE c.section_id = ?", [section_id]
    ).fetchall()
    if not rows:
        return []
    records = sorted(
        (_to_record(row) for row in rows), key=lambda record: chunk_ordinal(record.chunk_id)
    )
    index = next(
        (i for i, record in enumerate(records) if record.chunk_id == chunk_id), None
    )
    if index is None:
        return []
    low = max(0, index - n)
    return records[low : index + n + 1]


@dataclass(frozen=True, slots=True)
class FilingRecord:
    accession: str
    cik: int
    ticker: str
    sector: str
    form: str
    filed_at: datetime
    period_end: date | None
    url: str
    items: tuple[str, ...]


def load_filings(
    con: duckdb.DuckDBPyConnection,
    *,
    form: str | None = None,
    ticker: str | None = None,
) -> list[FilingRecord]:
    """Filings with the set of items actually extracted for each.

    `items` matters for question building: nine of the 120 indexed 10-Ks have
    no item 7, because those registrants incorporate MD&A by reference rather
    than printing it, and a numeric question pointed at one of them would be
    unanswerable for a reason that has nothing to do with retrieval.
    """
    clauses, params = [], []
    if form is not None:
        clauses.append("f.form = ?")
        params.append(form)
    if ticker is not None:
        clauses.append("f.ticker = ?")
        params.append(ticker.upper())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = con.execute(
        f"""
        SELECT f.accession, f.cik, f.ticker, f.sector, f.form, f.filed_at,
               f.period_end, f.url, list(sec.item ORDER BY sec.item)
        FROM filings f
        LEFT JOIN sections sec ON sec.accession = f.accession
        {where}
        GROUP BY ALL
        ORDER BY f.ticker, f.period_end
        """,
        params,
    ).fetchall()
    return [
        FilingRecord(
            accession=row[0],
            cik=row[1],
            ticker=row[2],
            sector=row[3],
            form=row[4],
            filed_at=row[5],
            period_end=row[6],
            url=row[7],
            items=tuple(item for item in (row[8] or []) if item is not None),
        )
        for row in rows
    ]


def section_text(
    con: duckdb.DuckDBPyConnection, accession: str, item: str
) -> str | None:
    row = con.execute(
        "SELECT text FROM sections WHERE accession = ? AND item = ?",
        [accession, item],
    ).fetchone()
    return row[0] if row else None


def chunks_in_section(
    con: duckdb.DuckDBPyConnection, accession: str, item: str
) -> list[ChunkRecord]:
    rows = con.execute(
        f"{_SELECT} WHERE sec.accession = ? AND sec.item = ?", [accession, item]
    ).fetchall()
    return sorted(
        (_to_record(row) for row in rows),
        key=lambda record: chunk_ordinal(record.chunk_id),
    )
