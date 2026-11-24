"""Finding passages by literal text, independently of the rankers.

This exists for one reason: the gold chunk ids on a qualitative question must
not be found using the retrieval system those questions are later used to
measure. If BM25 picks the gold passage and recall@k is then scored against it,
recall@k is guaranteed high and measures nothing -- the gold set would be
defined as "what the ranker returns".

So gold passages are located with a substring or regex scan over `chunks.text`
in SQL. That is a brute-force sequential scan over 148,097 rows with no
scoring, no term weighting, no vocabulary and no vectors. It shares no code
path and no ranking signal with either arm. Recall@k against a gold set built
this way is a real question: can the ranker surface the passage a dumb scan
already proved is in the corpus?

The scan is slow in the way `grep` is slow and that is fine; it runs while
questions are being written, never inside an evaluation run.
"""

from __future__ import annotations

import duckdb

from ticker.cited.corpus import _SELECT, ChunkRecord, _to_record

DEFAULT_LIMIT = 40


def grep_chunks(
    con: duckdb.DuckDBPyConnection,
    pattern: str,
    *,
    company: str | None = None,
    form_type: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    regex: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> list[ChunkRecord]:
    """Chunks whose text matches `pattern`.

    `regex=False` does a case-insensitive substring match, which is what most
    gold lookups want: a distinctive phrase copied out of a filing. `regex=True`
    switches to `regexp_matches` for the cases where it does not, such as a
    figure that may be written with or without a thousands separator.

    Ordered by chunk id so repeated runs return the same list in the same
    order. A match set is reviewed by eye, and a scan whose order shifts
    between runs makes the review unrepeatable.
    """
    predicate = (
        "regexp_matches(c.text, ?)" if regex else "lower(c.text) LIKE lower(?)"
    )
    needle = pattern if regex else f"%{pattern}%"
    clauses, params = [predicate], [needle]

    if company is not None:
        clauses.append("f.ticker = ?")
        params.append(company.upper())
    if form_type is not None:
        clauses.append("f.form = ?")
        params.append(form_type)
    if fiscal_year is not None:
        clauses.append("EXTRACT(year FROM f.period_end) = ?")
        params.append(fiscal_year)
    if item is not None:
        clauses.append("sec.item = ?")
        params.append(item)

    rows = con.execute(
        f"{_SELECT} WHERE {' AND '.join(clauses)} ORDER BY c.chunk_id LIMIT ?",
        [*params, limit],
    ).fetchall()
    return [_to_record(row) for row in rows]


def count_matching_chunks(
    con: duckdb.DuckDBPyConnection,
    pattern: str,
    *,
    company: str | None = None,
    form_type: str | None = None,
    regex: bool = False,
) -> int:
    """How many chunks match, without materialising them.

    Used by the answerability check: a gold figure that matches zero chunks is
    not present in the indexed text, whatever XBRL says the company reported.
    """
    predicate = (
        "regexp_matches(c.text, ?)" if regex else "lower(c.text) LIKE lower(?)"
    )
    needle = pattern if regex else f"%{pattern}%"
    clauses, params = [predicate], [needle]
    if company is not None:
        clauses.append("f.ticker = ?")
        params.append(company.upper())
    if form_type is not None:
        clauses.append("f.form = ?")
        params.append(form_type)
    row = con.execute(
        f"""
        SELECT count(*)
        FROM chunks c
        JOIN sections sec ON c.section_id = sec.section_id
        JOIN filings f ON sec.accession = f.accession
        WHERE {' AND '.join(clauses)}
        """,
        params,
    ).fetchone()
    return int(row[0]) if row else 0
