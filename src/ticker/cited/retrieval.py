"""The search surface the agent and the baselines call.

Thin by design. `ticker.retrieval.bm25`, `ticker.retrieval.dense` and
`ticker.fusion.rrf` are reused exactly as the evaluation harness uses them, so
a number measured here is a number about Ticker's retrieval rather than about a
second ranker written for this feature. What this module adds is the two things
a `Retriever` deliberately does not do: metadata filters, and joining results
back to their text and provenance.

Filtering is post-hoc over an over-fetched pool, matching `scripts/ask.py`. The
`Retriever` protocol has no filter argument and adding one would mean a
company-scoped index per company. Over-fetching candidates and discarding the
ones that miss the filter costs one ranking pass and leaves the ranker
untouched.

Pool depth scales with the filter's selectivity, which the fixed 20x of
`scripts/ask.py` does not. One company and one form is about 1/60th of this
corpus, so a 200-deep pool filtered to `NVDA` 10-K survives zero chunks -- the
filter is narrower than the pool is deep. Depth is therefore
`k * DEPTH_MULTIPLIER / selectivity`, where selectivity is counted in SQL
against the same predicate, capped at `MAX_POOL_DEPTH`. BM25 retrieves in
1.5 ms and the dense index is flat, so a few thousand candidates is not a cost
worth optimising around.

`SearchResult.exhausted_pool` reports a short list rather than hiding one,
because "the corpus holds nothing else that matches" and "the pool was too
shallow" are different findings and only the first is about retrieval quality.

Fusion is RRF at k=60, the published value, unchanged and untuned -- same
rationale as the harness: there is no held-out split here to fit it on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from ticker.cited.corpus import ChunkRecord, load_chunks
from ticker.fusion import DEFAULT_RRF_K, rrf
from ticker.retrieval_types import RankedChunk

DEFAULT_INDEX_ROOT = Path("data/index")

# Over-fetch factor before filtering, applied on top of the selectivity
# correction. 20 is `scripts/ask.py`'s value.
DEFAULT_DEPTH_MULTIPLIER = 20
MIN_POOL_DEPTH = 200
# Ceiling on the selectivity correction. A filter matching a single short
# section would otherwise ask for a pool larger than the corpus.
MAX_POOL_DEPTH = 20_000

DEFAULT_K = 8
MAX_K = 50

# The synthetic query id RRF is keyed on. `ticker.fusion.rrf` fuses runs shaped
# query_id -> chunk_id -> score; a single ad hoc query is a one-entry run.
_QUERY_ID = "q"

ARM_BM25 = "bm25"
ARM_DENSE = "dense"
ARM_RRF = "rrf"


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    score: float
    rank: int
    chunk: ChunkRecord
    arm_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    arm: str
    hits: list[SearchHit]
    pool_size: int
    exhausted_pool: bool
    filters: dict[str, object]

    @property
    def chunk_ids(self) -> list[str]:
        return [hit.chunk_id for hit in self.hits]


def _matches(
    chunk: ChunkRecord,
    *,
    company: str | None,
    form_type: str | None,
    fiscal_year: int | None,
) -> bool:
    if company is not None and chunk.ticker != company.upper():
        return False
    if form_type is not None and chunk.form != form_type:
        return False
    if fiscal_year is not None and chunk.fiscal_year != fiscal_year:
        return False
    return True


class HybridSearcher:
    """Hybrid retrieval over the indexed chunks, with filters and text.

    Holds a read-only DuckDB handle and one or more loaded retrievers. Loading
    the dense arm pulls a 440 MB FAISS index and the query encoder into memory,
    so construct this once per process and pass it around.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        retrievers: dict[str, object],
        *,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridSearcher needs at least one retriever")
        self._con = con
        self._retrievers = retrievers
        self._rrf_k = rrf_k
        self._selectivity_cache: dict[tuple, float] = {}

    @property
    def arms(self) -> tuple[str, ...]:
        single = tuple(sorted(self._retrievers))
        return single + ((ARM_RRF,) if len(single) > 1 else ())

    def _selectivity(
        self,
        company: str | None,
        form_type: str | None,
        fiscal_year: int | None,
    ) -> float:
        """Fraction of indexed chunks the filter admits, counted in SQL.

        Cached per filter combination: an agent session asks for the same
        company and form repeatedly, and this is a full scan over 148k chunks.
        """
        key = (company, form_type, fiscal_year)
        if key in self._selectivity_cache:
            return self._selectivity_cache[key]
        if key == (None, None, None):
            self._selectivity_cache[key] = 1.0
            return 1.0

        clauses, params = [], []
        if company is not None:
            clauses.append("f.ticker = ?")
            params.append(company.upper())
        if form_type is not None:
            clauses.append("f.form = ?")
            params.append(form_type)
        if fiscal_year is not None:
            clauses.append("EXTRACT(year FROM f.period_end) = ?")
            params.append(fiscal_year)
        matched, total = self._con.execute(
            f"""
            SELECT count(*) FILTER (WHERE {' AND '.join(clauses)}), count(*)
            FROM chunks c
            JOIN sections sec ON c.section_id = sec.section_id
            JOIN filings f ON sec.accession = f.accession
            """,
            params,
        ).fetchone()
        # A filter matching nothing gets the floor depth, not a divide by zero.
        # The empty result is then a fact about the corpus, reported through
        # `exhausted_pool` rather than masked by an ever-deeper pool.
        selectivity = (matched / total) if total and matched else 1.0
        self._selectivity_cache[key] = selectivity
        return selectivity

    def _chunk_metadata(
        self,
        chunk_ids: list[str],
        *,
        company: str | None,
        form_type: str | None,
        fiscal_year: int | None,
    ) -> set[str]:
        """Which of `chunk_ids` satisfy the filters, as a set of ids.

        Pushes the filter into SQL and projects only the id, so the pool scan
        moves no passage text across the boundary. Returning a set rather than
        rows makes the caller's loop a membership test, which preserves the
        ranker's ordering: the database is free to return rows in any order and
        the ranking must not depend on it.
        """
        if not chunk_ids:
            return set()
        clauses = ["c.chunk_id IN (SELECT unnest(?))"]
        params: list[object] = [chunk_ids]
        if company is not None:
            clauses.append("f.ticker = ?")
            params.append(company.upper())
        if form_type is not None:
            clauses.append("f.form = ?")
            params.append(form_type)
        if fiscal_year is not None:
            clauses.append("EXTRACT(year FROM f.period_end) = ?")
            params.append(fiscal_year)
        rows = self._con.execute(
            f"""
            SELECT c.chunk_id
            FROM chunks c
            JOIN sections sec ON c.section_id = sec.section_id
            JOIN filings f ON sec.accession = f.accession
            WHERE {' AND '.join(clauses)}
            """,
            params,
        ).fetchall()
        return {row[0] for row in rows}

    def search(
        self,
        query: str,
        *,
        company: str | None = None,
        form_type: str | None = None,
        fiscal_year: int | None = None,
        k: int = DEFAULT_K,
        arm: str = ARM_RRF,
        depth_multiplier: int = DEFAULT_DEPTH_MULTIPLIER,
    ) -> SearchResult:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        k = min(k, MAX_K)
        if arm != ARM_RRF and arm not in self._retrievers:
            raise ValueError(f"unknown arm {arm!r}; have {self.arms}")

        selectivity = self._selectivity(company, form_type, fiscal_year)
        depth = min(
            MAX_POOL_DEPTH,
            max(MIN_POOL_DEPTH, int(k * depth_multiplier / selectivity)),
        )
        runs = {
            name: {
                _QUERY_ID: {
                    hit.chunk_id: hit.score
                    for hit in _search_arm(retriever, query, depth)
                }
            }
            for name, retriever in self._retrievers.items()
            if arm in (ARM_RRF, name)
        }

        fused_arm = arm
        if arm == ARM_RRF and len(runs) > 1:
            fused = rrf(runs, k=self._rrf_k)[_QUERY_ID]
        else:
            # Asking for fusion with one arm loaded is single-arm retrieval.
            # Reporting it as "rrf" would put a fusion label on a number that
            # never went through fusion.
            fused_arm = next(iter(runs))
            fused = runs[fused_arm][_QUERY_ID]

        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        pool_size = len(ordered)

        # Filter on metadata before fetching any passage text. The pool runs to
        # thousands of candidates once the selectivity correction kicks in, and
        # loading every candidate's text to read its ticker off the join made
        # the tool ten times slower than the ranking it wraps -- 230 ms against
        # 3 ms for BM25 itself. Metadata is a narrow projection; text is fetched
        # only for the k that survive.
        meta = self._chunk_metadata(
            [chunk_id for chunk_id, _ in ordered],
            company=company,
            form_type=form_type,
            fiscal_year=fiscal_year,
        )

        surviving: list[tuple[str, float]] = []
        for chunk_id, score in ordered:
            # A chunk id in the index with no row in `chunks` means index and
            # database have drifted. Dropping it keeps a stale index from
            # producing a citation that cannot be read back.
            if chunk_id in meta:
                surviving.append((chunk_id, score))
            if len(surviving) == k:
                break

        records = load_chunks(self._con, [chunk_id for chunk_id, _ in surviving])
        hits: list[SearchHit] = []
        for chunk_id, score in surviving:
            record = records.get(chunk_id)
            if record is None:
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    score=float(score),
                    rank=len(hits) + 1,
                    chunk=record,
                    arm_scores={
                        name: run[_QUERY_ID][chunk_id]
                        for name, run in runs.items()
                        if chunk_id in run[_QUERY_ID]
                    },
                )
            )

        return SearchResult(
            query=query,
            arm=fused_arm,
            hits=hits,
            pool_size=pool_size,
            exhausted_pool=len(hits) < k,
            filters={
                "company": company,
                "form_type": form_type,
                "fiscal_year": fiscal_year,
            },
        )


def _search_arm(retriever: object, query: str, depth: int) -> list[RankedChunk]:
    return retriever.search(query, depth)  # type: ignore[attr-defined]


def load_searcher(
    con: duckdb.DuckDBPyConnection,
    *,
    index_root: Path | str = DEFAULT_INDEX_ROOT,
    arms: tuple[str, ...] = (ARM_BM25, ARM_DENSE),
) -> HybridSearcher:
    """Load the named retrievers from `index_root`.

    Imports are deferred per arm so a BM25-only caller does not pay the
    sentence-transformers and faiss import cost, which dominates start-up for
    a single `ask`.
    """
    index_root = Path(index_root)
    retrievers: dict[str, object] = {}
    if ARM_BM25 in arms:
        from ticker.retrieval.bm25 import load_retriever as load_bm25

        retrievers[ARM_BM25] = load_bm25(index_root / "bm25")
    if ARM_DENSE in arms:
        from ticker.retrieval.dense import load_retriever as load_dense

        retrievers[ARM_DENSE] = load_dense(index_root / "dense")
    unknown = set(arms) - {ARM_BM25, ARM_DENSE}
    if unknown:
        raise ValueError(f"unknown retrieval arms: {sorted(unknown)}")
    return HybridSearcher(con, retrievers)
