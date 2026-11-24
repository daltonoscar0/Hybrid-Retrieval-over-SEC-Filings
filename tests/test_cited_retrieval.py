"""The search tool: filtering, pool depth, ordering, neighbours, chunk ids.

Uses a stub ranker over an in-memory corpus. The rankers themselves are tested
in `tests/test_bm25.py` and `tests/test_dense.py`; what is under test here is
the layer around them -- that filters are applied without disturbing the
ranking, that a filter narrower than the pool is reported rather than silently
returning nothing, and that chunk ids resolve back to text and neighbours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from conftest_cited import StubRetriever, build_corpus  # noqa: E402

from ticker.cited.corpus import (
    chunk_ordinal,
    load_chunks,
    read_chunk,
    read_neighbors,
    section_id_of,
)
from ticker.cited.retrieval import ARM_RRF, HybridSearcher


@pytest.fixture
def corpus():
    return build_corpus()


class TestChunkIds:
    def test_parses_ordinal_and_section(self):
        chunk_id = "0000050863-25-000009#1A#chunk228"
        assert chunk_ordinal(chunk_id) == 228
        assert section_id_of(chunk_id) == "0000050863-25-000009#1A"

    def test_handles_items_with_spaces(self):
        chunk_id = "0001413447-21-000084#Part I Item 2#chunk14"
        assert chunk_ordinal(chunk_id) == 14
        assert section_id_of(chunk_id) == "0001413447-21-000084#Part I Item 2"

    @pytest.mark.parametrize("bad", ["not-a-chunk-id", "acc#1A", "acc#1A#chunkX", ""])
    def test_rejects_malformed(self, bad):
        # A sentinel return would sort an unparseable id to the front of its
        # section rather than failing.
        with pytest.raises(ValueError):
            chunk_ordinal(bad)


class TestCorpusReads:
    def test_reads_a_chunk_with_provenance(self, corpus):
        con, ids = corpus
        record = read_chunk(con, ids["nvda_2024"][0])
        assert record is not None
        assert record.ticker == "NVDA"
        assert record.form == "10-K"
        assert record.item == "1A"
        assert "53%" in record.text
        assert record.fiscal_year == 2025  # period end, not filing year

    def test_fiscal_year_is_the_period_not_the_filing_date(self, corpus):
        con, ids = corpus
        record = read_chunk(con, ids["nvda_2024"][0])
        # Filed February 2025 for a year ending January 2025.
        assert record.filed_at.year == 2025 and record.fiscal_year == 2025

    def test_missing_chunk_returns_none(self, corpus):
        con, _ = corpus
        assert read_chunk(con, "nope#1A#chunk0") is None

    def test_load_chunks_returns_only_what_exists(self, corpus):
        con, ids = corpus
        found = load_chunks(con, [ids["nvda_2024"][0], "nope#1A#chunk0"])
        assert set(found) == {ids["nvda_2024"][0]}

    def test_citation_names_the_filing(self, corpus):
        con, ids = corpus
        citation = read_chunk(con, ids["zion_2024"][0]).citation()
        assert "ZION" in citation and "0000000002-25-000001" in citation


class TestNeighbors:
    def test_returns_a_window_in_document_order(self, corpus):
        con, ids = corpus
        chunks = ids["nvda_2024"]
        middle = chunks[1]
        window = read_neighbors(con, middle, 1)
        ordinals = [chunk_ordinal(record.chunk_id) for record in window]
        assert ordinals == sorted(ordinals)
        assert middle in [record.chunk_id for record in window]

    def test_clamps_at_the_section_start(self, corpus):
        con, ids = corpus
        window = read_neighbors(con, ids["nvda_2024"][0], 2)
        assert window[0].chunk_id == ids["nvda_2024"][0]

    def test_does_not_cross_a_section_boundary(self, corpus):
        # A sentence past a section boundary is a different item of the
        # filing, and returning it as adjacent would invite a claim cited to
        # one item that is really supported by another.
        con, ids = corpus
        window = read_neighbors(con, ids["zion_2024"][0], 3)
        assert {record.item for record in window} == {"7"}
        assert all(record.ticker == "ZION" for record in window)

    def test_zero_returns_just_the_chunk(self, corpus):
        con, ids = corpus
        window = read_neighbors(con, ids["nvda_2024"][1], 0)
        assert [record.chunk_id for record in window] == [ids["nvda_2024"][1]]

    def test_negative_raises(self, corpus):
        con, ids = corpus
        with pytest.raises(ValueError):
            read_neighbors(con, ids["nvda_2024"][0], -1)


class TestSearchFiltering:
    def searcher(self, con, ranking):
        return HybridSearcher(con, {"stub": StubRetriever("stub", ranking)})

    def test_company_filter_excludes_other_companies(self, corpus):
        con, ids = corpus
        ranking = ids["zion_2024"] + ids["nvda_2024"]
        result = self.searcher(con, ranking).search("x", company="NVDA", k=3)
        assert result.hits
        assert all(hit.chunk.ticker == "NVDA" for hit in result.hits)

    def test_fiscal_year_filter_uses_the_period(self, corpus):
        con, ids = corpus
        ranking = ids["nvda_2023"] + ids["nvda_2024"]
        result = self.searcher(con, ranking).search(
            "x", company="NVDA", fiscal_year=2023, k=5
        )
        assert result.hits
        assert all(hit.chunk.fiscal_year == 2023 for hit in result.hits)

    def test_filter_preserves_the_rankers_order(self, corpus):
        con, ids = corpus
        # Interleave so a database-order result would differ from the ranking.
        ranking = [
            ids["zion_2024"][0], ids["nvda_2024"][2],
            ids["zion_2024"][1], ids["nvda_2024"][0],
        ]
        result = self.searcher(con, ranking).search("x", company="NVDA", k=2)
        assert result.chunk_ids == [ids["nvda_2024"][2], ids["nvda_2024"][0]]

    def test_ranks_are_contiguous_from_one(self, corpus):
        con, ids = corpus
        result = self.searcher(con, ids["nvda_2024"]).search("x", k=3)
        assert [hit.rank for hit in result.hits] == [1, 2, 3]

    def test_exhausted_pool_is_reported(self, corpus):
        con, ids = corpus
        # Asking for more than the corpus holds for this company.
        result = self.searcher(con, ids["zion_2024"]).search("x", company="ZION", k=50)
        assert result.exhausted_pool
        assert len(result.hits) < 50

    def test_filter_matching_nothing_returns_empty_not_an_error(self, corpus):
        con, ids = corpus
        result = self.searcher(con, ids["nvda_2024"]).search("x", company="AAPL", k=5)
        assert result.hits == [] and result.exhausted_pool

    def test_unknown_id_in_the_index_is_dropped(self, corpus):
        # Index and database having drifted must not produce a citation that
        # cannot be read back.
        con, ids = corpus
        ranking = ["ghost#1A#chunk0"] + ids["nvda_2024"]
        result = self.searcher(con, ranking).search("x", k=2)
        assert "ghost#1A#chunk0" not in result.chunk_ids

    def test_single_arm_is_not_labelled_as_fusion(self, corpus):
        # Reporting a single-arm result as "rrf" would put a fusion label on a
        # number that never went through fusion.
        con, ids = corpus
        result = self.searcher(con, ids["nvda_2024"]).search("x", k=1, arm=ARM_RRF)
        assert result.arm == "stub"

    def test_zero_k_is_rejected(self, corpus):
        con, ids = corpus
        with pytest.raises(ValueError):
            self.searcher(con, ids["nvda_2024"]).search("x", k=0)

    def test_unknown_arm_is_rejected(self, corpus):
        con, ids = corpus
        with pytest.raises(ValueError, match="unknown arm"):
            self.searcher(con, ids["nvda_2024"]).search("x", arm="magic")

    def test_no_retriever_is_rejected(self, corpus):
        con, _ = corpus
        with pytest.raises(ValueError):
            HybridSearcher(con, {})


class TestFusion:
    def test_two_arms_fuse_and_are_labelled_rrf(self, corpus):
        con, ids = corpus
        chunks = ids["nvda_2024"]
        searcher = HybridSearcher(
            con,
            {
                "a": StubRetriever("a", [chunks[1], chunks[0], chunks[2]]),
                "b": StubRetriever("b", [chunks[1], chunks[2], chunks[0]]),
            },
        )
        result = searcher.search("x", k=3, arm=ARM_RRF)
        assert result.arm == ARM_RRF
        # Ranked first by both arms, so it wins outright.
        assert result.hits[0].chunk_id == chunks[1]
        # Both arms contributed, so both scores are carried for inspection.
        assert set(result.hits[0].arm_scores) == {"a", "b"}

    def test_first_plus_third_narrowly_beats_second_plus_second(self, corpus):
        # A real property of RRF at k=60 rather than an accident: 1/61 + 1/63
        # is 0.032266 against 2/62 = 0.032258. Worth pinning, because a
        # reader comparing arm rankings by eye will not expect it.
        con, ids = corpus
        chunks = ids["nvda_2024"]
        searcher = HybridSearcher(
            con,
            {
                "a": StubRetriever("a", [chunks[0], chunks[1], chunks[2]]),
                "b": StubRetriever("b", [chunks[2], chunks[1], chunks[0]]),
            },
        )
        result = searcher.search("x", k=3, arm=ARM_RRF)
        assert result.chunk_ids[:2] == [chunks[0], chunks[2]]
        assert result.chunk_ids[2] == chunks[1]

    def test_arms_property_lists_fusion_only_with_two(self, corpus):
        con, ids = corpus
        chunks = ids["nvda_2024"]
        one = HybridSearcher(con, {"a": StubRetriever("a", chunks)})
        two = HybridSearcher(
            con, {"a": StubRetriever("a", chunks), "b": StubRetriever("b", chunks)}
        )
        assert ARM_RRF not in one.arms
        assert ARM_RRF in two.arms
