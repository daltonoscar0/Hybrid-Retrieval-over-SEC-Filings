"""Cited answers over the indexed filings, and the evaluation that scores them.

The retrieval layer under `ticker.retrieval` is reused unchanged. Nothing here
builds an index, re-chunks a filing, or touches the novelty layer; a citation is
a `chunks.chunk_id`, so every claim points at a passage that is already in the
corpus and can be fetched back out of DuckDB by primary key.
"""
