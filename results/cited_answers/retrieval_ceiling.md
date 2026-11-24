Gold passages exist for 30 of the sixty questions: the 25 qualitative and 5 multi-filing ones. Numeric golds are XBRL values rather than passages, and unanswerable questions have no gold passage by construction.

### Recall@k, with the question's company and form filters applied

| arm | R@1 | R@5 | R@10 | R@20 | R@50 | MRR |
|---|---|---|---|---|---|---|
| bm25 | 66.7% [50, 83] | 83.3% [70, 97] | 90.0% [77, 100] | 90.0% [77, 100] | 93.3% [83, 100] | 0.735 |
| dense | 36.7% [20, 53] | 70.0% [53, 83] | 83.3% [70, 93] | 86.7% [73, 97] | 90.0% [77, 100] | 0.525 |
| rrf | 50.0% [33, 67] | 80.0% [67, 93] | 86.7% [73, 97] | 90.0% [77, 100] | 90.0% [77, 100] | 0.646 |

### Recall@k, unfiltered, over all 148,097 chunks

| arm | R@1 | R@5 | R@10 | R@20 | R@50 | MRR |
|---|---|---|---|---|---|---|
| bm25 | 33.3% [17, 50] | 53.3% [33, 70] | 56.7% [37, 73] | 66.7% [50, 83] | 83.3% [70, 97] | 0.427 |
| dense | 16.7% [3, 30] | 23.3% [10, 40] | 26.7% [10, 43] | 36.7% [20, 53] | 53.3% [37, 70] | 0.218 |
| rrf | 20.0% [7, 33] | 33.3% [17, 50] | 46.7% [30, 63] | 63.3% [47, 80] | 83.3% [70, 97] | 0.283 |

### Read this before comparing the arms

These numbers compare BM25, dense and fusion at finding a passage, and they favour BM25 for a reason built into how the gold set was made. Each gold passage was located by substring scan for a distinctive phrase, and each question was then drafted from that passage, so question and passage share vocabulary by construction. That is what a lexical ranker is good at. The comparison is sound as a ceiling on citation recall for these questions -- which is what it is here for -- and unsound as a general claim that BM25 retrieves better than the dense arm on this corpus. The retrieval comparison that would settle that needs the human relevance judgments under `data/qrels/`, which are not written.

The filtered and unfiltered tables differ by much more than the arms do. Scoping to the right company and form is worth more than the choice of ranker, which is the useful finding for an agent: getting the filters right matters more than anything it can do with the query.

### Query latency

End to end for one `search` tool call: ranking, the metadata filter over the over-fetched pool, and loading the returned passages from DuckDB. BM25 ranking alone is about 3 ms at this pool depth, so most of what follows is the database work the tool wraps around it, not the ranker.

| arm | mean | p95 |
|---|---|---|
| bm25 | 107.2 ms | 244.6 ms |
| dense | 137.9 ms | 258.9 ms |
| rrf | 209.1 ms | 304.9 ms |

### Why citation recall is a lower bound

A gold passage set is the passages an answer was drafted from, found by one phrase. Filings restate the same fact in more than one indexed item, so a system can cite a passage that fully supports its claim and still be counted as a recall miss because that passage is not the one the drafter used. Measured at a content-word Jaccard of 0.5, over 30 gold questions:

- 4 have a near-duplicate passage adjacent to a gold passage. This is an artefact of chunking: 4-sentence windows at stride 2 share two sentences with each neighbour.
- 4 have a near-duplicate passage in a different section, or in the other filing of a multi-filing question. These are genuine second disclosures and they are what make this metric a lower bound.

Example found by this check: on `m01` the gold passage is item 1A (`0001212545-23-000093#1A#chunk8`), and `0001212545-24-000092#1A#chunk8` (item 1A) states the same thing.

The count above is itself a lower bound, because a word-overlap threshold only catches restatements that reuse the wording. Question `q08` is a case it misses: the gold passage is Microchip's Item 1 distributor discussion, and `0000827054-25-000077#7#chunk94` in MD&A gives the same two percentages and the same Arrow Electronics concentration in different words, which puts its overlap under the threshold. It is also the passage the search tool ranks first for that question, so the most likely citation a system would make there is correct and would score as a recall miss. See `example_retrieval.md`.

Citation support compares a claim against whatever passage was actually cited, so it does not have this problem. Read it, not citation recall, as the grounding measure.

### Numeric gold coverage

19 of 20 numeric gold values appear in the indexed text of the filing that reported them. The 1 that do not (n04) cannot be answered from retrieval at any k, and are reported separately in the numeric accuracy row.
