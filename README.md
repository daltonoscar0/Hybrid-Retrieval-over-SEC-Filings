# Ticker

Hybrid BM25 and dense retrieval over SEC filings, with a per sentence novelty
layer that separates language a company has never used before from the
boilerplate it reprints every year.

Search returns passages ranked by relevance. The novelty layer then tells you
which of those passages are actually new this period.

## Status

The system is built and the evaluation harness runs. The relevance judgments it
scores against are not written yet, so the retrieval comparison is unmeasured.
This section says which numbers exist and which do not, because the rest of the
repo is organised around not asserting the second kind.

Measured:

| quantity | value |
|---|---|
| corpus | 966 filings, 20 companies, 2 sectors, 2020 to 2025 |
| chunks indexed | 148,097 |
| sentences scored for novelty | 298,375 |
| novelty agreement with an independent diff (AUC) | 0.7132, CI [0.7013, 0.7244] |
| BM25 query latency | 1.5 ms mean |
| dense query latency | 54.2 ms mean, 164.1 ms p95 |
| cited answer question set | 60 questions, frozen, SHA-256 `712e2843e66cfbfb` |
| retrieval ceiling on citation recall | 90.0% R@10 filtered, CI [77, 100] |
| tests | 598 passing, 3 skipped |

Not measured yet:

* Whether dense retrieval beats BM25 on this corpus, or whether fusion beats
  either. The runs exist, the metrics code exists, the judgments do not.
* The two column table comparing standard relevance against novelty
  conditioned relevance, which is the result this project is built to produce.
* Whether the section extraction boundaries are correct. There are no approved
  fixtures, so every extraction rate currently reports the extractor grading
  itself.
* Every cited answer number: answer accuracy, citation support, hallucinated
  figures, cost. The agent, both baselines, the scorer and the judges are
  built and tested. No evaluation run has been made, so the results table is
  empty. See `Cited answers` below for what that table will hold and what the
  run costs.

## The novelty measure

For a sentence in a filing from firm f at time t:

```
novelty(s) = surprisal_firm(s) - surprisal_sector(s)
```

Both terms are interpolated Kneser Ney 5 gram models. The first is fit on
firm f's own filings before t. The second is fit on that firm's sector peers
before t, excluding f.

The subtraction is the point. Raw surprisal under a firm model flags anything
containing unusual English, including jargon that is unremarkable in context.
Subtracting a sector background cancels that. A sentence that is rare in
general and rare for this firm scores near zero. A sentence that is ordinary
English this firm has never written before scores high. That second thing is
what a reader actually wants.

Every fit takes an explicit `as_of` and filters on strict inequality. A model
scoring a document filed at t sees only documents filed before t, and that is
enforced in the data layer rather than by convention.

### Does it work

Two checks so far, neither of which needs a human label.

**Agreement with a sentence diff.** Align each section to the same firm's prior
period version, label every sentence inserted, modified or unchanged, and ask
whether novelty separates changed from unchanged. AUC is 0.7132, CI [0.7013,
0.7244] over 92,863 sentences, with a 100% join rate. The diff is computed from
text alone with no access to the surprisal code, so this is two independent
constructions agreeing. The threshold for abandoning the measure was an AUC
near 0.5.

**Concentration by item.** Novelty should be highest where the literature says
real change concentrates. Risk Factors sits above the Item 1 Business control
with disjoint intervals, which is the predicted result. MD&A does not separate
from the control, which is not. Items 3 and 7A average under 20 sentences per
section and are too short to read anything into.

That second check is a partial negative and it is reported as one.

## Corpus

Twenty companies across semiconductors and regional banks, six years, EDGAR
only. Two sectors rather than one because a per firm language model needs a
sector matched background to contrast against, and a sector matched control is
stronger than a corpus wide one.

| form | filings | sentences |
|---|---|---|
| 10-K | 120 | 114,667 |
| 10-Q | 360 | 144,659 |
| 8-K | 486 | 39,049 |

10-K items 1, 1A, 3, 7 and 7A. 10-Q Part I item 2 and Part II items 1 and 1A.
8-K exhibit 99.1 earnings releases.

There are no scraped earnings call transcripts. The readable ones are
copyrighted and a public repo that scrapes them is a liability. See `DATA.md`.

## How it works

Chunks are four sentence windows at stride two, carrying stable sentence ids so
a novelty score computed per sentence maps back onto a chunk without
recomputation.

**Lexical.** `bm25s` with the Lucene scoring variant, k1=1.5, b=0.75, no
stemmer. Naming the variant matters because production systems disagree about
five of them (Kamphuis et al., ECIR 2020). Lucene's IDF cannot go negative for
a term appearing in more than half the corpus, which "risk" and "quarter" both
do here. The index rebuilds byte identically, verified on the full corpus.

**Dense.** `BAAI/bge-base-en-v1.5` pinned to a resolved commit sha, never a
branch name, over a FAISS flat index. Flat means exact cosine with no
approximation and no recall knob, so the dense arm's numbers report the
embedding model rather than index hyperparameters. 461 MB, and the search is
faster than the query encode that feeds it.

**Fusion.** Reciprocal rank fusion at k=60, the published value, deliberately
not tuned. With 40 queries and no third split reserved for it, fitting k would
fit a free parameter on the same queries the result gets reported over. A
weighted variant tunes on a held out split and reports the held out number.

**Evaluation.** `ranx` for nDCG@10, MRR@10 and Recall@100 with bootstrap
confidence intervals and paired t tests. nDCG is cross checked against
`pytrec_eval` to four decimals as a test rather than a one off script, so it
re-checks on every run.

The harness was written before the second retriever existed. If evaluation
comes last it gets compromised to make the numbers look good, and everyone
reading the repo knows it.

## Query set and judging

Forty queries, graded 0 to 3, pooled from BM25, dense and a keyword seed list
at depth 20 each. The pool is 53.6 chunks per query, 2,144 in total.

Forty rather than the sixty originally planned. That is a deliberate trade and
it costs precision: every confidence interval is about 22% wider than the same
interval over sixty queries, and every paired test loses power by the same
factor. A non significant row should be read as underpowered rather than as a
null.

Judgments are human written. Nothing in this repo generates a relevance label,
and the paths that hold them are blocked at the tooling level. A model scoring
its own retrieval is not a result.

## Cited answers

An analyst asks a question about a company. An agent answers it using the
retrieval above, and every claim in the answer carries the chunk ids of the
passages that support it. Then an evaluation measures how often the answers are
right and how often the citations actually hold up.

The evaluation is the point. The agent is the thing being measured.

### Results

| system | answer accuracy | citation support | hallucinated figures | cost per question |
|---|---|---|---|---|
| closed book | not measured | not measured | not measured | not measured |
| single shot retrieval | not measured | not measured | not measured | not measured |
| agent | not measured | not measured | not measured | not measured |

**No evaluation run has been made.** Every cell above is empty because
generating it costs money and none has been spent. The agent, both baselines,
the scorer, the two judges and the reporting are built, and 598 tests pass
offline. `results/cited_answers/report.md` is the file this table comes from,
and it currently says the same thing.

One command fills it in:

```
ticker eval run --system closed
ticker eval run --system single
ticker eval run --system agent
ticker eval score
```

That costs about $8 end to end with `--preset cheap`, which answers on Haiku
4.5 and judges on Sonnet 5, or about $31 on the default Opus 5 and Sonnet 5.
`ticker eval run` prints its estimate and asks before spending anything.

### What is measured

These numbers need no model, so they exist. They are in
`results/cited_answers/retrieval_ceiling.json`.

Citation recall cannot exceed the rate at which retrieval returns the gold
passage at all. That ceiling is measured directly, over the 30 questions that
have gold passages, with the question text as the query:

| arm | R@1 | R@5 | R@10 | R@50 |
|---|---|---|---|---|
| BM25 | 66.7% [50, 83] | 83.3% [70, 97] | 90.0% [77, 100] | 93.3% [83, 100] |
| dense | 36.7% [20, 53] | 70.0% [53, 83] | 83.3% [70, 93] | 90.0% [77, 100] |
| fusion | 50.0% [33, 67] | 80.0% [67, 93] | 86.7% [73, 97] | 90.0% [77, 100] |

Intervals are 95% percentile bootstrap over questions. Filters set to the
question's company and form.

Two findings, and one warning about reading them.

Scoping matters more than ranking. Without the company and form filters, BM25
R@10 falls from 90.0% to 56.7% and dense from 83.3% to 26.7%. The gap between
the filtered and unfiltered tables is far wider than the gap between the arms,
which says that the most consequential thing an agent does on these questions
is pick the right company and form, not phrase the query well.

The warning: this comparison favours BM25 by construction and is not evidence
that lexical retrieval is better here in general. Each gold passage was found
by substring scan for a distinctive phrase, and each question was then written
from that passage, so question and passage share vocabulary. A lexical ranker
is good at exactly that. The comparison is sound as a ceiling, which is what it
is for, and unsound as a retrieval result. Settling the retrieval question
needs the human judgments under `data/qrels/`, which are not written.

Also measured: 19 of the 20 numeric gold values appear in the indexed text of
the filing that reported them. The exception is `n04`, Qualcomm's fiscal 2025
research and development expense, which is tagged in XBRL but stated nowhere in
the five indexed 10-K items. No system can answer it from retrieval, so it is
reported separately rather than counted as a retrieval failure.

### A worked example

No agent answer exists, so there is none to show. What can be shown is the
input: the passages the `search` tool actually returns for a real question,
beside the gold answer written from the filing in advance. Full version in
`results/cited_answers/example_retrieval.md`.

Question `q08`:

> What share of Microchip's fiscal 2025 net sales came through distributors,
> and which distributor accounted for 10% or more of net sales?

Gold answer, written before any system ran:

> Microchip derived 45% of fiscal 2025 net sales through distributors and 55%
> from customers serviced directly, compared with 47% and 53% in fiscal 2024.
> Arrow Electronics, its largest distributor, accounted for 10% of net sales in
> fiscal 2025 and 12% in fiscal 2024; no other distributor or direct customer
> exceeded 10%.

Top passage returned, `0000827054-25-000077#7#chunk94`, MCHP 10-K item 7:

> Distribution Distributors accounted for approximately 45% and 47% of our net
> sales in fiscal 2025 and fiscal 2024, respectively. With the exception of
> Arrow Electronics, our largest distributor, which accounted for 10% and 12%
> of our net sales in fiscal 2025 and in fiscal 2024, respectively, no other
> distributor or direct customer accounted for more than 10% of our net sales
> in fiscal 2025 or in fiscal 2024.

That passage fully supports the gold answer and is not in the gold set, because
the gold set holds the Item 1 passage the answer was drafted from. A system
citing it would be right and would score as a citation recall miss. This is the
main caveat on that metric and it is quantified below.

### How the questions were built and frozen

Sixty questions in `eval/questions.jsonl`, all twenty indexed companies, every
gold answer written before any answering system existed.

| kind | n | gold provenance |
|---|---|---|
| numeric | 20 | XBRL value from EDGAR company facts, with unit, period and accession |
| qualitative | 25 | one or two sentences drafted from located passages, with gold chunk ids |
| unanswerable | 10 | nothing in the corpus answers it; correct behaviour is to decline |
| multi filing | 5 | needs two filings, so neither alone answers it |

The numeric half is generated, not drafted. A plan names a company, a concept
and whether the question asks for a level or a year over year change; the value,
its unit, its period and its accession are read from EDGAR. Nothing about those
figures was written by hand, and the build fails rather than substituting a
concept that turns out not to be tagged. Twelve ask for a level and eight for a
change. Change questions take both operands from the same filing, because the
comparative a company prints this year is the figure it computed its own growth
rate against.

The text half records the substring pattern each answer was found with, not the
chunk ids it was found at. The build re-runs that scan to derive the gold chunk
ids, so the gold set has a reproducible chain back to the filing text.
`ticker eval build-questions --verify` re-derives all sixty and fails on any
drift; it passes at the recorded hash.

Gold passages were located by SQL substring scan over `chunks.text`, never by
BM25 or the dense index. Scoring retrieval against a gold set that retrieval
chose would guarantee high recall and measure nothing.

Unanswerability is checked, not asserted. Each unanswerable question carries
probe phrases that must match zero chunks. That check caught three drafts that
were wrong: NVIDIA's fiscal 2019 revenue is in the corpus, as a comparative in
the fiscal 2021 10-K; Qualcomm's effective tax rate is discussed in MD&A; and an
attempt to build these questions around balance sheet figures failed outright,
because the 486 indexed 8-K earnings releases carry full balance sheets even
though Item 8 is not extracted. What is structurally absent, and so safe ground,
is cover page facts, Item 5 content, proxy content, note level detail, fiscal
years outside the window, and events that did not happen.

The file is frozen. Its SHA-256 is recorded in every results file, and CI fails
if a results file claims a hash the question file no longer has.

### The three systems

All three answer the same frozen questions, in the same schema, through the same
response cache.

**Closed book** gets the question and no tools. It retrieves nothing, so every
citation it offers is rejected. That is the measurement: the count of rejected
citations is a count of fabricated chunk ids, and the figures in its answers are
a count of recalled or invented numbers.

**Single shot** retrieves once against the question text and answers from those
passages, with no loop. Its filters come from the question record, so it is
handed the company and form that the ceiling numbers above say matter most. Any
margin the agent shows over it is a margin over a baseline that got the scoping
for free.

**The agent** has `search`, `read_chunk` and `read_neighbors`, and a cap of
eight tool calls by default. It is told the cap and sees the remaining count in
every tool result, so spending it is a decision rather than a truncation. When
the budget runs out the loop forces a submission from what was retrieved, and
records that it did.

The answer is a tool call with a strict schema, not parsed prose. Each claim
carries the chunk ids supporting it, and declining is a field on that schema
rather than a phrase the scorer has to recognise. A citation to a chunk the
session never retrieved is stripped and counted, because chunk ids are
structural and so trivially fabricable from a known accession.

### How answers are scored

Answer correctness branches on question kind, because the kinds are not
comparable. Numeric answers are checked arithmetically against the XBRL value
with no model involved, correct within 0.5% after unit normalisation.
Unanswerable questions are correct only if the system declined. Everything else
goes to an LLM judge against a written rubric.

Citation support prefers the deterministic path. A claim that asserts figures is
checked by looking for those figures in the cited text, with no judge and so no
judge error rate. Only a claim with no reported figure in it reaches the
citation judge.

Two deliberate departures from a naive reading of the spec, both because the
naive version measures the wrong thing:

A percentage gold is matched on magnitude with the direction read from the
wording. Filings write a 2.08% decline as "revenue declined 2.1%", never as
"-2.08%", and a signed comparison marks that correct answer wrong. Direction
must still agree, and an answer that gives a magnitude with no direction fails.

A relative 0.5% tolerance on a gold of -2.08% admits only 0.0104 percentage
points, so an answer rounded to one decimal scores wrong. Percentage golds also
accept an absolute 0.05 percentage point allowance. Which of the two rules let
an answer through is recorded per answer, so the split is visible rather than
buried.

Passage matching tries several scales, because a filing table carries "(In
Millions)" in a column header that often sits in a different chunk from the
number. That search is constrained by a three significant digit floor: at scale
1e9 the bare token "83" is within 0.5% of $82.798 billion, and without the floor
that counts as a disclosure of the figure. It is a coincidence the scale search
produced.

### The judge agreement check

No judge based number should be read before the judge has been checked against a
person, and nothing in the judge's own output can do that.

`ticker eval judge-sample` draws 40 claim and citation pairs, stratified by
system and seeded, and writes two files. The blind sample carries the claim, the
cited passage and the label options, and **not** the judge's verdict; showing
the verdict while labelling would measure compliance rather than agreement. The
judge's own labels go to a separate file at the same time. Raw agreement and
Cohen's kappa come from `ticker eval judge-agreement`, which exits non zero
below kappa 0.6, the conventional substantial agreement boundary.

The citation judge sees the claim and the cited passage and nothing else. Not
the question, not the gold answer, not which system produced it. A judge that
can see the gold will mark a citation supported because the claim is true, and
the question is not whether the claim is true but whether that passage supports
it. A claim can be correct and cited to a passage that says nothing about it,
which is the failure a citation metric exists to catch.

`eval/human_labels.jsonl` is written by a person. It is blocked at the tooling
level alongside `data/qrels/`, for the same reason: a model generated label
there would make the judge's agreement a measurement of a model against itself.

**Not run.** No answers exist, so there are no pairs to sample and no agreement
to report. Until that changes, treat answer accuracy on the 30 qualitative and
multi filing questions, and citation support on claims without figures, as
unvalidated. The numeric arm does not depend on the judge at all.

### Failure analysis

**Not done.** Sorting wrong answers into categories needs wrong answers, and no
system has produced any. The categories the analysis will use are the ones the
recorded data supports: retrieved the wrong year, wrong unit or scale, right
passage but misread, cited a passage that does not support the claim, should
have declined, and missed the second filing. Every answer records its full tool
call sequence with arguments, so a wrong year or a wrong filter is visible in
the record rather than inferred.

### What to be skeptical of

**There are no answer quality numbers.** The headline table is empty. Everything
measured here is about the inputs to the agent and the correctness of the
scoring machinery, not about how well the agent answers. A reader should treat
this as a built and tested harness with a verified question set, not as a
result.

**Citation recall is a lower bound.** A gold set is the passages an answer was
drafted from, found by one phrase. Filings restate the same fact in more than
one indexed item. At a content word overlap of 0.5, four of the thirty gold
questions have a near duplicate passage in another section, and four more have
one adjacent to a gold passage through the stride 2 chunk overlap. That count is
itself a lower bound, because word overlap only catches restatements that reuse
the wording: `q08` above is a case it misses, and the passage it misses is the
one the search tool ranks first. Citation support compares a claim against
whatever passage was cited and does not have this problem.

**Thirty of the sixty gold answers are model drafted and unreviewed.** The 20
numeric golds come from EDGAR with accession numbers and are machine checkable.
The 25 qualitative, 5 multi filing and 10 unanswerable golds were drafted from
located passages and have not been reviewed by a person. The locators and the
unanswerability probes make them reproducible and checkable, which is not the
same as checked. This is the same objection this repo takes seriously for
relevance judgments, and it applies here.

**Sixty questions, twenty companies, two sectors.** Every interval over the full
set is already wide, and the by type intervals are computed over 5 to 25
questions each. Semiconductors and regional banks are not the economy.

**One numeric question is unanswerable from the indexed text** and is known to
be. `n04` stays in the set because removing it would flatter the numeric
accuracy figure, and it is reported separately instead.

**Tool latency is not ranking latency.** The BM25 search tool averages 107 ms
end to end on the recorded run against about 3 ms for BM25 ranking at the same
pool depth. The rest is the metadata filter over an over fetched pool and
loading passage text from DuckDB. The 1.5 ms in the Status table is the ranker
alone and the two are not comparable. These figures move by tens of
milliseconds between runs on a laptop, so read them as the shape of the cost
rather than as a benchmark.

### Running the cited answer evaluation

```
ticker eval build-questions --verify
ticker eval retrieval-ceiling
ticker ask "How much did Microchip derive through distributors?" --company MCHP
```

The first two make no model calls and cost nothing. `ticker ask` and
`ticker eval run` do. `--offline` turns a cache miss into an error, so
re-running a completed evaluation can be shown to have sent nothing: response
caching is keyed on the prompt, the model and the settings, which is what makes
re-scoring free and the reported numbers reproducible from the cache.

The API key is read from the environment by the SDK. It is never written to a
file, a log or a results file.

## Running it

```
uv sync
uv run python scripts/download_corpus.py
uv run python scripts/ingest_corpus.py
uv run python scripts/index.py
uv run python scripts/index_dense.py
```

The dense build encodes 148,097 chunks in about three and a quarter hours on an
M series laptop at 12.5 chunks per second. Everything else is minutes.

Then search, fuse and pool:

```
uv run python scripts/search.py \
    --retriever ticker.retrieval.bm25:load_retriever \
    --retriever ticker.retrieval.dense:load_retriever
uv run python scripts/fuse.py
uv run python scripts/pool.py \
    --bm25 ticker.retrieval.bm25:load_retriever \
    --dense ticker.retrieval.dense:load_retriever
```

Score novelty and validate it:

```
uv run python scripts/score_novelty.py --form 10-K --as-of filing
uv run python scripts/validate.py
```

Judge, then evaluate:

```
uv run python scripts/judge.py
uv run python scripts/evaluate.py
```

`scripts/judge.py` grades one chunk per keypress with no Enter, writes every
decision before showing the next chunk, and resumes where it left off.

## What to be skeptical of

**Time discipline covers novelty, not retrieval.** BM25 fits its IDF over the
whole corpus with no time predicate and the dense index embeds every chunk
regardless of date. Retrieval here is static ad hoc, not walk forward. Someone
who knows the leakage literature will spot this, so it is better stated than
defended later.

**The sector background is fit on a survivorship selected peer set.** The 20
companies were chosen by requiring continuous filing from 2020 through 2025,
which is a decision made with information from 2026. SVB Financial, Signature
Bank and First Republic are absent because they failed in 2023, so a surviving
bank's 2023 language is scored against a background containing no bank that
failed that year. The direction of the resulting bias is not measured and is
not asserted.

**The 40 queries were filtered partly on corpus term frequency.** Twenty of
sixty candidates were dropped, several on how much material the corpus held.
That shapes the evaluation set using the corpus the systems are scored on. Every
drop is listed with its reason in `reports/query_set.md`.

**The dense index rebuild is unverified at full scale.** Byte identical rebuild
is checked for BM25 on the whole corpus and for dense only on subsets, because
verifying it doubles a three hour build.

## Stack

Python 3.12 with uv. `edgartools`, `duckdb`, `bm25s`, `sentence-transformers`,
`faiss-cpu`, `ranx`, and `anthropic` for the cited answer agent and judges. No
vector database, no orchestration framework, no RAG library. The agent loop is
a `while` loop over the Messages API with four tools. At this corpus size the
frameworks add dependencies and hide the parts worth showing.

## Layout

```
src/ticker/          library code
src/ticker/cited/    the cited answer agent, its evaluation and its judges
src/ticker/cli.py    the `ticker` command line
scripts/             one entry point per pipeline stage
tests/               598 tests
reports/             one report per phase, with the numbers
eval/questions.jsonl the frozen 60 question set
results/cited_answers/  what the cited answer evaluation has measured
data/qrels/          human written relevance judgments
PLAN.md              the specification
DATA.md              corpus scope and the licensing position
REFERENCES.md        citations
```

## License

MIT, see `LICENSE`. Filing text is public domain from EDGAR.
