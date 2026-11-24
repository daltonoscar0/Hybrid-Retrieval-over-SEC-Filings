"""Measurements that need no model: the retrieval ceiling, and corpus coverage.

These are the numbers this feature can report without spending anything, and
they are not filler. Citation recall is bounded above by whether the gold
passage is retrieved at all: if a question's gold chunk is not in the top k,
no agent reading those k passages can cite it, however well it reasons. So
recall@k over the gold sets is the ceiling on the agent arm's citation recall,
measured directly, and a citation-recall figure from an evaluation run has to
be read against it.

Three things are measured here.

**Recall@k** per retrieval arm, on the thirty questions that have gold
passages, at the question text as query. Reported both with the question's own
company and form filters applied -- what an agent that scopes correctly sees
-- and unfiltered, which is the harder problem of finding the passage in all
148,097 chunks. The gap between those two is how much of retrieval's job the
filters do.

**Reciprocal rank** of the first gold passage, which says whether a gold hit
at k=10 was at rank 2 or rank 10. Recall alone cannot distinguish those, and
an agent reading the top few passages cares.

**Corpus coverage of the numeric golds**: how many of the twenty XBRL values
appear in the indexed text of the filing they were reported in. A gold figure
that is not in the text is unanswerable from retrieval no matter what, and
that count is the honest caveat on the numeric arm.

The gold sets were located by substring scan, never by these rankers (see
`ticker.cited.locate`), so none of this is circular.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ticker.cited.corpus import connect
from ticker.cited.questions import (
    KIND_MULTI_FILING,
    KIND_NUMERIC,
    KIND_QUALITATIVE,
    Question,
    freeze_hash,
    load_questions,
)
from ticker.cited.retrieval import ARM_BM25, ARM_DENSE, ARM_RRF, load_searcher
from ticker.evaluation import DEFAULT_N_BOOT, bootstrap_ci

DEFAULT_K_VALUES = (1, 5, 10, 20, 50)

# The deepest k any measurement needs. Retrieved once per query per arm and
# sliced, rather than re-querying per k: the arms are deterministic, so the
# top 10 of a 50-deep result is identical to a 10-deep query, and this makes
# the sweep five times cheaper.
MAX_K = max(DEFAULT_K_VALUES)


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    question_id: str
    kind: str
    arm: str
    filtered: bool
    gold_chunk_ids: tuple[str, ...]
    retrieved: tuple[str, ...]
    latency_ms: float

    def hit_at(self, k: int) -> bool:
        """Did any gold chunk appear in the top k?"""
        return bool(set(self.retrieved[:k]) & set(self.gold_chunk_ids))

    def coverage_at(self, k: int) -> float:
        """Share of this question's gold chunks appearing in the top k."""
        if not self.gold_chunk_ids:
            return 0.0
        found = set(self.retrieved[:k]) & set(self.gold_chunk_ids)
        return len(found) / len(self.gold_chunk_ids)

    @property
    def reciprocal_rank(self) -> float:
        gold = set(self.gold_chunk_ids)
        for rank, chunk_id in enumerate(self.retrieved, start=1):
            if chunk_id in gold:
                return 1.0 / rank
        return 0.0


@dataclass(frozen=True, slots=True)
class ArmResult:
    arm: str
    filtered: bool
    n_questions: int
    recall_at_k: dict[int, list[float]] = field(default_factory=dict)
    coverage_at_k: dict[int, list[float]] = field(default_factory=dict)
    mrr: float = 0.0
    mrr_ci: tuple[float, float] = (0.0, 0.0)
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


def _ci(values: list[float], n_boot: int = DEFAULT_N_BOOT) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    return bootstrap_ci(np.asarray(values, dtype=float), n_boot=n_boot)


def gold_questions(questions: list[Question]) -> list[Question]:
    """Questions with a gold passage set: the qualitative and multi-filing ones."""
    return [
        question
        for question in questions
        if question.gold_chunk_ids
        and question.kind in (KIND_QUALITATIVE, KIND_MULTI_FILING)
    ]


def run_arm(
    searcher,
    questions: list[Question],
    *,
    arm: str,
    filtered: bool,
    k: int = MAX_K,
) -> list[QueryOutcome]:
    """Retrieve for every question with one arm, at depth `k`."""
    outcomes: list[QueryOutcome] = []
    for question in questions:
        started = time.perf_counter()
        result = searcher.search(
            question.text,
            company=question.company if filtered else None,
            form_type=question.form_type if filtered else None,
            # A multi-filing question spans two years, so its record carries no
            # single fiscal year and filtering on one would exclude half its
            # own gold set. `Question.fiscal_year` is already None for those.
            fiscal_year=question.fiscal_year if filtered else None,
            k=k,
            arm=arm,
        )
        outcomes.append(
            QueryOutcome(
                question_id=question.question_id,
                kind=question.kind,
                arm=arm,
                filtered=filtered,
                gold_chunk_ids=question.gold_chunk_ids,
                retrieved=tuple(result.chunk_ids),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        )
    return outcomes


def summarise_arm(
    outcomes: list[QueryOutcome],
    *,
    arm: str,
    filtered: bool,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> ArmResult:
    selected = [o for o in outcomes if o.arm == arm and o.filtered == filtered]
    latencies = [o.latency_ms for o in selected]
    reciprocal = [o.reciprocal_rank for o in selected]
    return ArmResult(
        arm=arm,
        filtered=filtered,
        n_questions=len(selected),
        recall_at_k={k: [float(o.hit_at(k)) for o in selected] for k in k_values},
        coverage_at_k={k: [o.coverage_at(k) for o in selected] for k in k_values},
        mrr=float(np.mean(reciprocal)) if reciprocal else 0.0,
        mrr_ci=_ci(reciprocal),
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
    )


CONTENT_WORD = __import__("re").compile(r"[a-z0-9][a-z0-9%$.,-]*")

# Jaccard overlap on content words at which two passages are treated as
# restating the same thing. 0.5 is high: half the vocabulary shared. Adjacent
# chunks reach it by construction, since 4-sentence windows at stride 2 share
# two sentences, which is why adjacency is reported separately below.
RESTATEMENT_THRESHOLD = 0.5

_STOPWORDS = frozenset(
    "the a an and or of to in for on at by with from as is are was were be been "
    "that this these those it its our we us their his her not no but if then "
    "than which who whom whose will would may might can could should have has "
    "had do does did i you he she they them there here when where how why all "
    "any both each few more most other some such only own same so very s t".split()
)


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in CONTENT_WORD.findall(text.lower())
        if word not in _STOPWORDS and len(word) > 2
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def alternative_support(
    con, questions: list[Question], *, threshold: float = RESTATEMENT_THRESHOLD
) -> dict:
    """How often a passage outside the gold set restates a gold passage.

    This quantifies the main caveat on citation recall. A gold set is the
    passages an answer was drafted from, located by one phrase. Filings
    restate the same fact in more than one indexed item -- Microchip gives its
    distributor share in both Item 1 and MD&A -- so a system can cite a
    passage that fully supports its claim, be right, and still be scored as a
    recall miss because that passage is not the one the drafter happened to
    use.

    Two kinds of restatement are counted separately because they mean
    different things. An *adjacent* near-duplicate is an artefact of chunking:
    4-sentence windows at stride 2 share two sentences with each neighbour, so
    the neighbour almost always carries the claim too. A near-duplicate
    *elsewhere in the filing* is a genuine second disclosure, and it is the
    case that makes citation recall a lower bound rather than a measurement.
    """
    from ticker.cited.corpus import chunk_ordinal, section_id_of

    adjacent_questions = 0
    elsewhere_questions = 0
    examples: list[dict] = []

    for question in questions:
        gold_ids = set(question.gold_chunk_ids)
        if not gold_ids:
            continue
        rows = con.execute(
            """
            SELECT c.chunk_id, c.text, sec.item
            FROM chunks c
            JOIN sections sec ON c.section_id = sec.section_id
            JOIN filings f ON sec.accession = f.accession
            WHERE f.accession IN (SELECT unnest(?))
            """,
            [list(question.gold_accessions)],
        ).fetchall()
        texts = {row[0]: row[1] for row in rows}
        items = {row[0]: row[2] for row in rows}

        has_adjacent = False
        has_elsewhere = False
        for gold_id in question.gold_chunk_ids:
            if gold_id not in texts:
                continue
            gold_words = _content_words(texts[gold_id])
            gold_section = section_id_of(gold_id)
            gold_ordinal = chunk_ordinal(gold_id)
            for chunk_id, text in texts.items():
                if chunk_id in gold_ids:
                    continue
                if _jaccard(gold_words, _content_words(text)) < threshold:
                    continue
                same_section = section_id_of(chunk_id) == gold_section
                near = same_section and abs(chunk_ordinal(chunk_id) - gold_ordinal) <= 2
                if near:
                    has_adjacent = True
                else:
                    has_elsewhere = True
                    if len(examples) < 5:
                        examples.append(
                            {
                                "question_id": question.question_id,
                                "gold_chunk_id": gold_id,
                                "gold_item": items.get(gold_id),
                                "restating_chunk_id": chunk_id,
                                "restating_item": items.get(chunk_id),
                            }
                        )
        adjacent_questions += int(has_adjacent)
        elsewhere_questions += int(has_elsewhere)

    total = sum(1 for q in questions if q.gold_chunk_ids)
    return {
        "threshold": threshold,
        "n_gold_questions": total,
        "questions_with_adjacent_restatement": adjacent_questions,
        "questions_with_restatement_elsewhere": elsewhere_questions,
        "examples": examples,
    }


def numeric_coverage(questions: list[Question]) -> dict:
    """How many numeric golds are present in the indexed text of their filing.

    `figure_in_indexed_text` is set at question-build time by searching the
    whole filing, so this only tallies it.
    """
    numeric = [q for q in questions if q.kind == KIND_NUMERIC]
    present = [q for q in numeric if q.figure_in_indexed_text]
    absent = [q for q in numeric if not q.figure_in_indexed_text]
    return {
        "n_numeric": len(numeric),
        "n_present_in_text": len(present),
        "n_absent_from_text": len(absent),
        "absent_question_ids": [q.question_id for q in absent],
        "absent_detail": [
            {
                "question_id": q.question_id,
                "company": q.companies[0] if q.companies else None,
                "concept": q.gold_concepts[0] if q.gold_concepts else None,
            }
            for q in absent
        ],
    }


def measure(
    *,
    db_path: Path,
    index_root: Path,
    questions_path: Path,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    arms: tuple[str, ...] = (ARM_BM25, ARM_DENSE, ARM_RRF),
) -> dict:
    """Run the whole zero-cost measurement and return it as plain data."""
    questions = load_questions(questions_path)
    targets = gold_questions(questions)
    con = connect(db_path)
    searcher = load_searcher(con, index_root=index_root)

    outcomes: list[QueryOutcome] = []
    for arm in arms:
        for filtered in (True, False):
            outcomes.extend(
                run_arm(searcher, targets, arm=arm, filtered=filtered, k=max(k_values))
            )

    results = [
        summarise_arm(outcomes, arm=arm, filtered=filtered, k_values=k_values)
        for arm in arms
        for filtered in (True, False)
    ]

    def rendered(result: ArmResult) -> dict:
        return {
            "arm": result.arm,
            "filtered": result.filtered,
            "n_questions": result.n_questions,
            "recall_at_k": {
                str(k): {
                    "value": float(np.mean(values)),
                    "ci": list(_ci(values)),
                }
                for k, values in result.recall_at_k.items()
            },
            "gold_coverage_at_k": {
                str(k): {
                    "value": float(np.mean(values)),
                    "ci": list(_ci(values)),
                }
                for k, values in result.coverage_at_k.items()
            },
            "mrr": result.mrr,
            "mrr_ci": list(result.mrr_ci),
            "mean_latency_ms": result.mean_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
        }

    return {
        "questions_path": str(questions_path),
        "questions_sha256": freeze_hash(questions_path),
        "n_gold_questions": len(targets),
        "k_values": list(k_values),
        "arms": [rendered(result) for result in results],
        "numeric_coverage": numeric_coverage(questions),
        "alternative_support": alternative_support(con, targets),
        "per_query": [asdict(outcome) for outcome in outcomes],
    }


def render_markdown(payload: dict) -> str:
    """The retrieval-ceiling tables, in the form the README carries them."""
    lines: list[str] = []
    lines.append(
        f"Gold passages exist for {payload['n_gold_questions']} of the sixty "
        "questions: the 25 qualitative and 5 multi-filing ones. Numeric golds "
        "are XBRL values rather than passages, and unanswerable questions have "
        "no gold passage by construction."
    )
    lines.append("")
    k_values = payload["k_values"]

    for filtered in (True, False):
        label = (
            "with the question's company and form filters applied"
            if filtered
            else "unfiltered, over all 148,097 chunks"
        )
        lines.append(f"### Recall@k, {label}")
        lines.append("")
        header = ["arm"] + [f"R@{k}" for k in k_values] + ["MRR"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for arm in payload["arms"]:
            if arm["filtered"] != filtered:
                continue
            cells = [arm["arm"]]
            for k in k_values:
                entry = arm["recall_at_k"][str(k)]
                low, high = entry["ci"]
                cells.append(
                    f"{entry['value'] * 100:.1f}% [{low * 100:.0f}, {high * 100:.0f}]"
                )
            cells.append(f"{arm['mrr']:.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("### Read this before comparing the arms")
    lines.append("")
    lines.append(
        "These numbers compare BM25, dense and fusion at finding a passage, "
        "and they favour BM25 for a reason built into how the gold set was "
        "made. Each gold passage was located by substring scan for a "
        "distinctive phrase, and each question was then drafted from that "
        "passage, so question and passage share vocabulary by construction. "
        "That is what a lexical ranker is good at. The comparison is sound as "
        "a ceiling on citation recall for these questions -- which is what it "
        "is here for -- and unsound as a general claim that BM25 retrieves "
        "better than the dense arm on this corpus. The retrieval comparison "
        "that would settle that needs the human relevance judgments under "
        "`data/qrels/`, which are not written."
    )
    lines.append("")
    lines.append(
        "The filtered and unfiltered tables differ by much more than the arms "
        "do. Scoping to the right company and form is worth more than the "
        "choice of ranker, which is the useful finding for an agent: getting "
        "the filters right matters more than anything it can do with the query."
    )
    lines.append("")

    lines.append("### Query latency")
    lines.append("")
    lines.append(
        "End to end for one `search` tool call: ranking, the metadata filter "
        "over the over-fetched pool, and loading the returned passages from "
        "DuckDB. BM25 ranking alone is about 3 ms at this pool depth, so most "
        "of what follows is the database work the tool wraps around it, not "
        "the ranker."
    )
    lines.append("")
    lines.append("| arm | mean | p95 |")
    lines.append("|---|---|---|")
    for arm in payload["arms"]:
        if not arm["filtered"]:
            continue
        lines.append(
            f"| {arm['arm']} | {arm['mean_latency_ms']:.1f} ms | "
            f"{arm['p95_latency_ms']:.1f} ms |"
        )
    lines.append("")

    alternative = payload.get("alternative_support")
    if alternative:
        lines.append("### Why citation recall is a lower bound")
        lines.append("")
        lines.append(
            "A gold passage set is the passages an answer was drafted from, "
            "found by one phrase. Filings restate the same fact in more than "
            "one indexed item, so a system can cite a passage that fully "
            "supports its claim and still be counted as a recall miss because "
            "that passage is not the one the drafter used. Measured at a "
            f"content-word Jaccard of {alternative['threshold']}, over "
            f"{alternative['n_gold_questions']} gold questions:"
        )
        lines.append("")
        lines.append(
            f"- {alternative['questions_with_adjacent_restatement']} have a "
            "near-duplicate passage adjacent to a gold passage. This is an "
            "artefact of chunking: 4-sentence windows at stride 2 share two "
            "sentences with each neighbour."
        )
        lines.append(
            f"- {alternative['questions_with_restatement_elsewhere']} have a "
            "near-duplicate passage in a different section, or in the other "
            "filing of a multi-filing question. These are genuine second "
            "disclosures and they are what make this metric a lower bound."
        )
        if alternative["examples"]:
            example = alternative["examples"][0]
            lines.append("")
            lines.append(
                f"Example found by this check: on `{example['question_id']}` the "
                f"gold passage is item {example['gold_item']} "
                f"(`{example['gold_chunk_id']}`), and "
                f"`{example['restating_chunk_id']}` (item "
                f"{example['restating_item']}) states the same thing."
            )
        lines.append("")
        lines.append(
            "The count above is itself a lower bound, because a word-overlap "
            "threshold only catches restatements that reuse the wording. "
            "Question `q08` is a case it misses: the gold passage is Microchip's "
            "Item 1 distributor discussion, and `0000827054-25-000077#7#chunk94` "
            "in MD&A gives the same two percentages and the same Arrow "
            "Electronics concentration in different words, which puts its "
            "overlap under the threshold. It is also the passage the search "
            "tool ranks first for that question, so the most likely citation a "
            "system would make there is correct and would score as a recall "
            "miss. See `example_retrieval.md`."
        )
        lines.append("")
        lines.append(
            "Citation support compares a claim against whatever passage was "
            "actually cited, so it does not have this problem. Read it, not "
            "citation recall, as the grounding measure."
        )
        lines.append("")

    coverage = payload["numeric_coverage"]
    lines.append("### Numeric gold coverage")
    lines.append("")
    lines.append(
        f"{coverage['n_present_in_text']} of {coverage['n_numeric']} numeric gold "
        "values appear in the indexed text of the filing that reported them. "
        f"The {coverage['n_absent_from_text']} that do not "
        f"({', '.join(coverage['absent_question_ids']) or 'none'}) cannot be "
        "answered from retrieval at any k, and are reported separately in the "
        "numeric accuracy row."
    )
    return "\n".join(lines)


def write(payload: dict, *, results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "retrieval_ceiling.json"
    md_path = results_dir / "retrieval_ceiling.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(payload) + "\n")
    return json_path, md_path
