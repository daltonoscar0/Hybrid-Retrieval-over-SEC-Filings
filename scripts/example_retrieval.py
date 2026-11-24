#!/usr/bin/env python3
"""Render a worked retrieval example for the README, from the real corpus.

The README needs to show what a cited answer looks like. No evaluation run has
been made, so there is no agent answer to show, and inventing one would put a
fabricated result in the README. What can be shown truthfully is the input: the
passages the `search` tool actually returns for a real question, with their
real chunk ids and real filing text, next to the gold answer that was written
from them in advance.

Usage:
  uv run python scripts/example_retrieval.py --question q08
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ticker.cited.corpus import connect  # noqa: E402
from ticker.cited.questions import load_questions  # noqa: E402
from ticker.cited.retrieval import load_searcher  # noqa: E402

DEFAULT_OUT = Path("results/cited_answers/example_retrieval.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="q08")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--db", type=Path, default=Path("data/ticker.duckdb"))
    parser.add_argument("--index-root", type=Path, default=Path("data/index"))
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    questions = {q.question_id: q for q in load_questions(args.questions)}
    question = questions[args.question]

    con = connect(args.db)
    searcher = load_searcher(con, index_root=args.index_root)
    result = searcher.search(
        question.text,
        company=question.company,
        form_type=question.form_type,
        fiscal_year=question.fiscal_year,
        k=args.k,
    )

    lines: list[str] = []
    lines.append(f"Question `{question.question_id}` ({question.kind}):")
    lines.append("")
    lines.append(f"> {question.text}")
    lines.append("")
    lines.append("Gold answer, written from the filing before any system ran:")
    lines.append("")
    lines.append(f"> {question.gold_answer}")
    lines.append("")
    lines.append(
        f"Gold passages, located by substring scan for `{question.gold_locator}`: "
        + ", ".join(f"`{chunk_id}`" for chunk_id in question.gold_chunk_ids)
    )
    lines.append("")
    lines.append(
        f"What the `search` tool returns for this question (hybrid, k={args.k}, "
        f"filtered to {question.company} {question.form_type}). A tick marks a "
        "passage that is in the gold set."
    )
    lines.append("")

    gold = set(question.gold_chunk_ids)
    for hit in result.hits:
        mark = " **gold**" if hit.chunk_id in gold else ""
        lines.append(f"{hit.rank}. `{hit.chunk_id}`{mark}")
        lines.append(f"   {hit.chunk.citation()}")
        lines.append("")
        lines.append(f"   > {hit.chunk.text}")
        lines.append("")

    lines.append(
        "An answer over these passages would carry one claim per assertion, "
        "each citing the chunk ids above, and would be rejected if it cited an "
        "id the session never retrieved. No such answer exists yet: no "
        "evaluation run has been made."
    )

    rendered = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered + "\n")
    print(rendered)
    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
