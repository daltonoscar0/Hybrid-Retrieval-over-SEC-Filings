"""The `ticker` command line: ask a question, or run the evaluation.

One entry point per thing a reader might want to do, and every subcommand that
can spend money says what it will cost before it does. `--offline` turns a
cache miss into an error, so a re-run of a completed evaluation can be proven
to have sent nothing.

The existing per-stage scripts under `scripts/` are untouched. They are the
corpus and index pipeline; this is the query and evaluation surface on top of
the artefacts they produce.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from ticker.cited.config import PRESETS, Config, price_usd

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
OFF = "\033[0m"


def _config_from_args(args: argparse.Namespace) -> Config:
    config = PRESETS[getattr(args, "preset", "default")]
    config = replace(config, **{
        key: value
        for key, value in {
            "answer_model": getattr(args, "answer_model", None),
            "judge_model": getattr(args, "judge_model", None),
            "max_tool_calls": getattr(args, "max_tool_calls", None),
            "offline": getattr(args, "offline", None) or None,
            "questions_path": getattr(args, "questions", None),
            "results_dir": getattr(args, "results_dir", None),
            "db_path": getattr(args, "db", None),
            "index_root": getattr(args, "index_root", None),
        }.items()
        if value is not None
    })
    return config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=sorted(PRESETS), default="default",
                        help="default uses the stronger models; cheap uses Haiku to answer")
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--index-root", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--offline", action="store_true",
                        help="fail on a cache miss instead of calling the API")


# ---------------------------------------------------------------- ask

def cmd_ask(args: argparse.Namespace) -> int:
    from ticker.cited.agent import answer_question
    from ticker.cited.corpus import connect, load_chunks
    from ticker.cited.llm import CachedClient
    from ticker.cited.questions import KIND_QUALITATIVE, Question
    from ticker.cited.retrieval import load_searcher

    config = _config_from_args(args)
    question = Question(
        question_id="ask",
        kind=KIND_QUALITATIVE,
        text=args.question,
        gold_answer="",
        gold_source="filing_text",
        answerable=True,
        rationale="ad hoc question, no gold",
        companies=(args.company.upper(),) if args.company else (),
        form_type=args.form,
        gold_chunk_ids=("placeholder",),
    )

    con = connect(config.db_path)
    searcher = load_searcher(con, index_root=config.index_root)
    client = CachedClient(config)
    answer = answer_question(
        question, con=con, searcher=searcher, client=client, config=config
    )

    print()
    print(f"{BOLD}{CYAN}{args.question}{OFF}")
    print(f"{DIM}{config.answer_model}, {len(answer.tool_calls)} tool calls"
          f"{', hit the cap' if answer.hit_tool_cap else ''}{OFF}")
    print()

    if answer.error:
        print(f"{YELLOW}error: {answer.error}{OFF}")
    if answer.declined:
        print("Not answerable from the indexed filings.")
        if answer.summary:
            print(f"  {answer.summary}")
    else:
        if answer.summary:
            print(answer.summary)
            print()
        records = load_chunks(con, list(answer.cited_chunk_ids))
        for index, claim in enumerate(answer.claims, start=1):
            print(f"{BOLD}{index}.{OFF} {claim.text}")
            if not claim.chunk_ids:
                print(f"   {YELLOW}no citation{OFF}")
            for chunk_id in claim.chunk_ids:
                record = records.get(chunk_id)
                if record is None:
                    continue
                print(f"   {DIM}{record.citation()}{OFF}")
                print(f"   {DIM}[{chunk_id}]{OFF}")
                for line in _wrap(record.text, 92):
                    print(f"     {line}")
            print()

    if answer.rejected_citations:
        print(f"{YELLOW}rejected {len(answer.rejected_citations)} citation(s) to "
              f"chunks never retrieved: {', '.join(answer.rejected_citations)}{OFF}")
    cost = f"${answer.cost_usd:.4f}" if answer.cost_usd is not None else "unknown"
    print(f"{DIM}cost {cost}, {answer.latency_s:.1f}s"
          f"{', served from cache' if answer.from_cache else ''}{OFF}")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# ------------------------------------------------- eval build-questions

def cmd_build_questions(args: argparse.Namespace) -> int:
    from ticker.cited.build import build_all
    from ticker.cited.corpus import connect
    from ticker.cited.questions import (
        composition,
        freeze_hash,
        load_questions,
        validate_composition,
        write_questions,
    )

    config = _config_from_args(args)
    con = connect(config.db_path)

    if args.verify:
        if not config.questions_path.exists():
            print(f"no question file at {config.questions_path}", file=sys.stderr)
            return 1
        existing = load_questions(config.questions_path)
        before = freeze_hash(config.questions_path)
        rebuilt = build_all(con)
        validate_composition(rebuilt)
        existing_map = {q.question_id: q for q in existing}
        drift = [
            q.question_id
            for q in rebuilt
            if q.question_id not in existing_map
            or existing_map[q.question_id].gold_chunk_ids != q.gold_chunk_ids
            or existing_map[q.question_id].gold_value != q.gold_value
        ]
        print(f"questions: {len(existing)}  sha256: {before}")
        if drift:
            print(f"{YELLOW}drift in {len(drift)} question(s): "
                  f"{', '.join(drift)}{OFF}", file=sys.stderr)
            return 1
        print("every gold value and gold chunk set re-derives from its recorded "
              "locator and accession")
        return 0

    if config.questions_path.exists() and not args.force:
        print(f"{config.questions_path} exists and is frozen. Its SHA-256 is "
              f"recorded in the results, so rebuilding it invalidates them.\n"
              f"Use --verify to check it still re-derives, or --force to rebuild.",
              file=sys.stderr)
        return 1

    questions = build_all(con)
    validate_composition(questions)
    count = write_questions(config.questions_path, questions)
    print(f"wrote {count} questions to {config.questions_path}")
    print(f"composition: {composition(questions)}")
    print(f"sha256: {freeze_hash(config.questions_path)}")
    return 0


# ----------------------------------------------------------- eval run

def _estimate_cost(config: Config, system: str, n_questions: int) -> tuple[float | None, str]:
    """Rough cost of a run, before making it.

    The per-question token figures are order-of-magnitude estimates from the
    prompt sizes, not measurements, and the estimate says so. Its job is to
    stop someone starting a run that costs ten times what they expected, not
    to predict the bill.
    """
    profiles = {
        "closed": (700, 700),
        "single": (3_000, 800),
        "agent": (55_000, 3_000),
    }
    per_in, per_out = profiles[system]
    per_question = price_usd(
        config.answer_model, input_tokens=per_in, output_tokens=per_out
    )
    if per_question is None:
        return None, f"no published price for {config.answer_model}"
    total = per_question * n_questions
    return total, (
        f"~${total:.2f} for {n_questions} questions on {config.answer_model} "
        f"(~{per_in:,} in / {per_out:,} out per question, estimated)"
    )


def cmd_run(args: argparse.Namespace) -> int:
    from ticker.cited.questions import load_questions
    from ticker.cited.run import run_system

    config = _config_from_args(args)
    questions = load_questions(config.questions_path)
    total, description = _estimate_cost(config, args.system, len(questions))
    print(f"estimated cost: {description}")
    if config.offline:
        print("offline: a cache miss will error rather than spend anything")
    elif not args.yes:
        reply = input("proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("aborted")
            return 1

    outcome = run_system(
        args.system,
        config=config,
        questions=questions,
        progress=lambda message: print(f"{DIM}{message}{OFF}", file=sys.stderr),
    )
    spend = "unknown" if outcome.spend_usd is None else f"${outcome.spend_usd:.4f}"
    print(f"wrote {len(outcome.answers)} answers to {outcome.path}")
    print(f"spend {spend} over {outcome.calls} calls "
          f"({outcome.cache_hits} served from cache)")
    errors = [a for a in outcome.answers if a.error]
    if errors:
        print(f"{YELLOW}{len(errors)} question(s) errored: "
              f"{', '.join(a.question_id for a in errors)}{OFF}")
    return 0


# --------------------------------------------------------- eval score

def cmd_score(args: argparse.Namespace) -> int:
    from ticker.cited.run import report_path, score_all, scores_path

    config = _config_from_args(args)
    scores, _payload = score_all(
        config=config,
        progress=lambda message: print(f"{DIM}{message}{OFF}", file=sys.stderr),
    )
    print(f"scored {len(scores)} answers to {scores_path(config)}")
    print(f"report written to {report_path(config)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from ticker.cited.run import report_path

    config = _config_from_args(args)
    path = report_path(config)
    if not path.exists():
        print(f"no report at {path}. Run `ticker eval score` first.", file=sys.stderr)
        return 1
    print(path.read_text())
    return 0


# ---------------------------------------------- eval retrieval-ceiling

def cmd_retrieval_ceiling(args: argparse.Namespace) -> int:
    from ticker.cited import retrieval_eval

    config = _config_from_args(args)
    payload = retrieval_eval.measure(
        db_path=config.db_path,
        index_root=config.index_root,
        questions_path=config.questions_path,
    )
    json_path, md_path = retrieval_eval.write(payload, results_dir=config.results_dir)
    print(retrieval_eval.render_markdown(payload))
    print()
    print(f"{DIM}wrote {json_path} and {md_path}{OFF}")
    return 0


# ------------------------------------------------ eval judge-agreement

def cmd_judge_sample(args: argparse.Namespace) -> int:
    from ticker.cited import agreement
    from ticker.cited.corpus import connect
    from ticker.cited.report import load_scores
    from ticker.cited.run import scores_path

    config = _config_from_args(args)
    path = scores_path(config)
    if not path.exists():
        print(f"no scores at {path}. Run `ticker eval score` first.", file=sys.stderr)
        return 1

    con = connect(config.db_path)
    scores = load_scores(path)
    pairs = agreement.sample_pairs(
        agreement.eligible_pairs(scores, con=con), size=args.size, seed=args.seed
    )
    if not pairs:
        print("no judge-rated claim-citation pairs to sample. Either no answers "
              "have been scored, or every claim was checked deterministically.",
              file=sys.stderr)
        return 1

    agreement.write_sample(agreement.DEFAULT_SAMPLE_PATH, pairs)
    agreement.write_judge_verdicts(agreement.DEFAULT_VERDICTS_PATH, scores, pairs)
    print(f"wrote {len(pairs)} blind pairs to {agreement.DEFAULT_SAMPLE_PATH}")
    print(f"judge verdicts for the same pairs: {agreement.DEFAULT_VERDICTS_PATH}")
    print()
    print("Label each pair in the blind file and write your labels to "
          f"{agreement.DEFAULT_HUMAN_LABELS_PATH} as one JSON object per line:")
    print('  {"pair_id": "agent:q01:0", "label": "supported"}')
    print(f"Options: {', '.join(agreement.SUPPORT_LABELS)}. The rubric is in "
          "src/ticker/cited/judge.py.")
    print("Then run `ticker eval judge-agreement`.")
    return 0


def cmd_judge_agreement(args: argparse.Namespace) -> int:
    from ticker.cited import agreement

    config = _config_from_args(args)
    judge_labels = agreement.load_labels(agreement.DEFAULT_VERDICTS_PATH)
    human_labels = agreement.load_labels(agreement.DEFAULT_HUMAN_LABELS_PATH)
    if not judge_labels:
        print(f"no judge verdicts at {agreement.DEFAULT_VERDICTS_PATH}. "
              "Run `ticker eval judge-sample` first.", file=sys.stderr)
        return 1

    report = agreement.compare(
        judge_labels=judge_labels, human_labels=human_labels, seed=args.seed
    )
    rendered = agreement.render_markdown(report)
    print(rendered)
    out = config.results_dir / "judge_agreement.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered + "\n")
    (config.results_dir / "judge_agreement.json").write_text(
        json.dumps(
            {
                "n_pairs": report.n_pairs,
                "n_labelled": report.n_labelled,
                "raw_agreement": report.raw_agreement,
                "kappa": report.kappa,
                "rubric_version": report.rubric_version,
                "seed": report.seed,
                "confusion": report.confusion,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0 if report.n_labelled and report.is_acceptable else 1


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ticker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer one question with citations")
    ask.add_argument("question")
    ask.add_argument("--company", default=None, help="ticker to scope to, e.g. NVDA")
    ask.add_argument("--form", default=None, choices=["10-K", "10-Q", "8-K"])
    ask.add_argument("--max-tool-calls", type=int, default=None)
    _add_common(ask)
    ask.set_defaults(func=cmd_ask)

    evaluation = sub.add_parser("eval", help="build, run and score the evaluation")
    evaluation_sub = evaluation.add_subparsers(dest="eval_command", required=True)

    build = evaluation_sub.add_parser(
        "build-questions", help="build or verify the frozen question set"
    )
    build.add_argument("--verify", action="store_true",
                       help="check the existing file still re-derives, and exit")
    build.add_argument("--force", action="store_true",
                       help="overwrite a frozen question file")
    _add_common(build)
    build.set_defaults(func=cmd_build_questions)

    run = evaluation_sub.add_parser("run", help="answer the question set with one system")
    run.add_argument("--system", required=True, choices=["closed", "single", "agent"])
    run.add_argument("--max-tool-calls", type=int, default=None)
    run.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    _add_common(run)
    run.set_defaults(func=cmd_run)

    score = evaluation_sub.add_parser("score", help="score every answer file present")
    _add_common(score)
    score.set_defaults(func=cmd_score)

    report = evaluation_sub.add_parser("report", help="print the scored report")
    _add_common(report)
    report.set_defaults(func=cmd_report)

    ceiling = evaluation_sub.add_parser(
        "retrieval-ceiling",
        help="measure recall@k against the gold passages; makes no model calls",
    )
    _add_common(ceiling)
    ceiling.set_defaults(func=cmd_retrieval_ceiling)

    sample = evaluation_sub.add_parser(
        "judge-sample", help="draw a blind claim-citation sample for human labelling"
    )
    sample.add_argument("--size", type=int, default=40)
    sample.add_argument("--seed", type=int, default=20260101)
    _add_common(sample)
    sample.set_defaults(func=cmd_judge_sample)

    agree = evaluation_sub.add_parser(
        "judge-agreement", help="compare the judge against the human labels"
    )
    agree.add_argument("--seed", type=int, default=20260101)
    _add_common(agree)
    agree.set_defaults(func=cmd_judge_agreement)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
