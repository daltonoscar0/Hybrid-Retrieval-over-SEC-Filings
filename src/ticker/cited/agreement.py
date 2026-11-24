"""Checking the citation judge against a human, before trusting its numbers.

Every judge-based number in the results depends on the citation judge being
roughly right, and nothing in the judge's own output can establish that. So a
sample of claim-citation pairs is drawn, labelled by a human against the same
rubric and the same two fields the judge saw, and the two label sets are
compared by raw agreement and Cohen's kappa.

Two files, deliberately:

`judge_sample.jsonl`
    The blind sample. Pair id, claim text, cited passage, and the rubric's
    label options. **It does not contain the judge's verdict.** Showing the
    verdict while labelling would anchor the human to it and the agreement
    number would measure compliance rather than agreement.

`judge_sample_verdicts.jsonl`
    The judge's labels for the same pair ids, written at the same time so the
    comparison cannot be accused of having been drawn after the fact.

`eval/human_labels.jsonl` is written by a person, never by this code. It is
listed in the write guard alongside `data/qrels/` for the same reason the
relevance judgments are: a model-generated label here would make the judge's
agreement a measurement of the model against itself. This module builds the
sample and computes the statistic; producing the labels is the human's job.

Sampling is stratified by system and seeded. Stratified because the systems
produce different kinds of citation -- the closed-book arm cites nothing it
retrieved, the agent cites passages it chose to read -- and an unstratified
draw would under-sample whichever arm made fewer cited claims, which is
usually the one whose citations are most suspect. Seeded so the sample is
reproducible from the recorded seed.

Only judge-rated pairs are eligible. A claim checked by the deterministic
number-in-passage rule has no judge verdict to agree or disagree with, and
including those would inflate agreement with cases the judge never saw.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from ticker.agreement import cohens_kappa
from ticker.cited.corpus import load_chunks
from ticker.cited.judge import RUBRIC_VERSION, SUPPORT_LABELS
from ticker.cited.scoring import METHOD_JUDGE, AnswerScore

DEFAULT_SAMPLE_SIZE = 40
DEFAULT_SEED = 20260101

DEFAULT_SAMPLE_PATH = Path("eval/judge_sample.jsonl")
DEFAULT_VERDICTS_PATH = Path("eval/judge_sample_verdicts.jsonl")
DEFAULT_HUMAN_LABELS_PATH = Path("eval/human_labels.jsonl")


@dataclass(frozen=True, slots=True)
class JudgePair:
    """One claim-citation pair, with everything a labeller needs and no more."""

    pair_id: str
    question_id: str
    system: str
    claim_index: int
    claim_text: str
    cited_chunk_ids: tuple[str, ...]
    cited_text: str


@dataclass(frozen=True, slots=True)
class AgreementReport:
    n_pairs: int
    n_labelled: int
    raw_agreement: float | None
    kappa: float | None
    rubric_version: str
    seed: int
    confusion: dict[str, dict[str, int]]
    missing_pair_ids: tuple[str, ...]
    unknown_pair_ids: tuple[str, ...]

    @property
    def is_acceptable(self) -> bool:
        """Whether the judge's numbers can be reported.

        The threshold is kappa >= 0.6, conventionally read as substantial
        agreement. Below that the rubric is rewritten and the check re-run
        before any judge-based figure is published; that sequencing is the
        point of running this at all.
        """
        return self.kappa is not None and self.kappa >= 0.6


def eligible_pairs(
    scores: Iterable[AnswerScore], *, con: duckdb.DuckDBPyConnection
) -> list[JudgePair]:
    """Every claim-citation pair the LLM judge actually rated."""
    pairs: list[JudgePair] = []
    for score in scores:
        for claim in score.claim_scores:
            if claim.method != METHOD_JUDGE or not claim.chunk_ids:
                continue
            records = load_chunks(con, list(claim.chunk_ids))
            cited_text = "\n\n".join(
                records[chunk_id].text
                for chunk_id in claim.chunk_ids
                if chunk_id in records
            )
            if not cited_text:
                continue
            pairs.append(
                JudgePair(
                    pair_id=f"{score.system}:{score.question_id}:{claim.index}",
                    question_id=score.question_id,
                    system=score.system,
                    claim_index=claim.index,
                    claim_text=claim.text,
                    cited_chunk_ids=claim.chunk_ids,
                    cited_text=cited_text,
                )
            )
    return pairs


def sample_pairs(
    pairs: list[JudgePair],
    *,
    size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[JudgePair]:
    """Draw `size` pairs, stratified by system, deterministically.

    Each system gets an equal share, and any shortfall from a system with too
    few eligible pairs is redistributed to the others rather than left as a
    short sample. A system contributing nothing is simply absent; the report
    records the realised composition.
    """
    if size <= 0:
        raise ValueError(f"sample size must be positive, got {size}")

    by_system: dict[str, list[JudgePair]] = {}
    for pair in sorted(pairs, key=lambda item: item.pair_id):
        by_system.setdefault(pair.system, []).append(pair)
    if not by_system:
        return []

    rng = random.Random(seed)
    quota = max(1, size // len(by_system))
    drawn: list[JudgePair] = []
    leftovers: list[JudgePair] = []

    for system in sorted(by_system):
        available = by_system[system][:]
        rng.shuffle(available)
        drawn.extend(available[:quota])
        leftovers.extend(available[quota:])

    rng.shuffle(leftovers)
    drawn.extend(leftovers[: max(0, size - len(drawn))])
    return sorted(drawn[:size], key=lambda item: item.pair_id)


def write_sample(
    path: Path,
    pairs: list[JudgePair],
    *,
    rubric_version: str = RUBRIC_VERSION,
) -> int:
    """Write the blind sample. Carries no judge verdict, by design."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for pair in pairs:
            payload = asdict(pair)
            payload["cited_chunk_ids"] = list(pair.cited_chunk_ids)
            payload["label_options"] = list(SUPPORT_LABELS)
            payload["rubric_version"] = rubric_version
            payload["label"] = None
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    return len(pairs)


def write_judge_verdicts(
    path: Path, scores: Iterable[AnswerScore], pairs: list[JudgePair]
) -> int:
    """Write the judge's labels for the sampled pairs, in a separate file."""
    wanted = {pair.pair_id for pair in pairs}
    rows: list[dict] = []
    for score in scores:
        for claim in score.claim_scores:
            pair_id = f"{score.system}:{score.question_id}:{claim.index}"
            if pair_id in wanted:
                rows.append(
                    {
                        "pair_id": pair_id,
                        "label": claim.support,
                        "reason": claim.reason,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in sorted(rows, key=lambda item: item["pair_id"]):
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return len(rows)


def load_labels(path: Path) -> dict[str, str]:
    """Read a `{pair_id: label}` mapping from a labels file.

    Last line wins for a repeated pair id, matching how `ticker.qrels` treats
    its append-only files: a correction is a new line, not an edit.
    """
    labels: dict[str, str] = {}
    if not path.exists():
        return labels
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label")
            if label:
                labels[row["pair_id"]] = label
    return labels


def compare(
    *,
    judge_labels: dict[str, str],
    human_labels: dict[str, str],
    seed: int = DEFAULT_SEED,
    rubric_version: str = RUBRIC_VERSION,
) -> AgreementReport:
    """Raw agreement and Cohen's kappa over the pairs both sides labelled."""
    shared = sorted(set(judge_labels) & set(human_labels))
    missing = tuple(sorted(set(judge_labels) - set(human_labels)))
    unknown = tuple(sorted(set(human_labels) - set(judge_labels)))

    confusion: dict[str, dict[str, int]] = {
        human: dict.fromkeys(SUPPORT_LABELS, 0) for human in SUPPORT_LABELS
    }
    for pair_id in shared:
        human = human_labels[pair_id]
        judge = judge_labels[pair_id]
        confusion.setdefault(human, dict.fromkeys(SUPPORT_LABELS, 0))
        confusion[human][judge] = confusion[human].get(judge, 0) + 1

    if not shared:
        return AgreementReport(
            n_pairs=len(judge_labels),
            n_labelled=0,
            raw_agreement=None,
            kappa=None,
            rubric_version=rubric_version,
            seed=seed,
            confusion=confusion,
            missing_pair_ids=missing,
            unknown_pair_ids=unknown,
        )

    human_sequence = [human_labels[pair_id] for pair_id in shared]
    judge_sequence = [judge_labels[pair_id] for pair_id in shared]
    agreement = sum(
        1 for a, b in zip(human_sequence, judge_sequence) if a == b
    ) / len(shared)

    return AgreementReport(
        n_pairs=len(judge_labels),
        n_labelled=len(shared),
        raw_agreement=agreement,
        kappa=cohens_kappa(human_sequence, judge_sequence),
        rubric_version=rubric_version,
        seed=seed,
        confusion=confusion,
        missing_pair_ids=missing,
        unknown_pair_ids=unknown,
    )


def render_markdown(report: AgreementReport) -> str:
    lines = [
        f"Sampled {report.n_pairs} claim-citation pairs (seed {report.seed}, "
        f"rubric version {report.rubric_version}).",
        "",
    ]
    if not report.n_labelled:
        lines.append(
            "No human labels found, so no agreement is reported and no "
            "judge-based number should be read as validated. Label "
            f"`{DEFAULT_SAMPLE_PATH}` into `{DEFAULT_HUMAN_LABELS_PATH}` and "
            "re-run `ticker eval judge-agreement`."
        )
        return "\n".join(lines)

    lines.append(f"Pairs labelled by both: {report.n_labelled}.")
    lines.append("")
    lines.append(f"- Raw agreement: {report.raw_agreement * 100:.1f}%")
    lines.append(f"- Cohen's kappa: {report.kappa:.3f}")
    lines.append("")
    lines.append(
        "Judge-based numbers are reportable."
        if report.is_acceptable
        else "Kappa is below 0.6. The rubric needs rewriting and this check "
        "re-running before any judge-based number is reported."
    )
    lines.append("")
    lines.append("Rows are the human label, columns the judge's.")
    lines.append("")
    header = ["human \\ judge"] + list(SUPPORT_LABELS)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for human in SUPPORT_LABELS:
        row = [human] + [
            str(report.confusion.get(human, {}).get(judge, 0))
            for judge in SUPPORT_LABELS
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
