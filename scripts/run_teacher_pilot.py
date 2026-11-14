#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Run the 150-example teacher pilot (F1.3).

Reads the F1.2 artifacts under B:\\p8\\data\\work\\, builds case-A contexts
with gold ensured, and appends one teacher answer per sampled train question.
Sol runs this detached; the pilot does not train anything.

    python scripts/run_teacher_pilot.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: E402, F401  (installs the docmind_service alias)
from docmind_ft.build_data import DATA_DIR, WORK_DIR  # noqa: E402
from docmind_ft.qasper import Answer, load_qasper  # noqa: E402
from docmind_ft.retrieval import TOP_K, Retrieved  # noqa: E402
from docmind_ft.teacher import (  # noqa: E402
    OllamaTeacher,
    TeacherError,
    build_teacher_prompt,
    ensure_gold_context,
    expected_yes_no,
    human_answer,
    reference_answers,
    verify_answer,
)
from docmind_service.domain import RetrievedChunk  # noqa: E402

PILOT_SIZE = 150
SEED = 20261207
QUESTIONS_PATH = WORK_DIR / "questions.jsonl"
CHUNKS_PATH = WORK_DIR / "chunks.jsonl"
OUT_PATH = WORK_DIR / "pilot_out.jsonl"
TRAIN_PATH = DATA_DIR / "qasper-train-v0.3.json"


class PilotError(RuntimeError):
    """The pilot's inputs are missing or inconsistent."""


def _rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PilotError(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def _require_inputs() -> None:
    missing = [str(path) for path in (QUESTIONS_PATH, CHUNKS_PATH, TRAIN_PATH) if not path.exists()]
    if missing:
        raise PilotError(
            "missing input(s): "
            + ", ".join(missing)
            + "\nrun `python -m docmind_ft.build_data` first (and fetch QASPER with "
            "`python scripts/fetch_qasper.py` if needed)"
        )


def _answers_by_question() -> dict[str, tuple[Answer, ...]]:
    answers = {}
    for paper in load_qasper(TRAIN_PATH):
        for question in paper.questions:
            answers[question.question_id] = question.answers
    return answers


def _context(
    row: dict,
    chunks_by_id: dict[str, dict],
) -> tuple[list[Retrieved], list[RetrievedChunk]]:
    question_id = row["question_id"]
    retrieved = [Retrieved(item["chunk_id"], float(item["score"])) for item in row["retrieved"]]
    gold_scores = row.get("gold_scores") or {}
    gold = []
    for chunk_id in row["gold_chunk_ids"]:
        score = gold_scores.get(chunk_id)
        if score is None:
            score = next(
                (item.score for item in retrieved if item.chunk_id == chunk_id),
                None,
            )
        if score is None:
            raise PilotError(
                f"{question_id}: no score for gold chunk {chunk_id}; "
                "rerun `python -m docmind_ft.build_data` to write gold_scores"
            )
        gold.append(Retrieved(chunk_id, float(score)))

    ranked = ensure_gold_context(retrieved, gold, TOP_K)
    prompt_chunks = []
    for item in ranked:
        chunk = chunks_by_id.get(item.chunk_id)
        if chunk is None:
            raise PilotError(f"{question_id}: unknown retrieved chunk {item.chunk_id}")
        prompt_chunks.append(
            RetrievedChunk(
                chunk_id=item.chunk_id,
                document_id=uuid5(NAMESPACE_URL, chunk["paper_id"]),
                page=int(chunk["page"]),
                text=chunk["text"],
                score=item.score,
            )
        )
    return ranked, prompt_chunks


def _run_one(
    row: dict,
    answers: Sequence[Answer],
    chunks_by_id: dict[str, dict],
    teacher: OllamaTeacher,
) -> dict:
    question_id = row["question_id"]
    reference = human_answer(answers)
    if reference is None:
        raise PilotError(f"{question_id}: no answering annotator")
    contexts, prompt_chunks = _context(row, chunks_by_id)
    canary = f"P8-{uuid4().hex[:12]}"
    prompt, mapping = build_teacher_prompt(
        row["question"],
        prompt_chunks,
        reference,
        row["gold_chunk_ids"],
        canary,
    )
    teacher_answer = teacher.answer(prompt)
    gold = set(row["gold_chunk_ids"])
    result = verify_answer(
        teacher_answer,
        prompt_chunk_ids=tuple(mapping),
        gold_chunk_ids=tuple(short for short, chunk in mapping.items() if chunk.chunk_id in gold),
        expected_yes_no=expected_yes_no(answers),
        reference_answers=reference_answers(answers),
        chunk_texts=[chunk.text for chunk in prompt_chunks],
        canary=canary,
    )
    return {
        "question_id": question_id,
        "teacher_answer": teacher_answer,
        "passed": result.passed,
        "failed_rule": result.failed_rule,
        "context_chunk_ids": [item.chunk_id for item in contexts],
    }


def _print_summary(rows: Sequence[dict], sample_ids: set[str]) -> None:
    selected = [row for row in rows if row["question_id"] in sample_ids]
    accepted = sum(bool(row["passed"]) for row in selected)
    failures = Counter(row["failed_rule"] for row in selected if not row["passed"])
    print(f"pass rate: {accepted}/{PILOT_SIZE} ({accepted / PILOT_SIZE:.1%})")
    print(
        "fail counts: "
        + " ".join(f"{rule}={failures.get(rule, 0)}" for rule in ("V1", "V2", "V3", "V4"))
    )
    print(f"processed: {len(selected)}/{PILOT_SIZE}")


def main() -> int:
    try:
        _require_inputs()
        questions = _rows(QUESTIONS_PATH)
        chunks = _rows(CHUNKS_PATH)
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        answers_by_id = _answers_by_question()
        candidates = [
            row
            for row in questions
            if row.get("split") == "train"
            and row.get("answerable") is True
            and row.get("gold_chunk_ids")
        ]
        if len(candidates) < PILOT_SIZE:
            raise PilotError(
                f"only {len(candidates)} answerable train questions with gold; need {PILOT_SIZE}"
            )
        candidates.sort(key=lambda row: row["question_id"])
        sampled = random.Random(SEED).sample(candidates, PILOT_SIZE)
        sample_ids = {row["question_id"] for row in sampled}
        print(f"seed {SEED}; sampled {len(sampled)} questions")

        existing = _rows(OUT_PATH) if OUT_PATH.exists() else []
        done = {row["question_id"] for row in existing}
        if done:
            print(f"resuming: {len(done & sample_ids)} sampled rows already in {OUT_PATH}")
        teacher = OllamaTeacher()
        with OUT_PATH.open("a", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(sampled, start=1):
                question_id = row["question_id"]
                if question_id in done:
                    continue
                answers = answers_by_id.get(question_id)
                if answers is None:
                    raise PilotError(f"{question_id}: not found in {TRAIN_PATH}")
                result = _run_one(row, answers, chunks_by_id, teacher)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                done.add(question_id)
                verdict = "pass" if result["passed"] else f"fail {result['failed_rule']}"
                print(f"[{index}/{PILOT_SIZE}] {question_id}: {verdict}", flush=True)

        _print_summary(_rows(OUT_PATH), sample_ids)
        print(f"output: {OUT_PATH}")
        return 0
    except (PilotError, TeacherError, OSError, KeyError, ValueError) as exc:
        print(f"pilot FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
