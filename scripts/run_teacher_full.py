#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Run the full F1.4 teacher build for train-A and train-U questions.

Reads the calibrated ``p0.json`` and appends one row per question to
``B:\\p8\\data\\work\\teacher_out.jsonl``.  A questions are re-run even if they
appeared in the F1.3 pilot (fresh canaries).  U questions never call the
teacher.  Sol runs this detached.

    python scripts/run_teacher_full.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: E402, F401  (installs the docmind_service alias)
from docmind_ft.teacher import OllamaTeacher  # noqa: E402
from docmind_ft.teacher_build import (  # noqa: E402
    CASE_A,
    CASE_EXCLUDED,
    CASE_U,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    P0_PATH,
    QUESTIONS_PATH,
    QASPER_TRAIN_PATH,
    TEACHER_OUT_PATH,
    QuestionEmbedder,
    assign_case,
    branch_p0,
    case_counts,
    load_chunk_store,
    load_embeddings,
    load_train_answers,
    load_train_questions,
    new_canary,
    process_question,
    read_jsonl,
    summarize_teacher_rows,
)


def _require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing {path}")


def _append(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> int:
    if not P0_PATH.exists():
        raise SystemExit(f"missing {P0_PATH}; run `python scripts/calibrate_p0.py` first")
    p0_payload = json.loads(P0_PATH.read_text(encoding="utf-8"))
    p0 = float(p0_payload["p0"])
    if not 0.0 <= p0 <= 1.0:
        raise SystemExit(f"invalid p0={p0} in {P0_PATH}")

    for path in (QUESTIONS_PATH, CHUNKS_PATH, QASPER_TRAIN_PATH):
        _require(path)

    questions = load_train_questions(QUESTIONS_PATH)
    counts = case_counts(questions)
    a_questions = sorted(
        (question for question in questions if assign_case(question) == CASE_A),
        key=lambda question: question.question_id,
    )
    u_questions = sorted(
        (question for question in questions if assign_case(question) == CASE_U),
        key=lambda question: question.question_id,
    )
    ordered = sorted(a_questions + u_questions, key=lambda question: question.question_id)
    if not ordered:
        raise SystemExit("no train A/U questions found")

    existing = read_jsonl(TEACHER_OUT_PATH) if TEACHER_OUT_PATH.exists() else []
    existing_ids = [str(row["question_id"]) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise SystemExit(f"duplicate question_id in {TEACHER_OUT_PATH}")
    seen = set(existing_ids)
    pending = [question for question in ordered if question.question_id not in seen]
    if not pending:
        print("nothing to do; all train A/U rows are already present")

    store = load_chunk_store(CHUNKS_PATH)
    answers_by_id = load_train_answers(QASPER_TRAIN_PATH)
    pending_p0 = [
        question
        for question in pending
        if assign_case(question) == CASE_A and branch_p0(question.question_id, p0)
    ]
    embeddings = (
        load_embeddings(EMBEDDINGS_PATH, expected_rows=len(store.ids)) if pending_p0 else None
    )
    embedder = QuestionEmbedder() if pending_p0 else None
    teacher = OllamaTeacher()

    total_prompts = len(a_questions)
    prompts_done = sum(row.get("case") == CASE_A for row in existing)
    new_prompts = 0
    started = time.monotonic()
    for index, question in enumerate(pending, start=1):
        row = process_question(
            question,
            answers_by_id[question.question_id],
            store,
            teacher,
            p0=p0,
            canary=new_canary(),
            embeddings=embeddings,
            embed_question=embedder.embed if embedder is not None else None,
        )
        _append(TEACHER_OUT_PATH, row)
        seen.add(question.question_id)
        if assign_case(question) == CASE_A:
            prompts_done += 1
            new_prompts += 1
            if new_prompts % 50 == 0:
                elapsed = time.monotonic() - started
                remaining = max(total_prompts - prompts_done, 0)
                eta = elapsed / new_prompts * remaining if new_prompts else 0.0
                print(
                    f"prompts {prompts_done}/{total_prompts} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        if index % 100 == 0:
            print(f"rows {index}/{len(pending)}", flush=True)

    rows = read_jsonl(TEACHER_OUT_PATH)
    summary = summarize_teacher_rows(rows)
    summary["excluded"] = counts.get(CASE_EXCLUDED, 0)
    summary["expected_rows"] = counts.get(CASE_A, 0) + counts.get(CASE_U, 0)
    summary["p0_calibration"] = p0
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
