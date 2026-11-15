#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Calibrate the p0 gold-removed abstention branch (F1.4).

Samples 200 train-A questions not used by the F1.3 pilot, probes the local
teacher on a gold-free top-10 context without the hint, and writes
``B:\\p8\\data\\work\\p0.json``.  Sol runs this detached before the full build.

    python scripts/calibrate_p0.py
"""

from __future__ import annotations

import json
import random
import sys
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
    PILOT_OUT_PATH,
    PILOT_PASS_RATE,
    P0_PATH,
    PROBE_SAMPLE,
    QUESTIONS_PATH,
    QASPER_TRAIN_PATH,
    SEED,
    TRAIN_A,
    TRAIN_EXCLUDED,
    TRAIN_U,
    QuestionEmbedder,
    assign_case,
    case_counts,
    compute_p0,
    load_chunk_store,
    load_embeddings,
    load_pilot_ids,
    load_train_answers,
    load_train_questions,
    new_canary,
    probe_question,
)


def _require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing {path}; run the F1.2 build first")


def main() -> int:
    for path in (QUESTIONS_PATH, CHUNKS_PATH, EMBEDDINGS_PATH, QASPER_TRAIN_PATH):
        _require(path)
    if not PILOT_OUT_PATH.exists():
        raise SystemExit(f"missing {PILOT_OUT_PATH}; the calibration must exclude the F1.3 pilot")

    questions = load_train_questions(QUESTIONS_PATH)
    counts = case_counts(questions)
    if (
        counts.get(CASE_A) != TRAIN_A
        or counts.get(CASE_U) != TRAIN_U
        or counts.get(CASE_EXCLUDED) != TRAIN_EXCLUDED
    ):
        raise SystemExit(
            f"unexpected train composition {counts}; expected A={TRAIN_A}, "
            f"U={TRAIN_U}, excluded={TRAIN_EXCLUDED}"
        )

    excluded_pilot = load_pilot_ids(PILOT_OUT_PATH)
    candidates = sorted(
        (
            question
            for question in questions
            if assign_case(question) == CASE_A and question.question_id not in excluded_pilot
        ),
        key=lambda question: question.question_id,
    )
    if len(candidates) < PROBE_SAMPLE:
        raise SystemExit(
            f"only {len(candidates)} A questions outside the pilot; need {PROBE_SAMPLE}"
        )

    sample = random.Random(SEED).sample(candidates, PROBE_SAMPLE)
    store = load_chunk_store(CHUNKS_PATH)
    embeddings = load_embeddings(EMBEDDINGS_PATH, expected_rows=len(store.ids))
    answers_by_id = load_train_answers(QASPER_TRAIN_PATH)
    teacher = OllamaTeacher()
    embedder = QuestionEmbedder()

    failures = 0
    for index, question in enumerate(sample, start=1):
        outcome = probe_question(
            question,
            answers_by_id[question.question_id],
            store,
            teacher,
            embeddings=embeddings,
            embed_question=embedder.embed,
            canary=new_canary(),
        )
        failures += int(outcome.failed)
        if index % 25 == 0:
            print(f"probe {index}/{PROBE_SAMPLE}: fails={failures}", flush=True)

    n = len(sample)
    r = failures / n
    p0 = compute_p0(TRAIN_A, TRAIN_U, PILOT_PASS_RATE, r)
    payload = {
        "p0": p0,
        "A": TRAIN_A,
        "U": TRAIN_U,
        "excluded": TRAIN_EXCLUDED,
        "q": PILOT_PASS_RATE,
        "r": r,
        "probe_n": n,
        "probe_fails": failures,
        "seed": SEED,
        "pilot_excluded": len(excluded_pilot),
    }
    with P0_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
