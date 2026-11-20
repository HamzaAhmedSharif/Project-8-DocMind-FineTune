#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Assemble the F1.4 training files and their I2/statistics report.

Reads the complete ``teacher_out.jsonl`` and writes the distilled file, the
human-target comparison file with identical prompts, and the deterministic
25% sample.  Sol runs this after the full teacher build.

    python scripts/build_train.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: E402, F401  (installs the docmind_service alias)
from docmind_ft.teacher_build import (  # noqa: E402
    CASE_A,
    CASE_EXCLUDED,
    CASE_U,
    CHUNKS_PATH,
    QUESTIONS_PATH,
    QASPER_TRAIN_PATH,
    REPORTS_DIR,
    TEACHER_OUT_PATH,
    TRAIN_DATA_DIR,
    abstention_share,
    answer_length_stats,
    assemble_trl_rows,
    assign_case,
    case_counts,
    i2_stats,
    is_data25,
    load_chunk_store,
    load_train_answers,
    load_train_questions,
    read_jsonl,
    summarize_teacher_rows,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    for path in (QUESTIONS_PATH, CHUNKS_PATH, QASPER_TRAIN_PATH):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    if not TEACHER_OUT_PATH.exists():
        raise SystemExit(
            f"missing {TEACHER_OUT_PATH}; run `python scripts/run_teacher_full.py` first"
        )

    questions = load_train_questions(QUESTIONS_PATH)
    by_id = {question.question_id: question for question in questions}
    counts = case_counts(questions)
    expected = {
        question.question_id for question in questions if assign_case(question) in (CASE_A, CASE_U)
    }
    teacher_rows = read_jsonl(TEACHER_OUT_PATH)
    teacher_ids = [str(row["question_id"]) for row in teacher_rows]
    if len(teacher_ids) != len(set(teacher_ids)):
        raise SystemExit(f"duplicate question_id in {TEACHER_OUT_PATH}")
    missing = expected - set(teacher_ids)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise SystemExit(
            f"{TEACHER_OUT_PATH} is incomplete: {len(missing)} rows missing ({preview})"
        )

    store = load_chunk_store(CHUNKS_PATH)
    answers_by_id = load_train_answers(QASPER_TRAIN_PATH)
    distilled, option_a = assemble_trl_rows(teacher_rows, by_id, store, answers_by_id)
    data25 = [row for row in distilled if is_data25(str(row["meta"]["question_id"]))]

    train_path = TRAIN_DATA_DIR / "train.jsonl"
    nodistill_path = TRAIN_DATA_DIR / "train_nodistill.jsonl"
    data25_path = TRAIN_DATA_DIR / "train_data25.jsonl"
    _write_jsonl(train_path, distilled)
    _write_jsonl(nodistill_path, option_a)
    _write_jsonl(data25_path, data25)

    files = {
        "train": train_path,
        "train_nodistill": nodistill_path,
        "train_data25": data25_path,
    }
    rows_by_file = {
        "train": distilled,
        "train_nodistill": option_a,
        "train_data25": data25,
    }
    # train_nodistill.jsonl holds verbatim human answers, so the I2
    # word/sentence rules do not apply to it (DECISIONS.md, F1.4).
    i2 = {
        name: i2_stats(path, check_length=(name != "train_nodistill"))
        for name, path in files.items()
    }
    i2_share = {name: stats["i2_valid_share"] for name, stats in i2.items()}

    meta_rows = [row["meta"] for row in distilled]
    by_type = Counter(str(meta["answer_type"]) for meta in meta_rows)
    by_case = Counter(str(meta["case"]) for meta in meta_rows)
    by_type_case = Counter(f"{meta['answer_type']}:{meta['case']}" for meta in meta_rows)
    teacher_summary = summarize_teacher_rows(teacher_rows)
    teacher_summary["excluded"] = counts.get(CASE_EXCLUDED, 0)

    stats = {
        "source_counts": counts,
        "excluded": counts.get(CASE_EXCLUDED, 0),
        "teacher": teacher_summary,
        "counts": {
            "train": len(distilled),
            "train_nodistill": len(option_a),
            "train_data25": len(data25),
            "target_1500": 1500,
            "below_target": len(distilled) < 1500,
            "by_case": dict(by_case),
            "by_type": dict(by_type),
            "by_type_case": dict(by_type_case),
        },
        "pass_rates": {
            "gold_v1_v4": teacher_summary["gold"]["pass_rate"],
            "p0_probe_fail": teacher_summary["p0"]["probe_fail_rate"],
            "u_kept": (
                teacher_summary["u"]["kept"] / teacher_summary["u"]["rows"]
                if teacher_summary["u"]["rows"]
                else 0.0
            ),
        },
        "answer_lengths": answer_length_stats(distilled),
        "abstention_share": {name: abstention_share(rows) for name, rows in rows_by_file.items()},
        "i2_valid_share": i2_share,
        "i2": i2,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "F1-train-stats.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    failed = [name for name, share in i2_share.items() if share < 0.99]
    if failed:
        print(
            f"I2 gate failed (<0.99): {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
