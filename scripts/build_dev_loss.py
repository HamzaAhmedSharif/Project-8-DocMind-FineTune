#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Build the F3.1 dev-loss set from QASPER validation questions.

Dev-loss is the held-out eval set for training loss: TRL rows built the
same way as data/train.jsonl (same local Ollama teacher, same p0 calibration,
same assemble_trl_rows), from QASPER validation questions that are NOT in
Q-DEV and from papers NOT in Q-DEV.  It is never mixed into training.
The eligible pool after Q-DEV exclusion is 374 questions, and the teacher
keep rate is ~60%, so ALL eligible questions are used (~220 kept rows
expected) rather than capping at 300.

Pipeline (mirrors scripts/run_teacher_full.py + scripts/build_train.py):
  1. Load dev questions from B:\\p8\\data\\work\\questions.jsonl (split == "dev").
  2. Exclude Q-DEV questions and Q-DEV papers (evaluation/splits/qdev.json,
     read-only — the frozen split is never modified).
  3. Deterministic seeded sample of 300 A/U questions (seed 20261207).
  4. Run the local Ollama teacher over them (resumable append to
     B:\\p8\\data\\work\\dev_loss_teacher_out.jsonl).
  5. Assemble TRL rows, drop any failing the I2 prototype checks (F1.4
     precedent) -> data/dev_loss.jsonl, verify zero overlap with
     Q-DEV, Q-TEST, and train.jsonl (questions + papers), and report I2 stats.
     The gate re-verifies the written file independently.

    python scripts/build_dev_loss.py [--limit N] [--force] [--dry-run N]

--dry-run N runs the teacher on N sampled questions into a temp file,
assembles in memory, prints the overlap + I2 checks, and deletes the temp
file — it writes no persistent output.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: E402, F401  (installs the docmind_service alias)
from docmind_ft.jsonl import read_jsonl  # noqa: E402
from docmind_ft.teacher import OllamaTeacher  # noqa: E402
from docmind_ft.teacher_build import (  # noqa: E402
    CASE_A,
    CASE_EXCLUDED,
    CASE_U,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    P0_PATH,
    QASPER_TRAIN_PATH,
    QUESTIONS_PATH,
    REPORTS_DIR,
    TRAIN_DATA_DIR,
    WORK_DIR,
    QuestionEmbedder,
    TrainQuestion,
    abstention_share,
    assemble_trl_rows,
    assign_case,
    branch_p0,
    case_counts,
    i2_stats,
    load_chunk_store,
    load_embeddings,
    load_train_answers,
    new_canary,
    process_question,
    summarize_teacher_rows,
    validate_trl_row,
)

DEV_QASPER_PATH = QASPER_TRAIN_PATH.parent / "qasper-dev-v0.3.json"
DEV_TEACHER_OUT_PATH = WORK_DIR / "dev_loss_teacher_out.jsonl"
QDEV_PATH = REPO_ROOT / "evaluation" / "splits" / "qdev.json"
QTEST_PATH = REPO_ROOT / "evaluation" / "splits" / "qtest.json"
TRAIN_JSONL_PATH = TRAIN_DATA_DIR / "train.jsonl"
OUT_PATH = TRAIN_DATA_DIR / "dev_loss.jsonl"
REPORT_PATH = REPORTS_DIR / "dev-loss-stats.json"

SEED = 20261207


def _require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing {path}")


def _append(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_split_ids(path: Path) -> set[str]:
    return {str(item) for item in json.loads(path.read_text(encoding="utf-8"))}


def _load_p0() -> float:
    if not P0_PATH.exists():
        raise SystemExit(f"missing {P0_PATH}; run `python scripts/calibrate_p0.py` first")
    p0_payload = json.loads(P0_PATH.read_text(encoding="utf-8"))
    p0 = float(p0_payload["p0"])
    if not 0.0 <= p0 <= 1.0:
        raise SystemExit(f"invalid p0={p0} in {P0_PATH}")
    return p0


def _run_teacher(
    sampled: list[TrainQuestion],
    teacher_out_path: Path,
    p0: float,
) -> list[dict[str, object]]:
    """Run the teacher over the sampled questions (resumable append)."""
    existing = read_jsonl(teacher_out_path) if teacher_out_path.exists() else []
    existing_ids = [str(row["question_id"]) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise SystemExit(f"duplicate question_id in {teacher_out_path}")
    seen = set(existing_ids)
    pending = [question for question in sampled if question.question_id not in seen]
    if not pending:
        print("nothing to do; all sampled rows are already present")

    store = load_chunk_store(CHUNKS_PATH)
    answers_by_id = load_train_answers(DEV_QASPER_PATH)
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

    total_prompts = sum(assign_case(q) == CASE_A for q in sampled)
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
        _append(teacher_out_path, row)
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

    return read_jsonl(teacher_out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap the sample size")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild the assembled set from the existing teacher output "
        "(teacher-out is kept, so no new teacher calls)",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=None,
        metavar="N",
        help="teacher on N sampled questions only; no persistent output",
    )
    args = parser.parse_args(argv)

    for path in (QUESTIONS_PATH, CHUNKS_PATH, DEV_QASPER_PATH, QDEV_PATH, QTEST_PATH):
        _require(path)
    p0 = _load_p0()

    if args.force:
        # Rebuild the assembled set only: the teacher-out file is the
        # expensive resumable intermediate (~25 min of teacher calls) and is
        # never deleted by --force. Delete it by hand for a full re-teach.
        if OUT_PATH.exists():
            OUT_PATH.unlink()
            print(f"removed {OUT_PATH}")
    if OUT_PATH.exists() and args.dry_run is None:
        raise SystemExit(f"{OUT_PATH} already exists; pass --force to rebuild it")

    # --- Sample dev questions, excluding everything in Q-DEV ---
    all_rows = read_jsonl(QUESTIONS_PATH)
    dev_questions = [TrainQuestion.from_row(row) for row in all_rows if row.get("split") == "dev"]
    test_paper_by_qid = {
        str(row["question_id"]): str(row["paper_id"])
        for row in all_rows
        if row.get("split") == "test"
    }
    by_id = {question.question_id: question for question in dev_questions}
    qdev_qids = _load_split_ids(QDEV_PATH)
    qtest_qids = _load_split_ids(QTEST_PATH)
    qdev_papers = {by_id[qid].paper_id for qid in qdev_qids if qid in by_id}
    qtest_papers = {test_paper_by_qid[qid] for qid in qtest_qids if qid in test_paper_by_qid}
    eligible = sorted(
        (
            question
            for question in dev_questions
            if assign_case(question) in (CASE_A, CASE_U)
            and question.question_id not in qdev_qids
            and question.paper_id not in qdev_papers
        ),
        key=lambda question: question.question_id,
    )
    pool_counts = case_counts(dev_questions)
    # Use ALL eligible questions: the pool (374) is the binding constraint,
    # and the ~60% teacher keep rate means a 300-cap would yield only ~180 rows.
    n_target = len(eligible) if args.limit is None else min(args.limit, len(eligible))
    if args.dry_run is not None:
        n_target = min(args.dry_run, len(eligible))
    sampled = random.Random(SEED).sample(eligible, min(n_target, len(eligible)))
    sampled = sorted(sampled, key=lambda question: question.question_id)
    print(
        f"dev pool: {len(dev_questions)} questions "
        f"(A={pool_counts.get(CASE_A, 0)} U={pool_counts.get(CASE_U, 0)} "
        f"excluded={pool_counts.get(CASE_EXCLUDED, 0)}); "
        f"Q-DEV excluded: {len(qdev_qids)} questions, {len(qdev_papers)} papers; "
        f"eligible: {len(eligible)}; sampled: {len(sampled)}"
    )

    # --- Teacher run ---
    teacher_out_path = (
        DEV_TEACHER_OUT_PATH
        if args.dry_run is None
        else WORK_DIR / "dev_loss_teacher_out_dry.jsonl"
    )
    if args.dry_run is not None and teacher_out_path.exists():
        teacher_out_path.unlink()
    rows = _run_teacher(sampled, teacher_out_path, p0)
    sampled_by_id = {q.question_id: q for q in sampled}
    rows = [row for row in rows if str(row["question_id"]) in sampled_by_id]

    # --- Assemble TRL rows ---
    store = load_chunk_store(CHUNKS_PATH)
    answers_by_id = load_train_answers(DEV_QASPER_PATH)
    distilled, _option_a = assemble_trl_rows(rows, sampled_by_id, store, answers_by_id)
    print(f"assembled {len(distilled)} dev-loss rows from {len(rows)} teacher rows")
    # Drop rows failing the I2 prototype checks (F1.4 precedent: invalid rows
    # are dropped at assembly, not gated out after the fact). The gate below
    # still re-verifies the written file independently.
    kept = [row for row in distilled if not validate_trl_row(row)]
    dropped = len(distilled) - len(kept)
    if dropped:
        print(f"dropped {dropped} rows failing I2 checks; {len(kept)} rows kept")
    distilled = kept

    # --- Overlap verification (questions + papers, all must be zero) ---
    dev_qids = {str(row["meta"]["question_id"]) for row in distilled}
    dev_papers = {str(row["meta"]["paper_id"]) for row in distilled}
    train_rows = read_jsonl(TRAIN_JSONL_PATH) if TRAIN_JSONL_PATH.exists() else []
    train_qids = {str(row["meta"]["question_id"]) for row in train_rows}
    train_papers = {str(row["meta"]["paper_id"]) for row in train_rows}
    overlaps = {
        "qdev_questions": sorted(dev_qids & qdev_qids),
        "qdev_papers": sorted(dev_papers & qdev_papers),
        "qtest_questions": sorted(dev_qids & qtest_qids),
        "qtest_papers": sorted(dev_papers & qtest_papers),
        "train_questions": sorted(dev_qids & train_qids),
        "train_papers": sorted(dev_papers & train_papers),
    }
    overlap_total = sum(len(v) for v in overlaps.values())
    for name, items in overlaps.items():
        print(f"overlap {name}: {len(items)}")

    if args.dry_run is not None:
        teacher_out_path.unlink(missing_ok=True)
        print(
            f"dry run complete: {len(distilled)} rows assembled in memory, "
            f"overlap total {overlap_total}, no files written"
        )
        return 0 if overlap_total == 0 else 1

    _write_jsonl(OUT_PATH, distilled)
    print(f"wrote {len(distilled)} rows -> {OUT_PATH}")

    # --- I2 stats on the written file ---
    stats_i2 = i2_stats(OUT_PATH, expected_rows=len(distilled))
    i2_share = stats_i2["i2_valid_share"]
    print(f"I2 valid share: {i2_share}")

    summary = {
        "seed": SEED,
        "dev_pool": len(dev_questions),
        "pool_case_counts": pool_counts,
        "qdev_excluded_questions": len(qdev_qids),
        "qdev_excluded_papers": len(qdev_papers),
        "eligible": len(eligible),
        "sampled": len(sampled),
        "sampled_case_counts": case_counts(sampled),
        "teacher_rows": len(rows),
        "teacher_summary": summarize_teacher_rows(rows),
        "kept_rows": len(distilled),
        "i2_dropped_rows": dropped,
        "overlaps": {name: len(items) for name, items in overlaps.items()},
        "abstention_share": abstention_share(distilled),
        "i2": stats_i2,
        "p0_calibration": p0,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    failed = []
    if overlap_total:
        failed.append("overlap with Q-DEV/Q-TEST/train is non-zero")
    if i2_share < 0.99:
        failed.append(f"I2 valid share {i2_share} < 0.99")
    if failed:
        print(f"dev-loss gate FAILED: {'; '.join(failed)}", file=sys.stderr)
        return 1
    print("dev-loss gate PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
