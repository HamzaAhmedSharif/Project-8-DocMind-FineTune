#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""F5.6: error analysis for FT-main (Q-TEST + HOLD).

Reads the FT-main raw outputs + the aggregate metrics, categorizes failures,
and writes the analysis tables to evaluation/results/error_analysis.json
(consumed by docs/EVALUATION.md).

Categories (seeded sample):
  retrieval_miss     no gold chunk in the context
  wrong_answer       gold in context but the answer is wrong (answer_f1 < 0.5)
  over_refusal       answerable-with-gold but the model abstained
  citation_error     answer otherwise reasonable but cites a wrong/unknown chunk
  format_error       format_valid is False
  other              anything else

A "failure" on Q-TEST is an answerable question with answer_f1 < 0.5, or an
unanswerable question the model did not abstain on, or a format error.
A "failure" on HOLD is a question where not all key facts were found, or a
format error, or an over-refusal.
Samples: 30 random Q-TEST failures + 20 HOLD failures (seed 20261207).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: F401, E402  # installs the docmind_service alias

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from final_eval_compute import build_rows, per_question_scores  # noqa: E402
from docmind_ft.metrics import (  # noqa: E402
    citation_ids,
    format_valid,
    is_abstention,
)

RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
OUT_PATH = RESULTS_DIR / "error_analysis.json"
SEED = 20261207


def categorize_qtest(row, score_f1: float) -> str | None:
    """Return the failure category for a Q-TEST row, or None if not a failure."""
    if not format_valid(row.output, row.prompt_chunk_ids):
        return "format_error"
    if row.answerable:
        gold_in_ctx = bool(set(row.gold_chunk_ids) & set(row.prompt_chunk_ids))
        if is_abstention(row.output):
            return "over_refusal" if gold_in_ctx else "other"
        if not gold_in_ctx:
            return "retrieval_miss"
        if score_f1 < 0.5:
            # Check citation: if it cites a non-gold chunk only, it's a citation error.
            stats = citation_ids(row.output)
            gold = set(row.gold_chunk_ids)
            if stats.ids and not any(cid in gold for cid in stats.ids):
                return "citation_error"
            return "wrong_answer"
        return None
    # Unanswerable: failure if it did not abstain.
    if not is_abstention(row.output):
        return "wrong_answer"
    return None


def categorize_hold(row, keyfact_frac: float) -> str | None:
    """Return the failure category for a HOLD row, or None if not a failure."""
    if not format_valid(row.output, row.prompt_chunk_ids):
        return "format_error"
    if row.answerable:
        gold_in_ctx = bool(set(row.gold_chunk_ids) & set(row.prompt_chunk_ids))
        if is_abstention(row.output):
            return "over_refusal" if gold_in_ctx else "other"
        if keyfact_frac >= 1.0:
            return None
        if not gold_in_ctx:
            return "retrieval_miss"
        stats = citation_ids(row.output)
        gold = set(row.gold_chunk_ids)
        if stats.ids and not any(cid in gold for cid in stats.ids):
            return "citation_error"
        return "wrong_answer"
    if not is_abstention(row.output):
        return "wrong_answer"
    return None


def main() -> int:
    from final_eval_compute import row_keyfacts

    rng = random.Random(SEED)
    analysis: dict = {"seed": SEED}

    # ---- Q-TEST ----
    row_keyfacts.clear()
    q_rows = build_rows("ft_main", "qtest")
    q_scores = per_question_scores(q_rows, "qtest")
    q_failures: list[dict] = []
    for i, row in enumerate(q_rows):
        cat = categorize_qtest(row, q_scores["answer_f1"][i])
        if cat is None:
            continue
        q_failures.append(
            {
                "question_id": row.question_id,
                "paper_id": row.paper_id,
                "category": cat,
                "answer_f1": round(q_scores["answer_f1"][i], 4),
                "output": row.output,
            }
        )
    q_counts: dict[str, int] = {}
    for f in q_failures:
        q_counts[f["category"]] = q_counts.get(f["category"], 0) + 1
    q_sample = rng.sample(q_failures, min(30, len(q_failures)))
    analysis["qtest"] = {
        "n_questions": len(q_rows),
        "n_failures": len(q_failures),
        "counts": q_counts,
        "sampled": q_sample,
    }

    # ---- HOLD ----
    row_keyfacts.clear()
    h_rows = build_rows("ft_main", "hold")
    h_scores = per_question_scores(h_rows, "hold")
    h_failures: list[dict] = []
    for i, row in enumerate(h_rows):
        cat = categorize_hold(row, h_scores["keyfact_cov"][i])
        if cat is None:
            continue
        h_failures.append(
            {
                "question_id": row.question_id,
                "paper_id": row.paper_id,
                "category": cat,
                "keyfact_cov": round(h_scores["keyfact_cov"][i], 4),
                "output": row.output,
            }
        )
    h_counts: dict[str, int] = {}
    for f in h_failures:
        h_counts[f["category"]] = h_counts.get(f["category"], 0) + 1
    h_sample = rng.sample(h_failures, min(20, len(h_failures)))
    analysis["hold"] = {
        "n_questions": len(h_rows),
        "n_failures": len(h_failures),
        "counts": h_counts,
        "sampled": h_sample,
    }

    # ---- B1-beats-FT: five largest answer_f1 gaps on Q-TEST ----
    row_keyfacts.clear()
    b1_rows = build_rows("b1", "qtest")
    b1_scores = per_question_scores(b1_rows, "qtest")
    b1_by_id = {r.question_id: s for r, s in zip(b1_rows, b1_scores["answer_f1"], strict=False)}
    ft_by_id = {r.question_id: s for r, s in zip(q_rows, q_scores["answer_f1"], strict=False)}
    gaps = []
    for qid, b1_f1 in b1_by_id.items():
        ft_f1 = ft_by_id.get(qid, 0.0)
        gaps.append((b1_f1 - ft_f1, qid, b1_f1, ft_f1))
    gaps.sort(reverse=True)
    analysis["b1_beats_ft_top5"] = [
        {
            "question_id": qid,
            "b1_answer_f1": round(b1, 4),
            "ft_answer_f1": round(ft, 4),
            "gap": round(g, 4),
        }
        for g, qid, b1, ft in gaps[:5]
    ]

    # ---- Answer-length stats (from aggregate metrics) ----
    agg = {}
    for name in ("ft_main_qtest", "b1_qtest", "b0_qtest", "ft_main_hold", "b1_hold"):
        agg[name] = json.loads((RESULTS_DIR / f"metrics_{name}.json").read_text(encoding="utf-8"))
    analysis["answer_length"] = {name: agg[name].get("mean_answer_length") for name in agg}
    # Per-type answer_f1 (Q-TEST) for FT-main vs B1.
    analysis["answer_f1_by_type"] = {
        "ft_main_qtest": (agg["ft_main_qtest"].get("answer") or {}).get("answer_f1_by_type"),
        "b1_qtest": (agg["b1_qtest"].get("answer") or {}).get("answer_f1_by_type"),
        "b0_qtest": (agg["b0_qtest"].get("answer") or {}).get("answer_f1_by_type"),
    }

    OUT_PATH.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT_PATH}")
    print(f"Q-TEST failures: {len(q_failures)} of {len(q_rows)}; counts={q_counts}")
    print(f"HOLD failures: {len(h_failures)} of {len(h_rows)}; counts={h_counts}")
    print("B1-beats-FT top5:", analysis["b1_beats_ft_top5"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
