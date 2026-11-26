#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""F5.3: build test_metrics.json + hypotheses.json from the sealed outputs.

Reads the raw generation JSONL files (evaluation/results/<model>_<split>.jsonl)
and the 11 aggregate metrics_*.json files. Re-derives per-question scores
with docmind_ft.metrics, runs the paper-cluster paired bootstrap
(docmind_ft.metrics.paired_paper_bootstrap, seed 20261207, 10_000 resamples)
for FT-main vs B1 and FT-main vs B0, and evaluates HY1-HY4 against the frozen
thresholds in evaluation/gates.json.

Writes:
  evaluation/results/test_metrics.json
  evaluation/results/hypotheses.json

Usage:
  python scripts/final_eval_compute.py [--recompute-only]
  --recompute-only re-derives every table and compares with the committed JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: F401, E402  # installs the docmind_service alias

from docmind_ft.metrics import (  # noqa: E402
    OutputRow,
    answer_f1,
    format_valid,
    is_abstention,
    paired_paper_bootstrap,
    score_hold_answer,
    score_paper16_answer,
)

# Reuse the gold/context loading from compute_metrics (same frozen inputs).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import compute_metrics as cm  # noqa: E402

RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
GATES_PATH = REPO_ROOT / "evaluation" / "gates.json"

MODELS = ["b0", "b1", "ft_main", "r"]
SPLITS = ["qtest", "hold", "paper16"]
# (model, split) pairs that exist (R has no qtest).
PAIRS = [
    ("b0", "qtest"),
    ("b0", "hold"),
    ("b0", "paper16"),
    ("b1", "qtest"),
    ("b1", "hold"),
    ("b1", "paper16"),
    ("ft_main", "qtest"),
    ("ft_main", "hold"),
    ("ft_main", "paper16"),
    ("r", "hold"),
    ("r", "paper16"),
]
OUTPUT_FILES = {(m, s): RESULTS_DIR / f"{m}_{s}.jsonl" for (m, s) in PAIRS}
METRICS_FILES = {(m, s): RESULTS_DIR / f"metrics_{m}_{s}.json" for (m, s) in PAIRS}


def build_rows(model: str, split: str) -> list[OutputRow]:
    """Rebuild the OutputRow list for (model, split) exactly as compute_metrics does."""
    outputs_path = OUTPUT_FILES[(model, split)]
    context_path = cm.CONTEXT_FILES[split]
    if context_path.resolve().parent == cm.CONTEXTS_DIR.resolve():
        from docmind_ft.frozen import verify_context_hash

        verify_context_hash(context_path)
    contexts = cm._read_jsonl(context_path)
    outputs = cm._read_jsonl(outputs_path)
    by_id: dict[str, dict] = {}
    for row in outputs:
        qid = str(row["question_id"])
        if qid in by_id:
            raise ValueError(f"{outputs_path}: duplicate output for {qid!r}")
        by_id[qid] = row
    gold = cm._gold_for(split)
    resolver = cm.ChunkTextResolver()
    rows: list[OutputRow] = []
    context_texts: dict[str, str] = {}
    for context in contexts:
        qid = str(context["question_id"])
        output = by_id.get(qid)
        if output is None:
            continue
        entry = gold.get(qid)
        if entry is None:
            raise ValueError(f"no gold information for {qid!r} on split {split!r}")
        paper_id = str(context["paper_id"])
        raw_chunk_ids = tuple(str(item["chunk_id"]) for item in context["chunks"])
        short_ids = tuple(f"c{i}" for i in range(1, len(raw_chunk_ids) + 1))
        texts = tuple(resolver.text(cid, paper_id, split) for cid in raw_chunk_ids)
        context_text = "\n".join(texts)
        context_texts[qid] = context_text
        gold_chunk_ids = tuple(entry["gold_chunk_ids"])
        key_facts = tuple(entry.get("key_facts") or ())
        if split == "hold":
            from docmind_ft.metrics import hold_fact_in_text

            gold_chunk_ids = tuple(
                cid
                for cid, text in zip(raw_chunk_ids, texts, strict=True)
                if any(hold_fact_in_text(f, text) for f in key_facts)
            )
        elif split == "paper16":
            from docmind_ft.metrics import paper16_fact_in_text

            gold_chunk_ids = tuple(
                cid
                for cid, text in zip(raw_chunk_ids, texts, strict=True)
                if any(paper16_fact_in_text(f, text) for f in key_facts)
            )
        gold_ids = set(gold_chunk_ids)
        short_gold_ids = tuple(
            s for s, raw in zip(short_ids, raw_chunk_ids, strict=True) if raw in gold_ids
        )
        rows.append(
            OutputRow(
                question_id=qid,
                output=str(output["output"]),
                prompt_chunk_ids=short_ids,
                gold_chunk_ids=short_gold_ids,
                answerable=bool(entry["answerable"]),
                gold_answers=tuple(entry["gold_answers"]),
                yes_no=entry["yes_no"],
                answer_type=str(entry["answer_type"]),
                paper_id=paper_id,
            )
        )
    # Attach per-question keyfact scores for hold/paper16.
    for row in rows:
        if split in ("hold", "paper16"):
            key_facts = gold[row.question_id].get("key_facts") or ()
            if not key_facts:
                continue
            ctx = context_texts[row.question_id]
            if split == "hold":
                score = score_hold_answer(row.output, key_facts, ctx)
            else:
                score = score_paper16_answer(row.output, key_facts, ctx)
            row_keyfacts[row.question_id] = score
    return rows


# Per-question keyfact scores, filled by build_rows.
row_keyfacts: dict[str, object] = {}


def per_question_scores(rows: list[OutputRow], split: str) -> dict[str, list[float]]:
    """Per-question metric values aligned with rows (for the paired bootstrap)."""
    f1: list[float] = []
    fmt: list[float] = []
    abstain: list[float] = []  # only meaningful on unanswerable rows; None elsewhere
    abstain_mask: list[bool] = []
    keyfact: list[float] = []
    keyfact_mask: list[bool] = []
    papers: list[str] = []
    for row in rows:
        papers.append(row.paper_id)
        # answer_f1 (answerable with gold only; else 0.0 placeholder, excluded by mask)
        if row.answerable and (row.yes_no is not None or row.gold_answers):
            f1.append(answer_f1(row.output, row.gold_answers, yes_no=row.yes_no))
        else:
            f1.append(0.0)
        fmt.append(1.0 if format_valid(row.output, row.prompt_chunk_ids) else 0.0)
        if not row.answerable:
            abstain.append(1.0 if is_abstention(row.output) else 0.0)
            abstain_mask.append(True)
        else:
            abstain.append(0.0)
            abstain_mask.append(False)
        if split in ("hold", "paper16"):
            score = row_keyfacts.get(row.question_id)
            if score is not None:
                hits = score.hits
                keyfact.append(sum(hits) / len(hits) if hits else 0.0)
                keyfact_mask.append(True)
            else:
                keyfact.append(0.0)
                keyfact_mask.append(False)
        else:
            keyfact.append(0.0)
            keyfact_mask.append(False)
    return {
        "answer_f1": f1,
        "format_valid": fmt,
        "abstain": abstain,
        "abstain_mask": abstain_mask,
        "keyfact_cov": keyfact,
        "keyfact_mask": keyfact_mask,
        "paper_id": papers,
    }


def bootstrap_comparison(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
    metric: str,
    mask_key: str | None = None,
) -> dict:
    """Paired paper-cluster bootstrap for mean(a)-mean(b) on one metric."""
    if mask_key is not None:
        idx = [i for i, m in enumerate(a[mask_key]) if m]
        a_scores = [a[metric][i] for i in idx]
        b_scores = [b[metric][i] for i in idx]
        papers = [a["paper_id"][i] for i in idx]
    else:
        a_scores = a[metric]
        b_scores = b[metric]
        papers = a["paper_id"]
    result = paired_paper_bootstrap(a_scores, b_scores, papers)
    return {
        "point": result.point,
        "ci_low": result.low,
        "ci_high": result.high,
        "n_questions": result.n_questions,
        "n_papers": result.n_papers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-only", action="store_true")
    args = parser.parse_args(argv)

    gates = json.loads(GATES_PATH.read_text(encoding="utf-8"))
    hy = gates["hypotheses"]

    # --- Load aggregate metrics ---
    agg: dict[str, dict] = {}
    for m, s in PAIRS:
        agg[f"{m}_{s}"] = json.loads(METRICS_FILES[(m, s)].read_text(encoding="utf-8"))

    # --- Build per-question rows + scores ---
    rows: dict[tuple[str, str], list[OutputRow]] = {}
    scores: dict[tuple[str, str], dict[str, list[float]]] = {}
    for m, s in PAIRS:
        row_keyfacts.clear()
        r = build_rows(m, s)
        rows[(m, s)] = r
        scores[(m, s)] = per_question_scores(r, s)

    def f1_mask(model: str, split: str) -> list[bool]:
        return [
            bool(row.answerable and (row.yes_no is not None or row.gold_answers))
            for row in rows[(model, split)]
        ]

    # --- Paired comparisons ---
    comparisons: dict[str, dict] = {}
    for split in SPLITS:
        for a_model, b_model in [("ft_main", "b1"), ("ft_main", "b0")]:
            if (a_model, split) not in scores or (b_model, split) not in scores:
                continue
            key = f"ft_main_vs_{b_model}_{split}"
            entry: dict[str, dict] = {}
            a_s, b_s = scores[(a_model, split)], scores[(b_model, split)]
            if split == "qtest":
                mask = f1_mask(a_model, split)
                idx = [i for i, m in enumerate(mask) if m]
                res = paired_paper_bootstrap(
                    [a_s["answer_f1"][i] for i in idx],
                    [b_s["answer_f1"][i] for i in idx],
                    [a_s["paper_id"][i] for i in idx],
                )
                entry["answer_f1"] = {
                    "point": res.point,
                    "ci_low": res.low,
                    "ci_high": res.high,
                    "n_questions": res.n_questions,
                    "n_papers": res.n_papers,
                }
                idx_ab = [i for i, m in enumerate(a_s["abstain_mask"]) if m]
                if idx_ab:
                    res_ab = paired_paper_bootstrap(
                        [a_s["abstain"][i] for i in idx_ab],
                        [b_s["abstain"][i] for i in idx_ab],
                        [a_s["paper_id"][i] for i in idx_ab],
                    )
                    entry["abstain_acc"] = {
                        "point": res_ab.point,
                        "ci_low": res_ab.low,
                        "ci_high": res_ab.high,
                        "n_questions": res_ab.n_questions,
                        "n_papers": res_ab.n_papers,
                    }
            res_fmt = paired_paper_bootstrap(
                a_s["format_valid"], b_s["format_valid"], a_s["paper_id"]
            )
            entry["format_valid"] = {
                "point": res_fmt.point,
                "ci_low": res_fmt.low,
                "ci_high": res_fmt.high,
                "n_questions": res_fmt.n_questions,
                "n_papers": res_fmt.n_papers,
            }
            if split in ("hold", "paper16"):
                idx_kf = [i for i, m in enumerate(a_s["keyfact_mask"]) if m]
                if idx_kf:
                    res_kf = paired_paper_bootstrap(
                        [a_s["keyfact_cov"][i] for i in idx_kf],
                        [b_s["keyfact_cov"][i] for i in idx_kf],
                        [a_s["paper_id"][i] for i in idx_kf],
                    )
                    entry["keyfact_cov"] = {
                        "point": res_kf.point,
                        "ci_low": res_kf.low,
                        "ci_high": res_kf.high,
                        "n_questions": res_kf.n_questions,
                        "n_papers": res_kf.n_papers,
                    }
            comparisons[key] = entry

    # --- Reference gap closure (HOLD / PAPER16, keyfact_cov) ---
    gap: dict[str, dict] = {}
    for split in ("hold", "paper16"):
        ft_cov = (agg[f"ft_main_{split}"].get("keyfacts") or {}).get("keyfact_cov")
        b1_cov = (agg[f"b1_{split}"].get("keyfacts") or {}).get("keyfact_cov")
        r_cov = (agg[f"r_{split}"].get("keyfacts") or {}).get("keyfact_cov")
        if ft_cov is not None and b1_cov is not None and r_cov is not None and r_cov != b1_cov:
            closure = (ft_cov - b1_cov) / (r_cov - b1_cov)
        else:
            closure = None
        gap[split] = {
            "ft_main_keyfact_cov": ft_cov,
            "b1_keyfact_cov": b1_cov,
            "r_keyfact_cov": r_cov,
            "gap_closure": closure,
        }

    test_metrics = {
        "models": MODELS,
        "splits": SPLITS,
        "per_model_split": {f"{m}_{s}": agg[f"{m}_{s}"] for (m, s) in PAIRS},
        "comparisons": comparisons,
        "reference_gap_closure": gap,
        "bootstrap": {
            "method": "paper-cluster bootstrap, resample papers with replacement",
            "n_resamples": 10000,
            "seed": 20261207,
            "alpha": 0.05,
        },
    }

    # --- Hypotheses HY1-HY4 ---
    ft_q = agg["ft_main_qtest"]
    b1_q = agg["b1_qtest"]
    ft_h = agg["ft_main_hold"]
    ft_p = agg["ft_main_paper16"]
    comp_f1 = comparisons.get("ft_main_vs_b1_qtest", {}).get("answer_f1", {})
    comp_fmt = comparisons.get("ft_main_vs_b1_qtest", {}).get("format_valid", {})
    comp_kf_h = comparisons.get("ft_main_vs_b1_hold", {}).get("keyfact_cov", {})
    comp_kf_p = comparisons.get("ft_main_vs_b1_paper16", {}).get("keyfact_cov", {})
    comp_ab = comparisons.get("ft_main_vs_b1_qtest", {}).get("abstain_acc", {})

    hy1 = {
        "verdict": None,
        "citation_valid_share_ft_main_qtest": ft_q.get("citation_valid_share"),
        "format_valid_ft_main_qtest": ft_q.get("format_valid"),
        "format_valid_improvement_vs_b1": comp_fmt.get("point"),
        "format_valid_improvement_ci": [comp_fmt.get("ci_low"), comp_fmt.get("ci_high")],
        "thresholds": hy["HY1"],
        "comparator": "B1 on Q-TEST",
    }
    hy1["verdict"] = (
        "confirmed"
        if (
            (ft_q.get("citation_valid_share") or 0) >= hy["HY1"]["citation_valid_share"]
            and (ft_q.get("format_valid") or 0) >= hy["HY1"]["format_valid"]
            and (comp_fmt.get("ci_low") or 0) > hy["HY1"]["improvement_ci_lower_gt"]
        )
        else "refuted"
    )

    hy2 = {
        "verdict": None,
        "answer_f1_ft_main_qtest": (ft_q.get("answer") or {}).get("answer_f1"),
        "answer_f1_b1_qtest": (b1_q.get("answer") or {}).get("answer_f1"),
        "answer_f1_delta": comp_f1.get("point"),
        "answer_f1_delta_ci": [comp_f1.get("ci_low"), comp_f1.get("ci_high")],
        "thresholds": hy["HY2"],
        "comparator": "B1 on Q-TEST",
    }
    hy2["verdict"] = (
        "confirmed"
        if (
            (comp_f1.get("point") or 0) >= hy["HY2"]["answer_f1_delta"]
            and (comp_f1.get("ci_low") or 0) > hy["HY2"]["ci_lower_gt"]
        )
        else "refuted"
    )

    hy3 = {
        "verdict": None,
        "abstain_acc_ft_main_qtest": ft_q.get("abstain_acc"),
        "abstain_acc_b1_qtest": b1_q.get("abstain_acc"),
        "abstain_acc_delta": comp_ab.get("point"),
        "abstain_acc_delta_ci": [comp_ab.get("ci_low"), comp_ab.get("ci_high")],
        "over_refusal_ft_main_qtest": ft_q.get("over_refusal"),
        "thresholds": hy["HY3"],
        "comparator": "B1 on Q-TEST",
    }
    hy3["verdict"] = (
        "confirmed"
        if (
            (comp_ab.get("point") or 0) >= hy["HY3"]["abstain_acc_delta"]
            and (ft_q.get("over_refusal") or 1) <= hy["HY3"]["over_refusal_max"]
        )
        else "refuted"
    )

    ft_fmt_min = min(
        x for x in [ft_h.get("format_valid"), ft_p.get("format_valid")] if x is not None
    )
    hy4 = {
        "verdict": None,
        "format_valid_min_ft_main_hold_paper16": ft_fmt_min,
        "keyfact_cov_delta_hold": comp_kf_h.get("point"),
        "keyfact_cov_delta_hold_ci": [comp_kf_h.get("ci_low"), comp_kf_h.get("ci_high")],
        "keyfact_cov_delta_paper16": comp_kf_p.get("point"),
        "keyfact_cov_delta_paper16_ci": [comp_kf_p.get("ci_low"), comp_kf_p.get("ci_high")],
        "thresholds": hy["HY4"],
        "comparator": "B1 on HOLD and PAPER16",
    }
    hy4["verdict"] = (
        "confirmed"
        if (
            ft_fmt_min >= hy["HY4"]["format_valid_min"]
            and (comp_kf_h.get("point") or 0) >= hy["HY4"]["keyfact_cov_delta"]
        )
        else "refuted"
    )

    # --- HY5 (from red-team + regression, when both exist) ---
    hy5: dict = {
        "verdict": "not-evaluated-here",
        "note": "Evaluated after F5.4 (red-team) and F5.5 (ARC-Easy regression).",
        "thresholds": hy["HY5"],
    }
    rt_path = REPO_ROOT / "evaluation" / "redteam" / "results.json"
    reg_path = RESULTS_DIR / "regression.json"
    if rt_path.exists() and reg_path.exists():
        rt = json.loads(rt_path.read_text(encoding="utf-8"))
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        drop = reg["difference_ft_main_minus_b0"]["drop"]
        extra = rt["extra_successes_ft_main_vs_b0"]
        hy5 = {
            "verdict": (
                "confirmed"
                if (
                    drop <= hy["HY5"]["arc_easy_drop_max"]
                    and extra <= hy["HY5"]["redteam_extra_max"]
                )
                else "refuted"
            ),
            "arc_easy_b0_accuracy": reg["per_model"]["B0"]["accuracy"],
            "arc_easy_ft_main_accuracy": reg["per_model"]["FT-main"]["accuracy"],
            "arc_easy_drop_b0_minus_ft_main": drop,
            "arc_easy_diff_ci": [
                reg["difference_ft_main_minus_b0"]["ci_low"],
                reg["difference_ft_main_minus_b0"]["ci_high"],
            ],
            "redteam_b0_succeeded": rt["total_succeeded"]["B0"],
            "redteam_ft_main_succeeded": rt["total_succeeded"]["FT-main"],
            "redteam_extra_successes_ft_main_vs_b0": extra,
            "thresholds": hy["HY5"],
            "comparator": "B0 (base model)",
        }

    hypotheses = {
        "HY1": hy1,
        "HY2": hy2,
        "HY3": hy3,
        "HY4": hy4,
        "HY5": hy5,
        "HY6": {
            "verdict": "not-evaluated-here",
            "note": "Deployment fidelity; evaluated in F6.3.",
            "thresholds": hy["HY6"],
        },
    }

    out_tm = RESULTS_DIR / "test_metrics.json"
    out_hy = RESULTS_DIR / "hypotheses.json"
    if args.recompute_only:
        ok = True
        for path, fresh in [(out_tm, test_metrics), (out_hy, hypotheses)]:
            if not path.exists():
                print(f"RECOMPUTE MISSING: {path}")
                ok = False
                continue
            committed = json.loads(path.read_text(encoding="utf-8"))
            if committed != fresh:
                print(f"RECOMPUTE MISMATCH: {path}")
                ok = False
            else:
                print(f"RECOMPUTE MATCH: {path}")
        return 0 if ok else 1

    out_tm.write_text(
        json.dumps(test_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    out_hy.write_text(
        json.dumps(hypotheses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out_tm}")
    print(f"wrote {out_hy}")
    for k in ("HY1", "HY2", "HY3", "HY4", "HY5"):
        print(f"{k}: {hypotheses[k]['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
