#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Compute the section 3.8 metrics from raw generation rows.

    python scripts/compute_metrics.py --split dev
    python scripts/compute_metrics.py --split hold --outputs evaluation/results/hold.jsonl

Test splits are refused unless ``DOCMIND_FT_ALLOW_TEST=1`` is set; only
``scripts\\final_eval.ps1`` sets it, in F5.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: F401, E402  # installs the docmind_service alias

from docmind_ft.config import Settings  # noqa: E402
from docmind_ft.contexts import CONTEXTS_DIR, HOLD_QUESTIONS_PATH, PAPER16_QUESTIONS_PATH  # noqa: E402
from docmind_ft.generate import ChunkTextResolver  # noqa: E402
from docmind_ft.metrics import (  # noqa: E402
    OutputRow,
    abstain_metrics,
    answer_metrics,
    citation_metrics,
    citation_support,
    efficiency_summary,
    format_valid,
    gold_recall,
    hold_fact_in_text,
    mean_answer_length,
    paper16_fact_in_text,
    score_hold_answer,
    score_paper16_answer,
    summarize_facts,
)
from docmind_ft.qasper import load_qasper  # noqa: E402
from docmind_ft.teacher import expected_yes_no, reference_answers  # noqa: E402

QASPER_DIR = Path(r"B:\p8\data\qasper")
QASPER_FILES = {"dev": "qasper-dev-v0.3.json", "qtest": "qasper-test-v0.3.json"}
CONTEXT_FILES = {
    "dev": CONTEXTS_DIR / "contexts_qdev.jsonl",
    "qtest": CONTEXTS_DIR / "contexts_qtest.jsonl",
    "hold": CONTEXTS_DIR / "contexts_hold.jsonl",
    "paper16": CONTEXTS_DIR / "contexts_paper16.jsonl",
}
WORK_QUESTIONS = Path(r"B:\p8\data\work\questions.jsonl")
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
TEST_SPLITS = frozenset({"qtest", "hold", "paper16"})


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def _share(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _work_rows() -> dict[str, dict]:
    return {str(row["question_id"]): row for row in _read_jsonl(WORK_QUESTIONS)}


def _qasper_gold(split: str) -> dict[str, dict]:
    work = _work_rows()
    gold: dict[str, dict] = {}
    for paper in load_qasper(QASPER_DIR / QASPER_FILES[split]):
        for question in paper.questions:
            entry = work.get(question.question_id)
            if entry is None:
                continue
            gold[question.question_id] = {
                "gold_answers": tuple(reference_answers(question.answers)),
                "yes_no": expected_yes_no(question.answers),
                "gold_chunk_ids": tuple(entry.get("gold_chunk_ids") or ()),
                "answerable": bool(entry.get("answerable")),
                "answer_type": str(entry.get("answer_type") or ""),
            }
    return gold


def _hold_gold() -> dict[str, dict]:
    gold: dict[str, dict] = {}
    for item in _read_jsonl(HOLD_QUESTIONS_PATH):
        answerable = bool(item.get("answerable"))
        gold[str(item["id"])] = {
            "gold_answers": (str(item["gold_answer"]),) if answerable else (),
            "yes_no": None,
            "gold_chunk_ids": (),
            "answerable": answerable,
            "answer_type": str(item.get("type") or ""),
            "key_facts": tuple(item.get("key_facts") or ()),
        }
    return gold


def _paper16_questions() -> list[dict]:
    spec = importlib.util.spec_from_file_location("p8_paper16_questions", PAPER16_QUESTIONS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return list(module.QUESTIONS)


def _paper16_gold() -> dict[str, dict]:
    gold: dict[str, dict] = {}
    for item in _paper16_questions():
        gold[str(item["id"])] = {
            "gold_answers": (),
            "yes_no": None,
            "gold_chunk_ids": (),
            "answerable": item.get("type") == "doc",
            "answer_type": str(item.get("type") or ""),
            "key_facts": tuple(item.get("key_facts") or ()),
        }
    return gold


def _gold_for(split: str) -> dict[str, dict]:
    if split in ("dev", "qtest"):
        return _qasper_gold(split)
    if split == "hold":
        return _hold_gold()
    if split == "paper16":
        return _paper16_gold()
    raise ValueError(f"unknown split: {split!r}")


def compute(split: str, outputs_path: Path) -> dict:
    contexts = _read_jsonl(CONTEXT_FILES[split])
    outputs = _read_jsonl(outputs_path)
    by_id: dict[str, dict] = {}
    for row in outputs:
        question_id = str(row["question_id"])
        if question_id in by_id:
            raise ValueError(f"{outputs_path}: duplicate output for {question_id!r}")
        by_id[question_id] = row
    context_ids = {str(context["question_id"]) for context in contexts}
    extra = sorted(set(by_id) - context_ids)
    if extra:
        raise ValueError(
            f"{outputs_path}: {len(extra)} output ids are not in split {split!r} (e.g. {extra[:5]})"
        )

    gold = _gold_for(split)
    resolver = ChunkTextResolver()
    rows: list[OutputRow] = []
    raw_rows: list[dict] = []
    context_texts: dict[str, str] = {}
    for context in contexts:
        question_id = str(context["question_id"])
        output = by_id.get(question_id)
        if output is None:
            continue
        entry = gold.get(question_id)
        if entry is None:
            raise ValueError(f"no gold information for {question_id!r} on split {split!r}")
        paper_id = str(context["paper_id"])
        raw_chunk_ids = tuple(str(item["chunk_id"]) for item in context["chunks"])
        short_ids = tuple(f"c{i}" for i in range(1, len(raw_chunk_ids) + 1))
        texts = tuple(resolver.text(chunk_id, paper_id, split) for chunk_id in raw_chunk_ids)
        context_text = "\n".join(texts)
        context_texts[question_id] = context_text
        gold_chunk_ids = tuple(entry["gold_chunk_ids"])
        key_facts = tuple(entry.get("key_facts") or ())
        if split == "hold":
            gold_chunk_ids = tuple(
                chunk_id
                for chunk_id, text in zip(raw_chunk_ids, texts, strict=True)
                if any(hold_fact_in_text(fact, text) for fact in key_facts)
            )
        elif split == "paper16":
            gold_chunk_ids = tuple(
                chunk_id
                for chunk_id, text in zip(raw_chunk_ids, texts, strict=True)
                if any(paper16_fact_in_text(fact, text) for fact in key_facts)
            )
        gold_ids = set(gold_chunk_ids)
        short_gold_ids = tuple(
            short for short, raw in zip(short_ids, raw_chunk_ids, strict=True) if raw in gold_ids
        )
        rows.append(
            OutputRow(
                question_id=question_id,
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
        raw_rows.append(output)

    result = {
        "split": split,
        "outputs": str(outputs_path),
        "n_outputs": len(rows),
        "n_expected": len(contexts),
        "format_valid": _share([format_valid(row.output, row.prompt_chunk_ids) for row in rows]),
        **citation_metrics(rows),
        "citation_support": citation_support(rows),
        **abstain_metrics(rows),
        "gold_recall": gold_recall(rows),
        "mean_answer_length": mean_answer_length(rows),
        "efficiency": efficiency_summary(
            [float(row.get("seconds") or 0.0) for row in raw_rows],
            [row.get("generated_tokens") for row in raw_rows],
            [row.get("prompt_tokens") for row in raw_rows],
        ),
    }
    if split in ("dev", "qtest"):
        result["answer"] = answer_metrics(rows)
    else:
        scores = []
        per_question = []
        for row in rows:
            key_facts = gold[row.question_id].get("key_facts") or ()
            if not key_facts:
                continue
            context_text = context_texts[row.question_id]
            if split == "hold":
                score = score_hold_answer(row.output, key_facts, context_text)
            else:
                score = score_paper16_answer(row.output, key_facts, context_text)
            scores.append(score)
            per_question.append(
                {
                    "question_id": row.question_id,
                    "found": sum(score.hits),
                    "total": len(score.hits),
                    "all_found": all(score.hits),
                }
            )
        result["keyfacts"] = summarize_facts(scores)
        result["keyfacts"]["per_question"] = per_question
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=tuple(CONTEXT_FILES))
    parser.add_argument(
        "--outputs",
        type=Path,
        default=None,
        help="raw generation JSONL (default: evaluation/results/<split>.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="metrics JSON (default: evaluation/results/metrics_<split>.json)",
    )
    args = parser.parse_args(argv)

    if args.split in TEST_SPLITS and not Settings.from_env().docmind_ft_allow_test:
        print(
            f"refusing split {args.split!r}: DOCMIND_FT_ALLOW_TEST=1 is required for a test split",
            file=sys.stderr,
        )
        return 2

    outputs_path = args.outputs or RESULTS_DIR / f"{args.split}.jsonl"
    if not outputs_path.exists():
        print(f"error: outputs file not found: {outputs_path}", file=sys.stderr)
        return 2
    result = compute(args.split, outputs_path)

    out_path = args.out or RESULTS_DIR / f"metrics_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: result[key] for key in ("split", "n_outputs", "n_expected")}))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
