"""Tests for the leakage gate I1 (F1.7) on hand-made fixtures; no GPU/network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docmind_ft.leakage import (  # noqa: E402
    P7_MANIFEST,
    QASPER_DIR,
    TRAIN_DIR,
    WORK_DIR,
    check_ngram_overlap,
    check_paper_id_overlap,
    check_title_overlap,
    run_all_checks,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _make_work_questions(tmp_path: Path, entries: dict[str, tuple[str, str, str]]) -> Path:
    """question_id -> (paper_id, question, split) written as work-style JSONL."""
    rows = []
    for qid, value in entries.items():
        paper_id, question, split = value
        rows.append(
            {
                "split": split,
                "paper_id": paper_id,
                "question_id": qid,
                "question": question,
            }
        )
    path = tmp_path / "questions.jsonl"
    _write_jsonl(path, rows)
    return path


def _make_split(path: Path, ids: list[str]) -> None:
    path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Check 1: paper-id overlap
# ---------------------------------------------------------------------------


def test_check1_passing_no_paper_overlap(tmp_path: Path) -> None:
    work = _make_work_questions(
        tmp_path,
        {
            "q1": ("p1", "Q one", "train"),
            "q2": ("p2", "Q two", "train"),
            "q3": ("p3", "Q three", "dev"),
            "q4": ("p4", "Q four", "test"),
        },
    )
    _make_split(tmp_path / "qdev.json", ["q3"])
    _make_split(tmp_path / "qtest.json", ["q4"])

    result = check_paper_id_overlap(work, tmp_path / "qdev.json", tmp_path / "qtest.json")
    assert result["passed"] is True
    assert result["violations"] == []
    assert result["counts"]["train_papers"] == 2
    assert result["counts"]["dev_papers"] == 1
    assert result["counts"]["test_papers"] == 1


def test_check1_detects_shared_paper_id(tmp_path: Path) -> None:
    # p1 appears in both train (q1) and dev (q3).
    work = _make_work_questions(
        tmp_path,
        {
            "q1": ("p1", "Q one", "train"),
            "q2": ("p2", "Q two", "train"),
            "q3": ("p1", "Q three", "dev"),
            "q4": ("p4", "Q four", "test"),
        },
    )
    _make_split(tmp_path / "qdev.json", ["q3"])
    _make_split(tmp_path / "qtest.json", ["q4"])

    result = check_paper_id_overlap(work, tmp_path / "qdev.json", tmp_path / "qtest.json")
    assert result["passed"] is False
    assert any(v["pair"] == "train-dev" and "p1" in v["paper_ids"] for v in result["violations"])


# ---------------------------------------------------------------------------
# Check 2: title overlap
# ---------------------------------------------------------------------------


def _make_qasper_split(path: Path, title: str) -> None:
    raw = {
        "paper1": {
            "title": title,
            "abstract": "abstract",
            "full_text": [],
            "qas": [],
        }
    }
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8", newline="\n")


def _make_qasper_dir(qasper_dir: Path, titles: dict[str, str]) -> None:
    qasper_dir.mkdir(parents=True, exist_ok=True)
    for split, title in titles.items():
        _make_qasper_split(qasper_dir / f"qasper-{split}-v0.3.json", title)


def _make_manifest(path: Path, titles: list[str]) -> None:
    path.write_text(
        json.dumps({"papers": [{"title": t} for t in titles]}, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )


def test_check2_passing_no_title_overlap(tmp_path: Path) -> None:
    qasper_dir = tmp_path / "qasper"
    _make_qasper_dir(
        qasper_dir,
        {
            "train": "Some Unique Paper",
            "dev": "Another Unique Paper",
            "test": "Yet Another Unique Paper",
        },
    )
    manifest = tmp_path / "manifest.json"
    _make_manifest(manifest, ["Other Completely Different Title"])

    result = check_title_overlap(qasper_dir, manifest)
    assert result["passed"] is True
    assert result["violations"] == []


def test_check2_detects_shared_title_with_hold(tmp_path: Path) -> None:
    qasper_dir = tmp_path / "qasper"
    _make_qasper_dir(
        qasper_dir,
        {
            "train": "Hold Paper One",
            "dev": "Another Unique Paper",
            "test": "Yet Another Unique Paper",
        },
    )
    manifest = tmp_path / "manifest.json"
    _make_manifest(manifest, ["hold paper one"])  # normalized match

    result = check_title_overlap(qasper_dir, manifest)
    assert result["passed"] is False
    assert any(v["side"] == "hold-manifest" for v in result["violations"])


# ---------------------------------------------------------------------------
# Check 3: 13-gram matches
# ---------------------------------------------------------------------------


_HOLD_13GRAM = "the quick brown fox jumps over the lazy dog today and tomorrow morning"


def _make_hold(path: Path) -> None:
    rows = [
        {
            "id": "h01",
            "paper_id": "x1",
            "question": _HOLD_13GRAM,
            "gold_answer": "some unrelated answer",
            "key_facts": [{"verbatim": "unrelated fact"}],
            "evidence_pages": [1],
            "type": "factual",
            "answerable": True,
            "distinctive_terms": [],
            "authored_by": "agent",
        }
    ]
    _write_jsonl(path, rows)


def _make_paper16(path: Path) -> None:
    source = (
        "QUESTIONS = [\n"
        '    {"id": "d01", "question": "unrelated paper16 question", '
        '"key_facts": [{"fact": "unrelated fact"}]}\n'
        "]\n"
    )
    path.write_text(source, encoding="utf-8", newline="\n")


def test_check3_passing_no_ngram_overlap(tmp_path: Path) -> None:
    train_dir = tmp_path / "data"
    train_dir.mkdir()
    # Train text has 13+ tokens but shares no 13-gram with the hold question.
    train_row = {
        "prompt": [
            {
                "role": "user",
                "content": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu",
            }
        ],
        "completion": [{"role": "assistant", "content": "answer"}],
        "meta": {"question_id": "t1"},
    }
    _write_jsonl(train_dir / "train.jsonl", [train_row])
    _write_jsonl(train_dir / "train_nodistill.jsonl", [])
    _write_jsonl(train_dir / "train_data25.jsonl", [])

    hold = tmp_path / "hold.jsonl"
    _make_hold(hold)
    paper16 = tmp_path / "paper16_questions.py"
    _make_paper16(paper16)

    result = check_ngram_overlap(train_dir, hold, paper16)
    assert result["passed"] is True
    assert result["violations"] == []
    assert result["counts"]["train_13gram_instances"] == 2
    assert result["counts"]["eval_13gram_distinct"] >= 1


def test_check3_detects_ngram_leak(tmp_path: Path) -> None:
    train_dir = tmp_path / "data"
    train_dir.mkdir()
    train_row = {
        "prompt": [
            {
                "role": "user",
                "content": "prefix " + _HOLD_13GRAM + " suffix",
            }
        ],
        "completion": [{"role": "assistant", "content": "answer"}],
        "meta": {"question_id": "t_leak"},
    }
    _write_jsonl(train_dir / "train.jsonl", [train_row])
    _write_jsonl(train_dir / "train_nodistill.jsonl", [])
    _write_jsonl(train_dir / "train_data25.jsonl", [])

    hold = tmp_path / "hold.jsonl"
    _make_hold(hold)
    paper16 = tmp_path / "paper16_questions.py"
    _make_paper16(paper16)

    result = check_ngram_overlap(train_dir, hold, paper16)
    assert result["passed"] is False
    assert any(
        v["question_id"] == "t_leak" and v["ngram"] == _HOLD_13GRAM for v in result["violations"]
    )


# ---------------------------------------------------------------------------
# run_all_checks smoke
# ---------------------------------------------------------------------------


def test_run_all_checks_returns_three_results() -> None:
    if not (TRAIN_DIR / "train.jsonl").exists():
        pytest.skip("F1 train artifacts are not available")
    if not (WORK_DIR / "questions.jsonl").exists():
        pytest.skip("F1.2 artifacts are not available")
    if not QASPER_DIR.exists() or not P7_MANIFEST.exists():
        pytest.skip("QASPER data or the P7 held-out manifest is not available")
    results = run_all_checks()
    assert len(results) == 3
    assert [r["name"] for r in results] == [
        "paper-id overlap",
        "title overlap",
        "13-gram matches",
    ]
    assert all("passed" in r and "counts" in r and "violations" in r for r in results)
