"""Tests for the HOLD-set question verifier (F1.5); no network is used."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from docmind_ft.holdout import verify_questions


class AccuracyEmbedder:
    """Tiny deterministic embedder: chunks with the word accuracy rank first."""

    dim = 2

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if "accuracy" in text.lower() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


class MarkerEmbedder:
    """Tiny deterministic embedder that puts the fact chunk last."""

    dim = 2

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[0.0, 1.0] if "hiddenfact" in text.lower() else [1.0, 0.0] for text in texts],
            dtype=np.float32,
        )


def _write_pages(path: Path, paper_id: str, pages: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{paper_id}.pages.json").write_text(
        json.dumps(pages, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )


def _write_questions(path: Path, items: list[dict]) -> Path:
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _item(**overrides) -> dict:
    item = {
        "id": "h00",
        "paper_id": "p1",
        "question": "What is reported?",
        "gold_answer": "a fact",
        "key_facts": [{"verbatim": "a fact", "accept": [r"a\s+fact"]}],
        "evidence_pages": [1],
        "type": "factual",
        "answerable": True,
        "distinctive_terms": [],
        "authored_by": "agent",
    }
    item.update(overrides)
    return item


def _main_fixture(tmp_path: Path) -> tuple[Path, Path]:
    pages = [
        (
            "Alpha study reports 42 percent accuracy on the benchmark. "
            "The method uses sparse retrieval and a small model."
        ),
        "A second page describes gamma analysis and delta results.",
    ]
    pages_dir = tmp_path / "holdout"
    _write_pages(pages_dir, "p1", pages)
    long_question = " ".join(["word"] * 41) + "?"
    items = [
        _item(
            id="h_good",
            question="What accuracy did the Alpha study report?",
            gold_answer="42 percent accuracy",
            key_facts=[{"verbatim": "42 percent accuracy", "accept": [r"42\s+percent\s+accuracy"]}],
        ),
        _item(
            id="h_badfact",
            question="What recall did the study report?",
            gold_answer="99 percent recall",
            key_facts=[{"verbatim": "99 percent recall", "accept": [r"99\s+percent\s+recall"]}],
        ),
        _item(
            id="h_unanswerable",
            question="What is the Delta secret?",
            gold_answer="I could not find relevant information in the indexed documents.",
            key_facts=[],
            evidence_pages=[],
            type="unanswerable",
            answerable=False,
            distinctive_terms=["alpha"],
        ),
        _item(
            id="h_duplicate",
            question="What accuracy did the Alpha study report?",
            gold_answer="42 percent accuracy",
            key_facts=[{"verbatim": "42 percent accuracy", "accept": [r"42\s+percent\s+accuracy"]}],
        ),
        _item(
            id="h_question",
            question="Does the method use sparse retrieval?",
            gold_answer="sparse retrieval",
            key_facts=[{"verbatim": "sparse retrieval", "accept": [r"sparse\s+retrieval"]}],
        ),
        _item(
            id="h_short",
            question="Which architecture is used?",
            gold_answer="small model",
            key_facts=[{"verbatim": "small model", "accept": ["m"]}],
        ),
        _item(
            id="h_stop",
            question="Which approach is used?",
            gold_answer="the method",
            key_facts=[{"verbatim": "the method", "accept": ["the"]}],
        ),
        _item(
            id="h_long",
            question=long_question,
            gold_answer="benchmark",
            key_facts=[{"verbatim": "benchmark", "accept": ["benchmark"]}],
        ),
        _item(
            id="h_mismatch",
            question="What was found?",
            gold_answer="gamma analysis",
            key_facts=[{"verbatim": "gamma analysis", "accept": [r"delta\s+results"]}],
            evidence_pages=[2],
        ),
        _item(
            id="h_good",
            question="What did the method use?",
            gold_answer="sparse retrieval",
            key_facts=[{"verbatim": "sparse retrieval", "accept": [r"sparse\s+retrieval"]}],
        ),
    ]
    questions = _write_questions(tmp_path / "holdout_questions.jsonl", items)
    return questions, pages_dir


@pytest.fixture
def main_result(tmp_path: Path):
    questions, pages_dir = _main_fixture(tmp_path)
    return verify_questions(questions, pages_dir, embedder=AccuracyEmbedder())


def test_good_item_passes_and_is_retrievable(main_result) -> None:
    good = main_result.results[0]

    assert good.item_id == "h_good"
    assert good.passed is True
    assert good.failures == []
    assert good.retrievable is True


def test_key_fact_not_in_paper_is_caught(main_result) -> None:
    assert any(
        failure.item_id == "h_badfact" and failure.check == "verbatim_not_on_evidence_page"
        for failure in main_result.failures
    )


def test_unanswerable_distinctive_term_present_is_caught(main_result) -> None:
    assert any(
        failure.item_id == "h_unanswerable" and failure.check == "distinctive_term_present"
        for failure in main_result.failures
    )


def test_duplicate_question_is_caught(main_result) -> None:
    assert any(
        failure.item_id == "h_duplicate" and failure.check == "duplicate_question"
        for failure in main_result.failures
    )


def test_duplicate_id_is_caught(main_result) -> None:
    assert any(failure.check == "duplicate_id" for failure in main_result.failures)


def test_accept_pattern_matching_question_is_caught(main_result) -> None:
    assert any(
        failure.item_id == "h_question" and failure.check == "accept_matches_question"
        for failure in main_result.failures
    )


def test_accept_pattern_shorter_than_three_chars_is_rejected(main_result) -> None:
    assert any(
        failure.item_id == "h_short" and failure.check == "accept_too_short"
        for failure in main_result.failures
    )


def test_bare_common_word_pattern_is_rejected(main_result) -> None:
    assert any(
        failure.item_id == "h_stop" and failure.check == "accept_stopword"
        for failure in main_result.failures
    )


def test_question_over_40_words_is_rejected(main_result) -> None:
    assert any(
        failure.item_id == "h_long" and failure.check == "question_too_long"
        for failure in main_result.failures
    )


def test_accept_pattern_not_matching_verbatim_is_rejected(main_result) -> None:
    assert any(
        failure.item_id == "h_mismatch" and failure.check == "accept_mismatch"
        for failure in main_result.failures
    )


def test_unretrievable_item_is_kept_not_failed(tmp_path: Path) -> None:
    pages_dir = tmp_path / "holdout"
    filler = " ".join(f"w{index}" for index in range(1500))
    _write_pages(pages_dir, "p_out", [f"{filler} hiddenfact"])
    item = _item(
        id="h_out",
        paper_id="p_out",
        question="Find the hidden fact",
        gold_answer="hiddenfact",
        key_facts=[{"verbatim": "hiddenfact", "accept": ["hiddenfact"]}],
    )
    questions = _write_questions(tmp_path / "holdout_questions.jsonl", [item])

    result = verify_questions(questions, pages_dir, embedder=MarkerEmbedder())

    assert result.items == 1
    assert result.failed == 0
    assert result.results[0].passed is True
    assert result.results[0].retrievable is False
    assert result.retrievable == 0
