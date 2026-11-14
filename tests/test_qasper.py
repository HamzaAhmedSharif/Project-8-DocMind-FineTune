"""Tests for the QASPER v0.3 loader (F1.1), on a two-paper synthetic fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docmind_ft.qasper import Paper, load_qasper


def _answer(
    *,
    annotation_id: str = "a1",
    worker_id: str = "w1",
    unanswerable: bool = False,
    extractive_spans: tuple[str, ...] = (),
    yes_no: bool = False,
    free_form_answer: str = "",
    evidence: tuple[str, ...] = (),
    highlighted_evidence: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "answer": {
            "unanswerable": unanswerable,
            "extractive_spans": list(extractive_spans),
            "yes_no": yes_no,
            "free_form_answer": free_form_answer,
            "evidence": list(evidence),
            "highlighted_evidence": list(highlighted_evidence),
        },
        "annotation_id": annotation_id,
        "worker_id": worker_id,
    }


def _question(question_id: str, question: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    return {"question": question, "question_id": question_id, "answers": answers}


RAW: dict[str, Any] = {
    "p1": {
        "title": "Paper One",
        "abstract": "An abstract about one.",
        "full_text": [
            {"section_name": "FLOAT SELECTED", "paragraphs": ["Table 1: results."]},
            {"section_name": "Introduction", "paragraphs": ["One two three four.", "Five six."]},
            {"section_name": "Method", "paragraphs": ["Seven eight nine ten eleven."]},
        ],
        "qas": [
            _question(
                "q-extractive",
                "What is one?",
                [_answer(extractive_spans=("one two",), evidence=("One two three four.",))],
            ),
            _question("q-yesno", "Is five six?", [_answer(yes_no=True, evidence=("Five six.",))]),
            _question(
                "q-freeform",
                "Why seven?",
                [
                    _answer(
                        free_form_answer="Because eight.",
                        evidence=("Seven eight nine ten eleven.",),
                    )
                ],
            ),
            _question("q-unanswerable", "What is missing?", [_answer(unanswerable=True)]),
            _question(
                "q-mixed",
                "What is five?",
                [
                    _answer(
                        annotation_id="a2",
                        worker_id="w2",
                        extractive_spans=("five six",),
                        evidence=("Five six.",),
                    ),
                    _answer(annotation_id="a3", worker_id="w3", unanswerable=True),
                ],
            ),
        ],
        "figures_and_tables": [],
    },
    "p2": {
        "title": "Paper Two",
        "abstract": None,
        "full_text": [{"section_name": None, "paragraphs": [None, "Only paragraph here."]}],
        "qas": [_question("q-test", "What is here?", [])],
        "figures_and_tables": [],
    },
}

EXPECTED_PARAGRAPHS = [
    ("FLOAT SELECTED", "FLOAT SELECTED", 0, 2),
    ("FLOAT SELECTED", "Table 1: results.", 2, 5),
    ("Introduction", "Introduction", 5, 6),
    ("Introduction", "One two three four.", 6, 10),
    ("Introduction", "Five six.", 10, 12),
    ("Method", "Method", 12, 13),
    ("Method", "Seven eight nine ten eleven.", 13, 18),
]


def _load_fixture(tmp_path: Path) -> list[Paper]:
    path = tmp_path / "qasper.json"
    path.write_text(json.dumps(RAW), encoding="utf-8")
    return load_qasper(path)


def _questions(tmp_path: Path) -> dict[str, Any]:
    return {question.question_id: question for question in _load_fixture(tmp_path)[0].questions}


def test_three_answer_types(tmp_path: Path) -> None:
    questions = _questions(tmp_path)

    extractive = questions["q-extractive"].answers[0]
    assert extractive.extractive_spans == ("one two",)
    assert extractive.yes_no is False
    assert extractive.free_form_answer == ""

    yes_no = questions["q-yesno"].answers[0]
    assert yes_no.yes_no is True
    assert yes_no.extractive_spans == ()
    assert yes_no.free_form_answer == ""

    free_form = questions["q-freeform"].answers[0]
    assert free_form.free_form_answer == "Because eight."
    assert free_form.extractive_spans == ()
    assert free_form.yes_no is False


def test_unanswerable_question(tmp_path: Path) -> None:
    answer = _questions(tmp_path)["q-unanswerable"].answers[0]
    assert answer.unanswerable is True
    assert answer.extractive_spans == ()
    assert answer.evidence == ()


def test_mixed_annotations_are_preserved(tmp_path: Path) -> None:
    answers = _questions(tmp_path)["q-mixed"].answers
    assert [answer.unanswerable for answer in answers] == [False, True]
    assert answers[0].extractive_spans == ("five six",)
    assert answers[1].extractive_spans == ()
    assert (answers[0].annotation_id, answers[0].worker_id) == ("a2", "w2")


def test_float_selected_paragraphs_kept(tmp_path: Path) -> None:
    paragraphs = _load_fixture(tmp_path)[0].paragraphs
    floats = [paragraph for paragraph in paragraphs if paragraph.section == "FLOAT SELECTED"]
    assert [paragraph.text for paragraph in floats] == ["FLOAT SELECTED", "Table 1: results."]


def test_paragraph_offsets_across_sections(tmp_path: Path) -> None:
    paragraphs = _load_fixture(tmp_path)[0].paragraphs
    assert [
        (paragraph.section, paragraph.text, paragraph.word_start, paragraph.word_end)
        for paragraph in paragraphs
    ] == EXPECTED_PARAGRAPHS
    words = [word for paragraph in paragraphs for word in paragraph.text.split()]
    for paragraph in paragraphs:
        assert words[paragraph.word_start : paragraph.word_end] == paragraph.text.split()


def test_none_section_names_and_paragraphs_skipped(tmp_path: Path) -> None:
    paper = _load_fixture(tmp_path)[1]
    assert paper.abstract == ""
    assert [
        (paragraph.section, paragraph.text, paragraph.word_start, paragraph.word_end)
        for paragraph in paper.paragraphs
    ] == [("", "Only paragraph here.", 0, 3)]


def test_test_split_questions_have_no_annotations(tmp_path: Path) -> None:
    papers = _load_fixture(tmp_path)
    assert [paper.paper_id for paper in papers] == ["p1", "p2"]
    assert papers[1].title == "Paper Two"
    assert papers[1].questions[0].answers == ()
