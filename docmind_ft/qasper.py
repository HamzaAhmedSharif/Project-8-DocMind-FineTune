"""QASPER v0.3 loader (F1.1).

Normalizes one split file into papers whose paragraphs carry half-open word
offsets over the document's whitespace-split stream. Section titles are
paragraphs in their own right, so offsets and citations follow the same word
count as the text a model sees. Everything here is a pure function except
:func:`load_qasper`, which reads one JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Answer:
    """One annotator answer; an unanswerable annotation leaves the fields empty."""

    annotation_id: str
    worker_id: str
    unanswerable: bool
    extractive_spans: tuple[str, ...]
    yes_no: bool | None  # None for non-yes/no answers (2266 in train)
    free_form_answer: str
    evidence: tuple[str, ...]
    highlighted_evidence: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    answers: tuple[Answer, ...]


@dataclass(frozen=True)
class Paragraph:
    section: str
    text: str
    word_start: int
    word_end: int  # exclusive


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    paragraphs: tuple[Paragraph, ...]
    questions: tuple[Question, ...]


def _answer(raw: dict[str, Any]) -> Answer:
    answer = raw["answer"]
    return Answer(
        annotation_id=raw["annotation_id"],
        worker_id=raw["worker_id"],
        unanswerable=answer["unanswerable"],
        extractive_spans=tuple(answer["extractive_spans"]),
        yes_no=answer["yes_no"],
        free_form_answer=answer["free_form_answer"],
        evidence=tuple(answer["evidence"]),
        highlighted_evidence=tuple(answer["highlighted_evidence"]),
    )


def _question(raw: dict[str, Any]) -> Question:
    return Question(
        question_id=raw["question_id"],
        question=raw["question"],
        answers=tuple(_answer(answer) for answer in raw["answers"]),
    )


def _paragraphs(full_text: list[dict[str, Any]]) -> tuple[Paragraph, ...]:
    paragraphs = []
    offset = 0
    for section in full_text:
        # Real data has null section names and null/blank paragraphs; they
        # carry no words, so skip them (offsets are unaffected).
        name = section["section_name"] or ""
        for text in (name, *section["paragraphs"]):
            if text is None or not text.strip():
                continue
            end = offset + len(text.split())
            paragraphs.append(Paragraph(name, text, offset, end))
            offset = end
    return tuple(paragraphs)


def normalize_paper(paper_id: str, raw: dict[str, Any]) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=raw["title"] or "",
        abstract=raw["abstract"] or "",
        paragraphs=_paragraphs(raw["full_text"]),
        questions=tuple(_question(question) for question in raw["qas"]),
    )


def load_qasper(path: str | Path) -> list[Paper]:
    """Load one split file (train, dev, or test) preserving file order."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [normalize_paper(paper_id, paper) for paper_id, paper in raw.items()]
