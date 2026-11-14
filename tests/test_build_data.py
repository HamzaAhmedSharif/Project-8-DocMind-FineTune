"""Tests for the QASPER data engine (F1.2) on hand-made fixtures."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from docmind_ft.build_data import (
    ChunkData,
    QuestionData,
    chunk_paper,
    embeddings_for,
    gold_chunk_ids,
    label_question,
    select_split,
    word_page,
)
from docmind_ft.qasper import Answer, Paper, Paragraph, Question


def _answer(
    *,
    unanswerable: bool = False,
    extractive_spans: tuple[str, ...] = (),
    yes_no: bool | None = None,
    free_form_answer: str = "",
    evidence: tuple[str, ...] = (),
) -> Answer:
    return Answer(
        annotation_id="a1",
        worker_id="w1",
        unanswerable=unanswerable,
        extractive_spans=extractive_spans,
        yes_no=yes_no,
        free_form_answer=free_form_answer,
        evidence=evidence,
        highlighted_evidence=(),
    )


def _paper(*paragraphs: Paragraph, questions: tuple[Question, ...] = ()) -> Paper:
    return Paper(
        paper_id="p1",
        title="Paper",
        abstract="Abstract",
        paragraphs=paragraphs,
        questions=questions,
    )


def test_word_pages_start_at_one_and_flip_every_500_words() -> None:
    assert [word_page(offset) for offset in (0, 499, 500, 999, 1000)] == [1, 1, 2, 2, 3]


def test_chunk_offsets_and_pages_across_a_pseudo_page_boundary() -> None:
    words = [f"w{index}" for index in range(700)]
    paper = _paper(Paragraph("Intro", " ".join(words), 0, 700))
    chunks = chunk_paper(paper, "dev")

    assert [chunk.chunk_id for chunk in chunks] == [f"p1:{index}" for index in range(6)]
    assert [(chunk.word_start, chunk.word_end, chunk.page) for chunk in chunks] == [
        (0, 150, 1),
        (125, 275, 1),
        (250, 400, 1),
        (375, 525, 1),
        (500, 650, 2),
        (625, 700, 2),
    ]
    for chunk in chunks:
        assert chunk.text == " ".join(words[chunk.word_start : chunk.word_end])


def test_gold_eighty_word_paragraph_spans_two_chunks() -> None:
    before = " ".join(f"a{index}" for index in range(100))
    evidence_text = " ".join(f"t{index}" for index in range(80))
    after = " ".join(f"z{index}" for index in range(20))
    paper = _paper(
        Paragraph("Intro", before, 0, 100),
        Paragraph("Evidence", evidence_text, 100, 180),
        Paragraph("End", after, 180, 200),
    )
    chunks = chunk_paper(paper, "dev")
    question = Question(
        "q1",
        "What happened?",
        (_answer(extractive_spans=("t0",), evidence=(evidence_text,)),),
    )

    assert [(chunk.word_start, chunk.word_end) for chunk in chunks] == [(0, 150), (125, 200)]
    assert gold_chunk_ids(paper, question, chunks) == ("p1:0", "p1:1")


def test_gold_threshold_shrinks_for_a_short_paragraph() -> None:
    before = " ".join(f"a{index}" for index in range(138))
    evidence_text = " ".join(f"t{index}" for index in range(40))
    paper = _paper(
        Paragraph("Intro", before, 0, 138),
        Paragraph("Evidence", evidence_text, 138, 178),
    )
    chunks = chunk_paper(paper, "dev")
    question = Question("q1", "Q?", (_answer(evidence=(evidence_text,)),))

    assert gold_chunk_ids(paper, question, chunks) == ("p1:1",)


def test_gold_skips_float_selected_evidence() -> None:
    paper = _paper(
        Paragraph("FLOAT SELECTED", "FLOAT SELECTED", 0, 2),
        Paragraph("FLOAT SELECTED", "Table 1: results.", 2, 5),
        Paragraph("Intro", "alpha beta gamma delta", 5, 9),
    )
    chunks = chunk_paper(paper, "dev")
    question = Question(
        "q1",
        "Q?",
        (
            _answer(
                evidence=(
                    "FLOAT SELECTED: Table 1: results.",
                    "Table 1: results.",
                    "alpha beta gamma delta",
                )
            ),
        ),
    )

    assert gold_chunk_ids(paper, question, chunks) == ("p1:0",)


def test_question_labels_cover_all_annotator_combinations() -> None:
    unanswerable = _answer(unanswerable=True)
    extractive = _answer(extractive_spans=("span",))
    yes = _answer(yes_no=False)
    free_form = _answer(free_form_answer="Because.")

    all_unanswerable = Question("q1", "Q?", (unanswerable, unanswerable))
    mixed = Question("q2", "Q?", (extractive, unanswerable))
    yes_no = Question("q3", "Q?", (yes,))
    abstractive = Question("q4", "Q?", (free_form,))
    empty = Question("q5", "Q?", ())

    assert label_question(all_unanswerable).answer_type == "unanswerable"
    assert label_question(all_unanswerable).answerable is False
    assert label_question(mixed).answerable is True
    assert (label_question(mixed).answer_type, label_question(mixed).mixed) == ("extractive", True)
    assert label_question(yes_no).answer_type == "yes_no"
    assert label_question(abstractive).answer_type == "free_form"
    assert label_question(empty).answer_type == "unknown"
    assert label_question(empty).answerable is False


def test_split_is_stratified_capped_and_deterministic() -> None:
    records = []
    types = ["extractive"] * 5 + ["free_form"] * 2 + ["yes_no"] * 2 + ["unanswerable"]
    for paper_index in range(10):
        for offset, answer_type in enumerate(types):
            records.append(
                QuestionData(
                    split="dev",
                    paper_id=f"p{paper_index}",
                    question_id=f"q{paper_index * 10 + offset}",
                    question="Q?",
                    answer_type=answer_type,
                    answerable=answer_type != "unanswerable",
                    mixed=False,
                    gold_chunk_ids=(),
                )
            )

    ids = select_split(records, size=20, seed=20261207, min_papers=5)
    assert len(ids) == 20
    assert list(ids) == sorted(ids)
    assert ids == select_split(records, size=20, seed=20261207, min_papers=5)

    by_id = {record.question_id: record for record in records}
    selected = [by_id[question_id] for question_id in ids]
    assert Counter(record.answer_type for record in selected) == {
        "extractive": 10,
        "free_form": 4,
        "yes_no": 4,
        "unanswerable": 2,
    }
    per_paper = Counter(record.paper_id for record in selected)
    assert max(per_paper.values()) <= 3
    assert len(per_paper) >= 5


def test_split_refuses_an_impossible_size() -> None:
    record = QuestionData("dev", "p1", "q1", "Q?", "extractive", True, False, ())
    with pytest.raises(ValueError):
        select_split([record], size=2)


class _FakeEmbedder:
    dim = 4

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.ones((len(texts), self.dim), dtype=np.float32)


def test_embedding_cache_round_trip_and_invalidation(tmp_path: Path) -> None:
    embedder = _FakeEmbedder()
    chunk = ChunkData("dev", "p1", "p1:0", 1, 0, 0, 1, "one word")
    first = embeddings_for([chunk], tmp_path, embedder)
    second = embeddings_for([chunk], tmp_path, embedder)

    assert embedder.calls == 1
    assert np.array_equal(first, second)

    changed = ChunkData("dev", "p1", "p1:0", 1, 0, 0, 1, "different text")
    embeddings_for([changed], tmp_path, embedder)
    assert embedder.calls == 2
