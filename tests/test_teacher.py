# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Tests for teacher prompts, the V1-V4 filter, and the Ollama client (F1.3)."""

from __future__ import annotations

from uuid import UUID

import pytest
import requests

import docmind_ft.vendor  # noqa: F401  (installs the docmind_service alias)
from docmind_ft import teacher
from docmind_ft.qasper import Answer
from docmind_ft.retrieval import Retrieved
from docmind_ft.teacher import (
    Check,
    OllamaTeacher,
    TeacherError,
    build_teacher_prompt,
    ensure_gold_context,
    expected_yes_no,
    human_answer,
    reference_answers,
    token_f1,
    v1_format,
    v2_cites_gold,
    v3_correct,
    v4_not_near_copy,
    verify_answer,
)
from docmind_service.domain import RetrievedChunk


def _answer(
    *,
    unanswerable: bool = False,
    extractive: tuple[str, ...] = (),
    yes_no: bool | None = None,
    free_form: str = "",
) -> Answer:
    return Answer(
        annotation_id="a1",
        worker_id="w1",
        unanswerable=unanswerable,
        extractive_spans=extractive,
        yes_no=yes_no,
        free_form_answer=free_form,
        evidence=(),
        highlighted_evidence=(),
    )


def _chunk(text: str, chunk_id: str = "p:0", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=UUID(int=1),
        page=page,
        text=text,
        score=0.9,
    )


def test_human_answer_uses_first_answering_annotator() -> None:
    answers = (
        _answer(unanswerable=True),
        _answer(extractive=("first", "second")),
        _answer(free_form="later"),
    )

    assert human_answer(answers) == "first; second"
    assert reference_answers(answers) == ["first; second", "later"]
    assert human_answer((_answer(yes_no=False),)) == "No."
    assert human_answer((_answer(yes_no=True),)) == "Yes."
    assert expected_yes_no((_answer(unanswerable=True), _answer(yes_no=False))) is False


def test_v1_rejects_an_unknown_citation() -> None:
    check = v1_format("The answer is X [c99].", prompt_chunk_ids=("c1", "c2"))
    assert check == Check(False, "unknown citation [c99]")


def test_v1_rejects_leaked_hint_wording() -> None:
    check = v1_format("The human expert says X [c1].", prompt_chunk_ids=("c1",))
    assert not check.passed
    assert "hint wording" in check.reason


def test_v1_rejects_a_leaked_canary() -> None:
    check = v1_format("X [c1] P8-CANARY", prompt_chunk_ids=("c1",), canary="P8-CANARY")
    assert check == Check(False, "canary leaked")


def test_v2_requires_a_gold_citation() -> None:
    assert not v2_cites_gold("X [c1].", gold_chunk_ids=("c2",)).passed
    assert v2_cites_gold("X [c2].", gold_chunk_ids=("c2",)).passed


def test_v3_rejects_wrong_yes_no_polarity() -> None:
    check = v3_correct("No, it is not [c1].", expected_yes_no=True, reference_answers=())
    assert check == Check(False, "expected yes, got no")
    assert v3_correct("Yes [c1].", expected_yes_no=True, reference_answers=()).passed


def test_v3_token_f1_boundary_just_below_and_above() -> None:
    below = token_f1("alpha beta x y", "alpha beta c d e f g h i j")
    above = token_f1("alpha beta x y", "alpha beta c d e f g h i")

    assert 0.28 < below < 0.30
    assert 0.30 < above < 0.31
    assert not v3_correct(
        "alpha beta x y", expected_yes_no=None, reference_answers=["alpha beta c d e f g h i j"]
    ).passed
    assert v3_correct(
        "alpha beta x y", expected_yes_no=None, reference_answers=["alpha beta c d e f g h i"]
    ).passed


def test_v4_rejects_a_41_word_copy_but_accepts_40() -> None:
    words = [f"w{index}" for index in range(41)]
    chunk_text = " ".join(words)

    assert not v4_not_near_copy(" ".join(words), chunk_texts=[chunk_text]).passed
    assert v4_not_near_copy(" ".join(words[:40]), chunk_texts=[chunk_text]).passed


def test_context_assembly_replaces_the_lowest_with_the_best_gold() -> None:
    retrieved = [Retrieved(f"p:{index}", 1.0 - index * 0.05) for index in range(10)]
    gold = [
        Retrieved("g:low", 0.2),
        Retrieved("g:best", 0.8),
        Retrieved("g:mid", 0.5),
    ]

    context = ensure_gold_context(retrieved, gold)

    assert len(context) == 10
    assert "g:best" in {item.chunk_id for item in context}
    assert "p:9" not in {item.chunk_id for item in context}
    assert [item.score for item in context] == sorted(
        (item.score for item in context), reverse=True
    )
    assert context == sorted(context, key=lambda item: (-item.score, item.chunk_id))


def test_context_assembly_leaves_a_present_gold_chunk_alone() -> None:
    retrieved = [Retrieved(f"p:{index}", 1.0 - index * 0.05) for index in range(10)]
    assert ensure_gold_context(retrieved, [Retrieved("p:2", 0.9)]) == retrieved


def test_teacher_prompt_puts_hint_after_question_and_uses_gold_short_ids() -> None:
    chunks = [_chunk(f"chunk {index}", f"p:{index}") for index in range(10)]
    prompt, mapping = build_teacher_prompt("What is X?", chunks, "Yes.", ["p:3"], "TEST-CANARY")

    assert mapping["c4"].chunk_id == "p:3"
    assert "Internal reference: TEST-CANARY" in prompt
    assert prompt.index("Question: What is X?") < prompt.index("Reference answer")
    assert prompt.index("Reference answer") < prompt.index("Answer:")
    assert "The supporting chunks are c4." in prompt
    assert "Yes. The supporting chunks" in prompt
    assert prompt.count("Reference answer") == 1


def test_full_filter_accepts_a_good_answer() -> None:
    result = verify_answer(
        "alpha beta gamma [c1]",
        prompt_chunk_ids=("c1",),
        gold_chunk_ids=("c1",),
        expected_yes_no=None,
        reference_answers=("alpha beta gamma",),
        chunk_texts=("unrelated evidence",),
        canary=None,
    )
    assert result.passed
    assert result.failed_rule is None


class _Response:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self.text = "error body"
        self._payload = payload if payload is not None else {"message": {"content": "Yes [c1]."}}

    def json(self) -> dict:
        return self._payload


def test_ollama_client_sends_think_false_and_fixed_options(monkeypatch) -> None:
    calls: dict = {}

    def fake_post(url: str, *, json: dict, timeout: float) -> _Response:
        calls.update(url=url, payload=json, timeout=timeout)
        return _Response()

    monkeypatch.setattr(teacher.requests, "post", fake_post)
    answer = OllamaTeacher().answer("prompt")

    assert answer == "Yes [c1]."
    assert calls["url"] == "http://localhost:11434/api/chat"
    assert calls["payload"]["think"] is False
    assert calls["payload"]["stream"] is False
    assert calls["payload"]["messages"] == [{"role": "user", "content": "prompt"}]
    assert calls["payload"]["options"] == {
        "temperature": 0.2,
        "num_ctx": 4096,
        "num_predict": 160,
        "seed": 20261207,
    }


def test_ollama_client_raises_a_clear_timeout_error(monkeypatch) -> None:
    def fake_post(url: str, *, json: dict, timeout: float) -> _Response:
        raise requests.Timeout("slow")

    monkeypatch.setattr(teacher.requests, "post", fake_post)
    with pytest.raises(TeacherError, match="timed out"):
        OllamaTeacher().answer("prompt")


def test_ollama_client_raises_on_non_200(monkeypatch) -> None:
    def fake_post(url: str, *, json: dict, timeout: float) -> _Response:
        return _Response(status_code=500)

    monkeypatch.setattr(teacher.requests, "post", fake_post)
    with pytest.raises(TeacherError, match="HTTP 500"):
        OllamaTeacher().answer("prompt")
