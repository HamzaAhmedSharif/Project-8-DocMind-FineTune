# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Tests for the F1.4 teacher build, branching, targets, and I2 validator."""

from __future__ import annotations

import copy
import json
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest

import docmind_ft.vendor  # noqa: F401  (installs the docmind_service alias)
from docmind_ft.qasper import Answer
from docmind_ft.retrieval import TOP_K, Retrieved, retrieve
from docmind_ft.teacher import build_teacher_prompt, check_correctness
from docmind_ft.teacher_build import (
    BRANCH_GOLD,
    BRANCH_P0,
    CASE_A,
    CASE_U,
    CHUNKS_PATH,
    ChunkRecord,
    ChunkStore,
    EMBEDDINGS_PATH,
    FASTEMBED_CACHE,
    QUESTIONS_PATH,
    QuestionEmbedder,
    TrainQuestion,
    abstention_share,
    answer_length_stats,
    assemble_trl_rows,
    assign_case,
    best_gold_short_id,
    branch_p0,
    build_prompt_from_ids,
    compute_p0,
    gold_context,
    gold_free_context,
    i2_failures,
    i2_stats,
    is_data25,
    load_chunk_store,
    load_embeddings,
    load_train_questions,
    make_trl_row,
    option_a_target,
    process_question,
    read_jsonl,
    sentence_count,
    validate_trl_row,
)
from docmind_service.domain import RetrievedChunk
from docmind_service.query import NO_EVIDENCE_ANSWER


def _answer(
    *,
    unanswerable: bool = False,
    extractive: tuple[str, ...] = (),
    yes_no: bool | None = None,
    free_form: str = "",
) -> Answer:
    return Answer(
        annotation_id="a",
        worker_id="w",
        unanswerable=unanswerable,
        extractive_spans=extractive,
        yes_no=yes_no,
        free_form_answer=free_form,
        evidence=(),
        highlighted_evidence=(),
    )


def _question(
    *,
    qid: str = "q1",
    paper: str = "p1",
    answerable: bool = True,
    gold: tuple[str, ...] = ("p1:g",),
    gold_scores: dict[str, float] | None = None,
    retrieved: tuple[Retrieved, ...] | None = None,
    answer_type: str = "extractive",
) -> TrainQuestion:
    if retrieved is None:
        retrieved = tuple(Retrieved(f"{paper}:{index}", 0.90 - 0.01 * index) for index in range(10))
    if gold_scores is None:
        gold_scores = {chunk_id: 0.95 for chunk_id in gold}
    return TrainQuestion(
        question_id=qid,
        paper_id=paper,
        question="What is alpha?",
        answer_type=answer_type,
        answerable=answerable,
        gold_chunk_ids=gold,
        retrieved=retrieved,
        gold_scores=gold_scores,
    )


def _chunk(cid: str, *, paper: str = "p1", text: str | None = None) -> ChunkRecord:
    if text is None:
        text = "alpha beta gamma" if cid.endswith(":g") or cid.endswith(":0") else f"filler {cid}"
    return ChunkRecord(chunk_id=cid, paper_id=paper, page=1, text=text)


def _store_for(question: TrainQuestion, *, extras: tuple[str, ...] = ()) -> ChunkStore:
    ids = [item.chunk_id for item in question.retrieved]
    ids.extend(question.gold_chunk_ids)
    ids.extend(extras)
    records = [_chunk(cid) for cid in dict.fromkeys(ids) if cid != "p2:0"]
    records.append(_chunk("p2:0", paper="p2"))
    return ChunkStore.from_records(records)


def _embeddings_for(store: ChunkStore) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for cid in store.ids:
        if cid == "p1:g" or cid == "p1:0" or cid == "p2:0":
            rows.append([1.0, 0.0])
        elif cid.startswith("p1:"):
            index = int(cid.split(":")[1])
            rows.append([max(0.95 - 0.02 * index, 0.1), 0.1])
        else:
            rows.append([0.1, 1.0])
    return np.asarray(rows, dtype=np.float64), np.asarray([1.0, 0.0])


class _FakeTeacher:
    def __init__(self, answer: str, *, explode: bool = False) -> None:
        self.answer_text = answer
        self.explode = explode
        self.prompts: list[str] = []

    def answer(self, prompt: str) -> str:
        if self.explode:
            raise AssertionError("teacher must not be called")
        self.prompts.append(prompt)
        return self.answer_text


def _vchunk(cid: str, text: str = "alpha") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=uuid5(NAMESPACE_URL, "p1"),
        page=1,
        text=text,
        score=0.0,
    )


def test_check_correctness_is_the_shared_v3_check() -> None:
    assert check_correctness(
        "alpha beta", expected_yes_no=None, reference_answers=["alpha beta"]
    ).passed
    assert not check_correctness(
        "completely unrelated", expected_yes_no=None, reference_answers=["alpha beta"]
    ).passed


def test_branch_hash_is_deterministic() -> None:
    assert not branch_p0("q1", 0.0)
    assert branch_p0("q1", 1.0)
    assert branch_p0("q1", 0.5) == branch_p0("q1", 0.5)


def test_gold_context_inserts_a_gold_chunk_and_orders_by_score() -> None:
    question = _question(gold=("p1:g",), gold_scores={"p1:g": 0.95})
    context = gold_context(question)
    assert len(context) == TOP_K
    assert context[0].chunk_id == "p1:g"
    assert [item.score for item in context] == sorted(
        (item.score for item in context), reverse=True
    )


def test_gold_context_inserts_the_overall_best_gold_not_only_a_present_one() -> None:
    retrieved = (Retrieved("p1:g2", 0.91),) + tuple(
        Retrieved(f"p1:{index}", 0.90 - 0.01 * index) for index in range(9)
    )
    question = _question(
        gold=("p1:g1", "p1:g2"),
        gold_scores={"p1:g1": 0.99, "p1:g2": 0.60},
        retrieved=retrieved,
    )
    context = gold_context(question)
    assert len(context) == TOP_K
    assert "p1:g1" in {item.chunk_id for item in context}
    assert context[0].chunk_id == "p1:g1"


def test_gold_free_context_removes_gold_and_fills_from_same_paper() -> None:
    question = _question(gold=("p1:0",), gold_scores={"p1:0": 0.95})
    store = _store_for(question, extras=("p1:10",))
    embeddings, query = _embeddings_for(store)
    context = gold_free_context(question, store, embeddings, query)
    assert len(context) == TOP_K
    assert all(item.chunk_id.startswith("p1:") for item in context)
    assert all(item.chunk_id != "p1:0" for item in context)
    assert "p1:10" in {item.chunk_id for item in context}
    assert "p2:0" not in {item.chunk_id for item in context}


def test_gold_process_keeps_a_passing_v1_v4_answer_and_hint_stays_out_of_prompt() -> None:
    question = _question(gold=("p1:g",), gold_scores={"p1:g": 0.95})
    store = _store_for(question)
    teacher = _FakeTeacher("alpha beta gamma [c1]")
    answers = (_answer(extractive=("alpha beta gamma",)),)
    row = process_question(
        question,
        answers,
        store,
        teacher,
        p0=0.0,
        canary="P8-test",
    )
    assert row["passed"] is True
    assert row["branch"] == BRANCH_GOLD
    assert "Reference answer" in teacher.prompts[0]
    prompt, _ = build_prompt_from_ids(question, row["context_chunk_ids"], store, str(row["canary"]))
    assert "Reference answer" not in prompt
    assert "P8-test" in prompt

    distilled, option_a = assemble_trl_rows(
        [row], {question.question_id: question}, store, {question.question_id: answers}
    )
    assert distilled[0]["prompt"] == option_a[0]["prompt"]
    assert distilled[0]["completion"][0]["content"] == "alpha beta gamma [c1]"
    assert option_a[0]["completion"][0]["content"] == "alpha beta gamma [c1]"
    assert validate_trl_row(distilled[0]) == []
    assert validate_trl_row(option_a[0]) == []


def test_probe_failure_keeps_abstention_and_probe_success_drops() -> None:
    question = _question(gold=("p1:0",), gold_scores={"p1:0": 0.95})
    store = _store_for(question, extras=("p1:10",))
    embeddings, query = _embeddings_for(store)
    answers = (_answer(extractive=("alpha beta gamma",)),)
    embed = lambda _: query  # noqa: E731
    failed_probe = _FakeTeacher("completely unrelated")
    kept = process_question(
        question,
        answers,
        store,
        failed_probe,
        p0=1.0,
        canary="P8-test",
        embeddings=embeddings,
        embed_question=embed,
    )
    assert kept["branch"] == BRANCH_P0
    assert kept["passed"] is True
    assert kept["probe_failed"] is True
    assert kept["failed_rule"] is None
    assert "Reference answer" not in failed_probe.prompts[0]
    distilled, option_a = assemble_trl_rows(
        [kept], {question.question_id: question}, store, {question.question_id: answers}
    )
    for example in (distilled[0], option_a[0]):
        assert example["completion"][0]["content"] == NO_EVIDENCE_ANSWER

    passed_probe = _FakeTeacher("alpha beta gamma [c1]")
    dropped = process_question(
        question,
        answers,
        store,
        passed_probe,
        p0=1.0,
        canary="P8-test",
        embeddings=embeddings,
        embed_question=embed,
    )
    assert dropped["passed"] is False
    assert dropped["probe_failed"] is False
    assert dropped["failed_rule"] == "probe_pass"
    dropped_rows, dropped_option = assemble_trl_rows(
        [dropped], {question.question_id: question}, store, {question.question_id: answers}
    )
    assert dropped_rows == []
    assert dropped_option == []


def test_u_question_abstains_without_teacher_call() -> None:
    question = _question(answerable=False, gold=())
    store = _store_for(question)
    teacher = _FakeTeacher("unused", explode=True)
    row = process_question(question, (), store, teacher, p0=1.0, canary="P8-test")
    assert assign_case(question) == CASE_U
    assert row["case"] == CASE_U
    assert row["branch"] is None
    assert row["teacher_answer"] is None
    assert row["passed"] is True
    assert teacher.prompts == []

    distilled, option_a = assemble_trl_rows([row], {question.question_id: question}, store, {})
    for example in (distilled[0], option_a[0]):
        assert example["completion"][0]["content"] == NO_EVIDENCE_ANSWER
        assert "Reference answer" not in example["prompt"][0]["content"]
        assert example["meta"]["canary"] == "P8-test"


def test_option_a_targets_yes_no_spans_and_free_form() -> None:
    question = _question(
        gold=("p1:g1", "p1:g2"),
        gold_scores={"p1:g1": 0.9, "p1:g2": 0.5},
    )
    mapping = {"c1": _vchunk("p1:g1"), "c2": _vchunk("p1:g2")}
    assert option_a_target(question, (_answer(yes_no=True),), mapping) == "Yes. [c1]"
    assert (
        option_a_target(question, (_answer(extractive=("alpha", "beta")),), mapping)
        == "alpha; beta [c1]"
    )
    assert option_a_target(question, (_answer(free_form="free text"),), mapping) == "free text [c1]"
    assert best_gold_short_id(question, mapping) == "c1"


def test_trl_format_is_conversational() -> None:
    row = make_trl_row("prompt text", "target text", {"canary": "P8-test"})
    assert row["prompt"] == [{"role": "user", "content": "prompt text"}]
    assert row["completion"] == [{"role": "assistant", "content": "target text"}]
    assert row["meta"]["canary"] == "P8-test"


def test_data25_hash_is_deterministic_and_about_a_quarter() -> None:
    ids = [f"q{index}" for index in range(10000)]
    kept = sum(is_data25(question_id) for question_id in ids)
    assert 2300 <= kept <= 2700
    assert is_data25("same-id") == is_data25("same-id")


def _i2_row() -> dict[str, object]:
    prompt = (
        "Internal reference: P8-test\n\n"
        "<<<c1 page=1>>>\nalpha beta gamma\n<<<end c1>>>\n\n"
        "Question: What is alpha?\n\nAnswer:"
    )
    return make_trl_row(
        prompt,
        "alpha beta [c1]",
        {
            "question_id": "q1",
            "paper_id": "p1",
            "case": CASE_A,
            "answer_type": "extractive",
            "branch": BRANCH_GOLD,
            "canary": "P8-test",
        },
    )


def test_i2_accepts_a_valid_row() -> None:
    assert validate_trl_row(_i2_row()) == []
    assert sentence_count("alpha beta [c1]") == 1


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda row: row.pop("prompt"), "missing prompt"),
        (lambda row: row.pop("completion"), "missing completion"),
        (lambda row: row["completion"][0].update(content="alpha [c99]"), "unknown citation"),
        (
            lambda row: row["completion"][0].update(content=" ".join(["word"] * 91) + " [c1]"),
            "max 90",
        ),
        (
            lambda row: row["completion"][0].update(content="A. B. C. D. [c1]"),
            "sentences",
        ),
        (lambda row: row["completion"][0].update(content="[c1]"), "sentences"),
        (
            lambda row: row["completion"][0].update(content="alpha beta gamma"),
            "no citation",
        ),
        (
            lambda row: row["completion"][0].update(content="alpha P8-test [c1]"),
            "canary",
        ),
        (
            lambda row: row["completion"][0].update(content=f"{NO_EVIDENCE_ANSWER} [c1]"),
            "abstention target is not exact",
        ),
        (
            lambda row: row["completion"][0].update(content=f"{NO_EVIDENCE_ANSWER} "),
            "abstention target is not exact",
        ),
        (lambda row: row["meta"].pop("canary"), "missing canary"),
    ],
)
def test_i2_rejects_each_invalid_case(mutate, needle: str) -> None:
    row = copy.deepcopy(_i2_row())
    mutate(row)
    errors = validate_trl_row(row)
    assert errors
    assert any(needle in error for error in errors)


def test_read_jsonl_recovers_a_truncated_final_line(tmp_path, caplog) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"question_id": "q1"}\n{"question_id": "q2"', encoding="utf-8")
    with caplog.at_level("WARNING", logger="docmind_ft.jsonl"):
        rows = read_jsonl(path)
    assert rows == [{"question_id": "q1"}]
    assert "truncated incomplete final JSONL line" in caplog.text
    assert path.read_text(encoding="utf-8") == '{"question_id": "q1"}\n'


def test_read_jsonl_rejects_a_bad_mid_file_line(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"question_id": "q1"}\nnot-json\n{"question_id": "q2"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        read_jsonl(path)


def test_i2_jsonl_counts_invalid_json(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(_i2_row()) + "\n{not json}\n",
        encoding="utf-8",
    )
    stats = i2_stats(path)
    assert stats["rows"] == 2
    assert stats["valid"] == 1
    assert stats["i2_valid_share"] == 0.5
    assert stats["errors"]


def test_i2_nodistill_exception_skips_length_only() -> None:
    long_row = copy.deepcopy(_i2_row())
    long_row["completion"][0]["content"] = " ".join(["word"] * 150) + " [c1]"
    assert validate_trl_row(long_row)
    assert validate_trl_row(long_row, check_length=False) == []

    many_sentences = copy.deepcopy(_i2_row())
    many_sentences["completion"][0]["content"] = "A. B. C. D. E. [c1]"
    assert validate_trl_row(many_sentences)
    assert validate_trl_row(many_sentences, check_length=False) == []

    bad_citation = copy.deepcopy(_i2_row())
    bad_citation["completion"][0]["content"] = "alpha [c99]"
    assert validate_trl_row(bad_citation, check_length=False)

    no_citation = copy.deepcopy(_i2_row())
    no_citation["completion"][0]["content"] = " ".join(["word"] * 150)
    assert validate_trl_row(no_citation, check_length=False)


def test_i2_stats_passes_check_length_through(tmp_path) -> None:
    row = copy.deepcopy(_i2_row())
    row["completion"][0]["content"] = " ".join(["word"] * 150) + " [c1]"
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert i2_stats(path)["valid"] == 0
    assert i2_stats(path, check_length=False)["valid"] == 1


def test_i2_empty_file_scores_zero_and_fails_the_row_floor(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    stats = i2_stats(path, expected_rows=1)
    assert stats["rows"] == 0
    assert stats["i2_valid_share"] == 0.0
    assert stats["below_expected"] is True
    assert i2_failures({"train": stats}) == ["train"]


def test_i2_all_abstention_file_fails_the_abstention_ceiling(tmp_path) -> None:
    row = copy.deepcopy(_i2_row())
    row["completion"][0]["content"] = NO_EVIDENCE_ANSWER
    path = tmp_path / "abstain.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    stats = i2_stats(path, expected_rows=2)
    assert stats["i2_valid_share"] == 1.0
    assert stats["abstention_share"] == 1.0
    assert stats["below_expected"] is False
    assert i2_failures({"train": stats}) == ["train"]


def test_p0_formula_and_clamp() -> None:
    assert compute_p0(2101, 271, 0.527, 0.8) == pytest.approx(0.002966, abs=1e-6)
    assert compute_p0(10, 100, 0.527, 0.8) == 0.0
    assert compute_p0(1000, 0, 1.0, 0.0) == 0.30


def test_answer_stats_ignore_abstentions() -> None:
    rows = [
        make_trl_row("p", "one two [c1]", {"canary": "P8-test"}),
        make_trl_row("p", NO_EVIDENCE_ANSWER, {"canary": "P8-test"}),
    ]
    stats = answer_length_stats(rows)
    assert stats["n"] == 1
    assert stats["mean"] == 3.0
    assert abstention_share(rows) == 0.5


def test_include_hint_flag_controls_the_teacher_hint() -> None:
    chunk = _vchunk("p1:g", "alpha")
    with_hint, _ = build_teacher_prompt("What is alpha?", [chunk], "alpha", ["p1:g"], "P8-test")
    without_hint, _ = build_teacher_prompt(
        "What is alpha?", [chunk], "", (), "P8-test", include_hint=False
    )
    assert "Reference answer" in with_hint
    assert "Reference answer" not in without_hint
    assert "P8-test" in without_hint


def test_cached_embeddings_align_with_questions_jsonl() -> None:
    pytest.importorskip("fastembed")
    if not all(path.exists() for path in (QUESTIONS_PATH, CHUNKS_PATH, EMBEDDINGS_PATH)):
        pytest.skip("F1.2 artifacts are not available")
    if not FASTEMBED_CACHE.exists():
        pytest.skip("fastembed cache is not available")
    questions = load_train_questions(QUESTIONS_PATH)
    question = next(item for item in questions if item.retrieved)
    store = load_chunk_store(CHUNKS_PATH)
    matrix = load_embeddings(EMBEDDINGS_PATH, expected_rows=len(store.ids))
    rows = store.rows_by_paper[question.paper_id]
    ids = [store.ids[row] for row in rows]
    vector = QuestionEmbedder().embed(question.question)
    top = retrieve(vector, matrix[rows], ids, TOP_K)
    assert [item.chunk_id for item in top] == [item.chunk_id for item in question.retrieved]
    assert np.allclose(
        [item.score for item in top],
        [item.score for item in question.retrieved],
    )
