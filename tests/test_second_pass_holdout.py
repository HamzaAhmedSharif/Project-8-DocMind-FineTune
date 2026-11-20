"""Unit tests for the HOLD-set second-pass mismatch rule; no network/GPU."""

from __future__ import annotations

import json

import scripts.second_pass_holdout as sp


def make_item(accept) -> dict:
    return {
        "id": "h01",
        "paper_id": "2601.00000",
        "question": "Q?",
        "gold_answer": "gold only, never sent to the teacher",
        "key_facts": [{"accept": accept}],
        "evidence_pages": [1],
        "type": "numeric",
        "answerable": True,
    }


def test_matching_answer_not_flagged():
    item = make_item([r"42\s+percent"])
    assert sp.is_flagged(item, "The model says 42 percent.") is False


def test_matching_answer_case_insensitive():
    item = make_item([r"42\s+percent"])
    assert sp.is_flagged(item, "ANSWER: 42 PERCENT.") is False


def test_no_match_flagged():
    item = make_item([r"42\s+percent"])
    assert sp.is_flagged(item, "The model says 43 percent.") is True


def test_abstention_flagged():
    item = make_item([r"42\s+percent"])
    abstain = "I could not find relevant information in the indexed documents."
    assert sp.is_flagged(item, abstain) is True


def test_empty_patterns_flagged():
    item = make_item([])
    assert sp.is_flagged(item, "42 percent.") is True


def test_build_prompt_contains_only_evidence_and_question(tmp_path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "2601.00000.pages.json").write_text(
        json.dumps(["page one text here"]), encoding="utf-8"
    )
    pages = sp.load_pages("2601.00000", pages_dir)
    item = make_item([r"42\s+percent"])
    prompt = sp.build_prompt(item, pages)
    assert "page one text here" in prompt
    assert item["question"] in prompt
    assert item["gold_answer"] not in prompt
    assert "key_facts" not in prompt


def test_run_with_fake_teacher_skips_done(tmp_path, monkeypatch):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "2601.00000.pages.json").write_text(
        json.dumps(["the answer is 42 percent here"]), encoding="utf-8"
    )
    q_path = tmp_path / "questions.jsonl"
    q_path.write_text(
        json.dumps(make_item([r"42\s+percent"]), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"

    calls = []

    def fake(prompt: str) -> str:
        calls.append(prompt)
        return "42 percent."

    assert sp.run(q_path, pages_dir, out, answer_fn=fake, stop=False) == 0
    rows = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert rows == [{"id": "h01", "model_answer": "42 percent.", "flagged": False}]
    assert len(calls) == 1

    assert sp.run(q_path, pages_dir, out, answer_fn=fake, stop=False) == 0
    rows = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) == 1
    assert len(calls) == 1


def test_run_flags_unmatched_answer(tmp_path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "2601.00000.pages.json").write_text(
        json.dumps(["the answer is 42 percent here"]), encoding="utf-8"
    )
    q_path = tmp_path / "questions.jsonl"
    q_path.write_text(
        json.dumps(make_item([r"42\s+percent"]), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    out = tmp_path / "out.jsonl"

    assert sp.run(q_path, pages_dir, out, answer_fn=lambda prompt: "43 percent.", stop=False) == 0
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["flagged"] is True
