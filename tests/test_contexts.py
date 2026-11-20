"""Unit tests for the F1.6 frozen-context builder; no GPU/Ollama in the suite."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from docmind_ft.contexts import (  # noqa: E402
    PAPER16_EXPECTED_IDS,
    _build_contexts,
    _load_paper16_questions,
    _write_jsonl,
    build_paper16_contexts,
    chunk_pages,
)

PAPER_ID = "p1"
_WORD = "word"
PAGES = [" ".join([_WORD] * 1500) + " ."]


class FakeEmbedder:
    """Deterministic, CPU-free embedder; score varies with chunk word count."""

    dim = 3

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            k = float(len(text.split()))
            vectors.append([1.0, k, 0.0])
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)


def _fixture_rows():
    return [
        (f"{PAPER_ID}:q{i}", PAPER_ID, f"question number {i} about the paper") for i in range(3)
    ]


def _paper_loader():
    chunks = chunk_pages(PAGES)
    chunk_ids = [f"{PAPER_ID}:{chunk.ordinal}" for chunk in chunks]
    pages = [chunk.page for chunk in chunks]
    vectors = FakeEmbedder().embed([chunk.text for chunk in chunks])

    def loader(paper_id: str):
        assert paper_id == PAPER_ID
        return chunk_ids, pages, vectors

    return loader


def _serialize(rows) -> bytes:
    tmp = Path("C:/Users/ha684/AppData/Local/Temp/opencode/contexts_test.jsonl")
    _write_jsonl(tmp, rows)
    return tmp.read_bytes()


def test_row_format_and_descending_scores():
    loader = _paper_loader()
    rows = _build_contexts(_fixture_rows(), loader, FakeEmbedder())
    assert len(rows) == 3
    for row in rows:
        assert set(row) == {"question_id", "paper_id", "chunks"}
        assert row["paper_id"] == PAPER_ID
        chunks = row["chunks"]
        assert len(chunks) == 10
        for chunk in chunks:
            assert set(chunk) == {"chunk_id", "score", "page"}
            assert isinstance(chunk["page"], int)
            assert chunk["score"] == round(chunk["score"], 6)
        scores = [chunk["score"] for chunk in chunks]
        assert all(a + 1e-9 >= b for a, b in zip(scores, scores[1:], strict=False)), scores


def test_chunk_ids_well_formed_and_own_paper():
    loader = _paper_loader()
    rows = _build_contexts(_fixture_rows(), loader, FakeEmbedder())
    pattern = re.compile(r"^p1:\d+$")
    for row in rows:
        for chunk in row["chunks"]:
            assert pattern.match(chunk["chunk_id"]), chunk["chunk_id"]
            assert chunk["chunk_id"].startswith(PAPER_ID + ":")


def test_determinism_identical_bytes():
    loader = _paper_loader()
    rows_a = _build_contexts(_fixture_rows(), loader, FakeEmbedder())
    loader_b = _paper_loader()
    rows_b = _build_contexts(_fixture_rows(), loader_b, FakeEmbedder())
    assert _serialize(rows_a) == _serialize(rows_b)
    assert rows_a == rows_b


def test_paper16_selection_exactly_16_no_web():
    questions = _load_paper16_questions()
    ids = [q[0] for q in questions]
    assert ids == PAPER16_EXPECTED_IDS
    assert len(ids) == 16
    assert not any(qid.startswith("w") for qid in ids)


def test_paper16_chunk_ids_use_transformer_paper():
    rows = build_paper16_contexts(FakeEmbedder())
    assert len(rows) == 16
    for row in rows:
        assert row["paper_id"] == "1706.03762"
        assert len(row["chunks"]) == 10
        for chunk in row["chunks"]:
            assert chunk["chunk_id"].startswith("1706.03762:")


def test_chunk_pages_mirror_holdout_layout():
    chunks = chunk_pages(PAGES)
    assert chunks[0].page == 1
    assert chunks[0].ordinal == 0
    assert chunks[0].text.startswith("word word word")
    assert len(chunks) >= 10
