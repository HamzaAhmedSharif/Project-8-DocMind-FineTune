"""Tests for deterministic cosine retrieval (F1.2)."""

from __future__ import annotations

import numpy as np
import pytest

from docmind_ft.retrieval import Retrieved, cosine_scores, insert_gold, rank_chunks, retrieve


def test_cosine_scores_are_exact_for_simple_vectors() -> None:
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [2.0, 0.0]])

    assert cosine_scores(query, matrix).tolist() == [1.0, 0.0, -1.0, 1.0]


def test_rank_is_deterministic_and_breaks_ties_by_chunk_id() -> None:
    scores = [0.5, 0.5, 0.9]
    ids = ["b", "a", "c"]

    first = rank_chunks(scores, ids, top_k=3)
    assert first == rank_chunks(scores, ids, top_k=3)
    assert [(item.chunk_id, item.score) for item in first] == [("c", 0.9), ("a", 0.5), ("b", 0.5)]


def test_top_ten_ordering_drops_the_lowest_scores() -> None:
    ids = [f"c{index:02d}" for index in range(12)]
    scores = [index / 10 for index in range(12)]

    ranked = rank_chunks(scores, ids, top_k=10)

    assert len(ranked) == 10
    assert [item.score for item in ranked] == [1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    assert {item.chunk_id for item in ranked}.isdisjoint({"c00", "c01"})


def test_retrieval_is_deterministic() -> None:
    matrix = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    ids = ["b", "a", "c"]
    query = np.array([1.0, 0.2])

    assert retrieve(query, matrix, ids, top_k=2) == retrieve(query, matrix, ids, top_k=2)


def test_gold_insertion_keeps_score_order_and_does_not_duplicate() -> None:
    retrieved = [Retrieved("a", 0.9), Retrieved("c", 0.7), Retrieved("d", 0.6)]
    gold = [Retrieved("b", 0.8), Retrieved("c", 0.7)]

    assert insert_gold(retrieved, gold) == [
        Retrieved("a", 0.9),
        Retrieved("b", 0.8),
        Retrieved("c", 0.7),
        Retrieved("d", 0.6),
    ]


def test_cosine_scores_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_scores(np.array([1.0, 2.0]), np.array([[1.0, 0.0, 3.0]]))
