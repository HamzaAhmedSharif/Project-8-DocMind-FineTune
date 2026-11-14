"""Tests for docmind_ft.masking (gate I3 helper, script F0.4)."""

from __future__ import annotations

from docmind_ft.masking import check_completion_mask


def test_fully_masked_prompt_passes() -> None:
    result = check_completion_mask([1, 2, 3], [1, 2, 3, 4, 5], [-100, -100, -100, 4, 5])
    assert result["ok"] is True
    assert result["offset"] == 0
    assert result["completion_trained_frac"] == 1.0


def test_bos_offset_detected() -> None:
    result = check_completion_mask([1, 2, 3], [0, 1, 2, 3, 4, 5], [-100, -100, -100, -100, 4, 5])
    assert result["ok"] is True
    assert result["offset"] == 1


def test_unmasked_prompt_token_fails() -> None:
    result = check_completion_mask([1, 2, 3], [1, 2, 3, 4, 5], [-100, 2, -100, 4, 5])
    assert result["ok"] is False
    assert "unmasked" in result["reason"]


def test_mostly_masked_completion_fails() -> None:
    result = check_completion_mask([1, 2], [1, 2, 3, 4, 5, 6], [-100, -100, -100, -100, -100, 6])
    assert result["ok"] is False
    assert result["completion_trained_frac"] == 0.25


def test_length_mismatch_fails() -> None:
    result = check_completion_mask([1, 2, 3], [1, 2, 3, 4], [-100, -100, -100])
    assert result["ok"] is False


def test_prompt_not_a_prefix_fails() -> None:
    result = check_completion_mask([9, 9], [1, 2, 3, 4], [-100, -100, 3, 4])
    assert result["ok"] is False
