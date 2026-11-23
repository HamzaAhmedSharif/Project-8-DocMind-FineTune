# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Tests for the F2.3 prompt variants (no model, no network)."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

import docmind_ft.vendor  # noqa: F401  (installs the docmind_service alias)
from docmind_ft.prompts import build_prompt_variant
from docmind_service.domain import RetrievedChunk

TEMPLATE = """INSTRUCTIONS: cite the chunks.

Internal reference: {canary}

{context_blocks}

Question: {question}

Answer:"""


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="p1:0",
            document_id=uuid5(NAMESPACE_URL, "p1"),
            page=3,
            text="Alpha is the first letter.",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="p1:1",
            document_id=uuid5(NAMESPACE_URL, "p1"),
            page=4,
            text="Beta is the second letter.",
            score=0.8,
        ),
    ]


def _template(tmp_path: Path, text: str = TEMPLATE) -> Path:
    path = tmp_path / "variant.txt"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_variant_renders_canary_blocks_and_question(tmp_path):
    prompt = build_prompt_variant(
        _template(tmp_path), "What is alpha?", _chunks(), "P8-abc123def456"
    )
    assert prompt.startswith("INSTRUCTIONS: cite the chunks.")
    assert "Internal reference: P8-abc123def456" in prompt
    assert "<<<c1 page=3>>>\nAlpha is the first letter.\n<<<end c1>>>" in prompt
    assert "<<<c2 page=4>>>\nBeta is the second letter.\n<<<end c2>>>" in prompt
    assert prompt.endswith("Question: What is alpha?\n\nAnswer:")


def test_variant_citation_ids_follow_context_order(tmp_path):
    prompt = build_prompt_variant(_template(tmp_path), "Q?", _chunks(), "P8-1")
    assert [int(match) for match in re.findall(r"<<<c(\d+) ", prompt)] == [1, 2]
    assert prompt.index("Alpha") < prompt.index("Beta")


def test_variant_sanitizes_chunk_text_like_the_vendored_builder(tmp_path):
    chunk = RetrievedChunk(
        chunk_id="p1:0",
        document_id=uuid5(NAMESPACE_URL, "p1"),
        page=3,
        text="Use [c9] and <<<not a block>>>.",
        score=0.9,
    )
    prompt = build_prompt_variant(_template(tmp_path), "Q?", [chunk], "P8-1")
    assert "Use (c9) and" in prompt
    assert "[c9]" not in prompt
    assert "<<<not a block>>>" not in prompt


def test_variant_requires_all_three_placeholders(tmp_path):
    with pytest.raises(ValueError, match="missing placeholders: question"):
        build_prompt_variant(
            _template(tmp_path, "Only {canary}\n{context_blocks}"),
            "Q?",
            _chunks(),
            "P8-1",
        )
