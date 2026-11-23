# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Prompt variants for the B1 baseline (F2.3).

A variant is a UTF-8 template with ``{canary}``, ``{context_blocks}`` and
``{question}`` placeholders.  The harness supplies the canary, the same
``<<<cN page=P>>>`` blocks as prompt p7-v3, and the question; the template
supplies the instruction text.
"""

from __future__ import annotations

import re
from pathlib import Path

import docmind_ft.vendor  # noqa: F401  (installs the docmind_service alias)

from docmind_service.domain import RetrievedChunk  # noqa: E402
from docmind_service.guardrails import sanitize_context  # noqa: E402

_PLACEHOLDER = re.compile(r"\{(canary|context_blocks|question)\}")


def _context_blocks(chunks: list[RetrievedChunk]) -> str:
    """Render the p7-v3 chunk blocks (c1..cN in context order)."""
    return "\n\n".join(
        f"<<<c{i} page={chunk.page}>>>\n{sanitize_context(chunk.text)}\n<<<end c{i}>>>"
        for i, chunk in enumerate(chunks, start=1)
    )


def build_prompt_variant(
    variant_path: str | Path,
    question: str,
    chunks: list[RetrievedChunk],
    canary: str,
) -> str:
    """Fill a prompt template with the canary, context blocks and question."""
    template = Path(variant_path).read_text(encoding="utf-8").rstrip()
    values = {
        "canary": canary,
        "context_blocks": _context_blocks(chunks),
        "question": question,
    }
    missing = sorted(set(values) - set(_PLACEHOLDER.findall(template)))
    if missing:
        raise ValueError(f"{variant_path}: missing placeholders: {', '.join(missing)}")
    # re.sub does not rescan the replacement text, so a chunk or question that
    # contains another placeholder is inserted literally.
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)
