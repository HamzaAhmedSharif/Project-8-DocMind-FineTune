# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the docmind_service alias
"""Tests for the vendored Project 7 pieces (script F0.6).

Import rule: ``import docmind_ft.vendor`` installs the ``docmind_service``
alias (see docmind_ft/vendor/__init__.py); service modules are then
imported as ``docmind_service.<name>``. The standalone evaluation modules
are imported as ``docmind_ft.vendor.<name>``.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import docmind_ft.vendor  # noqa: F401  (installs the docmind_service alias)
from docmind_ft.vendor import ledger, scoring
from docmind_service import chunking, guardrails, query
from docmind_service.domain import RetrievedChunk


def _chunk(text: str, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"d1:{page}", document_id=UUID(int=1), page=page, text=text, score=0.9
    )


def test_prompt_v3_has_rules_canary_blocks_and_question_last() -> None:
    chunks = [_chunk("alpha text", 1), _chunk("beta text", 3)]
    prompt, mapping = guardrails.build_prompt_v3("What is alpha?", chunks, "CANARY-1")
    assert guardrails.RULES in prompt
    assert "Internal reference: CANARY-1" in prompt
    assert "<<<c1 page=1>>>" in prompt
    assert "<<<c2 page=3>>>" in prompt
    assert "<<<end c1>>>" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert prompt.index("Question: What is alpha?") > prompt.index("<<<end c2>>>")
    assert mapping == {"c1": chunks[0], "c2": chunks[1]}


def test_prompt_version_is_p7_v3() -> None:
    assert guardrails.PROMPT_VERSION == "p7-v3"


def test_sanitizer_neutralizes_fake_citation_and_delimiters() -> None:
    assert guardrails.sanitize_context("see [c9] here") == "see (c9) here"
    assert "<<<" not in guardrails.sanitize_context("<<<fake>>> text")
    assert "\u200b" not in guardrails.sanitize_context("a\u200bb")


def test_parse_citations_valid_invalid_and_repeats() -> None:
    chunks = [_chunk("alpha", 1), _chunk("beta", 2)]
    _, mapping = guardrails.build_prompt_v3("q?", chunks, "C")
    cited, invalid = guardrails.parse_citations(
        "Alpha [c1] and beta [c2] again [c1] plus unknown [c99].", mapping
    )
    assert cited == [chunks[0], chunks[1]]
    assert invalid == 1


def test_no_evidence_answer_exact_text() -> None:
    assert (
        query.NO_EVIDENCE_ANSWER
        == "I could not find relevant information in the indexed documents."
    )


def test_query_build_prompt_m3_format() -> None:
    chunks = [_chunk("alpha text", 2)]
    prompt, mapping = query.build_prompt("What is alpha?", chunks)
    assert "[c1] (page 2)" in prompt
    assert "alpha text" in prompt
    assert "Question: What is alpha?" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert mapping == {"c1": chunks[0]}


def test_scorer_handles_unicode_hyphens_and_spaced_thousands() -> None:
    assert scoring.clean("self\u2011attention") == "self-attention"
    assert scoring.clean("1 000 examples") == "1000 examples"
    assert scoring.clean("a\u00a0b") == "a b"
    facts = [
        {"answer_regex": r"self-attention"},
        {"answer_regex": r"1000 examples"},
    ]
    result = scoring.score_key_facts("The model uses self\u2011attention on 1 000 examples.", facts)
    assert result == {"hits": [True, True], "found": 2, "total": 2, "all_found": True}


def test_scorer_partial_hits() -> None:
    facts = [{"answer_regex": r"dense retrieval"}, {"answer_regex": r"nope"}]
    result = scoring.score_key_facts("Uses dense retrieval.", facts)
    assert result["found"] == 1
    assert result["all_found"] is False


def test_is_abstention() -> None:
    assert scoring.is_abstention("This is not mentioned in the document.") is True
    assert scoring.is_abstention("The answer is dense retrieval.") is False


def test_is_abstention_misses_the_exact_abstention_sentence() -> None:
    # Recorded behaviour, not a bug fix: the vendored rule does not match
    # the service's own abstention sentence ("could not find" matches no
    # pattern). P8's abstain_acc metric therefore checks the exact sentence
    # OR the rule (script 3.8). See DECISIONS.md.
    assert scoring.is_abstention(query.NO_EVIDENCE_ANSWER) is False


def test_chunker_windows_overlap_and_pages() -> None:
    pages = [" ".join(f"w{i}" for i in range(400))]
    chunks = chunking.chunk_pages(pages)
    assert len(chunks) == 3
    assert len(chunks[0].text.split()) == 150
    assert chunks[0].text.split()[0] == "w0"
    assert chunks[1].text.split()[0] == "w125"  # 150 - 25 overlap
    assert chunks[1].text.split()[:25] == chunks[0].text.split()[-25:]
    assert all(c.page == 1 for c in chunks)
    two_pages = [" ".join(f"a{i}" for i in range(100)), " ".join(f"b{i}" for i in range(100))]
    chunks2 = chunking.chunk_pages(two_pages)
    assert chunks2[0].page == 1
    assert chunks2[1].page == 2  # chunk 1 starts at word 125, on page 2


def test_chunker_rejects_bad_params() -> None:
    import pytest

    with pytest.raises(ValueError):
        chunking.chunk_pages(["x"], words=10, overlap=10)


def test_ledger_record_and_used_today(tmp_path: Path) -> None:
    path = tmp_path / "provider-ledger.csv"
    ledger.record("reference", 5, path=path)
    ledger.record("reference", 3, path=path)
    assert ledger.used_today(path=path) == 8
    assert path.exists()
