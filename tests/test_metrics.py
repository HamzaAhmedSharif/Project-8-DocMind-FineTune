# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Unit tests for the section 3.8 metrics (F2.1).

All functions are pure; fixtures are hand-made.  Run before any real model
output exists.
"""

from __future__ import annotations

import math

import pytest

import docmind_ft.vendor  # noqa: F401  # installs the docmind_service alias
from docmind_service.query import NO_EVIDENCE_ANSWER

from docmind_ft.metrics import (
    OutputRow,
    abstain_metrics,
    answer_f1,
    answer_metrics,
    citation_ids,
    citation_metrics,
    citation_support,
    clean_text,
    efficiency_summary,
    format_valid,
    gap_closure,
    gold_recall,
    hold_fact_in_text,
    is_abstention,
    mean_answer_length,
    paired_paper_bootstrap,
    paper16_fact_in_text,
    score_hold_answer,
    score_paper16_answer,
    squad_normalize,
    summarize_facts,
    token_f1,
)


def test_clean_text_normalizes_unicode_hyphens():
    assert clean_text("state\u2011of\u2011the\u2011art") == "state-of-the-art"
    assert clean_text("two\u2013thirds") == "two-thirds"


def test_clean_text_normalizes_curly_quotes():
    assert clean_text("\u201cYes\u201d") == '"Yes"'
    assert clean_text("it\u2019s") == "it's"


def test_clean_text_joins_spaced_thousands():
    assert clean_text("1 000 items") == "1000 items"
    assert clean_text("12,345 items") == "12345 items"
    assert clean_text("2024") == "2024"


def test_squad_normalize_strips_articles_and_punctuation():
    assert squad_normalize("The \u201cCat\u201d, sat.") == "cat sat"
    assert squad_normalize("a 1 000 an") == "1000"


def test_token_f1_exact_and_half_overlap():
    assert token_f1("The cat sat.", "cat sat") == 1.0
    assert math.isclose(token_f1("alpha beta gamma delta", "alpha beta"), 2 / 3)


def test_token_f1_empty_prediction_is_zero():
    assert token_f1("", "alpha") == 0.0
    assert token_f1("alpha", "") == 0.0


def test_answer_f1_takes_maximum_over_annotators():
    value = answer_f1("alpha beta", ("alpha", "alpha beta"))
    assert value == 1.0


def test_answer_f1_removes_citation_markers_only():
    assert answer_f1("alpha [c1] beta", ("alpha beta",)) == 1.0
    assert answer_f1("alpha [0, 1] beta", ("alpha 0 1 beta",)) == 1.0


def test_yes_no_polarity_uses_first_word():
    assert answer_f1("Yes [c1].", (), yes_no=True) == 1.0
    assert answer_f1("No.", (), yes_no=False) == 1.0
    assert answer_f1("No.", (), yes_no=True) == 0.0
    assert answer_f1("Yes [c1].", (), yes_no=False) == 0.0


def test_answer_f1_without_gold_is_zero():
    assert answer_f1("anything", ()) == 0.0


def test_citation_ids_single_and_adjacent():
    assert citation_ids("answer [c1]").ids == ("c1",)
    assert citation_ids("answer [c1][c2]").ids == ("c1", "c2")


def test_citation_ids_repeat_is_deduplicated():
    stats = citation_ids("[c1][c1]")
    assert stats.ids == ("c1",)
    assert stats.malformed == 0


def test_citation_ids_unknown_id_is_still_an_id():
    assert citation_ids("[c99]").ids == ("c99",)


def test_citation_ids_comma_form_is_unrecognized():
    stats = citation_ids("[c1, c2]")
    assert stats.ids == ()
    assert stats.malformed == 1


def test_citation_marker_inside_quote_is_found():
    assert citation_ids('the authors write "as shown [c1]" here').ids == ("c1",)


@pytest.mark.parametrize(
    "output",
    [
        NO_EVIDENCE_ANSWER,
        f"  {NO_EVIDENCE_ANSWER}\n",
        "The paper does not contain this information.",
        "This is not mentioned in the text.",
        "No information is available.",
        "I cannot find the answer.",
        "The result was not reported.",
        "This detail is not in the provided context.",
        "The authors did not specify the value.",
        "There is no evidence for that claim.",
        "I am unable to determine this.",
        "The paper doesn't mention it.",
    ],
)
def test_abstention_positive(output):
    assert is_abstention(output)


@pytest.mark.parametrize(
    "output",
    [
        "The answer is 42.",
        "Yes, it does.",
        "No, it uses 8 heads.",
        "It cannot be used on long sequences.",
        "The information is discussed in Section 3.",
        "They mention the dataset in Appendix B.",
        "The model includes eight attention heads.",
        "Training took four days.",
        "It achieves 28.4 BLEU on English-to-German.",
        "The model is not large but it is fast.",
    ],
)
def test_abstention_negative(output):
    assert not is_abstention(output)


def test_empty_output_tests():
    assert not is_abstention("")
    assert citation_ids("").ids == ()
    assert not format_valid("", ("c1",))
    assert answer_f1("", ("alpha",)) == 0.0


def test_format_valid_accepts_known_citation_and_abstention():
    assert format_valid("answer [c1]", ("c1", "c2"))
    assert format_valid(NO_EVIDENCE_ANSWER, ())
    assert format_valid(f"  {NO_EVIDENCE_ANSWER}  ", ())


def test_format_valid_rejects_unknown_or_missing_citation():
    assert not format_valid("answer [c99]", ("c1",))
    assert not format_valid("answer without a citation", ("c1",))
    assert not format_valid("[c1][c99]", ("c1",))


def test_format_valid_ignores_an_unrecognized_comma_marker():
    assert not format_valid("[c1, c2]", ("c1", "c2"))
    assert format_valid("[c1] [c1, c2]", ("c1", "c2"))


def test_format_valid_word_limit():
    at_limit = " ".join(["word"] * 119) + " [c1]"
    over_limit = " ".join(["word"] * 120) + " [c1]"
    assert format_valid(at_limit, ("c1",))
    assert not format_valid(over_limit, ("c1",))


def test_citation_metrics_micro_average_and_abstention_skip():
    rows = [
        OutputRow("q1", "[c1][c2]", ("c1", "c2")),
        OutputRow("q2", "[c99]", ("c1", "c2")),
        OutputRow("q3", "[c1][c1]", ("c1",)),
        OutputRow("q4", NO_EVIDENCE_ANSWER, ("c1",), answerable=False),
    ]
    result = citation_metrics(rows)
    assert result["citation_valid_share"] == 0.75
    assert result["answers_with_valid_citation"] == 2 / 3


def test_citation_support_uses_gold_chunks():
    rows = [
        OutputRow("q1", "answer [c1]", ("c1",), gold_chunk_ids=("c1",)),
        OutputRow("q2", "answer [c2]", ("c1", "c2"), gold_chunk_ids=("c1",)),
        OutputRow("q3", NO_EVIDENCE_ANSWER, ("c1",), gold_chunk_ids=("c1",)),
        OutputRow("q4", "answer [c1]", ("c1",), gold_chunk_ids=(), answerable=False),
    ]
    assert citation_support(rows) == 0.5


def test_gold_recall_over_answerable_questions():
    rows = [
        OutputRow("q1", "a", ("c1",), gold_chunk_ids=("c1",)),
        OutputRow("q2", "a", ("c1",), gold_chunk_ids=("c2",)),
        OutputRow("q3", "a", ("c1",), gold_chunk_ids=(), answerable=False),
    ]
    assert gold_recall(rows) == 0.5


def test_answer_metrics_by_type_skips_unscorable_rows():
    rows = [
        OutputRow(
            "q1",
            "alpha beta [c1]",
            ("c1",),
            gold_answers=("alpha beta",),
            answer_type="extractive",
        ),
        OutputRow(
            "q2",
            "Yes [c1]",
            ("c1",),
            gold_answers=("Yes.",),
            yes_no=True,
            answer_type="yes_no",
        ),
        OutputRow("q3", "I don't know", ("c1",), answerable=False),
        OutputRow("q4", "answer", ("c1",), answer_type="free_form"),
    ]
    result = answer_metrics(rows)
    assert result["answer_f1"] == 1.0
    assert result["n_scored"] == 2
    assert result["answer_f1_by_type"] == {"extractive": 1.0, "yes_no": 1.0}


def test_mean_answer_length_ignores_citation_markers():
    rows = [OutputRow("q1", "one two [c1]", ("c1",)), OutputRow("q2", "three", ("c1",))]
    assert mean_answer_length(rows) == 1.5


def test_abstain_metrics():
    rows = [
        OutputRow("u1", NO_EVIDENCE_ANSWER, ("c1",), answerable=False),
        OutputRow("u2", "Paris", ("c1",), answerable=False),
        OutputRow("a1", NO_EVIDENCE_ANSWER, ("c1",), gold_chunk_ids=("c1",)),
        OutputRow("a2", "answer [c1]", ("c1",), gold_chunk_ids=("c1",)),
        OutputRow("a3", NO_EVIDENCE_ANSWER, ("c1",), gold_chunk_ids=("c2",)),
    ]
    result = abstain_metrics(rows)
    assert result["abstain_acc"] == 0.5
    assert result["over_refusal"] == 0.5
    assert result["n_unanswerable"] == 2
    assert result["n_answerable_with_gold"] == 2


def test_hold_fact_in_text_uses_frozen_normalization():
    fact = {"accept": [r"790,?000\s+applications"]}
    assert hold_fact_in_text(fact, "It received 790,000 applications.")
    assert hold_fact_in_text(fact, "It received 790000 applications.")
    assert not hold_fact_in_text(fact, "It received 790 applications.")


def test_score_hold_answer_hits_groundable_and_strict():
    key_facts = [
        {"verbatim": "statutory constraints", "accept": [r"statutory\s+constraints"]},
        {"verbatim": "sequential analysis", "accept": [r"sequential\s+analysis"]},
    ]
    answer = "It adds statutory constraints and performs a sequential analysis."
    context = "The stage injects statutory constraints."
    score = score_hold_answer(answer, key_facts, context)
    assert score.hits == (True, True)
    assert score.groundable == (True, False)
    assert score.strict_hits == (True, True)


def test_score_hold_answer_strict_verbatim_is_verbatim_only():
    key_facts = [{"verbatim": "statutory constraints", "accept": [r"legal\s+limits"]}]
    loose = score_hold_answer("It adds legal limits.", key_facts, "legal limits")
    exact = score_hold_answer("It adds statutory constraints.", key_facts, "legal limits")
    assert loose.hits == (True,)
    assert loose.strict_hits == (False,)
    assert exact.strict_hits == (True,)


def test_paper16_fact_in_text_and_strict_score():
    fact = {
        "answer_regex": r"different\s*positions",
        "paper_regex": r"different\s*positions\s*of\s*a\s*single\s*sequence",
    }
    assert paper16_fact_in_text(fact, "relates different positions of a single sequence")
    assert not paper16_fact_in_text(fact, "relates tokens")
    score = score_paper16_answer(
        "Self-attention relates different positions of the sequence.",
        [fact],
        "relates different positions of a single sequence",
    )
    assert score.hits == (True,)
    assert score.groundable == (True,)


def test_score_paper16_uses_vendored_clean_for_numbers():
    fact = {"answer_regex": r"41\.[08]", "paper_regex": r"41\.[08]"}
    score = score_paper16_answer("It reached 41.8 BLEU.", [fact], "41.8")
    assert score.hits == (True,)


def test_summarize_facts_all_variants():
    scores = [
        score_hold_answer(
            "statutory constraints",
            [
                {"verbatim": "statutory constraints", "accept": [r"statutory\s+constraints"]},
                {"verbatim": "sequential analysis", "accept": [r"sequential\s+analysis"]},
            ],
            "statutory constraints",
        ),
        score_hold_answer(
            "statutory constraints and sequential analysis",
            [
                {"verbatim": "statutory constraints", "accept": [r"statutory\s+constraints"]},
                {"verbatim": "sequential analysis", "accept": [r"sequential\s+analysis"]},
            ],
            "unrelated context",
        ),
    ]
    result = summarize_facts(scores)
    assert result["keyfact_cov"] == 0.75
    assert result["keyfact_found"] == 3
    assert result["keyfact_total"] == 4
    assert result["strict_verbatim_cov"] == 0.75
    assert result["groundable_ceiling"] == 1
    assert result["keyfact_cov_groundable"] == "1 of 1 reachable"
    assert result["keyfact_cov_groundable_value"] == 1.0
    assert result["memory_bleed"] == 2
    assert result["all_facts_found"] == 0.5


def test_gap_closure_and_zero_denominator():
    assert gap_closure(50, 40, 60) == 0.5
    assert gap_closure(40, 40, 40) == 0.0


def test_bootstrap_degenerate_toy_interval():
    result = paired_paper_bootstrap(
        [1.0, 1.0],
        [0.5, 0.5],
        ["p1", "p2"],
        n_resamples=100,
    )
    assert result.point == 0.5
    assert result.low == 0.5
    assert result.high == 0.5
    assert result.n_questions == 2
    assert result.n_papers == 2


def test_bootstrap_keeps_whole_papers():
    result = paired_paper_bootstrap(
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 0.0],
        ["A", "A", "B"],
        n_resamples=200,
    )
    assert result.point == 50.0
    assert result.low == 50.0
    assert result.high == 50.0
    assert result.n_questions == 3
    assert result.n_papers == 2


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_paper_bootstrap([1.0], [1.0, 2.0], ["p1"])


def test_efficiency_summary():
    result = efficiency_summary([1.0, 2.0, 3.0, 4.0], [10, 10, 10, 10], [100, 100, 100, 100])
    assert result["answers"] == 4
    assert result["seconds_p50"] == 2.5
    assert result["seconds_p95"] == pytest.approx(3.85)
    assert result["prompt_tokens_mean"] == 100.0
    assert result["generated_tokens_mean"] == 10.0
    assert result["tokens_per_second"] == 4.0


def test_efficiency_summary_empty_and_zero_seconds():
    empty = efficiency_summary([], [])
    assert empty["answers"] == 0
    assert empty["seconds_p50"] is None
    assert empty["tokens_per_second"] is None
    assert efficiency_summary([0.0], [5])["tokens_per_second"] is None
