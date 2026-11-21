# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Evaluation metrics (script section 3.8): pure functions over raw outputs.

No model, network or file access here.  ``scripts/compute_metrics.py`` loads
the raw generation rows and calls these functions; the same functions are used
by the unit tests on hand-made fixtures.

Percentile intervals use ordinary linear interpolation.  The bootstrap for a
paired difference resamples *papers* with replacement and keeps every question
of a resampled paper in the resample.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import docmind_ft.vendor  # noqa: F401  (installs the docmind_service alias)
from docmind_ft.vendor.scoring import (
    clean,
    is_abstention as _vendored_is_abstention,
    score_key_facts,
)
from docmind_service.query import NO_EVIDENCE_ANSWER

MAX_FORMAT_WORDS = 120
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261207
BOOTSTRAP_ALPHA = 0.05
ARTICLES = {"a", "an", "the"}

_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_CITATION_MARKER = re.compile(r"\[c\d+\]", re.IGNORECASE)
_SINGLE_CITATION = re.compile(r"c\d+")
_ANY_CITATION = re.compile(r"\bc\d+\b", re.IGNORECASE)
_FIRST_WORD = re.compile(r"[a-z]+")
_PUNCTUATION = re.compile(r"[^\w\s]")

#: Normalization used by the vendored scorer, re-exported for tests.
clean_text = clean


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _share(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def hold_normalize(text: str) -> str:
    """Frozen HOLD normalization: lowercase and collapse whitespace."""
    return " ".join(text.split()).lower()


def strip_citations(text: str) -> str:
    """Remove ``[cN]`` markers before text scoring; other brackets stay."""
    return _CITATION_MARKER.sub(" ", text)


def is_abstention(output: str) -> bool:
    """Exact vendored sentence or the vendored abstention rule."""
    return output.strip() == NO_EVIDENCE_ANSWER or _vendored_is_abstention(output)


@dataclass(frozen=True)
class CitationStats:
    ids: tuple[str, ...]
    malformed: int


def citation_ids(text: str) -> CitationStats:
    """Distinct ``[cN]`` ids and malformed citation markers.

    ``[c1, c2]`` is not a ``[cN]`` marker: it is counted once as malformed
    (unrecognized), by design, and contributes no ids.
    """
    ids: list[str] = []
    seen: set[str] = set()
    malformed = 0
    for match in _BRACKET.finditer(text):
        inner = match.group(1).strip()
        if _SINGLE_CITATION.fullmatch(inner):
            if inner not in seen:
                seen.add(inner)
                ids.append(inner)
        elif _ANY_CITATION.search(inner):
            malformed += 1
    return CitationStats(tuple(ids), malformed)


def format_valid(
    output: str,
    prompt_chunk_ids: Sequence[str],
    *,
    abstention: str = NO_EVIDENCE_ANSWER,
) -> bool:
    """Script 3.8: exact abstention, or >=1 valid citation and <=120 words."""
    text = output.strip()
    if text == abstention:
        return True
    if len(text.split()) > MAX_FORMAT_WORDS:
        return False
    stats = citation_ids(text)
    if not stats.ids:
        return False
    prompt_ids = set(prompt_chunk_ids)
    return all(cid in prompt_ids for cid in stats.ids)


def squad_normalize(text: str) -> str:
    """SQuAD-style normalization: typography, lowercase, punctuation, articles."""
    text = clean(text).lower()
    text = _PUNCTUATION.sub("", text)
    return " ".join(token for token in text.split() if token not in ARTICLES)


def token_f1(prediction: str, reference: str) -> float:
    pred = Counter(squad_normalize(prediction).split())
    ref = Counter(squad_normalize(reference).split())
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def answer_f1(
    output: str,
    gold_answers: Sequence[str],
    *,
    yes_no: bool | None = None,
) -> float:
    """Token F1 against the best gold answer, or yes/no polarity."""
    if yes_no is not None:
        match = _FIRST_WORD.search(strip_citations(output).casefold())
        first = match.group(0) if match else ""
        expected = "yes" if yes_no else "no"
        return 1.0 if first == expected else 0.0
    if not gold_answers:
        return 0.0
    prediction = strip_citations(output)
    return max(token_f1(prediction, gold) for gold in gold_answers)


@dataclass(frozen=True)
class OutputRow:
    """One scored output joined with its context and gold information."""

    question_id: str
    output: str
    prompt_chunk_ids: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...] = ()
    answerable: bool = True
    gold_answers: tuple[str, ...] = ()
    yes_no: bool | None = None
    answer_type: str = ""
    paper_id: str = ""


def citation_metrics(rows: Sequence[OutputRow]) -> dict:
    """Script 3.8: micro valid-id share and answers with a valid citation."""
    numerator = 0
    denominator = 0
    answers = 0
    answers_with_valid = 0
    for row in rows:
        if is_abstention(row.output):
            continue
        stats = citation_ids(row.output)
        prompt_ids = set(row.prompt_chunk_ids)
        valid = sum(1 for cid in stats.ids if cid in prompt_ids)
        unknown = len(stats.ids) - valid
        numerator += valid
        denominator += valid + unknown + stats.malformed
        answers += 1
        if valid:
            answers_with_valid += 1
    return {
        "citation_valid_share": numerator / denominator if denominator else None,
        "answers_with_valid_citation": answers_with_valid / answers if answers else None,
    }


def citation_support(rows: Sequence[OutputRow]) -> float | None:
    """Share of non-abstention answerable answers citing a gold chunk."""
    flags = []
    for row in rows:
        if not row.answerable or is_abstention(row.output):
            continue
        gold = set(row.gold_chunk_ids)
        stats = citation_ids(row.output)
        flags.append(any(cid in gold for cid in stats.ids))
    return _share(flags)


def answer_metrics(rows: Sequence[OutputRow]) -> dict:
    """QASPER answer F1, overall and by answer type."""
    values: list[float] = []
    by_type: dict[str, list[float]] = {}
    for row in rows:
        if not row.answerable:
            continue
        if row.yes_no is None and not row.gold_answers:
            continue
        value = answer_f1(row.output, row.gold_answers, yes_no=row.yes_no)
        values.append(value)
        by_type.setdefault(row.answer_type or "unknown", []).append(value)
    return {
        "answer_f1": _mean(values),
        "n_scored": len(values),
        "answer_f1_by_type": {
            answer_type: _mean(scores) for answer_type, scores in sorted(by_type.items())
        },
    }


def gold_recall(rows: Sequence[OutputRow]) -> float | None:
    """Share of answerable questions whose gold chunk is in the context."""
    flags = [
        bool(set(row.gold_chunk_ids) & set(row.prompt_chunk_ids)) for row in rows if row.answerable
    ]
    return _share(flags)


def mean_answer_length(rows: Sequence[OutputRow]) -> float | None:
    return _mean([len(strip_citations(row.output).split()) for row in rows])


def abstain_metrics(rows: Sequence[OutputRow]) -> dict:
    """Abstain accuracy on unanswerables and over-refusal on answerables."""
    unanswerable = [row for row in rows if not row.answerable]
    answerable_with_gold = [
        row
        for row in rows
        if row.answerable and set(row.gold_chunk_ids) & set(row.prompt_chunk_ids)
    ]
    return {
        "abstain_acc": _share([is_abstention(row.output) for row in unanswerable]),
        "over_refusal": _share([is_abstention(row.output) for row in answerable_with_gold]),
        "n_unanswerable": len(unanswerable),
        "n_answerable_with_gold": len(answerable_with_gold),
    }


@dataclass(frozen=True)
class FactScore:
    """Per-fact answer hits, context groundability, and strict verbatim hits."""

    hits: tuple[bool, ...]
    groundable: tuple[bool, ...]
    strict_hits: tuple[bool, ...] = ()


def hold_fact_in_text(fact: dict, text: str) -> bool:
    """True if one of a HOLD fact's frozen ``accept`` patterns matches the text."""
    normalized = hold_normalize(text)
    return any(
        re.search(pattern, normalized, re.IGNORECASE) for pattern in fact.get("accept") or []
    )


def paper16_fact_in_text(fact: dict, text: str) -> bool:
    """True if a PAPER16 fact's ``paper_regex`` matches the text."""
    return bool(re.search(fact["paper_regex"], hold_normalize(text), re.IGNORECASE))


def score_hold_answer(
    answer: str,
    key_facts: Sequence[dict],
    context_text: str,
) -> FactScore:
    """HOLD key facts: frozen ``accept`` patterns and strict verbatim coverage."""
    hits = tuple(hold_fact_in_text(fact, answer) for fact in key_facts)
    groundable = tuple(hold_fact_in_text(fact, context_text) for fact in key_facts)
    normalized_answer = hold_normalize(answer)
    strict = tuple(
        bool(fact.get("verbatim")) and hold_normalize(str(fact["verbatim"])) in normalized_answer
        for fact in key_facts
    )
    return FactScore(hits, groundable, strict)


def score_paper16_answer(
    answer: str,
    key_facts: Sequence[dict],
    context_text: str,
) -> FactScore:
    """PAPER16 key facts with the vendored strict scorer (``answer_regex``)."""
    hits = tuple(score_key_facts(answer, list(key_facts))["hits"])
    groundable = tuple(paper16_fact_in_text(fact, context_text) for fact in key_facts)
    return FactScore(hits, groundable)


def summarize_facts(scores: Sequence[FactScore]) -> dict:
    """Aggregate key-fact coverage, groundable ceiling, bleed and all-found."""
    total = sum(len(score.hits) for score in scores)
    found = sum(sum(score.hits) for score in scores)
    groundable_total = sum(sum(score.groundable) for score in scores)
    found_groundable = sum(
        sum(1 for hit, ok in zip(score.hits, score.groundable, strict=True) if hit and ok)
        for score in scores
    )
    strict_total = sum(len(score.strict_hits) for score in scores)
    strict_found = sum(sum(score.strict_hits) for score in scores)
    memory_bleed = sum(
        sum(1 for hit, ok in zip(score.hits, score.groundable, strict=True) if hit and not ok)
        for score in scores
    )
    with_facts = [score for score in scores if score.hits]
    all_found = (
        sum(1 for score in with_facts if all(score.hits)) / len(with_facts) if with_facts else None
    )
    return {
        "keyfact_cov": found / total if total else None,
        "keyfact_found": found,
        "keyfact_total": total,
        "strict_verbatim_cov": strict_found / strict_total if strict_total else None,
        "strict_verbatim_found": strict_found,
        "groundable_ceiling": groundable_total,
        "keyfact_cov_groundable": f"{found_groundable} of {groundable_total} reachable",
        "keyfact_cov_groundable_value": (
            found_groundable / groundable_total if groundable_total else None
        ),
        "memory_bleed": memory_bleed,
        "all_facts_found": all_found,
    }


def gap_closure(score: float, baseline: float, ceiling: float) -> float:
    """Fraction of the baseline-to-ceiling gap closed; zero gap -> 0.0."""
    gap = ceiling - baseline
    if gap == 0:
        return 0.0
    return (score - baseline) / gap


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    position = (len(sorted_values) - 1) * q / 100.0
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    low: float
    high: float
    n_questions: int
    n_papers: int


def paired_paper_bootstrap(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    paper_ids: Sequence[str],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> BootstrapResult:
    """Cluster bootstrap 95% interval for mean(a) - mean(b), clustered by paper."""
    if not (len(a_scores) == len(b_scores) == len(paper_ids)):
        raise ValueError("a_scores, b_scores and paper_ids must have the same length")
    if not a_scores:
        raise ValueError("bootstrap needs at least one question")
    clusters: dict[str, list[float]] = {}
    for a_score, b_score, paper_id in zip(a_scores, b_scores, paper_ids, strict=True):
        clusters.setdefault(str(paper_id), []).append(float(a_score) - float(b_score))
    all_deltas = [delta for deltas in clusters.values() for delta in deltas]
    point = sum(all_deltas) / len(all_deltas)
    cluster_list = list(clusters.values())
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        picked = [cluster_list[rng.randrange(len(cluster_list))] for _ in cluster_list]
        resampled = [delta for cluster in picked for delta in cluster]
        means.append(sum(resampled) / len(resampled))
    means.sort()
    return BootstrapResult(
        point=point,
        low=_percentile(means, 100 * alpha / 2),
        high=_percentile(means, 100 * (1 - alpha / 2)),
        n_questions=len(all_deltas),
        n_papers=len(clusters),
    )


def efficiency_summary(
    seconds: Sequence[float],
    generated_tokens: Sequence[int | None],
    prompt_tokens: Sequence[int | None] = (),
) -> dict:
    """p50/p95 seconds, token means and generated tokens per second."""
    elapsed = [float(value) for value in seconds]
    generated = [int(value) for value in generated_tokens if value is not None]
    prompts = [int(value) for value in prompt_tokens if value is not None]
    total_seconds = sum(elapsed)
    return {
        "answers": len(elapsed),
        "seconds_p50": _percentile(sorted(elapsed), 50) if elapsed else None,
        "seconds_p95": _percentile(sorted(elapsed), 95) if elapsed else None,
        "prompt_tokens_mean": _mean(prompts),
        "generated_tokens_mean": _mean(generated),
        "tokens_per_second": sum(generated) / total_seconds if total_seconds > 0 else None,
    }
