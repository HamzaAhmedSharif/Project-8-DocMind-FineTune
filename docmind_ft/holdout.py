"""HOLD-set question checks (F1.5).

The authoring step writes ``evaluation/holdout/holdout_questions.jsonl``;
this module validates that file against the extracted texts in
``B:\\p8\\data\\holdout\\``.  The file contains one JSON object per line with
this schema::

    {"id": "h01", "paper_id": "2601.00166", "question": "...",
     "gold_answer": "...",
     "key_facts": [{"verbatim": "<=6 words, verbatim substring of the paper text>",
                    "accept": ["<regex>", "<regex>"]}],
     "evidence_pages": [3],
     "type": "factual|numeric|explanation|detail|unanswerable",
     "answerable": true, "distinctive_terms": [], "authored_by": "agent"}

``evidence_pages`` are 1-based page numbers.  ``distinctive_terms`` is
non-empty only for unanswerable items; those terms are verified absent from
the whole paper.  Normalization is applied consistently to paper text, key
facts, answers, and questions: lowercase and collapse all whitespace.  A
verbatim must be at most six words.  An accept pattern must match both its
verbatim and the gold answer, must not match the question, and must be at
least three literal characters (after removing regex syntax) and not a bare
common word.

Retrievability is informational only: an answerable item is retrievable when
a top-10 chunk for its question contains one of its verbatim key facts.  The
paper is chunked with the vendored chunker and embedded with the cached
MiniLM model exactly as ``docmind_ft.retrieval`` does.  Unretrievable items
are kept.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from docmind_ft.retrieval import TOP_K, FastEmbedder, cosine_scores, rank_chunks
from docmind_ft.vendor.chunking import chunk_pages

PAGES_DIR = Path(r"B:\p8\data\holdout")
QUESTIONS_PATH = (
    Path(__file__).resolve().parent.parent / "evaluation" / "holdout" / "holdout_questions.jsonl"
)
FASTEMBED_CACHE = Path(r"B:\p8\data\work\fastembed-cache")

ALLOWED_TYPES = {"factual", "numeric", "explanation", "detail", "unanswerable"}
STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "and",
    "or",
    "to",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "with",
    "by",
    "for",
    "from",
    "that",
    "this",
    "it",
    "as",
    "at",
}
MAX_QUESTION_WORDS = 40
MAX_ANSWER_WORDS = 25
MAX_VERBATIM_WORDS = 6


@dataclass(frozen=True)
class Failure:
    item_id: str
    check: str
    detail: str


@dataclass
class ItemResult:
    item_id: str
    answerable: bool
    passed: bool = True
    retrievable: bool | None = None
    failures: list[Failure] = field(default_factory=list)


@dataclass(frozen=True)
class VerificationResult:
    items: int
    passed: int
    failed: int
    retrievable: int
    failures: list[Failure]
    results: list[ItemResult]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace; used for every text comparison."""
    return " ".join(text.split()).lower()


def _words(text: str) -> int:
    return len(normalize(text).split())


def _failure(item_id: str, check: str, detail: str) -> Failure:
    return Failure(item_id=item_id, check=check, detail=detail)


def regex_literal(pattern: str) -> str:
    """Return the pattern with regex syntax removed, for the length/stopword gate."""
    text = re.sub(r"\\.", "", pattern)
    return re.sub(r"[\^$.*+?()[\]{}|]", "", text).strip()


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def load_pages(paper_id: str, pages_dir: Path = PAGES_DIR) -> list[str]:
    if not paper_id or Path(paper_id).name != paper_id:
        raise ValueError(f"unsafe paper id: {paper_id!r}")
    path = pages_dir / f"{paper_id}.pages.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(page, str) for page in data):
        raise ValueError(f"{path}: expected a JSON list of page strings")
    return data


def _schema_failures(row: dict, item_id: str) -> list[Failure]:
    failures: list[Failure] = []

    def fail(check: str, detail: str) -> None:
        failures.append(_failure(item_id, check, detail))

    for key in ("id", "paper_id", "question", "gold_answer", "authored_by"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            fail("schema", f"{key} must be a non-empty string")
    if row.get("type") not in ALLOWED_TYPES:
        fail("schema", "type must be factual, numeric, explanation, detail, or unanswerable")
    answerable = row.get("answerable")
    if not isinstance(answerable, bool):
        fail("schema", "answerable must be a boolean")
    elif row.get("type") == "unanswerable" and answerable:
        fail("schema", "unanswerable item must have answerable=false")
    elif row.get("type") in ALLOWED_TYPES - {"unanswerable"} and not answerable:
        fail("schema", "answerable item must have answerable=true")

    evidence = row.get("evidence_pages")
    if not isinstance(evidence, list) or any(
        not isinstance(page, int) or isinstance(page, bool) or page < 1 for page in evidence
    ):
        fail("schema", "evidence_pages must be a list of 1-based page numbers")

    facts = row.get("key_facts")
    if not isinstance(facts, list):
        fail("schema", "key_facts must be a list")
    else:
        for index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                fail("schema", f"key_facts[{index}] must be an object")
                continue
            verbatim = fact.get("verbatim")
            if not isinstance(verbatim, str) or not verbatim.strip():
                fail("schema", f"key_facts[{index}].verbatim must be a non-empty string")
            accept = fact.get("accept")
            if not isinstance(accept, list) or not accept:
                fail("schema", f"key_facts[{index}].accept must be a non-empty list")
            elif any(not isinstance(pattern, str) or not pattern for pattern in accept):
                fail("schema", f"key_facts[{index}].accept must contain non-empty strings")

    terms = row.get("distinctive_terms")
    if not isinstance(terms, list):
        fail("schema", "distinctive_terms must be a list")
    else:
        if any(not isinstance(term, str) or not term.strip() for term in terms):
            fail("schema", "distinctive_terms must contain non-empty strings")
        if row.get("type") == "unanswerable" and not terms:
            fail("schema", "unanswerable item must have distinctive_terms")
        if answerable is True and terms:
            fail("schema", "answerable item must have no distinctive_terms")
    return failures


class _PaperIndex:
    def __init__(self, paper_id: str, pages: list[str], embedder: object):
        self.chunks = chunk_pages(pages)
        self.chunk_ids = [f"{paper_id}:{chunk.ordinal}" for chunk in self.chunks]
        self.vectors = embedder.embed([chunk.text for chunk in self.chunks])
        self.by_id = {
            chunk_id: chunk for chunk_id, chunk in zip(self.chunk_ids, self.chunks, strict=True)
        }

    def top_texts(self, question: str, embedder: object) -> list[str]:
        if not self.chunks:
            return []
        query = embedder.embed([question])[0]
        ranked = rank_chunks(cosine_scores(query, self.vectors), self.chunk_ids, TOP_K)
        return [self.by_id[item.chunk_id].text for item in ranked]


def _retrievable(
    question: str,
    facts: list[dict],
    index: _PaperIndex,
    embedder: object,
) -> bool:
    top_texts = [normalize(text) for text in index.top_texts(question, embedder)]
    if not top_texts:
        return False
    for fact in facts:
        verbatim = fact.get("verbatim")
        if not isinstance(verbatim, str):
            continue
        normalized = normalize(verbatim)
        if normalized and any(normalized in text for text in top_texts):
            return True
    return False


def _make_embedder() -> FastEmbedder:
    if not FASTEMBED_CACHE.exists():
        raise RuntimeError(f"cached fastembed model directory is missing: {FASTEMBED_CACHE}")
    return FastEmbedder(cache_dir=str(FASTEMBED_CACHE))


def verify_questions(
    questions_path: Path = QUESTIONS_PATH,
    pages_dir: Path = PAGES_DIR,
    *,
    embedder: object | None = None,
) -> VerificationResult:
    """Run every check; return item-level failures and the retrievable count."""
    rows = load_questions(questions_path)
    results: list[ItemResult] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    paper_cache: dict[str, list[str] | None] = {}
    index_cache: dict[str, _PaperIndex] = {}
    active_embedder = embedder

    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            item_id = "<missing>"
        answerable = row.get("answerable") is True
        result = ItemResult(item_id=item_id, answerable=answerable)
        failures = result.failures

        failures.extend(_schema_failures(row, item_id))

        question = row.get("question") if isinstance(row.get("question"), str) else ""
        gold_answer = row.get("gold_answer") if isinstance(row.get("gold_answer"), str) else ""
        paper_id = row.get("paper_id") if isinstance(row.get("paper_id"), str) else ""
        question_norm = normalize(question)
        gold_norm = normalize(gold_answer)

        if item_id != "<missing>":
            if item_id in seen_ids:
                failures.append(_failure(item_id, "duplicate_id", f"duplicate id {item_id!r}"))
            else:
                seen_ids.add(item_id)
        if question_norm:
            if question_norm in seen_questions:
                failures.append(
                    _failure(item_id, "duplicate_question", "duplicate normalized question")
                )
            else:
                seen_questions.add(question_norm)

        if question and _words(question) > MAX_QUESTION_WORDS:
            failures.append(
                _failure(
                    item_id,
                    "question_too_long",
                    f"{_words(question)} words (maximum {MAX_QUESTION_WORDS})",
                )
            )
        if gold_answer and _words(gold_answer) > MAX_ANSWER_WORDS:
            failures.append(
                _failure(
                    item_id,
                    "answer_too_long",
                    f"{_words(gold_answer)} words (maximum {MAX_ANSWER_WORDS})",
                )
            )

        pages: list[str] | None = None
        if paper_id:
            if paper_id not in paper_cache:
                try:
                    paper_cache[paper_id] = load_pages(paper_id, pages_dir)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    paper_cache[paper_id] = None
                    failures.append(_failure(item_id, "paper_pages", str(exc)))
            pages = paper_cache[paper_id]
            if pages is None and not any(f.check == "paper_pages" for f in failures):
                failures.append(_failure(item_id, "paper_pages", f"pages missing for {paper_id}"))
        else:
            failures.append(_failure(item_id, "paper_pages", "paper_id is missing"))

        normalized_pages = [normalize(page) for page in pages] if pages is not None else []
        whole_text = " ".join(normalized_pages)
        evidence_pages = (
            row.get("evidence_pages") if isinstance(row.get("evidence_pages"), list) else []
        )
        facts = row.get("key_facts") if isinstance(row.get("key_facts"), list) else []

        for index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                continue
            verbatim = fact.get("verbatim")
            accept = fact.get("accept")
            if not isinstance(verbatim, str) or not verbatim.strip():
                continue
            verbatim_norm = normalize(verbatim)
            if len(verbatim_norm.split()) > MAX_VERBATIM_WORDS:
                failures.append(
                    _failure(
                        item_id,
                        "verbatim_too_long",
                        f"key_facts[{index}] has {len(verbatim_norm.split())} words",
                    )
                )

            if pages is not None:
                found = any(
                    isinstance(page, int)
                    and not isinstance(page, bool)
                    and 1 <= page <= len(normalized_pages)
                    and verbatim_norm in normalized_pages[page - 1]
                    for page in evidence_pages
                )
                if not found:
                    failures.append(
                        _failure(
                            item_id,
                            "verbatim_not_on_evidence_page",
                            f"key_facts[{index}] {verbatim!r} not found on pages {evidence_pages}",
                        )
                    )

            if not isinstance(accept, list):
                continue
            patterns: list[tuple[str, re.Pattern[str]]] = []
            for pattern in accept:
                if not isinstance(pattern, str):
                    continue
                try:
                    patterns.append((pattern, re.compile(pattern, re.IGNORECASE)))
                except re.error as exc:
                    failures.append(
                        _failure(
                            item_id,
                            "invalid_accept_pattern",
                            f"key_facts[{index}] {pattern!r}: {exc}",
                        )
                    )
            if not patterns:
                failures.append(
                    _failure(item_id, "accept_no_match", f"key_facts[{index}] has no valid pattern")
                )
                continue

            verbatim_matches = {
                pattern for pattern, compiled in patterns if compiled.search(verbatim_norm)
            }
            gold_matches = {pattern for pattern, compiled in patterns if compiled.search(gold_norm)}
            if not verbatim_matches & gold_matches:
                failures.append(
                    _failure(
                        item_id,
                        "accept_mismatch",
                        f"key_facts[{index}] has no pattern matching both verbatim and answer",
                    )
                )

            for pattern, compiled in patterns:
                if question_norm and compiled.search(question_norm):
                    failures.append(
                        _failure(
                            item_id,
                            "accept_matches_question",
                            f"key_facts[{index}] pattern {pattern!r} matches the question",
                        )
                    )
                literal = regex_literal(pattern)
                if len(literal) < 3:
                    failures.append(
                        _failure(
                            item_id,
                            "accept_too_short",
                            f"key_facts[{index}] pattern {pattern!r} has "
                            f"{len(literal)} literal characters",
                        )
                    )
                if literal.casefold() in STOPWORDS:
                    failures.append(
                        _failure(
                            item_id,
                            "accept_stopword",
                            f"key_facts[{index}] pattern {pattern!r} is a bare common word",
                        )
                    )

        if row.get("type") == "unanswerable":
            terms = (
                row.get("distinctive_terms")
                if isinstance(row.get("distinctive_terms"), list)
                else []
            )
            for term in terms:
                if not isinstance(term, str):
                    continue
                term_norm = normalize(term)
                if term_norm and term_norm in whole_text:
                    failures.append(
                        _failure(
                            item_id,
                            "distinctive_term_present",
                            f"unanswerable term {term!r} is present in the paper",
                        )
                    )

        if answerable and pages is not None:
            if not facts:
                result.retrievable = False
            else:
                if active_embedder is None:
                    active_embedder = _make_embedder()
                if paper_id not in index_cache:
                    index_cache[paper_id] = _PaperIndex(paper_id, pages, active_embedder)
                result.retrievable = _retrievable(
                    question, facts, index_cache[paper_id], active_embedder
                )

        result.passed = not failures
        results.append(result)

    failures = [failure for result in results for failure in result.failures]
    passed = sum(1 for result in results if result.passed)
    retrievable = sum(1 for result in results if result.retrievable is True)
    return VerificationResult(
        items=len(results),
        passed=passed,
        failed=len(results) - passed,
        retrievable=retrievable,
        failures=failures,
        results=results,
    )
