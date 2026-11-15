# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Teacher prompting, verification filter, and local Ollama client (F1.3).

The teacher writes distillation targets. Its prompt is the vendored student
prompt plus a human-answer hint after the question; the student never sees the
hint. The filter is pure so every rejection can be tested without Ollama.

Ollama parameter check (0.32.13): the request switch is the top-level boolean
``"think": false``; it is not an entry in ``options``. One short call returned
normal content and no ``message.thinking`` field.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import requests

import docmind_ft.vendor  # noqa: F401  (installs the docmind_service alias)
from docmind_ft.qasper import Answer
from docmind_ft.retrieval import TOP_K, Retrieved, insert_gold
from docmind_service.domain import RetrievedChunk
from docmind_service.guardrails import build_prompt_v3, canary_leaked

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
TEACHER_MODEL = "qwen3:8b"
TEMPERATURE = 0.2
NUM_CTX = 4096
NUM_PREDICT = 160
SEED = 20261207
TEACHER_CANARY = "P8-F1.3-TEACHER-CANARY"

MAX_ANSWER_WORDS = 90
MIN_TOKEN_F1 = 0.30
MAX_COPY_WORDS = 40
BANNED_HINT_WORDS = ("reference answer", "human expert", "hint", "expert")

_CITATION = re.compile(r"\[(c\d+)\]", re.IGNORECASE)
_FIRST_WORD = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class Check:
    passed: bool
    reason: str


@dataclass(frozen=True)
class FilterResult:
    passed: bool
    failed_rule: str | None
    reason: str


class TeacherError(RuntimeError):
    """The Ollama teacher could not produce an answer."""


def answer_text(answer: Answer) -> str | None:
    """One annotator's answer as text, or ``None`` when it is unanswerable."""
    if answer.unanswerable:
        return None
    if answer.yes_no is not None:
        return "Yes." if answer.yes_no else "No."
    if answer.extractive_spans:
        return "; ".join(answer.extractive_spans)
    text = answer.free_form_answer.strip()
    return text or None


def human_answer(answers: Sequence[Answer]) -> str | None:
    """The first answering annotator's answer, as used in the teacher hint."""
    for answer in answers:
        text = answer_text(answer)
        if text is not None:
            return text
    return None


def reference_answers(answers: Sequence[Answer]) -> list[str]:
    """Every answering annotator's answer, for the maximum V3 token-F1."""
    return [text for answer in answers if (text := answer_text(answer)) is not None]


def expected_yes_no(answers: Sequence[Answer]) -> bool | None:
    """Polarity of the first answering annotator, or ``None`` for other types."""
    for answer in answers:
        if not answer.unanswerable:
            return answer.yes_no
    return None


def ensure_gold_context(
    retrieved: Sequence[Retrieved],
    gold: Sequence[Retrieved],
    top_k: int = TOP_K,
) -> list[Retrieved]:
    """Return at most ``top_k`` chunks, replacing the lowest with the best gold.

    A gold chunk already in the ranked list means the context is left alone.
    Otherwise the highest-scored gold chunk replaces the lowest-ranked item,
    and the result is ordered by score then chunk id.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    context = list(retrieved[:top_k])
    if not gold:
        return context
    gold_ids = {item.chunk_id for item in gold}
    if any(item.chunk_id in gold_ids for item in context):
        return context
    best = min(gold, key=lambda item: (-item.score, item.chunk_id))
    base = context[:-1] if len(context) >= top_k else context
    return insert_gold(base, [best])


def build_teacher_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    human: str,
    gold_chunk_ids: Sequence[str],
    canary: str,
    *,
    include_hint: bool = True,
) -> tuple[str, dict[str, RetrievedChunk]]:
    """Render the p7-v3 prompt, optionally inserting the hint after the question."""
    prompt, mapping = build_prompt_v3(question, list(chunks)[:TOP_K], canary)
    if not include_hint:
        return prompt, mapping
    if not human.strip():
        raise ValueError("teacher prompt needs a non-empty human answer")
    gold = set(gold_chunk_ids)
    supporting = [short for short, chunk in mapping.items() if chunk.chunk_id in gold]
    if not supporting:
        raise ValueError("context does not contain a gold chunk")

    reference = human.strip()
    if not reference.endswith((".", "!", "?")):
        reference += "."
    hint = (
        "Reference answer from a human expert (use it to stay correct; never mention it): "
        f"{reference} The supporting chunks are {', '.join(supporting)}. Write the final "
        "answer in one to three sentences, cite the supporting chunks like [c3], and nothing else."
    )
    head, marker, tail = prompt.rpartition("\n\nAnswer:")
    if not marker:
        raise ValueError("vendored prompt has no Answer marker")
    return f"{head}\n\n{hint}{marker}{tail}", mapping


def cited_ids(answer: str) -> set[str]:
    return {match.casefold() for match in _CITATION.findall(answer)}


def v1_format(
    answer: str,
    *,
    prompt_chunk_ids: Sequence[str],
    canary: str | None = None,
) -> Check:
    """V1: valid citations only, at most 90 words, no hint wording or canary."""
    if canary_leaked(answer, canary):
        return Check(False, "canary leaked")
    unknown = cited_ids(answer) - set(prompt_chunk_ids)
    if unknown:
        return Check(False, f"unknown citation [{sorted(unknown)[0]}]")
    words = answer.split()
    if len(words) > MAX_ANSWER_WORDS:
        return Check(False, f"answer has {len(words)} words (max {MAX_ANSWER_WORDS})")
    lowered = answer.casefold()
    for phrase in BANNED_HINT_WORDS:
        if phrase in lowered:
            return Check(False, f"leaked hint wording: {phrase!r}")
    return Check(True, "format ok")


def v2_cites_gold(answer: str, *, gold_chunk_ids: Sequence[str]) -> Check:
    """V2: at least one citation points at a gold chunk."""
    if cited_ids(answer) & set(gold_chunk_ids):
        return Check(True, "gold chunk cited")
    return Check(False, "no gold chunk cited")


def _first_word(answer: str) -> str:
    match = _FIRST_WORD.search(_CITATION.sub(" ", answer))
    return match.group(0).casefold() if match else ""


def _content_tokens(text: str) -> list[str]:
    """Whitespace tokens without citation markers, case-folded."""
    return _CITATION.sub(" ", text).casefold().split()


def token_f1(prediction: str, reference: str) -> float:
    """Whitespace token F1 with multiset overlap."""
    pred = Counter(_content_tokens(prediction))
    ref = Counter(_content_tokens(reference))
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def v3_correct(
    answer: str,
    *,
    expected_yes_no: bool | None,
    reference_answers: Sequence[str],
) -> Check:
    """V3: yes/no polarity, else maximum token-F1 >= 0.30."""
    if expected_yes_no is not None:
        expected = "yes" if expected_yes_no else "no"
        got = _first_word(answer)
        if got == expected:
            return Check(True, f"yes/no polarity {expected}")
        return Check(False, f"expected {expected}, got {got or 'no word'}")
    if not reference_answers:
        return Check(False, "no reference answer")
    best = max(token_f1(answer, reference) for reference in reference_answers)
    if best >= MIN_TOKEN_F1:
        return Check(True, f"token F1 {best:.3f}")
    return Check(False, f"token F1 {best:.3f} < {MIN_TOKEN_F1:.2f}")


def check_correctness(
    answer: str,
    *,
    expected_yes_no: bool | None,
    reference_answers: Sequence[str],
) -> Check:
    """Public V3 check, shared by the filter and the gold-free probe."""
    return v3_correct(
        answer,
        expected_yes_no=expected_yes_no,
        reference_answers=reference_answers,
    )


def _longest_shared_run(answer: str, chunk: str) -> int:
    left = answer.split()
    right = chunk.split()
    previous = [0] * (len(right) + 1)
    longest = 0
    for word in left:
        current = [0] * (len(right) + 1)
        for index, other in enumerate(right, start=1):
            if word == other:
                current[index] = previous[index - 1] + 1
                longest = max(longest, current[index])
        previous = current
    return longest


def v4_not_near_copy(
    answer: str,
    *,
    chunk_texts: Sequence[str],
    max_run: int = MAX_COPY_WORDS,
) -> Check:
    """V4: reject a shared run of more than ``max_run`` consecutive words."""
    longest = max((_longest_shared_run(answer, text) for text in chunk_texts), default=0)
    if longest > max_run:
        return Check(False, f"shared {longest}-word run (max {max_run})")
    return Check(True, f"longest shared run {longest}")


def verify_answer(
    answer: str,
    *,
    prompt_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    expected_yes_no: bool | None,
    reference_answers: Sequence[str],
    chunk_texts: Sequence[str],
    canary: str | None = TEACHER_CANARY,
) -> FilterResult:
    """Run V1-V4 in order and report the first failing rule (one answer only)."""
    checks = (
        ("V1", v1_format(answer, prompt_chunk_ids=prompt_chunk_ids, canary=canary)),
        ("V2", v2_cites_gold(answer, gold_chunk_ids=gold_chunk_ids)),
        (
            "V3",
            check_correctness(
                answer,
                expected_yes_no=expected_yes_no,
                reference_answers=reference_answers,
            ),
        ),
        ("V4", v4_not_near_copy(answer, chunk_texts=chunk_texts)),
    )
    for rule, check in checks:
        if not check.passed:
            return FilterResult(False, rule, check.reason)
    return FilterResult(True, None, "accepted")


class OllamaTeacher:
    """One-at-a-time client for the local qwen3:8b teacher."""

    def __init__(
        self,
        url: str = OLLAMA_CHAT_URL,
        *,
        model: str = TEACHER_MODEL,
        timeout: float = 300.0,
    ) -> None:
        self.url = url
        self.model = model
        self.timeout = timeout
        self._lock = threading.Lock()

    def answer(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,  # top-level switch, verified on Ollama 0.32.13
            "options": {
                "temperature": TEMPERATURE,
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
                "seed": SEED,
            },
        }
        with self._lock:
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)
            except requests.Timeout as exc:
                raise TeacherError(f"Ollama timed out after {self.timeout:g}s") from exc
            except requests.RequestException as exc:
                raise TeacherError(f"Ollama request failed: {exc}") from exc
        if response.status_code != 200:
            detail = response.text[:200]
            raise TeacherError(f"Ollama returned HTTP {response.status_code}: {detail}")
        try:
            data = response.json()
        except ValueError as exc:
            raise TeacherError("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise TeacherError("Ollama returned an unexpected response")
        if data.get("error"):
            raise TeacherError(f"Ollama error: {data['error']}")
        content = data.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise TeacherError("Ollama returned an empty answer")
        return content.strip()
