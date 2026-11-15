# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Full teacher-build logic for the F1.4 training data (distilled targets).

The train questions split into three cases: A (answerable with gold), U
(unanswerable), and excluded (answerable without gold).  A questions take a
hash-assigned gold or p0 branch.  Gold contexts keep the teacher hint; p0
contexts remove every gold chunk and probe the teacher without the hint.  A
failed probe becomes an abstention example; a passed probe is ambiguous and
is dropped.  U questions are abstentions from the cached top-10 with no teacher
call.

Only the question text is embedded at run time for p0 context ranking.  The
chunk matrix in ``embeddings.npy`` is reused as-is and never rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np

import docmind_ft.vendor  # noqa: F401  (installs the docmind_service alias)
from docmind_ft.qasper import Answer, load_qasper
from docmind_ft.retrieval import (
    TOP_K,
    Retrieved,
    cosine_scores,
    rank_chunks,
)
from docmind_ft.teacher import (
    Check,
    build_teacher_prompt,
    check_correctness,
    ensure_gold_context,
    expected_yes_no,
    human_answer,
    reference_answers,
    verify_answer,
)
from docmind_service.domain import RetrievedChunk
from docmind_service.guardrails import canary_leaked
from docmind_service.query import NO_EVIDENCE_ANSWER

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = Path(r"B:\p8\data\work")
QUESTIONS_PATH = WORK_DIR / "questions.jsonl"
CHUNKS_PATH = WORK_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = WORK_DIR / "embeddings.npy"
FASTEMBED_CACHE = WORK_DIR / "fastembed-cache"
PILOT_OUT_PATH = WORK_DIR / "pilot_out.jsonl"
TEACHER_OUT_PATH = WORK_DIR / "teacher_out.jsonl"
P0_PATH = WORK_DIR / "p0.json"
QASPER_TRAIN_PATH = WORK_DIR.parent / "qasper" / "qasper-train-v0.3.json"

TRAIN_DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"

CASE_A = "A"
CASE_U = "U"
CASE_EXCLUDED = "excluded"
BRANCH_GOLD = "gold"
BRANCH_P0 = "p0"

SEED = 20261207
PILOT_PASS_RATE = 0.527
TRAIN_A = 2101
TRAIN_U = 271
TRAIN_EXCLUDED = 221
DATA25_SHARE = 0.25
PROBE_SAMPLE = 200
P0_MAX = 0.30

PROMPT_ID_RE = re.compile(r"<<<(c\d+) page=\d+>>>", re.IGNORECASE)
CITATION_RE = re.compile(r"\[(c\d+)\]", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class Teacher(Protocol):
    def answer(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class TrainQuestion:
    question_id: str
    paper_id: str
    question: str
    answer_type: str
    answerable: bool
    gold_chunk_ids: tuple[str, ...]
    retrieved: tuple[Retrieved, ...]
    gold_scores: dict[str, float]

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> TrainQuestion:
        return cls(
            question_id=str(row["question_id"]),
            paper_id=str(row["paper_id"]),
            question=str(row["question"]),
            answer_type=str(row["answer_type"]),
            answerable=bool(row["answerable"]),
            gold_chunk_ids=tuple(str(x) for x in row.get("gold_chunk_ids", [])),
            retrieved=tuple(
                Retrieved(str(item["chunk_id"]), float(item["score"]))
                for item in row.get("retrieved", [])
            ),
            gold_scores={
                str(key): float(value) for key, value in dict(row.get("gold_scores", {})).items()
            },
        )


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    paper_id: str
    page: int
    text: str


@dataclass
class ChunkStore:
    ids: list[str]
    by_id: dict[str, ChunkRecord]
    rows_by_paper: dict[str, list[int]]

    @classmethod
    def from_records(cls, records: Sequence[ChunkRecord]) -> ChunkStore:
        ids = [record.chunk_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk id in chunks.jsonl")
        by_id = {record.chunk_id: record for record in records}
        rows_by_paper: dict[str, list[int]] = {}
        for row, record in enumerate(records):
            rows_by_paper.setdefault(record.paper_id, []).append(row)
        return cls(ids=ids, by_id=by_id, rows_by_paper=rows_by_paper)


@dataclass(frozen=True)
class ProbeOutcome:
    answer: str
    check: Check
    context: tuple[Retrieved, ...]
    prompt: str

    @property
    def failed(self) -> bool:
        return not self.check.passed


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def load_train_questions(path: str | Path = QUESTIONS_PATH) -> list[TrainQuestion]:
    return [TrainQuestion.from_row(row) for row in read_jsonl(path) if row.get("split") == "train"]


def load_chunk_store(path: str | Path = CHUNKS_PATH) -> ChunkStore:
    records = [
        ChunkRecord(
            chunk_id=str(row["chunk_id"]),
            paper_id=str(row["paper_id"]),
            page=int(row["page"]),
            text=str(row["text"]),
        )
        for row in read_jsonl(path)
    ]
    return ChunkStore.from_records(records)


def load_embeddings(path: str | Path = EMBEDDINGS_PATH, expected_rows: int | None = None):
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D embedding matrix, got {matrix.shape}")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(f"{path}: {matrix.shape[0]} embedding rows for {expected_rows} chunks")
    return matrix


def load_train_answers(path: str | Path = QASPER_TRAIN_PATH) -> dict[str, tuple[Answer, ...]]:
    return {
        question.question_id: question.answers
        for paper in load_qasper(path)
        for question in paper.questions
    }


def load_pilot_ids(path: str | Path = PILOT_OUT_PATH) -> set[str]:
    if not Path(path).exists():
        return set()
    return {str(row["question_id"]) for row in read_jsonl(path)}


def new_canary() -> str:
    return f"P8-{uuid4().hex[:12]}"


def assign_case(question: TrainQuestion) -> str:
    if not question.answerable:
        return CASE_U
    return CASE_A if question.gold_chunk_ids else CASE_EXCLUDED


def branch_p0(question_id: str, p0: float) -> bool:
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 must be in [0, 1]")
    digest = hashlib.sha256(f"{question_id}:branch".encode()).hexdigest()
    return int(digest[:8], 16) / 2**32 < p0


def is_data25(question_id: str) -> bool:
    digest = hashlib.sha256(question_id.encode()).hexdigest()
    return int(digest[:8], 16) / 2**32 < DATA25_SHARE


def _gold_score(question: TrainQuestion, chunk_id: str) -> float:
    if chunk_id in question.gold_scores:
        return float(question.gold_scores[chunk_id])
    for item in question.retrieved:
        if item.chunk_id == chunk_id:
            return item.score
    raise ValueError(f"no score for gold chunk {chunk_id}")


def gold_retrieved(question: TrainQuestion) -> list[Retrieved]:
    if not question.gold_chunk_ids:
        raise ValueError(f"{question.question_id}: no gold chunks")
    return [
        Retrieved(chunk_id, _gold_score(question, chunk_id)) for chunk_id in question.gold_chunk_ids
    ]


def gold_context(question: TrainQuestion, top_k: int = TOP_K) -> list[Retrieved]:
    # The best gold chunk is the one Option A cites, so make sure it is present
    # even when a lower-scored gold chunk already sits in the top 10.
    best = min(
        gold_retrieved(question),
        key=lambda item: (-item.score, item.chunk_id),
    )
    return ensure_gold_context(question.retrieved, [best], top_k)


def best_gold_short_id(question: TrainQuestion, mapping: Mapping[str, RetrievedChunk]) -> str:
    gold = set(question.gold_chunk_ids)
    present = [(short, chunk) for short, chunk in mapping.items() if chunk.chunk_id in gold]
    if not present:
        raise ValueError(f"{question.question_id}: prompt has no gold chunk")
    return min(
        present,
        key=lambda pair: (-_gold_score(question, pair[1].chunk_id), pair[1].chunk_id),
    )[0]


def gold_free_context(
    question: TrainQuestion,
    store: ChunkStore,
    embeddings: np.ndarray,
    query_vector: np.ndarray,
    top_k: int = TOP_K,
) -> list[Retrieved]:
    """Top non-gold chunks from this paper only, ranked by cached cosine."""
    gold = set(question.gold_chunk_ids)
    rows = [row for row in store.rows_by_paper.get(question.paper_id, [])]
    keep_rows = [row for row in rows if store.ids[row] not in gold]
    if not keep_rows:
        return []
    chunk_ids = [store.ids[row] for row in keep_rows]
    scores = cosine_scores(query_vector, embeddings[keep_rows])
    return rank_chunks(scores, chunk_ids, top_k)


def context_chunks(context: Sequence[Retrieved], store: ChunkStore) -> list[RetrievedChunk]:
    chunks = []
    for item in context:
        record = store.by_id.get(item.chunk_id)
        if record is None:
            raise ValueError(f"context chunk {item.chunk_id} is not in chunks.jsonl")
        chunks.append(
            RetrievedChunk(
                chunk_id=record.chunk_id,
                document_id=uuid5(NAMESPACE_URL, record.paper_id),
                page=record.page,
                text=record.text,
                score=item.score,
            )
        )
    return chunks


def build_prompt_from_context(
    question: str,
    context: Sequence[Retrieved],
    store: ChunkStore,
    canary: str,
    *,
    include_hint: bool = True,
    human: str = "",
    gold_chunk_ids: Sequence[str] = (),
) -> tuple[str, dict[str, RetrievedChunk]]:
    return build_teacher_prompt(
        question,
        context_chunks(context, store),
        human,
        gold_chunk_ids,
        canary,
        include_hint=include_hint,
    )


def build_prompt_from_ids(
    question: TrainQuestion,
    context_chunk_ids: Sequence[str],
    store: ChunkStore,
    canary: str,
) -> tuple[str, dict[str, RetrievedChunk]]:
    context = [Retrieved(chunk_id, 0.0) for chunk_id in context_chunk_ids]
    return build_prompt_from_context(question.question, context, store, canary, include_hint=False)


class QuestionEmbedder:
    """Lazy local MiniLM wrapper for question text only (chunks are cached)."""

    def __init__(self, cache_dir: str | Path = FASTEMBED_CACHE) -> None:
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"fastembed cache {self.cache_dir} is missing; refusing to download"
            )
        self._model = None

    def _get_model(self):
        if self._model is None:
            from docmind_ft.retrieval import FastEmbedder

            self._model = FastEmbedder(cache_dir=str(self.cache_dir))
        return self._model

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self._get_model().embed([text])[0])


def probe_question(
    question: TrainQuestion,
    answers: Sequence[Answer],
    store: ChunkStore,
    teacher: Teacher,
    *,
    embeddings: np.ndarray,
    embed_question: Callable[[str], np.ndarray],
    canary: str,
    top_k: int = TOP_K,
) -> ProbeOutcome:
    query = embed_question(question.question)
    context = gold_free_context(question, store, embeddings, query, top_k)
    prompt, _ = build_prompt_from_context(
        question.question, context, store, canary, include_hint=False
    )
    answer = teacher.answer(prompt)
    check = check_correctness(
        answer,
        expected_yes_no=expected_yes_no(answers),
        reference_answers=reference_answers(answers),
    )
    return ProbeOutcome(answer, check, tuple(context), prompt)


def process_question(
    question: TrainQuestion,
    answers: Sequence[Answer],
    store: ChunkStore,
    teacher: Teacher,
    *,
    p0: float,
    canary: str,
    embeddings: np.ndarray | None = None,
    embed_question: Callable[[str], np.ndarray] | None = None,
) -> dict[str, object]:
    """Return one teacher_out row.  ``passed`` means the example is kept."""
    case = assign_case(question)
    if case == CASE_EXCLUDED:
        raise ValueError(f"{question.question_id}: excluded questions have no row")

    base: dict[str, object] = {
        "question_id": question.question_id,
        "case": case,
        "canary": canary,
    }

    if case == CASE_U:
        context = list(question.retrieved[:TOP_K])
        return {
            **base,
            "branch": None,
            "context_chunk_ids": [item.chunk_id for item in context],
            "teacher_answer": None,
            "passed": True,
            "failed_rule": None,
        }

    branch = BRANCH_P0 if branch_p0(question.question_id, p0) else BRANCH_GOLD
    if branch == BRANCH_GOLD:
        context = gold_context(question)
        human = human_answer(answers)
        if human is None:
            raise ValueError(f"{question.question_id}: no answering annotator")
        teacher_prompt, mapping = build_prompt_from_context(
            question.question,
            context,
            store,
            canary,
            include_hint=True,
            human=human,
            gold_chunk_ids=question.gold_chunk_ids,
        )
        answer = teacher.answer(teacher_prompt)
        chunks = context_chunks(context, store)
        gold_short = [
            short
            for short, chunk in mapping.items()
            if chunk.chunk_id in set(question.gold_chunk_ids)
        ]
        result = verify_answer(
            answer,
            prompt_chunk_ids=list(mapping),
            gold_chunk_ids=gold_short,
            expected_yes_no=expected_yes_no(answers),
            reference_answers=reference_answers(answers),
            chunk_texts=[chunk.text for chunk in chunks],
            canary=canary,
        )
        return {
            **base,
            "branch": branch,
            "context_chunk_ids": [item.chunk_id for item in context],
            "teacher_answer": answer,
            "passed": result.passed,
            "failed_rule": result.failed_rule,
        }

    if embeddings is None or embed_question is None:
        raise ValueError("p0 branch needs embeddings and a question embedder")
    outcome = probe_question(
        question,
        answers,
        store,
        teacher,
        embeddings=embeddings,
        embed_question=embed_question,
        canary=canary,
    )
    kept = outcome.failed
    return {
        **base,
        "branch": branch,
        "context_chunk_ids": [item.chunk_id for item in outcome.context],
        "teacher_answer": outcome.answer,
        "passed": kept,
        "failed_rule": None if kept else "probe_pass",
        "probe_failed": outcome.failed,
    }


def option_a_target(
    question: TrainQuestion,
    answers: Sequence[Answer],
    mapping: Mapping[str, RetrievedChunk],
) -> str:
    human = human_answer(answers)
    if human is None:
        raise ValueError(f"{question.question_id}: no answering annotator")
    short = best_gold_short_id(question, mapping)
    return f"{human} [{short}]"


def make_trl_row(prompt: str, target: str, meta: Mapping[str, object]) -> dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": target}],
        "meta": dict(meta),
    }


def assemble_trl_rows(
    teacher_rows: Sequence[Mapping[str, object]],
    questions: Mapping[str, TrainQuestion],
    store: ChunkStore,
    answers_by_id: Mapping[str, Sequence[Answer]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build (distilled, option-a) rows from the kept teacher_out rows."""
    distilled: list[dict[str, object]] = []
    option_a: list[dict[str, object]] = []

    for row in teacher_rows:
        if not row.get("passed"):
            continue
        question_id = str(row["question_id"])
        question = questions[question_id]
        case = str(row["case"])
        branch = row.get("branch")
        canary = str(row["canary"])
        context_ids = [str(item) for item in row["context_chunk_ids"]]
        prompt, mapping = build_prompt_from_ids(question, context_ids, store, canary)
        meta = {
            "question_id": question_id,
            "paper_id": question.paper_id,
            "case": case,
            "answer_type": question.answer_type,
            "branch": branch,
            "canary": canary,
        }

        if case == CASE_U or branch == BRANCH_P0:
            target = NO_EVIDENCE_ANSWER
            distilled.append(make_trl_row(prompt, target, meta))
            option_a.append(make_trl_row(prompt, target, meta))
            continue

        teacher_answer = row.get("teacher_answer")
        if not isinstance(teacher_answer, str) or not teacher_answer.strip():
            raise ValueError(f"{question_id}: kept gold row has no teacher answer")
        distilled.append(make_trl_row(prompt, teacher_answer, meta))
        option_a.append(
            make_trl_row(
                prompt,
                option_a_target(question, answers_by_id[question_id], mapping),
                meta,
            )
        )

    return distilled, option_a


def sentence_count(text: str) -> int:
    plain = CITATION_RE.sub(" ", text).strip()
    if not plain:
        return 0
    return len([part for part in SENTENCE_RE.split(plain) if part.strip()])


def validate_trl_row(row: Mapping[str, object]) -> list[str]:
    """Prototype I2 checks for one conversational prompt-completion row."""
    errors: list[str] = []
    meta = row.get("meta")
    if not isinstance(meta, Mapping):
        errors.append("missing meta")
        meta = {}
    canary = meta.get("canary")
    if not isinstance(canary, str) or not canary:
        errors.append("missing canary")

    prompt = row.get("prompt")
    prompt_content = ""
    if not isinstance(prompt, list) or not prompt:
        errors.append("missing prompt")
    else:
        first = prompt[0]
        if not isinstance(first, Mapping) or not isinstance(first.get("content"), str):
            errors.append("invalid prompt message")
        else:
            prompt_content = first["content"]
            if not prompt_content.strip():
                errors.append("empty prompt")

    completion = row.get("completion")
    target = ""
    if not isinstance(completion, list) or not completion:
        errors.append("missing completion")
    else:
        first = completion[0]
        if not isinstance(first, Mapping) or not isinstance(first.get("content"), str):
            errors.append("invalid completion message")
        else:
            target = first["content"]
            if not target.strip():
                errors.append("empty completion")

    if canary and isinstance(canary, str) and canary_leaked(target, canary):
        errors.append("canary in target")

    prompt_ids = set(PROMPT_ID_RE.findall(prompt_content))
    cited = {item.casefold() for item in CITATION_RE.findall(target)}
    unknown = sorted(cited - prompt_ids)
    if unknown:
        errors.append(f"unknown citation [{unknown[0]}]")

    if target == NO_EVIDENCE_ANSWER:
        return errors
    stripped = target.strip()
    if stripped == NO_EVIDENCE_ANSWER or stripped.casefold().startswith("i could not find"):
        errors.append("abstention target is not exact")

    if target:
        words = target.split()
        if len(words) > 90:
            errors.append(f"answer has {len(words)} words (max 90)")
        sentences = sentence_count(target)
        if not 1 <= sentences <= 3:
            errors.append(f"answer has {sentences} sentences (expected 1-3)")
        if not cited:
            errors.append("answered target has no citation")
    return errors


def i2_stats(path: str | Path) -> dict[str, object]:
    total = valid = 0
    errors: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            row_errors = validate_trl_row(row)
            if row_errors:
                errors.append(f"line {line_number}: {row_errors[0]}")
            else:
                valid += 1
    return {
        "rows": total,
        "valid": valid,
        "i2_valid_share": valid / total if total else 1.0,
        "errors": errors[:20],
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def answer_length_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    lengths = []
    for row in rows:
        completion = row.get("completion")
        if not isinstance(completion, list) or not completion:
            continue
        first = completion[0]
        if not isinstance(first, Mapping):
            continue
        target = first.get("content")
        if isinstance(target, str) and target != NO_EVIDENCE_ANSWER:
            lengths.append(len(target.split()))
    if not lengths:
        return {
            "n": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0,
        }
    return {
        "n": len(lengths),
        "min": min(lengths),
        "mean": round(mean(lengths), 3),
        "median": round(median(lengths), 3),
        "p90": round(percentile(lengths, 0.90), 3),
        "max": max(lengths),
    }


def abstention_share(rows: Sequence[Mapping[str, object]]) -> float:
    if not rows:
        return 0.0
    abstentions = 0
    for row in rows:
        completion = row.get("completion")
        if (
            isinstance(completion, list)
            and completion
            and isinstance(completion[0], Mapping)
            and completion[0].get("content") == NO_EVIDENCE_ANSWER
        ):
            abstentions += 1
    return abstentions / len(rows)


def summarize_teacher_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    a_rows = [row for row in rows if row.get("case") == CASE_A]
    gold = [row for row in a_rows if row.get("branch") == BRANCH_GOLD]
    p0 = [row for row in a_rows if row.get("branch") == BRANCH_P0]
    u_rows = [row for row in rows if row.get("case") == CASE_U]
    kept = sum(bool(row.get("passed")) for row in rows)
    gold_kept = sum(bool(row.get("passed")) for row in gold)
    p0_kept = sum(bool(row.get("passed")) for row in p0)
    fail_rules = Counter(
        str(row.get("failed_rule"))
        for row in rows
        if not row.get("passed") and row.get("failed_rule")
    )
    return {
        "rows": len(rows),
        "prompts": sum(row.get("teacher_answer") is not None for row in a_rows),
        "kept": kept,
        "dropped": len(rows) - kept,
        "by_case": dict(Counter(str(row.get("case")) for row in rows)),
        "by_branch": dict(Counter(str(row.get("branch")) for row in a_rows)),
        "by_case_branch": dict(Counter(f"{row.get('case')}:{row.get('branch')}" for row in rows)),
        "gold": {
            "rows": len(gold),
            "kept": gold_kept,
            "pass_rate": gold_kept / len(gold) if gold else 0.0,
        },
        "p0": {
            "rows": len(p0),
            "kept": p0_kept,
            "dropped": len(p0) - p0_kept,
            "probe_fail_rate": p0_kept / len(p0) if p0 else 0.0,
        },
        "u": {
            "rows": len(u_rows),
            "kept": sum(bool(row.get("passed")) for row in u_rows),
        },
        "failed_rules": dict(fail_rules),
    }


def compute_p0(A: int, U: int, q: float, r: float) -> float:
    """Abstention-branch share from the F1.4 formula, clamped to [0, 0.30]."""
    if A <= 0:
        raise ValueError("A must be positive")
    numerator = 0.20 * A * q - 0.80 * U
    denominator = A * (0.80 * r + 0.20 * q)
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return min(max(numerator / denominator, 0.0), P0_MAX)


def case_counts(questions: Sequence[TrainQuestion]) -> dict[str, int]:
    return dict(Counter(assign_case(question) for question in questions))
