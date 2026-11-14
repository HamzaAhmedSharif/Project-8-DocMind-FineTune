"""QASPER data engine: pseudo-pages, chunks, gold mapping, splits, embeddings.

Builds one chunk stream per paper from the loader's word offsets, maps every
annotator's evidence paragraphs to overlapping chunks, samples the Q-DEV and
Q-TEST question splits, and caches MiniLM embeddings plus top-10 retrieval
under ``B:\\p8\\data\\work\\`` (never in git). Each question is ranked only
against its own paper's chunks.

Run with ``python -m docmind_ft.build_data``. The full run embeds roughly
46,000 chunks; on the laptop CPU that takes several minutes.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from docmind_ft.qasper import Paper, Paragraph, Question, load_qasper
from docmind_ft.retrieval import (
    EMBEDDING_MODEL,
    TOP_K,
    FastEmbedder,
    cosine_scores,
    rank_chunks,
)
from docmind_ft.vendor.chunking import CHUNK_OVERLAP, CHUNK_WORDS, chunk_pages

DATA_DIR = Path(r"B:\p8\data\qasper")
WORK_DIR = Path(r"B:\p8\data\work")
SPLITS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "splits"
SPLITS = ("train", "dev", "test")
ANSWER_TYPES = ("extractive", "free_form", "yes_no", "unanswerable")
SEED = 20261207
PAGE_WORDS = 500
GOLD_MAX_OVERLAP = 30
GOLD_MIN_SHARE = 0.5


@dataclass(frozen=True)
class ChunkData:
    split: str
    paper_id: str
    chunk_id: str
    page: int
    ordinal: int
    word_start: int
    word_end: int  # exclusive
    text: str


@dataclass(frozen=True)
class QuestionData:
    split: str
    paper_id: str
    question_id: str
    question: str
    answer_type: str
    answerable: bool
    mixed: bool
    gold_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class QuestionLabel:
    answer_type: str
    answerable: bool
    mixed: bool


def word_page(word_offset: int) -> int:
    """1-based pseudo-page of one word in the document stream."""
    return 1 + word_offset // PAGE_WORDS


def chunk_paper(paper: Paper, split: str) -> tuple[ChunkData, ...]:
    """Chunk one paper's word stream; page is the pseudo-page of the first word."""
    words: list[str] = []
    for paragraph in paper.paragraphs:
        paragraph_words = paragraph.text.split()
        if paragraph.word_start != len(words):
            raise ValueError(
                f"{paper.paper_id}: paragraph starts at {paragraph.word_start}, "
                f"expected {len(words)}"
            )
        if paragraph.word_end != len(words) + len(paragraph_words):
            raise ValueError(f"{paper.paper_id}: paragraph offsets disagree with its text")
        words.extend(paragraph_words)
    if not words:
        return ()
    pages: dict[int, list[str]] = defaultdict(list)
    for offset, word in enumerate(words):
        pages[word_page(offset)].append(word)
    page_texts = [" ".join(pages[page]) for page in sorted(pages)]
    chunks = chunk_pages(page_texts)
    step = CHUNK_WORDS - CHUNK_OVERLAP
    records = []
    for chunk in chunks:
        start = chunk.ordinal * step
        chunk_words = chunk.text.split()
        end = start + len(chunk_words)
        if words[start:end] != chunk_words or chunk.page != word_page(start):
            raise ValueError(
                f"{paper.paper_id}: chunk {chunk.ordinal} does not match the word stream"
            )
        records.append(
            ChunkData(
                split=split,
                paper_id=paper.paper_id,
                chunk_id=f"{paper.paper_id}:{chunk.ordinal}",
                page=chunk.page,
                ordinal=chunk.ordinal,
                word_start=start,
                word_end=end,
                text=chunk.text,
            )
        )
    return tuple(records)


def label_question(question: Question) -> QuestionLabel:
    """Question-level label from annotator answers.

    Unanswerable only when every annotator marked it unanswerable. Mixed means
    some annotators answered and some did not. When answering annotators
    disagree on type, yes/no wins over extractive, which wins over free-form.
    """
    answering = [answer for answer in question.answers if not answer.unanswerable]
    if not question.answers:
        return QuestionLabel("unknown", False, False)
    mixed = bool(answering) and len(answering) < len(question.answers)
    if not answering:
        return QuestionLabel("unanswerable", False, mixed)
    if any(answer.yes_no is not None for answer in answering):
        answer_type = "yes_no"
    elif any(answer.extractive_spans for answer in answering):
        answer_type = "extractive"
    else:
        answer_type = "free_form"
    return QuestionLabel(answer_type, True, mixed)


def _paragraph_index(paper: Paper) -> dict[str, Paragraph]:
    """Evidence text -> paragraph, matching on whitespace-collapsed text.

    Real QASPER evidence sometimes drops a paragraph's leading whitespace, so
    exact character matching would lose about 3% of the gold paragraphs.
    """
    index: dict[str, Paragraph] = {}
    for paragraph in paper.paragraphs:
        index.setdefault(" ".join(paragraph.text.split()), paragraph)
    return index


def gold_chunk_ids(
    paper: Paper, question: Question, chunks: Sequence[ChunkData]
) -> tuple[str, ...]:
    """Chunks overlapping an answering annotator's evidence by the gold threshold."""
    paragraphs = _paragraph_index(paper)
    gold: set[str] = set()
    for answer in question.answers:
        if answer.unanswerable:
            continue
        for evidence in answer.evidence:
            if evidence.startswith("FLOAT SELECTED"):
                continue
            paragraph = paragraphs.get(" ".join(evidence.split()))
            if paragraph is None:
                raise ValueError(
                    f"{paper.paper_id}: evidence paragraph not found for "
                    f"{question.question_id}: {evidence[:80]!r}"
                )
            if paragraph.section == "FLOAT SELECTED":
                continue
            length = paragraph.word_end - paragraph.word_start
            if length <= 0:
                continue
            threshold = min(GOLD_MAX_OVERLAP, GOLD_MIN_SHARE * length)
            for chunk in chunks:
                overlap = min(chunk.word_end, paragraph.word_end) - max(
                    chunk.word_start, paragraph.word_start
                )
                if overlap >= threshold:
                    gold.add(chunk.chunk_id)
    return tuple(chunk.chunk_id for chunk in chunks if chunk.chunk_id in gold)


def build_question(
    split: str, paper: Paper, question: Question, chunks: Sequence[ChunkData]
) -> QuestionData:
    label = label_question(question)
    return QuestionData(
        split=split,
        paper_id=paper.paper_id,
        question_id=question.question_id,
        question=question.question,
        answer_type=label.answer_type,
        answerable=label.answerable,
        mixed=label.mixed,
        gold_chunk_ids=gold_chunk_ids(paper, question, chunks),
    )


def _proportional(size: int, counts: dict[str, int]) -> dict[str, int]:
    """Largest-remainder allocation of ``size`` over the answer-type strata."""
    total = sum(counts.values())
    if total <= 0 or size > total:
        raise ValueError(f"cannot allocate {size} questions over {counts}")
    quotas = {answer_type: counts[answer_type] * size // total for answer_type in ANSWER_TYPES}
    remainder = size - sum(quotas.values())
    fractions = {
        answer_type: counts[answer_type] * size / total - quotas[answer_type]
        for answer_type in ANSWER_TYPES
    }
    order = sorted(
        ANSWER_TYPES,
        key=lambda answer_type: (-fractions[answer_type], ANSWER_TYPES.index(answer_type)),
    )
    for answer_type in order[:remainder]:
        quotas[answer_type] += 1
    return quotas


def select_split(
    records: Sequence[QuestionData],
    *,
    size: int,
    seed: int = SEED,
    max_per_paper: int = 3,
    min_papers: int = 0,
) -> tuple[str, ...]:
    """Seeded stratified sample of question ids, at most ``max_per_paper`` per paper."""
    pools: dict[str, list[QuestionData]] = {answer_type: [] for answer_type in ANSWER_TYPES}
    for record in records:
        if record.answer_type in pools:
            pools[record.answer_type].append(record)
    if not 0 < size <= sum(len(pool) for pool in pools.values()):
        raise ValueError(f"cannot select {size} questions from {len(records)}")
    rng = random.Random(seed)
    for pool in pools.values():
        pool.sort(key=lambda record: record.question_id)
        rng.shuffle(pool)
    quotas = _proportional(size, {answer_type: len(pool) for answer_type, pool in pools.items()})
    chosen: list[QuestionData] = []
    per_paper: Counter[str] = Counter()
    leftovers: list[QuestionData] = []
    for answer_type in ANSWER_TYPES:
        quota = quotas[answer_type]
        for record in pools[answer_type]:
            if quota > 0 and per_paper[record.paper_id] < max_per_paper:
                chosen.append(record)
                per_paper[record.paper_id] += 1
                quota -= 1
            else:
                leftovers.append(record)
    if len(chosen) < size:
        rng.shuffle(leftovers)
        for record in leftovers:
            if len(chosen) >= size:
                break
            if per_paper[record.paper_id] < max_per_paper:
                chosen.append(record)
                per_paper[record.paper_id] += 1
    if len(chosen) < size:
        raise ValueError(f"cannot select {size} questions with {max_per_paper} per paper")
    if len(per_paper) < min_papers:
        raise ValueError(f"selected {len(per_paper)} papers, need {min_papers}")
    return tuple(sorted(record.question_id for record in chosen))


def write_split(name: str, question_ids: Sequence[str]) -> tuple[Path, str]:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{name}.json"
    text = json.dumps(list(question_ids), indent=2) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path, hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _chunk_fingerprint(chunks: Sequence[ChunkData]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.chunk_id}\t{chunk.text}\n".encode())
    return digest.hexdigest()


def embeddings_for(
    chunks: Sequence[ChunkData], work_dir: Path, embedder: FastEmbedder
) -> np.ndarray:
    """Load the cached embedding matrix when it matches, else embed and cache."""
    fingerprint = _chunk_fingerprint(chunks)
    meta = {
        "model": EMBEDDING_MODEL,
        "chunks_sha256": fingerprint,
        "count": len(chunks),
    }
    vectors_path = work_dir / "embeddings.npy"
    meta_path = work_dir / "embeddings.meta.json"
    if vectors_path.exists() and meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            vectors = np.load(vectors_path)
            if cached == meta and vectors.shape == (len(chunks), embedder.dim):
                print(f"embeddings cache hit: {vectors.shape}")
                return vectors
        except (OSError, EOFError, ValueError):
            pass
    print(f"embedding {len(chunks)} chunks ({EMBEDDING_MODEL}) ...", flush=True)
    started = time.perf_counter()
    vectors = embedder.embed([chunk.text for chunk in chunks])
    tmp_path = vectors_path.with_name(vectors_path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, vectors)
    tmp_path.replace(vectors_path)
    with meta_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    elapsed = time.perf_counter() - started
    print(f"embedded {vectors.shape} in {elapsed:.1f}s")
    return vectors


def _write_split_and_report(
    name: str, records: Sequence[QuestionData], size: int, min_papers: int
) -> tuple[str, ...]:
    question_ids = select_split(records, size=size, seed=SEED, min_papers=min_papers)
    path, digest = write_split(name, question_ids)
    by_id = {record.question_id: record for record in records}
    selected = [by_id[question_id] for question_id in question_ids]
    types = Counter(record.answer_type for record in selected)
    papers = len({record.paper_id for record in selected})
    breakdown = " ".join(
        f"{answer_type}={types.get(answer_type, 0)}" for answer_type in ANSWER_TYPES
    )
    mixed = sum(record.mixed for record in selected)
    print(f"{name}: {len(question_ids)} questions, {papers} papers, mixed={mixed}")
    print(f"  types {breakdown}")
    print(f"  file  {path}")
    print(f"  sha256 {digest}")
    return question_ids


def main() -> int:
    started = time.perf_counter()
    chunks: list[ChunkData] = []
    questions: list[QuestionData] = []
    for split in SPLITS:
        papers = load_qasper(DATA_DIR / f"qasper-{split}-v0.3.json")
        split_chunks = 0
        for paper in papers:
            paper_chunks = chunk_paper(paper, split)
            chunks.extend(paper_chunks)
            split_chunks += len(paper_chunks)
            for question in paper.questions:
                questions.append(build_question(split, paper, question, paper_chunks))
        print(f"{split}: {len(papers)} papers, {split_chunks} chunks")

    _write_split_and_report(
        "qdev", [record for record in questions if record.split == "dev"], 200, 60
    )
    _write_split_and_report(
        "qtest", [record for record in questions if record.split == "test"], 400, 150
    )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(WORK_DIR / "chunks.jsonl", (asdict(chunk) for chunk in chunks))
    print(f"chunks: {len(chunks)} rows -> {WORK_DIR / 'chunks.jsonl'}")

    embedder = FastEmbedder(cache_dir=str(WORK_DIR / "fastembed-cache"))
    vectors = embeddings_for(chunks, WORK_DIR, embedder)

    print(f"embedding {len(questions)} questions ...", flush=True)
    query_vectors = embedder.embed([question.question for question in questions])
    rows_by_paper: dict[str, list[int]] = defaultdict(list)
    for row, chunk in enumerate(chunks):
        rows_by_paper[chunk.paper_id].append(row)
    paper_rows = {
        paper_id: np.asarray(rows, dtype=np.int64) for paper_id, rows in rows_by_paper.items()
    }

    questions_path = WORK_DIR / "questions.jsonl"
    with questions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for question, query_vector in zip(questions, query_vectors, strict=True):
            rows = paper_rows.get(question.paper_id)
            if rows is not None:
                chunk_ids = [chunks[row].chunk_id for row in rows]
                scores = cosine_scores(query_vector, vectors[rows])
                retrieved = rank_chunks(scores, chunk_ids, TOP_K)
                scores_by_id = dict(zip(chunk_ids, scores, strict=True))
            else:
                retrieved = []
                scores_by_id = {}
            row = asdict(question)
            row["gold_chunk_ids"] = list(question.gold_chunk_ids)
            row["retrieved"] = [
                {"chunk_id": item.chunk_id, "score": item.score} for item in retrieved
            ]
            # Context assembly needs the score of a gold chunk even when it
            # falls outside the top 10, so keep gold scores alongside retrieval.
            row["gold_scores"] = {
                chunk_id: float(scores_by_id[chunk_id])
                for chunk_id in question.gold_chunk_ids
                if chunk_id in scores_by_id
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - started
    print(f"questions: {len(questions)} rows -> {questions_path}")
    print(f"build complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
