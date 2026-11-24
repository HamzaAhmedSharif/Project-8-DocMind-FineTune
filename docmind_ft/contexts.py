"""Frozen retrieval context builder (F1.6).

Builds top-10 retrieval contexts for the four frozen evaluation inputs
(Q-DEV, Q-TEST, HOLD, PAPER16).  Chunk texts are deliberately *not* stored;
they are re-derived deterministically from the paper text when the harness
runs.

Row format::

    {"question_id": "...", "paper_id": "...",
     "chunks": [{"chunk_id": "...", "score": 0.123456, "page": 3}, ...]}

Retrieval is plain top-10 by cosine similarity within the question's own
paper, reusing ``docmind_ft.retrieval`` (exact cosine, stable sort, chunk-id
tie-break).  No gold insertion, no filtering.
"""

from __future__ import annotations

# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: F401, E402  # installs the docmind_service alias

from docmind_ft.retrieval import TOP_K, FastEmbedder, cosine_scores, rank_chunks  # noqa: E402
from docmind_ft.vendor.chunking import chunk_pages  # noqa: E402

WORK_DIR = Path(r"B:\p8\data\work")
HOLD_PAGES_DIR = Path(r"B:\p8\data\holdout")
PAPER16_PAGES = Path(r"B:\p8\data\paper16\attention.pages.json")

CHUNKS_PATH = WORK_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = WORK_DIR / "embeddings.npy"
QUESTIONS_PATH = WORK_DIR / "questions.jsonl"
FASTEMBED_CACHE = WORK_DIR / "fastembed-cache"

SPLITS_DIR = REPO_ROOT / "evaluation" / "splits"
QDEV_PATH = SPLITS_DIR / "qdev.json"
QTEST_PATH = SPLITS_DIR / "qtest.json"
HOLD_QUESTIONS_PATH = REPO_ROOT / "evaluation" / "holdout" / "holdout_questions.jsonl"
PAPER16_QUESTIONS_PATH = REPO_ROOT / "evaluation" / "paper16" / "questions.py"

PAPER16_PAPER_ID = "1706.03762"

CONTEXTS_DIR = REPO_ROOT / "evaluation" / "contexts"

CONTEXT_SET_SPECS = (
    ("contexts_qdev", QDEV_PATH, 200),
    ("contexts_qtest", QTEST_PATH, 400),
    ("contexts_hold", HOLD_QUESTIONS_PATH, 96),
    ("contexts_paper16", PAPER16_QUESTIONS_PATH, 16),
)

PAPER16_EXPECTED_IDS = [f"d{i:02d}" for i in range(1, 13)] + [f"a{i:02d}" for i in range(1, 5)]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_work_questions() -> dict[str, dict[str, str]]:
    """question_id -> {"paper_id": ..., "question": ...} from work/questions.jsonl."""
    by_id: dict[str, dict[str, str]] = {}
    with QUESTIONS_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{QUESTIONS_PATH}:{line_number}: invalid JSON: {exc}") from exc
            qid = row.get("question_id")
            if not isinstance(qid, str) or not qid:
                continue
            by_id[qid] = {
                "paper_id": str(row.get("paper_id", "")),
                "question": str(row.get("question", "")),
            }
    return by_id


def load_split_questions(
    split_path: Path, work_questions: dict[str, dict[str, str]]
) -> list[tuple[str, str, str]]:
    """Return [(question_id, paper_id, question)] in split-file order."""
    ids = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(ids, list):
        raise ValueError(f"{split_path}: expected a list of question ids")
    rows: list[tuple[str, str, str]] = []
    missing: list[str] = []
    for qid in ids:
        entry = work_questions.get(str(qid))
        if entry is None:
            missing.append(str(qid))
            continue
        rows.append((str(qid), entry["paper_id"], entry["question"]))
    if missing:
        raise ValueError(
            f"{split_path}: {len(missing)} question ids not in questions.jsonl (e.g. {missing[:5]})"
        )
    return rows


class WorkChunkIndex:
    """Cached QASPER chunk embeddings, grouped by paper.

    ``embeddings.npy`` row i corresponds to ``chunks.jsonl`` line i.
    """

    def __init__(self) -> None:
        self.embeddings = np.load(EMBEDDINGS_PATH)
        self.by_paper: dict[str, list[tuple[str, int, int]]] = {}
        with CHUNKS_PATH.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                paper_id = str(row["paper_id"])
                self.by_paper.setdefault(paper_id, []).append(
                    (str(row["chunk_id"]), int(row["page"]), row_index)
                )

    def paper_chunks(self, paper_id: str):
        entries = self.by_paper.get(paper_id)
        if not entries:
            raise ValueError(f"no cached chunks for paper {paper_id!r}")
        chunk_ids = [e[0] for e in entries]
        pages = [e[1] for e in entries]
        matrix = self.embeddings[[e[2] for e in entries]]
        return chunk_ids, pages, matrix


def _load_holdout_questions() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with HOLD_QUESTIONS_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{HOLD_QUESTIONS_PATH}:{line_number}: invalid JSON: {exc}"
                ) from exc
            rows.append((str(item["id"]), str(item["paper_id"]), str(item["question"])))
    return rows


def _load_paper16_questions() -> list[tuple[str, str, str]]:
    spec = importlib.util.spec_from_file_location("paper16_questions", PAPER16_QUESTIONS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    questions = module.QUESTIONS
    if not isinstance(questions, list) or len(questions) != 16:
        raise ValueError(
            f"{PAPER16_QUESTIONS_PATH}: expected 16 questions, got "
            f"{len(questions) if isinstance(questions, list) else type(questions)}"
        )
    actual_ids = [str(q.get("id")) for q in questions]
    if actual_ids != PAPER16_EXPECTED_IDS:
        raise ValueError(f"{PAPER16_QUESTIONS_PATH}: ids mismatch: {actual_ids}")
    return [(str(q["id"]), PAPER16_PAPER_ID, str(q["question"])) for q in questions]


def _chunk_and_embed_paper(
    paper_id: str, pages_file: Path, embedder: FastEmbedder
) -> tuple[list[str], list[int], object]:
    data = json.loads(pages_file.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
        raise ValueError(f"{pages_file}: expected a JSON list of page strings")
    chunks = chunk_pages(data)
    chunk_ids = [f"{paper_id}:{chunk.ordinal}" for chunk in chunks]
    pages = [chunk.page for chunk in chunks]
    vectors = embedder.embed([chunk.text for chunk in chunks])
    return chunk_ids, pages, vectors


def _retrieve_top10(
    question_text: str,
    chunk_ids: list[str],
    pages: list[int],
    matrix,
    embedder: FastEmbedder,
) -> list[dict]:
    query = embedder.embed([question_text])[0]
    ranked = rank_chunks(cosine_scores(query, matrix), chunk_ids, TOP_K)
    page_by_id = {cid: page for cid, page in zip(chunk_ids, pages, strict=True)}
    return [
        {
            "chunk_id": item.chunk_id,
            "score": round(float(item.score), 6),
            "page": page_by_id[item.chunk_id],
        }
        for item in ranked
    ]


def _build_contexts(
    question_rows: list[tuple[str, str, str]],
    paper_loader,
    embedder: FastEmbedder,
) -> list[dict]:
    rows: list[dict] = []
    for question_id, paper_id, question in question_rows:
        chunk_ids, pages, matrix = paper_loader(paper_id)
        rows.append(
            {
                "question_id": question_id,
                "paper_id": paper_id,
                "chunks": _retrieve_top10(question, chunk_ids, pages, matrix, embedder),
            }
        )
    return rows


def build_q_contexts(
    split_path: Path,
    work_questions: dict[str, dict[str, str]],
    index: WorkChunkIndex,
    embedder: FastEmbedder,
) -> list[dict]:
    return _build_contexts(
        load_split_questions(split_path, work_questions),
        lambda pid: index.paper_chunks(pid),
        embedder,
    )


def build_hold_contexts(embedder: FastEmbedder) -> list[dict]:
    paper_cache: dict[str, tuple[list[str], list[int], object]] = {}

    def loader(paper_id: str):
        if paper_id not in paper_cache:
            pages_file = HOLD_PAGES_DIR / f"{paper_id}.pages.json"
            paper_cache[paper_id] = _chunk_and_embed_paper(paper_id, pages_file, embedder)
        return paper_cache[paper_id]

    return _build_contexts(_load_holdout_questions(), loader, embedder)


def build_paper16_contexts(embedder: FastEmbedder) -> list[dict]:
    chunk_ids, pages, vectors = _chunk_and_embed_paper(PAPER16_PAPER_ID, PAPER16_PAGES, embedder)
    loader = lambda pid: (chunk_ids, pages, vectors)  # noqa: E731
    return _build_contexts(_load_paper16_questions(), loader, embedder)


def build_all(embedder: FastEmbedder | None = None) -> dict[str, list[dict]]:
    """Build all four context sets.  Returns {name: rows}."""
    if embedder is None:
        if not FASTEMBED_CACHE.exists():
            raise RuntimeError(f"cached fastembed model directory is missing: {FASTEMBED_CACHE}")
        embedder = FastEmbedder(cache_dir=str(FASTEMBED_CACHE))

    work_questions = load_work_questions()
    index = WorkChunkIndex()

    return {
        "contexts_qdev": build_q_contexts(QDEV_PATH, work_questions, index, embedder),
        "contexts_qtest": build_q_contexts(QTEST_PATH, work_questions, index, embedder),
        "contexts_hold": build_hold_contexts(embedder),
        "contexts_paper16": build_paper16_contexts(embedder),
    }
