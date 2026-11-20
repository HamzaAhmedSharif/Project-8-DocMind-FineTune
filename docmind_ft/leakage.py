"""Leakage gate I1 (F1.7).

Proves the training data cannot leak the evaluation sets. Three independent
checks, all of which must pass:

  1. paper-id overlap: zero overlap among Q-TRAIN, Q-DEV and Q-TEST paper ids.
  2. title overlap: zero normalized-title overlap between the QASPER papers
     (all three splits) and (a) the 24 HOLD manifest titles and
     (b) the Transformer paper title "Attention Is All You Need".
  3. 13-gram matches: zero exact 13-gram matches between any train row
     (prompt + completion text) and any HOLD or PAPER16 question, gold answer
     or key fact.

The module is pure and deterministic.  ``scripts/check_leakage.py`` is the thin
CLI that prints results, writes ``reports/F1-leakage.md`` and exits 0 only
when all three checks pass; on any failure it exits 1 and lists every
violation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: F401, E402  # installs the docmind_service alias
from docmind_ft.contexts import (  # noqa: E402
    QDEV_PATH,
    QTEST_PATH,
    load_split_questions,
)
from docmind_ft.qasper import load_qasper  # noqa: E402

WORK_DIR = Path(r"B:\p8\data\work")
QASPER_DIR = Path(r"B:\p8\data\qasper")
P7_MANIFEST = Path(
    r"A:\HDD\Work\VS Code - Python\GitHub\Project-7-DocMind-Service"
    r"\evaluation\heldout\manifest.json"
)
HOLD_QUESTIONS_PATH = REPO_ROOT / "evaluation" / "holdout" / "holdout_questions.jsonl"
PAPER16_QUESTIONS_PATH = REPO_ROOT / "evaluation" / "paper16" / "questions.py"
TRAIN_DIR = REPO_ROOT / "data"

TRAIN_FILES = ("train.jsonl", "train_nodistill.jsonl", "train_data25.jsonl")
QASPER_SPLITS = ("qasper-train-v0.3.json", "qasper-dev-v0.3.json", "qasper-test-v0.3.json")
TRANSFORMER_TITLE = "Attention Is All You Need"
N_GRAM = 13

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_title(text: str) -> str:
    """Lowercase, strip all punctuation, collapse whitespace."""
    return " ".join(text.lower().translate(_PUNCT_TABLE).split())


def _normalize_tokens(text: str) -> list[str]:
    """Lowercase, split on whitespace, strip punctuation from tokens."""
    tokens = []
    for token in text.lower().split():
        token = token.translate(_PUNCT_TABLE)
        if token:
            tokens.append(token)
    return tokens


def _ngram_list(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return every length-``n`` n-gram of ``tokens`` as a list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _ngram_hash(tokens: tuple[str, ...]) -> int:
    """Deterministic 64-bit hash of an n-gram, for compact distinct counting."""
    joined = " ".join(tokens).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(joined, digest_size=8).digest(), "little")


def _read_manifest_titles(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(paper["title"]) for paper in data["papers"]]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def check_paper_id_overlap(
    work_questions_path: Path | None = None,
    qdev_path: Path | None = None,
    qtest_path: Path | None = None,
) -> dict:
    """Zero paper-id overlap among Q-TRAIN, Q-DEV and Q-TEST."""
    work_questions_path = work_questions_path or WORK_DIR / "questions.jsonl"
    qdev_path = qdev_path or QDEV_PATH
    qtest_path = qtest_path or QTEST_PATH

    work_questions = _work_questions_from_path(work_questions_path)

    train_paper_ids: set[str] = set()
    for row in _load_jsonl(work_questions_path):
        if str(row.get("split")) == "train":
            train_paper_ids.add(str(row["paper_id"]))

    dev_rows = load_split_questions(qdev_path, work_questions)
    test_rows = load_split_questions(qtest_path, work_questions)
    dev_paper_ids = {pid for _, pid, _ in dev_rows}
    test_paper_ids = {pid for _, pid, _ in test_rows}

    train_dev = sorted(train_paper_ids & dev_paper_ids)
    train_test = sorted(train_paper_ids & test_paper_ids)
    dev_test = sorted(dev_paper_ids & test_paper_ids)

    violations: list[dict] = []
    if train_dev:
        violations.append({"pair": "train-dev", "paper_ids": train_dev})
    if train_test:
        violations.append({"pair": "train-test", "paper_ids": train_test})
    if dev_test:
        violations.append({"pair": "dev-test", "paper_ids": dev_test})

    return {
        "name": "paper-id overlap",
        "passed": not violations,
        "counts": {
            "train_papers": len(train_paper_ids),
            "dev_papers": len(dev_paper_ids),
            "test_papers": len(test_paper_ids),
        },
        "violations": violations,
    }


def _work_questions_from_path(path: Path) -> dict[str, dict[str, str]]:
    """question_id -> {"paper_id": ..., "question": ...} from an arbitrary JSONL."""
    by_id: dict[str, dict[str, str]] = {}
    for row in _load_jsonl(path):
        qid = str(row.get("question_id"))
        if qid:
            by_id[qid] = {
                "paper_id": str(row.get("paper_id", "")),
                "question": str(row.get("question", "")),
            }
    return by_id


def check_title_overlap(
    qasper_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> dict:
    """Zero normalized-title overlap between QASPER papers and HOLD/Transformer titles."""
    qasper_dir = qasper_dir or QASPER_DIR
    manifest_path = manifest_path or P7_MANIFEST

    qasper_titles: set[str] = set()
    for split in QASPER_SPLITS:
        for paper in load_qasper(qasper_dir / split):
            title = (paper.title or "").strip()
            if title:
                qasper_titles.add(_normalize_title(title))

    hold_titles = {_normalize_title(t) for t in _read_manifest_titles(manifest_path)}
    transformer_title = _normalize_title(TRANSFORMER_TITLE)

    hold_overlap = sorted(qasper_titles & hold_titles)
    transformer_hit = transformer_title in qasper_titles

    violations: list[dict] = []
    if hold_overlap:
        violations.append({"side": "hold-manifest", "titles": hold_overlap})
    if transformer_hit:
        violations.append({"side": "transformer", "titles": [TRANSFORMER_TITLE]})

    return {
        "name": "title overlap",
        "passed": not violations,
        "counts": {
            "qasper_titles": len(qasper_titles),
            "hold_titles": len(hold_titles),
        },
        "violations": violations,
    }


def check_ngram_overlap(
    train_dir: Path | None = None,
    hold_path: Path | None = None,
    paper16_path: Path | None = None,
) -> dict:
    """Zero exact 13-gram matches between train rows and HOLD/PAPER16 text."""
    train_dir = train_dir or TRAIN_DIR
    hold_path = hold_path or HOLD_QUESTIONS_PATH
    paper16_path = paper16_path or PAPER16_QUESTIONS_PATH

    # Reference side: HOLD questions + gold answers + verbatim key facts, and
    # PAPER16 questions + key-fact text.
    eval_texts: list[str] = []
    for item in _load_jsonl(hold_path):
        eval_texts.append(str(item.get("question", "")))
        eval_texts.append(str(item.get("gold_answer", "")))
        for fact in item.get("key_facts", []) or []:
            eval_texts.append(str(fact.get("verbatim", "")))
    spec = importlib.util.spec_from_file_location("paper16_questions_leak", paper16_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    for q in module.QUESTIONS:
        eval_texts.append(str(q.get("question", "")))
        for fact in q.get("key_facts", []) or []:
            eval_texts.append(str(fact.get("fact", "")))

    eval_ngrams: set[tuple[str, ...]] = set()
    for text in eval_texts:
        eval_ngrams.update(_ngram_list(_normalize_tokens(text), N_GRAM))

    # Train side: stream rows, compare each n-gram against the reference set.
    train_ngram_hashes: set[int] = set()
    train_instances = 0
    violations: list[dict] = []
    for train_file in TRAIN_FILES:
        for row in _load_jsonl(train_dir / train_file):
            parts: list[str] = []
            for msg in row.get("prompt", []) or []:
                parts.append(str(msg.get("content", "")))
            for msg in row.get("completion", []) or []:
                parts.append(str(msg.get("content", "")))
            tokens = _normalize_tokens(" ".join(parts))
            row_ngrams = _ngram_list(tokens, N_GRAM)
            train_instances += len(row_ngrams)
            qid = str((row.get("meta") or {}).get("question_id", "<unknown>"))
            for ngram in row_ngrams:
                train_ngram_hashes.add(_ngram_hash(ngram))
                if ngram in eval_ngrams:
                    violations.append(
                        {
                            "file": train_file,
                            "question_id": qid,
                            "ngram": " ".join(ngram),
                        }
                    )

    return {
        "name": "13-gram matches",
        "passed": not violations,
        "counts": {
            "train_13gram_instances": train_instances,
            "train_13gram_distinct": len(train_ngram_hashes),
            "eval_13gram_distinct": len(eval_ngrams),
        },
        "violations": violations,
    }


def run_all_checks() -> list[dict]:
    """Run every gate check and return the per-check results."""
    return [
        check_paper_id_overlap(),
        check_title_overlap(),
        check_ngram_overlap(),
    ]
