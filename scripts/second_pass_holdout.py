#!/usr/bin/env python3
"""Run the independent HOLD-set second pass against local qwen3:8b (F1.5c).

Each answerable item is answered from its evidence pages only.  An item is
flagged when none of its accept patterns matches the model answer.  Results
are appended to a resumable JSONL file; unanswerable items are skipped.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docmind_ft.holdout import (  # noqa: E402
    PAGES_DIR,
    QUESTIONS_PATH,
    load_pages,
    load_questions,
    normalize,
)
from docmind_ft.teacher import TEACHER_MODEL, OllamaTeacher, TeacherError  # noqa: E402

OUTPUT_PATH = Path(r"B:\p8\data\work\second_pass_out.jsonl")


def build_prompt(item: dict, pages: list[str]) -> str:
    """Render evidence pages and the question; no gold answer or key facts."""
    evidence = []
    for page_number in item["evidence_pages"]:
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or not 1 <= page_number <= len(pages)
        ):
            raise ValueError(f"invalid evidence page {page_number!r}")
        evidence.append(f"[Page {page_number}]\n{pages[page_number - 1]}")
    if not evidence:
        raise ValueError(f"{item['id']}: no evidence pages")
    return "Evidence:\n" + "\n\n".join(evidence) + f"\n\nQuestion: {item['question']}\nAnswer:"


def _accept_patterns(item: dict) -> list[str]:
    patterns = []
    for fact in item.get("key_facts", []):
        if not isinstance(fact, dict):
            continue
        accept = fact.get("accept", [])
        if not isinstance(accept, list):
            continue
        patterns.extend(pattern for pattern in accept if isinstance(pattern, str))
    return patterns


def is_flagged(item: dict, model_answer: str) -> bool:
    """Flag unless any accept pattern matches the normalized model answer."""
    answer = normalize(model_answer)
    for pattern in _accept_patterns(item):
        try:
            if re.search(pattern, answer, re.IGNORECASE):
                return False
        except re.error:
            continue
    return True


def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                done.add(row["id"])
    return done


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _ollama_executable() -> str | None:
    executable = shutil.which("ollama")
    if executable:
        return executable
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(local) if local.exists() else None


def stop_model() -> None:
    """Release the GPU after the run, using the installed Ollama CLI."""
    executable = _ollama_executable()
    if executable is None:
        print("warning: ollama executable not found; could not stop qwen3:8b")
        return
    completed = subprocess.run(
        [executable, "stop", TEACHER_MODEL],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        print(f"warning: ollama stop failed: {completed.stderr.strip()}")


def run(
    questions_path: Path = QUESTIONS_PATH,
    pages_dir: Path = PAGES_DIR,
    output_path: Path = OUTPUT_PATH,
    *,
    answer_fn: Callable[[str], str] | None = None,
    stop: bool = True,
) -> int:
    """Second-pass every missing answerable item, one request at a time."""
    questions = load_questions(questions_path)
    answerable = [row for row in questions if row.get("answerable") is True]
    answer = answer_fn or OllamaTeacher().answer
    done = _load_done(output_path)
    checked = flagged = skipped = 0

    try:
        for item in answerable:
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                print("STOPPED: answerable item has no id")
                return 1
            if item_id in done:
                skipped += 1
                continue
            pages = load_pages(item["paper_id"], pages_dir)
            prompt = build_prompt(item, pages)
            try:
                model_answer = answer(prompt)
            except TeacherError as exc:
                print(f"STOPPED {item_id}: {exc}")
                return 1
            item_flagged = is_flagged(item, model_answer)
            _append_row(
                output_path,
                {"id": item_id, "model_answer": model_answer, "flagged": item_flagged},
            )
            checked += 1
            flagged += int(item_flagged)
            print(f"{'FLAG' if item_flagged else 'ok  '} {item_id}")
    finally:
        if stop:
            stop_model()

    print(f"items checked {checked}; flagged {flagged}; skipped {skipped}")
    return 0


def main() -> int:
    try:
        return run()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
