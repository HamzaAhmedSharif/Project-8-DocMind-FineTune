#!/usr/bin/env python3
"""Download and verify the QASPER v0.3 archives (F1.1).

Fetches both pinned archives, checks each SHA-256 before unpacking, then
checks the paper/question counts of every split. Exit 1 on any checksum or
count mismatch. An archive already on disk with a matching checksum is not
downloaded again.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

import requests

DATA_DIR = Path(r"B:\p8\data\qasper")
BASE_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com"
ATTEMPTS = 3

# (archive name, sha256, required members)
ARCHIVES = (
    (
        "qasper-train-dev-v0.3.tgz",
        "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a",
        ("qasper-train-v0.3.json", "qasper-dev-v0.3.json"),
    ),
    (
        "qasper-test-and-evaluator-v0.3.tgz",
        "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72",
        ("qasper-test-v0.3.json",),
    ),
)

# (split file, expected papers, expected questions; None = not pinned)
SPLITS = (
    ("qasper-train-v0.3.json", 888, 2593),
    ("qasper-dev-v0.3.json", 281, 1005),
    ("qasper-test-v0.3.json", 416, None),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, expected_sha: str) -> Path:
    dest = DATA_DIR / name
    if dest.exists() and file_sha256(dest) == expected_sha:
        print(f"cached  {name}")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        print(f"fetch   {name} (attempt {attempt}/{ATTEMPTS})")
        try:
            with requests.get(f"{BASE_URL}/{name}", stream=True, timeout=(15, 300)) as response:
                response.raise_for_status()
                with part.open("wb") as handle:
                    for block in response.iter_content(1 << 20):
                        handle.write(block)
            break
        except requests.RequestException as exc:
            print(f"error   {name}: {exc}")
            if attempt == ATTEMPTS:
                raise SystemExit(f"download failed after {ATTEMPTS} attempts: {name}") from exc
            time.sleep(5 * attempt)
    actual = file_sha256(part)
    if actual != expected_sha:
        raise SystemExit(f"checksum mismatch for {name}: expected {expected_sha}, got {actual}")
    part.replace(dest)
    print(f"ok      {name} sha256 {actual}")
    return dest


def unpack(archive: Path, members: tuple[str, ...]) -> None:
    root = DATA_DIR.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        missing = [name for name in members if name not in names]
        if missing:
            raise SystemExit(f"{archive.name}: missing member(s): {', '.join(missing)}")
        for member in tar.getmembers():
            if not (member.isfile() or member.isdir()):
                raise SystemExit(f"{archive.name}: unsupported member: {member.name}")
            if not (root / member.name).resolve().is_relative_to(root):
                raise SystemExit(f"{archive.name}: unsafe member path: {member.name}")
        tar.extractall(DATA_DIR, filter="data")
    print(f"unpack  {archive.name}")


def verify_counts() -> None:
    failures = []
    for name, expected_papers, expected_questions in SPLITS:
        path = DATA_DIR / name
        if not path.exists():
            raise SystemExit(f"missing split file: {path}")
        papers = json.loads(path.read_text(encoding="utf-8"))
        questions = sum(len(paper["qas"]) for paper in papers.values())
        print(f"check   {name}: {len(papers)} papers / {questions} questions")
        questions_ok = expected_questions is None or questions == expected_questions
        if len(papers) != expected_papers or not questions_ok:
            expected = f"{expected_papers} papers"
            if expected_questions is not None:
                expected += f" / {expected_questions} questions"
            failures.append(f"{name}: expected {expected}, got {len(papers)} / {questions}")
    if failures:
        raise SystemExit("count mismatch:\n" + "\n".join(failures))


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"target  {DATA_DIR}")
    for name, expected_sha, members in ARCHIVES:
        unpack(download(name, expected_sha), members)
    verify_counts()
    print("QASPER v0.3 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
