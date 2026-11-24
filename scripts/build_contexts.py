#!/usr/bin/env python3
# ruff: noqa: I001  # import order is load-bearing: docmind_ft.vendor installs the alias
"""Build the frozen retrieval contexts for F1.6.

Writes evaluation/contexts/{contexts_qdev,contexts_qtest,contexts_hold,contexts_paper16}.jsonl
and reports/hashes.json with the SHA-256 of each file.

    python scripts/build_contexts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import docmind_ft.vendor  # noqa: F401, E402  # installs the docmind_service alias

from docmind_ft.contexts import build_all, CONTEXTS_DIR, sha256_file  # noqa: E402
from docmind_ft.frozen import FrozenError, ensure_not_frozen  # noqa: E402

HASHES_PATH = REPO_ROOT / "reports" / "hashes.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild frozen contexts and update their hashes",
    )
    args = parser.parse_args(argv)
    try:
        ensure_not_frozen("evaluation contexts", force=args.force)
    except FrozenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    previous = {}
    if HASHES_PATH.exists():
        try:
            previous = json.loads(HASHES_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"error: {HASHES_PATH}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(previous, dict):
            print(f"error: {HASHES_PATH}: expected a JSON object", file=sys.stderr)
            return 2

    sets = build_all()
    hashes = {}
    for name, rows in sets.items():
        path = CONTEXTS_DIR / f"{name}.jsonl"
        _write_jsonl(path, rows)
        hashes[name] = {
            "path": f"evaluation/contexts/{name}.jsonl",
            "sha256": sha256_file(path),
            "rows": len(rows),
        }
        print(f"{name}: {len(rows)} rows -> {path} (sha256 {hashes[name]['sha256']})")

    contexts = previous.get("contexts")
    if not isinstance(contexts, dict):
        contexts = {}
    contexts.update(hashes)
    previous["contexts"] = contexts
    HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HASHES_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(previous, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {HASHES_PATH}")
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
