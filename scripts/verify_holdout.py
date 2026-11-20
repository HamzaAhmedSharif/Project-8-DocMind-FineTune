#!/usr/bin/env python3
"""Validate the authored HOLD-set questions (F1.5)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docmind_ft.holdout import QUESTIONS_PATH, verify_questions  # noqa: E402


def main() -> int:
    if not QUESTIONS_PATH.exists():
        print(f"ERROR: questions file not found: {QUESTIONS_PATH}")
        return 1
    try:
        result = verify_questions()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    for failure in result.failures:
        print(f"{failure.item_id} {failure.check}: {failure.detail}")
    print(
        f"items checked {result.items}; passed {result.passed}; "
        f"failed {result.failed}; retrievable {result.retrievable}"
    )
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
