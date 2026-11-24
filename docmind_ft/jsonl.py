"""JSONL reading with recovery for an interrupted final line."""

from __future__ import annotations

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def read_jsonl(path: str | Path, *, recover_tail: bool = True) -> list[dict]:
    """Read JSONL rows, truncating only an unterminated, unparseable final line.

    A complete bad line (newline-terminated) or a bad line in the middle still
    raises: only the last line of a file that does not end with a newline can
    be an interrupted write. ``recover_tail=False`` keeps strict reads for
    frozen inputs.
    """
    target = Path(path)
    if not target.exists():
        return []
    data = target.read_bytes()
    if recover_tail and data and not data.endswith(b"\n"):
        last_line = data.splitlines(keepends=True)[-1]
        try:
            json.loads(last_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            start = len(data) - len(last_line)
            with target.open("r+b") as handle:
                handle.truncate(start)
            LOGGER.warning("truncated incomplete final JSONL line in %s", target)
            data = data[:start]

    rows: list[dict] = []
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{target}:{line_number}: invalid JSON: {exc}") from exc
    return rows
