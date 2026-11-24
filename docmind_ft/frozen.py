"""Guards for the frozen evaluation inputs (F2.7).

The gate file is committed; ``reports/hashes.json`` is the local freeze
record.  Builders must refuse to rewrite frozen inputs unless ``--force`` is
explicit, and generation/scoring must verify the context bytes first.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GATES_PATH = REPO_ROOT / "evaluation" / "gates.json"
HASHES_PATH = REPO_ROOT / "reports" / "hashes.json"


class FrozenError(RuntimeError):
    """A frozen input is missing, changed, or about to be overwritten."""


def _load_object(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FrozenError(f"{path}: expected a JSON object")
    return data


def is_frozen(gates_path: Path | None = None) -> bool:
    path = gates_path or GATES_PATH
    if not path.exists():
        return False
    return bool(_load_object(path).get("meta", {}).get("frozen_at"))


def frozen_model_revision(gates_path: Path | None = None) -> str | None:
    path = gates_path or GATES_PATH
    if not path.exists():
        return None
    value = _load_object(path).get("meta", {}).get("model_revision")
    return str(value) if value else None


def frozen_b1_prompt_sha256(gates_path: Path | None = None) -> str | None:
    path = gates_path or GATES_PATH
    if not path.exists():
        return None
    value = _load_object(path).get("meta", {}).get("b1_prompt_sha256")
    return str(value) if value else None


def ensure_not_frozen(action: str, *, force: bool = False, gates_path: Path | None = None) -> None:
    if force:
        return
    path = gates_path or GATES_PATH
    if is_frozen(path):
        raise FrozenError(
            f"refusing to rebuild {action}: {path} has meta.frozen_at set; pass --force to override"
        )


def verify_context_hash(path: Path, *, hashes_path: Path | None = None) -> None:
    target = Path(path)
    if not target.is_file():
        raise FrozenError(f"frozen context file missing: {target}")
    record_path = hashes_path or HASHES_PATH
    entries = _load_object(record_path).get("contexts")
    if not isinstance(entries, dict):
        raise FrozenError(f"{record_path}: no contexts hash table")
    target_resolved = target.resolve()
    for entry in entries.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        recorded = Path(entry["path"])
        if not recorded.is_absolute():
            recorded = REPO_ROOT / recorded
        if recorded.resolve() != target_resolved:
            continue
        expected = entry.get("sha256")
        if not isinstance(expected, str) or not expected:
            raise FrozenError(f"{record_path}: no sha256 for {entry['path']}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise FrozenError(
                f"frozen context hash mismatch for {entry['path']}: "
                f"expected {expected}, got {actual}"
            )
        return
    raise FrozenError(f"{record_path}: no frozen hash for {target}")
