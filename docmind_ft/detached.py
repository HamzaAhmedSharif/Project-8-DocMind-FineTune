"""Name and path helpers for the detached runner (script 1.9).

Single source of truth for detached-job naming; ``scripts/run_detached.ps1``
mirrors these rules (same task-name prefix, same file layout under
``reports/logs/``). Pure functions, unit-tested in ``tests/test_runner.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

TASK_PREFIX = "p8-"


def valid_job_name(name: str) -> bool:
    """A job name is one token: letters, digits, '-' and '_', max 64 chars."""
    return bool(name) and bool(_JOB_NAME_RE.match(name))


def job_paths(repo_root: str | Path, name: str) -> dict[str, Path]:
    """Return the task name and file paths for a detached job.

    Raises ValueError on an invalid name. Layout (script 1.9):
    job output -> <name>.log, heartbeat -> <name>.heartbeat.log,
    completion markers -> <name>.done / <name>.failed, job arguments ->
    <name>-args.json, generated wrapper/launcher ->
    <name>-wrapper.ps1 / <name>-launcher.vbs.
    """
    if not valid_job_name(name):
        raise ValueError(f"invalid job name: {name!r}")
    logs = Path(repo_root) / "reports" / "logs"
    return {
        "task": TASK_PREFIX + name,
        "log": logs / f"{name}.log",
        "heartbeat": logs / f"{name}.heartbeat.log",
        "done": logs / f"{name}.done",
        "failed": logs / f"{name}.failed",
        "args": logs / f"{name}-args.json",
        "wrapper": logs / f"{name}-wrapper.ps1",
        "launcher": logs / f"{name}-launcher.vbs",
    }
