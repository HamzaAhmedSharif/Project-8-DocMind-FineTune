"""Windows-only integration test for the detached runner argument boundary."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or POWERSHELL is None,
    reason="scripts/run_detached.ps1 is Windows-only",
)


def _run_powershell(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ps_quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def test_detached_runner_preserves_apostrophe_and_metacharacter(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copy2(REPO_ROOT / "scripts" / "run_detached.ps1", scripts_dir / "run_detached.ps1")

    job = tmp_path / "job.ps1"
    job.write_text(
        "param([string]$Out)\n$args | Set-Content -LiteralPath $Out -Encoding utf8\n",
        encoding="utf-8",
        newline="\n",
    )
    out = tmp_path / "argument values.txt"
    name = f"pytest-{uuid.uuid4().hex}"

    caller = tmp_path / "caller.ps1"
    caller.write_text(
        f"& {_ps_quote(scripts_dir / 'run_detached.ps1')} "
        f"-Name {_ps_quote(name)} -Script {_ps_quote(job)} "
        f"-Args @({_ps_quote(out)}, 'O''Brien.jsonl', 'x & y') -DryRun\n",
        encoding="utf-8",
        newline="\n",
    )
    _run_powershell("-File", str(caller))

    logs = tmp_path / "reports" / "logs"
    payload = json.loads((logs / f"{name}-args.json").read_text(encoding="utf-8"))
    assert Path(payload["script"]).resolve() == job.resolve()
    assert payload["args"] == [str(out), "O'Brien.jsonl", "x & y"]
    assert payload["timeout_minutes"] == 0

    wrapper = (logs / f"{name}-wrapper.ps1").read_text(encoding="utf-8")
    assert "O'Brien.jsonl" not in wrapper
    assert "x & y" not in wrapper

    _run_powershell("-File", str(logs / f"{name}-wrapper.ps1"))

    assert out.read_text(encoding="utf-8-sig").splitlines() == ["O'Brien.jsonl", "x & y"]
    assert (logs / f"{name}.done").exists()
    assert not (logs / f"{name}.failed").exists()
