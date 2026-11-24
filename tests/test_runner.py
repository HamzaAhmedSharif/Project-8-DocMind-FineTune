"""Tests for docmind_ft.detached (runner name/path helpers, script F0.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from docmind_ft.detached import job_paths, valid_job_name


@pytest.mark.parametrize("name", ["f0-test", "train_main", "a", "x" * 64, "A1-b_2"])
def test_valid_job_names(name: str) -> None:
    assert valid_job_name(name) is True


@pytest.mark.parametrize("name", ["", "has space", "has/slash", "has.dot", "-leading", "x" * 65])
def test_invalid_job_names(name: str) -> None:
    assert valid_job_name(name) is False


def test_job_paths_layout(tmp_path: Path) -> None:
    paths = job_paths(tmp_path, "smoke-1")
    assert paths["task"] == "p8-smoke-1"
    assert paths["log"] == tmp_path / "reports" / "logs" / "smoke-1.log"
    assert paths["heartbeat"] == tmp_path / "reports" / "logs" / "smoke-1.heartbeat.log"
    assert paths["done"] == tmp_path / "reports" / "logs" / "smoke-1.done"
    assert paths["failed"] == tmp_path / "reports" / "logs" / "smoke-1.failed"
    assert paths["args"] == tmp_path / "reports" / "logs" / "smoke-1-args.json"
    assert paths["wrapper"] == tmp_path / "reports" / "logs" / "smoke-1-wrapper.ps1"
    assert paths["launcher"] == tmp_path / "reports" / "logs" / "smoke-1-launcher.vbs"


def test_job_paths_rejects_invalid_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        job_paths(tmp_path, "bad name")
