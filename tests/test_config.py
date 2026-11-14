"""Tests for docmind_ft.config (script F0.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from docmind_ft.config import DOTENV_NAMES, Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_repr_never_contains_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake values deliberately do NOT look like real tokens (no provider
    # prefix), so the verify.ps1 secret scan stays meaningful: anything
    # shaped like a real secret must be a real finding, not a test fixture.
    monkeypatch.setenv("HF_TOKEN", "test-token-value-aaaa")
    monkeypatch.setenv("HF_WRITE_TOKEN", "test-write-token-bbbb")
    monkeypatch.setenv("DOCMIND_LLM_API_KEY", "test-llm-key-cccc")
    settings = Settings.from_env()
    assert settings.hf_token == "test-token-value-aaaa"
    rendered = repr(settings) + str(settings)
    assert "test-token-value-aaaa" not in rendered
    assert "test-write-token-bbbb" not in rendered
    assert "test-llm-key-cccc" not in rendered
    assert "hf_token=set" in rendered
    assert "hf_write_token=set" in rendered


def test_unset_values_show_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_TOKEN", "HF_WRITE_TOKEN", "DOCMIND_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert "hf_token=unset" in repr(settings)


def test_dotenv_example_names_match_config_fields() -> None:
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    names = sorted(
        line.split("=", 1)[0].strip()
        for line in example.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert names == sorted(DOTENV_NAMES)


def test_allow_test_flag_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCMIND_FT_ALLOW_TEST", "1")
    assert Settings.from_env().docmind_ft_allow_test is True
    monkeypatch.setenv("DOCMIND_FT_ALLOW_TEST", "0")
    assert Settings.from_env().docmind_ft_allow_test is False
    monkeypatch.delenv("DOCMIND_FT_ALLOW_TEST", raising=False)
    assert Settings.from_env().docmind_ft_allow_test is False


def test_p7_dir_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCMIND_P7_DIR", r"A:\somewhere\p7")
    assert Settings.from_env().docmind_p7_dir == r"A:\somewhere\p7"
    assert "somewhere" not in repr(Settings.from_env())
