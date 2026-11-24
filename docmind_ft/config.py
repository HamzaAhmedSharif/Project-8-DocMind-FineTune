"""Settings for Project 8 (script 3.12).

Reads variables from the environment. Secret values are NEVER printed:
``repr``/``str`` of a Settings object show only field names and whether
each value is set. Check keys by name only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Names that must appear in .env.example (script F0.4).
DOTENV_NAMES = (
    "HF_TOKEN",
    "HF_WRITE_TOKEN",
    "DOCMIND_LLM_API_KEY",
    "DOCMIND_P7_DIR",
)


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


@dataclass(repr=False)
class Settings:
    hf_token: str | None = None
    hf_write_token: str | None = None
    docmind_llm_api_key: str | None = None
    docmind_p7_dir: str | None = None
    hf_home: str | None = None
    hf_endpoint: str | None = None
    docmind_ft_allow_test: bool = False
    ollama_host: str = "http://127.0.0.1:11434"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            hf_token=os.environ.get("HF_TOKEN") or None,
            hf_write_token=os.environ.get("HF_WRITE_TOKEN") or None,
            docmind_llm_api_key=os.environ.get("DOCMIND_LLM_API_KEY") or None,
            docmind_p7_dir=os.environ.get("DOCMIND_P7_DIR") or None,
            hf_home=os.environ.get("HF_HOME") or None,
            hf_endpoint=os.environ.get("HF_ENDPOINT") or None,
            docmind_ft_allow_test=_env_bool("DOCMIND_FT_ALLOW_TEST"),
            ollama_host=os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434",
        )

    def __repr__(self) -> str:  # never include a value
        parts = []
        for name in (
            "hf_token",
            "hf_write_token",
            "docmind_llm_api_key",
            "docmind_p7_dir",
            "hf_home",
            "hf_endpoint",
        ):
            parts.append(f"{name}={'set' if getattr(self, name) else 'unset'}")
        parts.append(f"docmind_ft_allow_test={self.docmind_ft_allow_test!r}")
        parts.append(f"ollama_host={self.ollama_host!r}")
        return f"Settings({', '.join(parts)})"

    __str__ = __repr__
