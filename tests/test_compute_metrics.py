"""Tests for the partial-run guard in the metrics CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
N_CONTEXTS = 96
N_OUTPUTS = 50


def _load_compute_metrics():
    path = REPO_ROOT / "scripts" / "compute_metrics.py"
    spec = importlib.util.spec_from_file_location("p8_compute_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _contexts(path: Path) -> None:
    _write_jsonl(
        path,
        [
            {
                "question_id": f"q{index}",
                "paper_id": "p1",
                "chunks": [{"chunk_id": f"p1:{index}", "score": 0.9, "page": 1}],
            }
            for index in range(N_CONTEXTS)
        ],
    )


def _outputs(path: Path) -> None:
    _write_jsonl(
        path,
        [
            {
                "question_id": f"q{index}",
                "output": "alpha [c1]",
                "seconds": 1.0,
                "prompt_tokens": 10,
                "generated_tokens": 2,
            }
            for index in range(N_OUTPUTS)
        ],
    )


class _FakeResolver:
    def text(self, chunk_id, paper_id, split):
        return "alpha"


def _configure(module, monkeypatch, contexts: Path) -> None:
    monkeypatch.setitem(module.CONTEXT_FILES, "dev", contexts)
    monkeypatch.setattr(
        module,
        "_gold_for",
        lambda split: {
            f"q{index}": {
                "gold_answers": ("alpha",),
                "yes_no": None,
                "gold_chunk_ids": (),
                "answerable": True,
                "answer_type": "extractive",
                "key_facts": (),
            }
            for index in range(N_CONTEXTS)
        },
    )
    monkeypatch.setattr(module, "ChunkTextResolver", _FakeResolver)
    monkeypatch.setattr(module, "verify_context_hash", lambda path: None)


def test_partial_run_exits_2_and_writes_nothing(tmp_path, monkeypatch):
    module = _load_compute_metrics()
    contexts = tmp_path / "contexts.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    _contexts(contexts)
    _outputs(outputs)
    _configure(module, monkeypatch, contexts)
    out = tmp_path / "metrics.json"

    code = module.main(["--split", "dev", "--outputs", str(outputs), "--out", str(out)])
    assert code == 2
    assert not out.exists()


def test_allow_partial_marks_the_output(tmp_path, monkeypatch):
    module = _load_compute_metrics()
    contexts = tmp_path / "contexts.jsonl"
    outputs = tmp_path / "outputs.jsonl"
    _contexts(contexts)
    _outputs(outputs)
    _configure(module, monkeypatch, contexts)
    out = tmp_path / "metrics.json"

    code = module.main(
        [
            "--split",
            "dev",
            "--outputs",
            str(outputs),
            "--out",
            str(out),
            "--allow-partial",
        ]
    )
    assert code == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["partial"] is True
    assert result["n_outputs"] == N_OUTPUTS
    assert result["n_expected"] == N_CONTEXTS
