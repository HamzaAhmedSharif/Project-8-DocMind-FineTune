"""Tests for the frozen-input guards and the hashes merge."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from docmind_ft import build_data, frozen
from docmind_ft.frozen import FrozenError, ensure_not_frozen, verify_context_hash

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_build_contexts():
    path = REPO_ROOT / "scripts" / "build_contexts.py"
    spec = importlib.util.spec_from_file_location("p8_build_contexts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_check_gates():
    path = REPO_ROOT / "scripts" / "check_gates.py"
    spec = importlib.util.spec_from_file_location("p8_check_gates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frozen_gates(tmp_path: Path) -> Path:
    path = tmp_path / "gates.json"
    path.write_text(json.dumps({"meta": {"frozen_at": "2026-11-24T00:00:00Z"}}), encoding="utf-8")
    return path


def test_ensure_not_frozen_refuses_until_forced(tmp_path):
    gates = _frozen_gates(tmp_path)
    with pytest.raises(FrozenError, match="force"):
        ensure_not_frozen("contexts", gates_path=gates)
    ensure_not_frozen("contexts", force=True, gates_path=gates)


def test_verify_context_hash_refuses_changed_bytes(tmp_path):
    context = tmp_path / "contexts_qdev.jsonl"
    context.write_text('{"question_id": "q1"}\n', encoding="utf-8", newline="\n")
    digest = hashlib.sha256(context.read_bytes()).hexdigest()
    hashes = tmp_path / "hashes.json"
    hashes.write_text(
        json.dumps({"contexts": {"contexts_qdev": {"path": str(context), "sha256": digest}}}),
        encoding="utf-8",
    )
    verify_context_hash(context, hashes_path=hashes)
    context.write_text('{"question_id": "q2"}\n', encoding="utf-8", newline="\n")
    with pytest.raises(FrozenError, match="mismatch"):
        verify_context_hash(context, hashes_path=hashes)


def test_build_contexts_refuses_frozen_without_force(tmp_path, monkeypatch):
    module = _load_build_contexts()
    monkeypatch.setattr(frozen, "GATES_PATH", _frozen_gates(tmp_path))
    called = []
    monkeypatch.setattr(module, "build_all", lambda: called.append(True) or {})
    assert module.main([]) == 2
    assert not called


def test_build_contexts_merges_the_existing_hashes(tmp_path, monkeypatch):
    module = _load_build_contexts()
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({"meta": {}}), encoding="utf-8")
    monkeypatch.setattr(frozen, "GATES_PATH", gates)
    monkeypatch.setattr(module, "CONTEXTS_DIR", tmp_path / "contexts")
    hashes_path = tmp_path / "hashes.json"
    monkeypatch.setattr(module, "HASHES_PATH", hashes_path)
    hashes_path.write_text(
        json.dumps(
            {
                "baselines": {"b1_frozen": {"sha256": "base"}},
                "gates.json": {"frozen_blob_sha256": "gate"},
                "contexts": {"old": {"sha256": "old"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "build_all",
        lambda: {"contexts_qdev": [{"question_id": "q1"}]},
    )

    assert module.main([]) == 0
    merged = json.loads(hashes_path.read_text(encoding="utf-8"))
    assert merged["baselines"] == {"b1_frozen": {"sha256": "base"}}
    assert merged["gates.json"] == {"frozen_blob_sha256": "gate"}
    assert merged["contexts"]["old"] == {"sha256": "old"}
    assert merged["contexts"]["contexts_qdev"]["rows"] == 1


def test_check_gates_verify_freeze_reads_lock_and_checks_inputs(tmp_path, monkeypatch):
    module = _load_check_gates()
    lock = json.loads(module.LOCK_PATH.read_text(encoding="utf-8"))
    lock_path = tmp_path / "gates.lock"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(module, "LOCK_PATH", lock_path)
    monkeypatch.setattr(module, "HASHES_PATH", tmp_path / "missing-hashes.json")

    assert module.verify_freeze() == 0

    lock["contexts"]["contexts_qdev"]["sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    assert module.verify_freeze() == 1


def test_build_data_refuses_frozen_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "GATES_PATH", _frozen_gates(tmp_path))
    assert build_data.main([]) == 2
