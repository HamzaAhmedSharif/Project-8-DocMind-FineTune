#!/usr/bin/env python3
"""Gate registry checks (script 3.11).

Modes:
  --schema-only    validate evaluation/gates.json shape (all I/TH/HY entries
                   present with metric/op/value or thresholds). Used by
                   check.ps1.
  --verify-freeze  verify gates.json and the frozen inputs against
                   evaluation/gates.lock (fallback: reports/hashes.json).
                   Used by CI and verify.ps1 -Milestone F2.
  --hypotheses     evaluate HY1-HY6 from evaluation/results/*.json (F5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GATES_PATH = REPO_ROOT / "evaluation" / "gates.json"
HASHES_PATH = REPO_ROOT / "reports" / "hashes.json"
LOCK_PATH = REPO_ROOT / "evaluation" / "gates.lock"

CONTEXT_NAMES = ("contexts_qdev", "contexts_qtest", "contexts_hold", "contexts_paper16")
SPLIT_FILES = {
    "qdev": ("evaluation/splits/qdev.json", "qdev_sha256"),
    "qtest": ("evaluation/splits/qtest.json", "qtest_sha256"),
    "holdout": ("evaluation/holdout/holdout_questions.jsonl", "holdout_sha256"),
}

REQUIRED = {
    "integrity": ["I1", "I2", "I3", "I4", "I5", "I6", "I7"],
    "training": ["TH1", "TH2", "TH3", "TH4"],
    "hypotheses": ["HY1", "HY2", "HY3", "HY4", "HY5", "HY6"],
}


def load_gates() -> dict:
    if not GATES_PATH.exists():
        raise SystemExit(f"FAIL: {GATES_PATH} missing")
    return json.loads(GATES_PATH.read_text(encoding="utf-8"))


def schema_only() -> int:
    gates = load_gates()
    failures = 0
    if "meta" not in gates or "version" not in gates.get("meta", {}):
        print("FAIL: meta.version missing")
        failures += 1
    for section, ids in REQUIRED.items():
        block = gates.get(section, {})
        for gate_id in ids:
            entry = block.get(gate_id)
            if not isinstance(entry, dict) or not entry:
                print(f"FAIL: {section}.{gate_id} missing or empty")
                failures += 1
                continue
            if section in ("integrity", "training"):
                for key in ("metric", "op", "value"):
                    if key not in entry:
                        print(f"FAIL: {section}.{gate_id} missing '{key}'")
                        failures += 1
    if failures:
        print(f"gates schema FAILED ({failures} problem(s))")
        return 1
    print("gates schema OK (I1-I7, TH1-TH4, HY1-HY6 present)")
    return 0


def _verify_file(label: str, record: dict) -> int:
    path_value = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        print(f"FAIL: {label}: no path recorded")
        return 1
    if not isinstance(expected, str) or not expected:
        print(f"FAIL: {label}: no sha256 recorded")
        return 1
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        print(f"FAIL: {label}: missing {path}")
        return 1
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        print(f"FAIL: {label}: expected {expected}, got {actual}")
        return 1
    print(f"{label} hash OK")
    return 0


def verify_freeze() -> int:
    gates = load_gates()
    if not gates.get("meta", {}).get("frozen_at"):
        print("gates NOT frozen (meta.frozen_at is null)")
        return 1
    record_path = LOCK_PATH if LOCK_PATH.is_file() else HASHES_PATH
    if not record_path.is_file():
        print("FAIL: evaluation/gates.lock and reports/hashes.json are both missing")
        return 1
    try:
        records = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read {record_path}: {exc}")
        return 1
    if not isinstance(records, dict):
        print(f"FAIL: {record_path}: expected a JSON object")
        return 1

    failures = 0
    gate_record = records.get("gates.json")
    frozen = gate_record.get("frozen_blob_sha256") if isinstance(gate_record, dict) else None
    if not frozen:
        print(f"FAIL: no gates.json frozen hash recorded in {record_path}")
        failures += 1
    else:
        current = hashlib.sha256(GATES_PATH.read_bytes()).hexdigest()
        if current != frozen:
            print("FAIL: gates.json differs from the frozen hash")
            failures += 1
        else:
            print("gates frozen hash OK")

    contexts = records.get("contexts")
    if not isinstance(contexts, dict):
        print(f"FAIL: {record_path}: no contexts hash table")
        failures += 1
    else:
        for name in CONTEXT_NAMES:
            entry = contexts.get(name)
            if not isinstance(entry, dict):
                print(f"FAIL: contexts.{name} missing")
                failures += 1
            else:
                failures += _verify_file(f"contexts.{name}", entry)

    splits = records.get("splits")
    if not isinstance(splits, dict):
        splits = {}
    for name, (path, key) in SPLIT_FILES.items():
        entry = splits.get(name)
        if not isinstance(entry, dict):
            entry = {"path": path, "sha256": gates.get("meta", {}).get(key)}
        failures += _verify_file(f"splits.{name}", entry)

    if failures:
        print(f"gates freeze + input hashes FAILED ({failures} problem(s))")
        return 1
    print("gates freeze + context/split hashes OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--schema-only", action="store_true")
    group.add_argument("--verify-freeze", action="store_true")
    group.add_argument("--hypotheses", action="store_true")
    args = parser.parse_args()
    if args.schema_only:
        return schema_only()
    if args.verify_freeze:
        return verify_freeze()
    print("hypotheses evaluation is implemented in F5")
    return 2


if __name__ == "__main__":
    sys.exit(main())
