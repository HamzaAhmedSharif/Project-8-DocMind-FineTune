#!/usr/bin/env python3
"""Gate registry checks (script 3.11).

Modes:
  --schema-only    validate evaluation/gates.json shape (all I/TH/HY entries
                   present with metric/op/value or thresholds). Used by
                   check.ps1 and CI from F0.
  --verify-freeze  confirm gates.json still matches the frozen blob hash in
                   reports/hashes.json (from the F2.7 freeze on).
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


def verify_freeze() -> int:
    gates = load_gates()
    if not gates.get("meta", {}).get("frozen_at"):
        print("gates NOT frozen (meta.frozen_at is null)")
        return 1
    if not HASHES_PATH.exists():
        print("FAIL: reports/hashes.json missing — cannot verify freeze")
        return 1
    hashes = json.loads(HASHES_PATH.read_text(encoding="utf-8"))
    frozen = hashes.get("gates.json", {}).get("frozen_blob_sha256")
    if not frozen:
        print("FAIL: no frozen gates.json hash recorded in hashes.json")
        return 1
    current = hashlib.sha256(GATES_PATH.read_bytes()).hexdigest()
    if current != frozen:
        print("FAIL: gates.json differs from the frozen hash")
        return 1
    print("gates frozen hash OK")
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
