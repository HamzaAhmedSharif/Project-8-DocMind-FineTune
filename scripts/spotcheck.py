#!/usr/bin/env python3
"""Spot re-derivation (protocol V step 2, script 1.5).

Recomputes three numbers from the milestone report by a different code path
than the one that produced them, and compares with the anchors recorded in
``reports/F<n>.md`` under ``## Spot-check anchors`` (``key: value`` lines).
Exit 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_anchors(milestone: str) -> dict[str, str]:
    report = REPO_ROOT / "reports" / f"{milestone}.md"
    text = report.read_text(encoding="utf-8")
    section = text.split("## Spot-check anchors", 1)[1]
    anchors = {}
    for line in section.splitlines():
        match = re.match(r"^([a-z_]+):\s*(.+?)\s*$", line)
        if match:
            anchors[match.group(1)] = match.group(2)
    return anchors


def count_vendor_files() -> int:
    vendor = REPO_ROOT / "docmind_ft" / "vendor"
    return sum(
        1
        for entry in vendor.iterdir()
        if entry.is_file() and entry.suffix == ".py" and entry.name != "__init__.py"
    )


def count_pinned_requirements() -> int:
    req = REPO_ROOT / "requirements.txt"
    return sum(
        1
        for line in req.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def count_collected_tests() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return sum(1 for line in result.stdout.splitlines() if "::" in line and "test" in line)


CHECKS_F0 = {
    "vendor_file_count": count_vendor_files,
    "requirements_pinned_count": count_pinned_requirements,
    "tests_collected": count_collected_tests,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone", required=True)
    args = parser.parse_args()
    if args.milestone != "F0":
        print(f"spotcheck: milestone {args.milestone} not implemented yet")
        return 2
    anchors = parse_anchors(args.milestone)
    failures = 0
    for name, fn in CHECKS_F0.items():
        recomputed = fn()
        anchor = anchors.get(name)
        match = anchor is not None and int(anchor) == recomputed
        print(f"{name}: report={anchor} recomputed={recomputed} match={match}")
        if not match:
            failures += 1
    # Cross-check: provenance entries must equal the vendor file count.
    prov = json.loads(
        (REPO_ROOT / "docmind_ft" / "vendor" / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    n_prov = len(prov["files"])
    n_disk = count_vendor_files()
    print(f"provenance entries={n_prov} vendor files on disk={n_disk}")
    if n_prov != n_disk:
        failures += 1
    if failures:
        print(f"spotcheck FAILED ({failures} mismatch(es))")
        return 1
    print("spotcheck PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
