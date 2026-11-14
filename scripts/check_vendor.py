#!/usr/bin/env python3
"""Verify vendored Project 7 files (script F0.6).

Default mode: recompute each vendored file's SHA-256 against
``git show <pinned sha>:<path>`` in the Project 7 checkout and against
PROVENANCE.json. ``--offline`` (CI, no P7 checkout): compare the committed
vendor files with PROVENANCE.json only. Exit 1 on any mismatch.

Usage:
    python scripts/check_vendor.py [--p7-dir <path>] [--offline]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "docmind_ft" / "vendor"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_show(p7_dir: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(p7_dir), "show", f"{commit}:{path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"git show failed for {commit}:{path}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p7-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    prov_path = VENDOR_DIR / "PROVENANCE.json"
    if not prov_path.exists():
        print("FAIL: PROVENANCE.json missing — run vendor_p7.py first")
        return 1
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    commit = provenance["p7_commit"]
    failures = 0
    for dest, info in sorted(provenance["files"].items()):
        target = VENDOR_DIR / dest
        if not target.exists():
            print(f"FAIL: {dest} missing from vendor/")
            failures += 1
            continue
        local_sha = sha256_file(target)
        if local_sha != info["sha256"]:
            print(f"FAIL: {dest} differs from PROVENANCE.json")
            failures += 1
            continue
        if not args.offline:
            p7_dir = Path(
                args.p7_dir
                or os.environ.get("DOCMIND_P7_DIR")
                or (REPO_ROOT.parent / "Project-7-DocMind-Service")
            )
            git_sha = hashlib.sha256(git_show(p7_dir, commit, info["source"])).hexdigest()
            if git_sha != info["sha256"]:
                print(f"FAIL: {dest} differs from git {commit[:12]}:{info['source']}")
                failures += 1
                continue
        print(f"OK: {dest} ({info['source']})")
    mode = "offline" if args.offline else f"git {commit[:12]}"
    if failures:
        print(f"vendor check FAILED ({failures} mismatch(es), mode={mode})")
        return 1
    print(f"vendor check PASSED ({len(provenance['files'])} files, mode={mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
