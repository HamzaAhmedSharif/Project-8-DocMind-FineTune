#!/usr/bin/env python3
"""Vendor Project 7 modules into docmind_ft/vendor/ (script F0.6).

Copies the pinned Project 7 commit's files byte-exactly (via
``git show <sha>:<path>`` so CRLF conversion cannot alter them) and writes
PROVENANCE.json with the commit, per-file SHA-256 and date. Never modifies
Project 7 (read-only git commands only).

Usage:
    python scripts/vendor_p7.py --p7-dir <path> --commit <sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "docmind_ft" / "vendor"

# Project 7 path -> vendored file name (script 3.12, extended by a 1.3
# decision: the seven listed files import docmind_service.domain,
# docmind_service.config and docmind_service.metrics, so those three are
# vendored too — without them the listed files cannot be imported at all.
# See DECISIONS.md "F0 vendor list extended".)
FILES = {
    "docmind_service/chunking.py": "chunking.py",
    "docmind_service/query.py": "query.py",
    "docmind_service/guardrails.py": "guardrails.py",
    "docmind_service/llm.py": "llm.py",
    "docmind_service/ingest.py": "ingest.py",
    "docmind_service/domain.py": "domain.py",
    "docmind_service/config.py": "config.py",
    "docmind_service/metrics.py": "metrics.py",
    "evaluation/scoring.py": "scoring.py",
    "evaluation/ledger.py": "ledger.py",
}


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
    parser.add_argument("--p7-dir", required=True, help="Project 7 checkout path")
    parser.add_argument("--commit", required=True, help="Pinned P7 commit SHA")
    args = parser.parse_args()
    p7_dir = Path(args.p7_dir)
    if not (p7_dir / ".git").exists():
        raise SystemExit(f"not a git checkout: {p7_dir}")

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    provenance: dict = {
        "p7_repo": "HamzaAhmedSharif/Project-7-DocMind-Service",
        "p7_commit": args.commit,
        "vendored_at": datetime.now(UTC).isoformat(),
        "files": {},
    }
    for source, dest in FILES.items():
        data = git_show(p7_dir, args.commit, source)
        target = VENDOR_DIR / dest
        target.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        provenance["files"][dest] = {"source": source, "sha256": sha}
        print(f"{dest}: {len(data)} bytes sha256={sha[:16]}...")

    (VENDOR_DIR / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"vendored {len(FILES)} files from {args.commit[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
